#!/usr/bin/env node

// Generic web image scraper for training data collection.
//
// Scrapes images from any web page using Playwright. Supports:
// - Single page: downloads all images matching a selector
// - Multi-page: follows pagination links or chapter-list links
// - Custom selectors for images and navigation
//
// Usage:
//   node scrape.mjs <url> [options]
//   node scrape.mjs --file=urls.txt [options]
//
// Options:
//   --output=<dir>       Output directory (default: ./dataset)
//   --file=<path>        Read URLs from file (one per line, # comments ok)
//   --selector=<css>     Image selector (default: img)
//   --links=<css>        Follow these links as sub-pages (e.g. chapter links)
//   --scroll             Scroll page to trigger lazy loading (default: false)
//   --delay=<ms>         Delay between pages (default: 2000)
//   --concurrency=<n>    Parallel image downloads (default: 3)
//   --headless           Run browser headless (default: true)
//   --min-width=<px>     Skip images smaller than this (default: 200)
//   --min-height=<px>    Skip images smaller than this (default: 200)
//   --dry-run            List images without downloading
//   --adapter=<path>     Load a custom adapter module (exports getChapters, getImages)

import { chromium } from "playwright";
import { mkdirSync, readdirSync, existsSync, readFileSync } from "node:fs";
import { writeFile } from "node:fs/promises";
import { join } from "node:path";

// ─── CLI parsing ──────────────────────────────────────────────────────────────

function parseArgs(argv) {
  const opts = {
    urls: [],
    output: join(import.meta.dirname, "dataset"),
    selector: "img",
    links: null,
    scroll: false,
    dryRun: false,
    delay: 2000,
    concurrency: 3,
    headless: true,
    minWidth: 200,
    minHeight: 200,
    adapter: null,
  };

  for (const arg of argv) {
    if (arg.startsWith("--output=")) opts.output = arg.split("=")[1];
    else if (arg.startsWith("--file=")) {
      const lines = readFileSync(arg.split("=")[1], "utf8")
        .split("\n")
        .map((l) => l.trim())
        .filter((l) => l && !l.startsWith("#"));
      opts.urls.push(...lines);
    } else if (arg.startsWith("--selector=")) opts.selector = arg.split("=").slice(1).join("=");
    else if (arg.startsWith("--links=")) opts.links = arg.split("=").slice(1).join("=");
    else if (arg === "--scroll") opts.scroll = true;
    else if (arg === "--dry-run") opts.dryRun = true;
    else if (arg === "--headless") opts.headless = true;
    else if (arg === "--no-headless") opts.headless = false;
    else if (arg.startsWith("--delay=")) opts.delay = parseInt(arg.split("=")[1]);
    else if (arg.startsWith("--concurrency=")) opts.concurrency = parseInt(arg.split("=")[1]);
    else if (arg.startsWith("--min-width=")) opts.minWidth = parseInt(arg.split("=")[1]);
    else if (arg.startsWith("--min-height=")) opts.minHeight = parseInt(arg.split("=")[1]);
    else if (arg.startsWith("--adapter=")) opts.adapter = arg.split("=")[1];
    else if (arg.startsWith("http")) opts.urls.push(arg);
  }

  return opts;
}

function slugify(url) {
  try {
    const u = new URL(url);
    return u.pathname.replace(/^\//, "").replace(/\/$/, "").replace(/[^a-z0-9]+/gi, "-") || u.hostname;
  } catch {
    return "unknown";
  }
}

// ─── Scroll to load lazy images ───────────────────────────────────────────────

async function autoScroll(page) {
  await page.evaluate(async () => {
    await new Promise((resolve) => {
      let total = 0;
      const distance = 500;
      const timer = setInterval(() => {
        window.scrollBy(0, distance);
        total += distance;
        if (total >= document.body.scrollHeight) {
          clearInterval(timer);
          resolve();
        }
      }, 200);
    });
  });
  await page.waitForTimeout(1000);
}

// ─── Extract images from page ─────────────────────────────────────────────────

async function extractImages(page, selector, minWidth, minHeight) {
  return page.evaluate(
    ({ sel, mw, mh }) => {
      const imgs = [];
      for (const el of document.querySelectorAll(sel)) {
        // Get the best URL: data-src (lazy) > src
        const url = el.dataset?.src || el.dataset?.lazySrc || el.src;
        if (!url || !url.startsWith("http")) continue;
        // Filter small images (icons, avatars)
        const w = el.naturalWidth || el.width || 0;
        const h = el.naturalHeight || el.height || 0;
        if (w > 0 && w < mw) continue;
        if (h > 0 && h < mh) continue;
        imgs.push(url);
      }
      return imgs;
    },
    { sel: selector, mw: minWidth, mh: minHeight },
  );
}

// ─── Extract links from page ──────────────────────────────────────────────────

async function extractLinks(page, linkSelector) {
  return page.evaluate((sel) => {
    return [...document.querySelectorAll(sel)]
      .map((a) => a.href)
      .filter((h) => h && h.startsWith("http"));
  }, linkSelector);
}

// ─── Download ─────────────────────────────────────────────────────────────────

async function downloadImage(url, dest, referer, cookies) {
  if (existsSync(dest)) return "skip";
  const headers = {
    Referer: referer,
    "User-Agent":
      "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
  };
  if (cookies) headers.Cookie = cookies;

  const resp = await fetch(url, { headers });
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  await writeFile(dest, Buffer.from(await resp.arrayBuffer()));
  return "ok";
}

async function downloadBatch(items, referer, concurrency, cookies) {
  let ok = 0,
    skipped = 0,
    failed = 0;

  for (let i = 0; i < items.length; i += concurrency) {
    const chunk = items.slice(i, i + concurrency);
    const results = await Promise.allSettled(
      chunk.map(({ url, dest }) => downloadImage(url, dest, referer, cookies)),
    );
    for (const r of results) {
      if (r.status === "fulfilled") (r.value === "skip" ? skipped++ : ok++);
      else failed++;
    }
  }

  return { ok, skipped, failed };
}

async function getCookieString(ctx, url) {
  const cookies = await ctx.cookies(url);
  return cookies.map((c) => `${c.name}=${c.value}`).join("; ");
}

// ─── Main ─────────────────────────────────────────────────────────────────────

const opts = parseArgs(process.argv.slice(2));

if (opts.urls.length === 0) {
  console.log(`Usage: node scrape.mjs <url> [options]

Options:
  --output=<dir>       Output directory (default: ./dataset)
  --file=<path>        Read URLs from file (one per line)
  --selector=<css>     Image CSS selector (default: img)
  --links=<css>        Follow links matching this selector as sub-pages
  --scroll             Scroll page to trigger lazy loading
  --delay=<ms>         Delay between pages (default: 2000)
  --concurrency=<n>    Parallel downloads (default: 3)
  --headless           Run headless (default)
  --no-headless        Run headed (for debugging / Turnstile)
  --min-width=<px>     Skip images narrower than this (default: 200)
  --min-height=<px>    Skip images shorter than this (default: 200)
  --adapter=<path>     Custom adapter module (exports getChapters, getImages)
  --dry-run            List images without downloading`);
  process.exit(0);
}

// Optional custom adapter
let adapter = null;
if (opts.adapter) {
  adapter = await import(opts.adapter);
}

const browser = await chromium.launch({ headless: opts.headless });
const ctx = await browser.newContext({
  userAgent:
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
  locale: "en-US",
  viewport: { width: 1440, height: 900 },
});
const page = await ctx.newPage();

let totalImages = 0;

for (const url of opts.urls) {
  const slug = slugify(url);
  console.log(`\n${"=".repeat(60)}`);
  console.log(slug);
  console.log("=".repeat(60));

  // Determine sub-pages (chapters/galleries)
  let subPages;
  if (adapter?.getChapters) {
    subPages = await adapter.getChapters(page, url);
  } else if (opts.links) {
    await page.goto(url, { waitUntil: "domcontentloaded", timeout: 30000 });
    if (opts.scroll) await autoScroll(page);
    subPages = await extractLinks(page, opts.links);
  } else {
    subPages = [url]; // Single page mode
  }

  console.log(`${subPages.length} pages to scrape`);

  if (opts.dryRun) {
    for (const p of subPages) console.log(`  ${p}`);
    continue;
  }

  const baseDir = join(opts.output, slug);
  mkdirSync(baseDir, { recursive: true });

  for (let i = 0; i < subPages.length; i++) {
    const pageUrl = subPages[i];
    const pageSlug = String(i + 1).padStart(4, "0");
    const pageDir = subPages.length > 1 ? join(baseDir, pageSlug) : baseDir;

    // Skip if already scraped — check BEFORE creating the directory.
    if (subPages.length > 1 && existsSync(pageDir)) {
      const files = readdirSync(pageDir).filter((f) => /\.(jpg|png|webp|gif)$/i.test(f));
      if (files.length > 0) {
        console.log(`  page ${i + 1}/${subPages.length} — ${files.length} files, skipped`);
        continue;
      }
    }
    mkdirSync(pageDir, { recursive: true });

    process.stdout.write(`  page ${i + 1}/${subPages.length}...`);

    try {
      let images;
      if (adapter?.getImages) {
        images = await adapter.getImages(page, pageUrl);
      } else {
        await page.goto(pageUrl, { waitUntil: "domcontentloaded", timeout: 30000 });
        if (opts.scroll) await autoScroll(page);
        images = await extractImages(page, opts.selector, opts.minWidth, opts.minHeight);
      }

      const cookies = await getCookieString(ctx, pageUrl);
      const downloads = images.map((imgUrl, idx) => {
        const ext = imgUrl.match(/\.(jpg|png|webp|gif)/i)?.[1] || "jpg";
        return { url: imgUrl, dest: join(pageDir, `${String(idx + 1).padStart(3, "0")}.${ext}`) };
      });

      const { ok, skipped, failed } = await downloadBatch(downloads, pageUrl, opts.concurrency, cookies);
      totalImages += ok;
      console.log(` ${images.length} imgs (${ok} new, ${skipped} skip, ${failed} fail)`);
    } catch (e) {
      console.log(` ERROR: ${e.message}`);
    }

    if (i < subPages.length - 1) await page.waitForTimeout(opts.delay);
  }
}

await ctx.close();
console.log(`\nDone. ${totalImages} images downloaded.`);
