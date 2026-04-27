#!/usr/bin/env node

// General-purpose web scraper for training data collection.
//
// Usage:
//   node scrape.mjs <url> [url2 ...] [options]
//   node scrape.mjs --file=urls.txt [options]
//
// Options:
//   --output=<dir>     Output directory (default: ./dataset)
//   --file=<path>      Read URLs from file (one per line, # comments ok)
//   --dry-run          List chapters without downloading
//   --skip-existing    Skip chapters that already have files (default: true)
//   --delay=<ms>       Delay between chapters (default: 2000)
//   --concurrency=<n>  Parallel image downloads per chapter (default: 3)
//   --headless         Run browser headless (default: headed for Turnstile)
//   --chapters=<range> Chapter range, e.g. "1-10" or "5-" or "-20"

import { chromium } from "playwright";
import { mkdirSync, readdirSync, existsSync, readFileSync } from "node:fs";
import { writeFile } from "node:fs/promises";
import { join } from "node:path";
import { findAdapter } from "./adapters.mjs";

// ─── CLI parsing ──────────────────────────────────────────────────────────────

function parseArgs(argv) {
  const opts = {
    urls: [],
    output: join(import.meta.dirname, "dataset"),
    dryRun: false,
    skipExisting: true,
    delay: 2000,
    concurrency: 3,
    headless: false,
    chapters: null,
  };

  for (const arg of argv) {
    if (arg.startsWith("--output=")) opts.output = arg.split("=")[1];
    else if (arg.startsWith("--file=")) {
      const lines = readFileSync(arg.split("=")[1], "utf8")
        .split("\n")
        .map((l) => l.trim())
        .filter((l) => l && !l.startsWith("#"));
      opts.urls.push(...lines);
    } else if (arg === "--dry-run") opts.dryRun = true;
    else if (arg === "--headless") opts.headless = true;
    else if (arg.startsWith("--delay=")) opts.delay = parseInt(arg.split("=")[1]);
    else if (arg.startsWith("--concurrency=")) opts.concurrency = parseInt(arg.split("=")[1]);
    else if (arg.startsWith("--chapters=")) opts.chapters = arg.split("=")[1];
    else if (arg.startsWith("http")) opts.urls.push(arg);
  }

  return opts;
}

function parseChapterRange(range) {
  if (!range) return null;
  const [start, end] = range.split("-").map((s) => (s ? parseFloat(s) : null));
  return { start, end };
}

function slugify(url) {
  const match = url.match(/\/manga\/([^/]+)/) || url.match(/\/serie\/([^/]+)/) || url.match(/\/webtoon\/([^/]+)/);
  return match?.[1]?.replace(/\/$/, "") || "unknown";
}

function chapterNum(url) {
  // "chapter-31-5" → "31.5", "chapter-90-net-..." → "90"
  // "chap-01-102" → "1", "chap-0-18" → "0"
  const m = url.match(/chap(?:ter)?-0*(\d+)(?:[.-](\d)(?=-|$|\/))?/);
  if (!m) return "0";
  return m[2] ? `${m[1]}.${m[2]}` : m[1];
}

// Deduplicate and sort chapters numerically by chapter number
function dedupeAndSort(chapters) {
  const seen = new Map(); // chapterNum → url
  for (const url of chapters) {
    const num = chapterNum(url);
    if (!seen.has(num)) seen.set(num, url);
  }
  return [...seen.entries()]
    .sort((a, b) => parseFloat(a[0]) - parseFloat(b[0]))
    .map(([, url]) => url);
}

function inRange(num, range) {
  if (!range) return true;
  const n = parseFloat(num);
  if (range.start != null && n < range.start) return false;
  if (range.end != null && n > range.end) return false;
  return true;
}

// ─── Download ─────────────────────────────────────────────────────────────────

async function downloadImage(url, dest, referer, cookies) {
  if (existsSync(dest)) return "skip";
  const headers = {
    Referer: referer,
    "User-Agent":
      "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
  };
  // Pass browser cookies for sites that require them (Cloudflare cf_clearance)
  if (cookies) headers.Cookie = cookies;

  const resp = await fetch(url, { headers });
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  await writeFile(dest, Buffer.from(await resp.arrayBuffer()));
  return "ok";
}

async function downloadBatch(items, referer, concurrency, cookies) {
  let ok = 0;
  let skipped = 0;
  let failed = 0;

  for (let i = 0; i < items.length; i += concurrency) {
    const chunk = items.slice(i, i + concurrency);
    const results = await Promise.allSettled(
      chunk.map(async ({ url, dest }) => downloadImage(url, dest, referer, cookies)),
    );
    for (const r of results) {
      if (r.status === "fulfilled") {
        r.value === "skip" ? skipped++ : ok++;
      } else {
        failed++;
      }
    }
  }

  return { ok, skipped, failed };
}

// ─── Cookie helper ────────────────────────────────────────────────────────────

async function getCookieString(ctx, url) {
  const cookies = await ctx.cookies(url);
  return cookies.map((c) => `${c.name}=${c.value}`).join("; ");
}

// ─── Main ─────────────────────────────────────────────────────────────────────

const opts = parseArgs(process.argv.slice(2));

if (opts.urls.length === 0) {
  console.log(`Usage: node scrape.mjs <url> [url2 ...] [options]

Options:
  --output=<dir>       Output directory (default: ./dataset)
  --file=<path>        Read URLs from file (one per line)
  --dry-run            List chapters without downloading
  --delay=<ms>         Delay between chapters (default: 2000)
  --concurrency=<n>    Parallel downloads per chapter (default: 3)
  --headless           Run headless (default: headed)
  --chapters=<range>   e.g. "1-10", "5-", "-20"

Supports any site via CSS selectors or custom adapters`);
  process.exit(0);
}

const range = parseChapterRange(opts.chapters);

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
  const adapter = findAdapter(url);

  console.log(`\n${"=".repeat(60)}`);
  console.log(`${slug}`);
  console.log(`${"=".repeat(60)}`);

  const titleDir = join(opts.output, slug);
  mkdirSync(titleDir, { recursive: true });

  const rawChapters = await adapter.getChapters(page, url);
  const chapters = dedupeAndSort(rawChapters);
  console.log(`${chapters.length} chapters found (${rawChapters.length} raw)`);

  if (opts.dryRun) {
    for (const ch of chapters) {
      const num = chapterNum(ch);
      const mark = inRange(num, range) ? "+" : "-";
      console.log(`  [${mark}] ch${num}`);
    }
    continue;
  }

  for (let i = 0; i < chapters.length; i++) {
    const chUrl = chapters[i];
    const num = chapterNum(chUrl);

    if (!inRange(num, range)) continue;

    const chDir = join(titleDir, `ch${num.padStart(4, "0")}`);

    // Skip if already scraped
    if (opts.skipExisting && existsSync(chDir)) {
      const files = readdirSync(chDir).filter((f) => /\.(jpg|png|webp|gif)$/i.test(f));
      if (files.length > 0) {
        console.log(`  ch${num} — ${files.length} files, skipped`);
        continue;
      }
    }

    mkdirSync(chDir, { recursive: true });
    process.stdout.write(`  ch${num} (${i + 1}/${chapters.length})...`);

    try {
      const images = await adapter.getImages(page, chUrl);

      // Get browser cookies for download requests (needed for CF-protected CDNs)
      const cookies = await getCookieString(ctx, chUrl);

      const downloads = images.map((imgUrl, idx) => {
        const ext = imgUrl.match(/\.(jpg|png|webp|gif)/i)?.[1] || "jpg";
        return { url: imgUrl, dest: join(chDir, `${String(idx + 1).padStart(3, "0")}.${ext}`) };
      });

      const { ok, skipped, failed } = await downloadBatch(downloads, chUrl, opts.concurrency, cookies);
      totalImages += ok;
      console.log(` ${images.length} imgs (${ok} new, ${skipped} skip, ${failed} fail)`);
    } catch (e) {
      console.log(` ERROR: ${e.message}`);
    }

    if (i < chapters.length - 1) await page.waitForTimeout(opts.delay);
  }
}

await ctx.close();
console.log(`\nDone. ${totalImages} images downloaded.`);
