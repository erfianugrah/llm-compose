#!/usr/bin/env node

// Image search scraper — downloads images for a search query.
// Uses Google Images via Playwright (scrolls to load more, extracts full-res URLs).
//
// Usage:
//   node image-search.mjs "search query" [options]
//
// Options:
//   --output=<dir>     Output directory (default: ./dataset/images/<query-slug>)
//   --max=<n>          Max images to download (default: 200)
//   --min-width=<px>   Minimum image width (default: 400)
//   --min-height=<px>  Minimum image height (default: 400)
//   --headless         Run headless (default)
//   --no-headless      Run headed (for debugging — also useful when Google
//                      shows a CAPTCHA that needs manual solving)
//   --concurrency=<n>  Parallel downloads (default: 5)

import { chromium } from "playwright";
import { mkdirSync, existsSync } from "node:fs";
import { writeFile } from "node:fs/promises";
import { join, extname } from "node:path";
import { createHash } from "node:crypto";

// ─── CLI ──────────────────────────────────────────────────────────────────────

const args = process.argv.slice(2);
const query = args.find((a) => !a.startsWith("--"));
if (!query) {
  console.log("Usage: node image-search.mjs \"search query\" [--max=200] [--output=dir]");
  process.exit(0);
}

const opts = {
  output: null,
  max: 200,
  minWidth: 400,
  minHeight: 400,
  headless: true,
  concurrency: 5,
};

for (const arg of args) {
  if (arg.startsWith("--output=")) opts.output = arg.split("=")[1];
  else if (arg.startsWith("--max=")) opts.max = parseInt(arg.split("=")[1]);
  else if (arg.startsWith("--min-width=")) opts.minWidth = parseInt(arg.split("=")[1]);
  else if (arg.startsWith("--min-height=")) opts.minHeight = parseInt(arg.split("=")[1]);
  else if (arg.startsWith("--concurrency=")) opts.concurrency = parseInt(arg.split("=")[1]);
  else if (arg === "--headless") opts.headless = true;
  else if (arg === "--no-headless") opts.headless = false;
}

const slug = query.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/-+$/, "");
const outputDir = opts.output || join(import.meta.dirname, "dataset", "images", slug);
mkdirSync(outputDir, { recursive: true });

// ─── Helpers ──────────────────────────────────────────────────────────────────

function urlHash(url) {
  return createHash("md5").update(url).digest("hex").slice(0, 12);
}

function guessExt(url, contentType) {
  if (contentType?.includes("jpeg") || contentType?.includes("jpg")) return ".jpg";
  if (contentType?.includes("png")) return ".png";
  if (contentType?.includes("webp")) return ".webp";
  if (contentType?.includes("gif")) return ".gif";
  const ext = extname(new URL(url).pathname).split("?")[0].toLowerCase();
  if ([".jpg", ".jpeg", ".png", ".webp", ".gif"].includes(ext)) return ext;
  return ".jpg";
}

async function downloadImage(url, dest) {
  if (existsSync(dest)) return "skip";
  try {
    const resp = await fetch(url, {
      headers: {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/136.0.0.0 Safari/537.36",
        Referer: "https://www.google.com/",
      },
      signal: AbortSignal.timeout(15000),
    });
    if (!resp.ok) return "fail";
    const contentType = resp.headers.get("content-type") || "";
    if (!contentType.includes("image")) return "fail";
    const buf = Buffer.from(await resp.arrayBuffer());
    if (buf.length < 5000) return "fail"; // too small, probably thumbnail/error
    const ext = guessExt(url, contentType);
    const finalDest = dest.replace(/\.[^.]+$/, ext);
    await writeFile(finalDest, buf);
    return "ok";
  } catch {
    return "fail";
  }
}

// ─── Google Images scraper ────────────────────────────────────────────────────

async function scrapeGoogleImages(page, searchQuery, maxImages) {
  const url = `https://www.google.com/search?q=${encodeURIComponent(searchQuery)}&tbm=isch&tbs=isz:l`; // isz:l = large images
  await page.goto(url, { waitUntil: "domcontentloaded", timeout: 20000 });
  await page.waitForTimeout(3000);

  // Accept cookies if prompted
  const acceptBtn = await page.$('button:has-text("Accept all")');
  if (acceptBtn) {
    await acceptBtn.click();
    await page.waitForTimeout(2000);
  }

  const imageUrls = new Set();
  let scrollAttempts = 0;
  const maxScrollAttempts = 50;

  while (imageUrls.size < maxImages && scrollAttempts < maxScrollAttempts) {
    // Click on thumbnails to reveal full-res URLs
    const thumbnails = await page.$$('div[data-ri] img, img.YQ4gaf, img.Q4LuWd');
    
    for (const thumb of thumbnails) {
      if (imageUrls.size >= maxImages) break;

      try {
        await thumb.click({ timeout: 2000 });
        await page.waitForTimeout(800);

        // Extract full-res image URL from the side panel
        const fullResUrls = await page.$$eval(
          'img[src^="http"][class*="sFlh5c"], img[src^="http"][class*="iPVvYb"], a[href^="/imgres"] img[src^="http"]',
          (imgs) =>
            imgs
              .map((img) => img.src)
              .filter((src) => src.startsWith("http") && !src.includes("google.com") && !src.includes("gstatic.com")),
        );

        for (const u of fullResUrls) {
          if (!imageUrls.has(u)) {
            imageUrls.add(u);
          }
        }
      } catch {
        // thumbnail click failed, skip
      }
    }

    // Scroll down to load more
    await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
    await page.waitForTimeout(2000);

    // Click "Show more results" if visible
    const showMore = await page.$('input[value="Show more results"], button:has-text("Show more")');
    if (showMore) {
      try {
        await showMore.click();
        await page.waitForTimeout(3000);
      } catch {}
    }

    scrollAttempts++;
    process.stdout.write(`\r  Collected ${imageUrls.size}/${maxImages} URLs (scroll ${scrollAttempts})...`);
  }

  console.log(`\n  Found ${imageUrls.size} unique image URLs`);
  return [...imageUrls];
}

// ─── Alternative: Bing Images (more reliable scraping) ────────────────────────

async function scrapeBingImages(page, searchQuery, maxImages) {
  const url = `https://www.bing.com/images/search?q=${encodeURIComponent(searchQuery)}&qft=+filterui:imagesize-large`;
  await page.goto(url, { waitUntil: "domcontentloaded", timeout: 20000 });
  await page.waitForTimeout(3000);

  // Accept cookies
  const acceptBtn = await page.$('#bnp_btn_accept, button:has-text("Accept")');
  if (acceptBtn) {
    await acceptBtn.click();
    await page.waitForTimeout(2000);
  }

  const imageUrls = new Set();
  let scrollAttempts = 0;

  while (imageUrls.size < maxImages && scrollAttempts < 30) {
    // Bing stores full-res URLs in data attributes
    const urls = await page.$$eval("a.iusc, .imgpt a", (els) =>
      els.map((a) => {
        try {
          const m = a.getAttribute("m");
          if (m) {
            const data = JSON.parse(m);
            return data.murl || "";
          }
          return "";
        } catch {
          return "";
        }
      }).filter((u) => u && u.startsWith("http")),
    );

    for (const u of urls) imageUrls.add(u);

    await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
    await page.waitForTimeout(2000);

    // Click "See more images"
    const seeMore = await page.$('.btn_seemore, a:has-text("See more images")');
    if (seeMore) {
      try {
        await seeMore.click();
        await page.waitForTimeout(3000);
      } catch {}
    }

    scrollAttempts++;
    process.stdout.write(`\r  Collected ${imageUrls.size}/${maxImages} URLs (scroll ${scrollAttempts})...`);
  }

  console.log(`\n  Found ${imageUrls.size} unique image URLs`);
  return [...imageUrls];
}

// ─── Main ─────────────────────────────────────────────────────────────────────

console.log(`Searching for: "${query}"`);
console.log(`Output: ${outputDir}`);
console.log(`Max: ${opts.max} images\n`);

const browser = await chromium.launch({ headless: opts.headless });
const ctx = await browser.newContext({
  userAgent: "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/136.0.0.0 Safari/537.36",
  viewport: { width: 1440, height: 900 },
});
const page = await ctx.newPage();

// Try Google first (better results), fall back to Bing
console.log("Scraping Google Images...");
let urls = await scrapeGoogleImages(page, query, opts.max);

if (urls.length < 20) {
  console.log("Google returned few results, trying Bing Images...");
  urls = await scrapeBingImages(page, query, opts.max);
}

await browser.close();

// Download
console.log(`\nDownloading ${urls.length} images...`);
let ok = 0;
let fail = 0;
let skip = 0;

for (let i = 0; i < urls.length; i += opts.concurrency) {
  const chunk = urls.slice(i, i + opts.concurrency);
  const results = await Promise.allSettled(
    chunk.map(async (url, j) => {
      const idx = i + j + 1;
      const hash = urlHash(url);
      const dest = join(outputDir, `${String(idx).padStart(4, "0")}_${hash}.jpg`);
      return downloadImage(url, dest);
    }),
  );
  for (const r of results) {
    if (r.status === "fulfilled") {
      if (r.value === "ok") ok++;
      else if (r.value === "skip") skip++;
      else fail++;
    } else {
      fail++;
    }
  }
  process.stdout.write(`\r  ${ok + skip + fail}/${urls.length} (${ok} ok, ${skip} skip, ${fail} fail)`);
}

console.log(`\n\nDone. ${ok} images downloaded to ${outputDir}`);
