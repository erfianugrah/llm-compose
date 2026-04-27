#!/usr/bin/env python3
"""
Curate images for LoRA training:
1. Filter by min resolution
2. Deduplicate using perceptual hashing
3. Resize/crop to target resolution
4. Output numbered files ready for captioning

Usage:
  python3 curate.py <input_dirs...> --output=<dir> [--target=1024] [--min-size=512] [--max-images=50]
"""

import sys, os, shutil
from pathlib import Path
from PIL import Image
import imagehash

def parse_args():
    args = sys.argv[1:]
    opts = {
        "inputs": [],
        "output": None,
        "target": 1024,
        "min_size": 512,
        "max_images": 60,
        "hash_threshold": 8,  # perceptual hash distance threshold for dedup
    }
    for arg in args:
        if arg.startswith("--output="):
            opts["output"] = arg.split("=", 1)[1]
        elif arg.startswith("--target="):
            opts["target"] = int(arg.split("=")[1])
        elif arg.startswith("--min-size="):
            opts["min_size"] = int(arg.split("=")[1])
        elif arg.startswith("--max-images="):
            opts["max_images"] = int(arg.split("=")[1])
        elif arg.startswith("--hash-threshold="):
            opts["hash_threshold"] = int(arg.split("=")[1])
        else:
            opts["inputs"].append(arg)
    return opts

def collect_images(dirs):
    exts = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
    images = []
    for d in dirs:
        p = Path(d)
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.suffix.lower() in exts:
                    images.append(f)
        elif p.is_file() and p.suffix.lower() in exts:
            images.append(p)
    return images

def filter_and_score(images, min_size):
    """Filter by min resolution and score by quality (prefer larger, squarer images)."""
    scored = []
    for path in images:
        try:
            with Image.open(path) as img:
                w, h = img.size
                if w < min_size or h < min_size:
                    continue
                # Score: prefer larger images, penalize extreme aspect ratios
                aspect = min(w, h) / max(w, h)
                size_score = min(w, h)  # smaller dimension
                score = size_score * (0.5 + 0.5 * aspect)  # penalize thin strips
                scored.append((path, score, w, h, img.mode))
        except Exception:
            continue
    scored.sort(key=lambda x: -x[1])  # best first
    return scored

def deduplicate(scored_images, threshold):
    """Remove perceptual duplicates, keeping highest-scored version."""
    seen_hashes = []
    unique = []
    for path, score, w, h, mode in scored_images:
        try:
            with Image.open(path) as img:
                phash = imagehash.phash(img.convert("RGB").resize((128, 128)))
                is_dup = False
                for existing_hash in seen_hashes:
                    if phash - existing_hash < threshold:
                        is_dup = True
                        break
                if not is_dup:
                    seen_hashes.append(phash)
                    unique.append((path, score, w, h))
        except Exception:
            continue
    return unique

def resize_and_save(images, output_dir, target_size, max_images):
    """Smart crop/resize to target resolution and save."""
    os.makedirs(output_dir, exist_ok=True)
    saved = 0
    for i, (path, score, w, h) in enumerate(images):
        if saved >= max_images:
            break
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                # Smart crop to square-ish, then resize
                # If image is very wide/tall, center crop first
                aspect = w / h
                if aspect > 1.5:
                    # Very wide — crop to center square
                    new_w = int(h * 1.2)
                    left = (w - new_w) // 2
                    img = img.crop((left, 0, left + new_w, h))
                elif aspect < 0.67:
                    # Very tall — crop to center
                    new_h = int(w * 1.2)
                    top = (h - new_h) // 2
                    img = img.crop((0, top, w, top + new_h))

                # Resize to target (maintaining aspect, then pad or just resize)
                img.thumbnail((target_size, target_size), Image.LANCZOS)

                # Save
                out_path = os.path.join(output_dir, f"{saved + 1:04d}.png")
                img.save(out_path, "PNG", quality=95)
                saved += 1
        except Exception as e:
            print(f"  SKIP {path}: {e}")

    return saved

def main():
    opts = parse_args()
    if not opts["inputs"] or not opts["output"]:
        print("Usage: python3 curate.py <input_dirs...> --output=<dir>")
        sys.exit(1)

    print(f"Collecting images from {len(opts['inputs'])} sources...")
    all_images = collect_images(opts["inputs"])
    print(f"  Found {len(all_images)} total images")

    print(f"Filtering (min {opts['min_size']}px)...")
    scored = filter_and_score(all_images, opts["min_size"])
    print(f"  {len(scored)} passed resolution filter")

    print(f"Deduplicating (hash threshold={opts['hash_threshold']})...")
    unique = deduplicate(scored, opts["hash_threshold"])
    print(f"  {len(unique)} unique images")

    print(f"Saving top {opts['max_images']} to {opts['output']}...")
    saved = resize_and_save(unique, opts["output"], opts["target"], opts["max_images"])
    print(f"  Saved {saved} images")
    print("Done.")

if __name__ == "__main__":
    main()
