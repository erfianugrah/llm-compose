#!/usr/bin/env python3
"""Slice tall webtoon strips into individual panels for LoRA training.

Strategy:
1. Detect horizontal white/near-white gaps (panel separators) in tall strips
2. Split at those gaps
3. Filter: skip panels that are too small, too narrow, or mostly white/text
4. Resize to max 1024px on longest side
5. Save to output directory
"""

import sys
from pathlib import Path
from PIL import Image
import numpy as np

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("dataset/in-the-summer-engsub")
DST = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("dataset/training/manhwa-panels")

MIN_HEIGHT = 300        # skip panels shorter than this
MAX_ASPECT = 3.0        # skip extremely tall/narrow panels
MIN_CONTENT = 0.15      # min fraction of non-white pixels
MAX_SIZE = 1024          # resize longest side
GAP_THRESHOLD = 245     # pixel value threshold for "white" row
GAP_MIN_ROWS = 5        # minimum consecutive white rows to count as gap


def find_gaps(img_array):
    """Find horizontal gaps (white rows) in image."""
    gray = np.mean(img_array, axis=(1, 2)) if img_array.ndim == 3 else np.mean(img_array, axis=1)
    is_white = gray > GAP_THRESHOLD
    
    gaps = []
    in_gap = False
    gap_start = 0
    
    for i, white in enumerate(is_white):
        if white and not in_gap:
            gap_start = i
            in_gap = True
        elif not white and in_gap:
            if i - gap_start >= GAP_MIN_ROWS:
                gaps.append((gap_start, i))
            in_gap = False
    
    if in_gap and len(is_white) - gap_start >= GAP_MIN_ROWS:
        gaps.append((gap_start, len(is_white)))
    
    return gaps


def content_ratio(img_array):
    """Fraction of non-white pixels."""
    gray = np.mean(img_array, axis=2) if img_array.ndim == 3 else img_array
    return np.mean(gray < GAP_THRESHOLD)


def slice_image(img_path):
    """Slice a tall strip into panels."""
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    
    # Skip if not a tall strip
    if h < 1500 or h / w < 2:
        return [img] if h >= MIN_HEIGHT else []
    
    arr = np.array(img)
    gaps = find_gaps(arr)
    
    if not gaps:
        return [img]
    
    panels = []
    prev_end = 0
    
    for gap_start, gap_end in gaps:
        if gap_start - prev_end >= MIN_HEIGHT:
            panel = img.crop((0, prev_end, w, gap_start))
            panels.append(panel)
        prev_end = gap_end
    
    # Last segment
    if h - prev_end >= MIN_HEIGHT:
        panel = img.crop((0, prev_end, w, h))
        panels.append(panel)
    
    return panels if panels else [img]


def resize_max(img, max_size):
    w, h = img.size
    if max(w, h) <= max_size:
        return img
    scale = max_size / max(w, h)
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def main():
    DST.mkdir(parents=True, exist_ok=True)
    
    images = sorted(SRC.rglob("*.jpg")) + sorted(SRC.rglob("*.png")) + sorted(SRC.rglob("*.webp"))
    print(f"Found {len(images)} source images in {SRC}")
    
    saved = 0
    skipped_small = 0
    skipped_aspect = 0
    skipped_content = 0
    
    for img_path in images:
        try:
            panels = slice_image(img_path)
        except Exception as e:
            print(f"  Error processing {img_path}: {e}")
            continue
        
        for panel in panels:
            w, h = panel.size
            
            if h < MIN_HEIGHT or w < 200:
                skipped_small += 1
                continue
            
            aspect = max(w, h) / min(w, h)
            if aspect > MAX_ASPECT:
                skipped_aspect += 1
                continue
            
            arr = np.array(panel)
            if content_ratio(arr) < MIN_CONTENT:
                skipped_content += 1
                continue
            
            panel = resize_max(panel, MAX_SIZE)
            out_path = DST / f"{saved:04d}.png"
            panel.save(out_path, "PNG")
            saved += 1
    
    print(f"\nDone! Saved {saved} panels to {DST}")
    print(f"Skipped: {skipped_small} too small, {skipped_aspect} bad aspect, {skipped_content} low content")


if __name__ == "__main__":
    main()
