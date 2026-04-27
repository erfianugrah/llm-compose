#!/usr/bin/env python3
"""Filter sliced panels for LoRA training quality.

Selects panels that:
- Are portrait-ish (height > width, aspect 1.2-2.5)
- Have high content density (lots of non-white, non-black pixels = actual art)
- Are large enough (min 400px on short side)
- Randomly samples N from the best candidates

Copies selected panels to output dir, renumbered.
"""

import sys
import random
import shutil
from pathlib import Path
from PIL import Image
import numpy as np

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("dataset/training/manhwa-panels")
DST = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("dataset/training/manhwa-curated")
N = int(sys.argv[3]) if len(sys.argv) > 3 else 150

MIN_SHORT_SIDE = 350
TARGET_ASPECT_MIN = 1.0   # roughly square to portrait
TARGET_ASPECT_MAX = 2.2
MIN_COLOR_STD = 25        # reject near-monotone panels (text-only, solid backgrounds)


def score_panel(img_path):
    """Return (valid, score) for a panel."""
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception:
        return False, 0
    
    w, h = img.size
    short = min(w, h)
    
    if short < MIN_SHORT_SIDE:
        return False, 0
    
    aspect = h / w
    if aspect < TARGET_ASPECT_MIN or aspect > TARGET_ASPECT_MAX:
        return False, 0
    
    arr = np.array(img)
    
    # Color std — higher = more colorful/detailed art
    color_std = np.std(arr.astype(float))
    if color_std < MIN_COLOR_STD:
        return False, 0
    
    # Prefer panels with moderate color variance (actual illustrated content)
    # and closer to our target aspect ratio (~1.46 = 832/1216 inverted)
    aspect_score = 1.0 - abs(aspect - 1.46) / 1.46
    size_score = min(short / 700, 1.0)  # prefer larger panels
    
    score = color_std * aspect_score * size_score
    return True, score


def main():
    DST.mkdir(parents=True, exist_ok=True)
    
    panels = sorted(SRC.glob("*.png"))
    print(f"Scoring {len(panels)} panels...")
    
    scored = []
    for p in panels:
        valid, score = score_panel(p)
        if valid:
            scored.append((p, score))
    
    print(f"{len(scored)} passed filters")
    
    # Sort by score, take top 2x N, then randomly sample N from those
    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:N * 3]
    selected = random.sample(top, min(N, len(top)))
    selected.sort(key=lambda x: x[0].name)
    
    for i, (src, score) in enumerate(selected):
        dst = DST / f"{i:04d}.png"
        shutil.copy2(src, dst)
    
    print(f"Selected {len(selected)} panels → {DST}")


if __name__ == "__main__":
    main()
