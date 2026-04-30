#!/usr/bin/env python3
"""Pick N images from a captioned dataset for a focused sub-dataset.

Default strategy: random with caption-quality filter. Rejects captions
that look like wrong-person matches (BLIP-2 hallucinating celebrity
names, describing men/boys when the subject is a young woman, etc).

Strategies:
  random            — random pick after quality filter (default)
  longest-caption   — longest BLIP-2 descriptions (WARNING: biases toward
                      noisy/confusing images that BLIP-2 over-describes)
  shortest-caption  — shortest descriptions (biases toward simple shots)

Usage:
  python3 scripts/pick-focus-subset.py <src> <dst> --n 40 \\
      [--positive-terms "young woman,red hair,freckles"] \\
      [--negative-terms "a man,a boy,lily collins"]

Runs inside lora_train container:
  docker exec lora_train python3 /pick-focus-subset.py \\
      /data/datasets/sophia-clean /data/datasets/sophia-focus --n 40
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

# Default caption-quality filter. BLIP-2 regularly mis-identifies androgynous
# photos of young women as men/boys. These phrases exclude those outright.
DEFAULT_NEGATIVE_TERMS = [
    "a man", "a boy", "a male", "old man", "old woman",
    "a group of people", "two people", "multiple people",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="source dataset directory")
    ap.add_argument("dst", help="destination directory (created if missing)")
    ap.add_argument("--n", type=int, default=40,
                    help="number of images to pick (default: 40)")
    ap.add_argument("--strategy", default="random",
                    choices=["longest-caption", "shortest-caption", "random"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--positive-terms", default="",
                    help="comma-sep — if set, caption must contain at least one")
    ap.add_argument("--negative-terms", default=",".join(DEFAULT_NEGATIVE_TERMS),
                    help="comma-sep — caption must NOT contain any of these")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    if not src.is_dir():
        print(f"error: {src} is not a directory", file=sys.stderr)
        return 1

    positive = [t.strip().lower() for t in args.positive_terms.split(",") if t.strip()]
    negative = [t.strip().lower() for t in args.negative_terms.split(",") if t.strip()]

    # Build (image_stem, caption_length) pairs with quality filtering
    candidates: list[tuple[str, int]] = []
    rejected = 0
    for img in src.iterdir():
        if img.suffix.lower() not in IMAGE_EXTS:
            continue
        cap = img.with_suffix(".txt")
        if not cap.exists():
            continue
        text = cap.read_text(encoding="utf-8", errors="replace").strip().lower()
        if not text:
            continue
        if any(term in text for term in negative):
            rejected += 1
            continue
        if positive and not any(term in text for term in positive):
            rejected += 1
            continue
        candidates.append((img.stem, len(text)))

    if not candidates:
        print(f"error: no (image, caption) pairs found in {src}", file=sys.stderr)
        return 1

    print(f"Pool: {len(candidates)} after quality filter (rejected {rejected})")

    if args.strategy == "longest-caption":
        candidates.sort(key=lambda x: -x[1])
    elif args.strategy == "shortest-caption":
        candidates.sort(key=lambda x: x[1])
    else:
        random.seed(args.seed)
        random.shuffle(candidates)

    picked = [c[0] for c in candidates[:args.n]]
    print(f"Picking {len(picked)} via strategy={args.strategy}")

    if args.dry_run:
        for stem in picked[:10]:
            print(f"  {stem}")
        if len(picked) > 10:
            print(f"  ... and {len(picked) - 10} more")
        return 0

    dst.mkdir(parents=True, exist_ok=True)
    for stem in picked:
        for ext in IMAGE_EXTS:
            src_img = src / f"{stem}{ext}"
            if src_img.exists():
                shutil.copy2(src_img, dst / src_img.name)
                break
        src_cap = src / f"{stem}.txt"
        if src_cap.exists():
            shutil.copy2(src_cap, dst / src_cap.name)

    print(f"Wrote {len(picked)} image+caption pairs to {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
