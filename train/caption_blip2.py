#!/usr/bin/env python3
"""Caption a dataset directory with BLIP-2 natural-language outputs.

BLIP-2 produces short English descriptions that align with Flux's T5-XXL
text encoder, unlike WD14 Danbooru tags. Uses Salesforce/blip2-opt-2.7b
(~5 GB) which works with current transformers (no remote-code headaches
like Florence-2).

Writes <stem>.txt next to each image. With --overwrite, replaces any
existing caption.

Usage (inside the lora-train container):
  python3 /train-hooks/caption_blip2.py /data/datasets/sophia-clean
    [--trigger-word sophia] [--prompt "a photograph of"]
    [--model Salesforce/blip2-opt-2.7b]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", help="directory of images to caption")
    ap.add_argument("--trigger-word", default="",
                    help="prepended to each caption (e.g. 'sophia, ')")
    ap.add_argument("--prompt", default="",
                    help="conditional prompt prefix, e.g. 'a photograph of' "
                         "(leave empty for pure image captioning)")
    ap.add_argument("--model", default="Salesforce/blip2-opt-2.7b",
                    help="HF model id")
    ap.add_argument("--max-new-tokens", type=int, default=60)
    ap.add_argument("--overwrite", action="store_true",
                    help="overwrite existing .txt captions (default: skip)")
    args = ap.parse_args()

    dataset = Path(args.dataset)
    if not dataset.is_dir():
        print(f"error: {dataset} is not a directory", file=sys.stderr)
        return 1

    images = sorted([p for p in dataset.iterdir()
                     if p.suffix.lower() in IMAGE_EXTS])
    print(f"Found {len(images)} images in {dataset}")

    import torch
    from PIL import Image
    from transformers import Blip2ForConditionalGeneration, Blip2Processor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Loading {args.model} on {device}...")

    processor = Blip2Processor.from_pretrained(args.model)
    model = Blip2ForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=dtype
    ).to(device).eval()

    done = 0
    skipped = 0
    for img_path in images:
        cap_path = img_path.with_suffix(".txt")
        if cap_path.exists() and not args.overwrite:
            skipped += 1
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  ✗ {img_path.name}: {e}")
            continue

        if args.prompt:
            inputs = processor(images=img, text=args.prompt,
                               return_tensors="pt").to(device, dtype)
        else:
            inputs = processor(images=img, return_tensors="pt").to(device, dtype)

        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                num_beams=5,
                do_sample=False,
            )
        caption = processor.decode(out[0], skip_special_tokens=True).strip()
        # Strip the echoed conditional prompt if BLIP-2 returned it verbatim
        if args.prompt and caption.lower().startswith(args.prompt.lower()):
            caption = caption[len(args.prompt):].strip()

        if args.trigger_word:
            caption = f"{args.trigger_word}, {caption}"

        cap_path.write_text(caption, encoding="utf-8")
        done += 1
        if done % 25 == 0:
            print(f"  [{done}/{len(images)-skipped}] {img_path.name}: {caption[:80]}...")

    print(f"\nDone: {done} captioned, {skipped} skipped (existing captions)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
