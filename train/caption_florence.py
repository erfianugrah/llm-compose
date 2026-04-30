#!/usr/bin/env python3
"""Caption a dataset directory with Florence-2-large natural-language outputs.

Florence-2 is Microsoft's vision-language model (~1.5 GB). It produces
proper English descriptions that align with Flux's T5-XXL text encoder,
unlike WD14 Danbooru tags which T5 was never trained on.

Writes <stem>.txt next to each image, overwriting any existing caption.

Usage (inside the lora-train container, which has PyTorch + CUDA):
  python3 /train-hooks/caption_florence.py /data/datasets/sophia-clean
    [--trigger-word sophia] [--task MORE_DETAILED_CAPTION]
    [--model microsoft/Florence-2-large]

Tasks:
  CAPTION                Short caption (~1 sentence)
  DETAILED_CAPTION       Medium (~2-3 sentences)
  MORE_DETAILED_CAPTION  Long (~5-7 sentences) — recommended for training
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
    ap.add_argument("--task", default="MORE_DETAILED_CAPTION",
                    choices=["CAPTION", "DETAILED_CAPTION", "MORE_DETAILED_CAPTION"])
    ap.add_argument("--model", default="microsoft/Florence-2-large")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="batch of images per forward pass (Florence-2 doesn't batch well; keep 1)")
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

    # Lazy import — model download + CUDA init takes time
    import torch
    from PIL import Image
    from transformers import AutoModelForCausalLM, AutoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Loading {args.model} on {device}...")

    # attn_implementation='eager' bypasses the _supports_sdpa check that breaks
    # on transformers>=4.50 with Florence-2's pinned remote code (known issue).
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, trust_remote_code=True,
        attn_implementation="eager",
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    prompt_token = f"<{args.task}>"
    print(f"Task prompt: {prompt_token}")

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

        inputs = processor(text=prompt_token, images=img, return_tensors="pt").to(device, dtype)
        with torch.inference_mode():
            gen = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
                do_sample=False,
            )
        raw = processor.batch_decode(gen, skip_special_tokens=False)[0]
        parsed = processor.post_process_generation(
            raw, task=prompt_token, image_size=(img.width, img.height)
        )
        caption = parsed.get(prompt_token, "").strip()

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
