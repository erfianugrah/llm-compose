#!/usr/bin/env python3
"""End-to-end eval runner.

Usage (all routed through proxy — triggers GPU swap automatically):

  # Single shot: one prompt + one stack
  python3 eval/run.py shot --prompt manhwa_stylized --stack face_manhwa_v5 --seed 111

  # Stage comparison: same seed through A=photo, B=manhwa_stylized, C=manhwa_prompt
  python3 eval/run.py stages --seed 111 --stack-b face_manhwa_v5

  # Stack sweep: all style stacks side-by-side (1 seed, fixed prompt)
  python3 eval/run.py sweep --prompt manhwa_stylized --seed 111

  # Seed matrix: N seeds × M stacks (build the identity robustness grid)
  python3 eval/run.py matrix --prompt manhwa_stylized \\
         --stacks face_manhwa_v5,face_manwha_web,face_illust \\
         --seeds 111,222,333,444

  # Checkpoint eval: swap the face LoRA across epochs, fixed seed
  python3 eval/run.py checkpoints --face-prefix <face-lora-prefix> \\
         --epochs 2,4,6,8,10,12 --weight 0.85

The face LoRA default is `face-lora` (a placeholder). Override via
`--face-lora <name>` or set `FACE_LORA` in `eval/presets_local.py`.

Outputs land in ~/docker-volumes/comfyui/output/<filename_prefix>/ with
deterministic filenames encoding prompt/stack/seed/epoch.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import comfyui  # noqa: E402
import presets  # noqa: E402
import workflows  # noqa: E402


def _negative_for(prompt_key: str) -> str:
    return presets.NEG_MANHWA if "manhwa" in prompt_key else presets.NEG_REALISTIC


def _stack(name: str) -> list[tuple[str, float]]:
    if name not in presets.STACKS:
        raise SystemExit(f"Unknown stack: {name}. Known: {', '.join(presets.STACKS)}")
    return presets.STACKS[name]


def _prompt(name: str, stack: str) -> str:
    if name not in presets.PROMPTS:
        raise SystemExit(f"Unknown prompt: {name}. Known: {', '.join(presets.PROMPTS)}")
    base = presets.PROMPTS[name]
    prefix = presets.STACK_PROMPT_PREFIX.get(stack, "")
    return f"{prefix}{base}" if prefix else base


def cmd_shot(args: argparse.Namespace) -> None:
    stack = _stack(args.stack)
    prompt = _prompt(args.prompt, args.stack)
    wf = workflows.txt2img(
        prompt=prompt,
        loras=stack,
        seed=args.seed,
        negative=_negative_for(args.prompt),
        filename_prefix=f"eval/{args.prompt}_{args.stack}",
    )
    print(f"[run] prompt={args.prompt} stack={args.stack} seed={args.seed}")
    print(f"[run] loras={stack}")
    files = comfyui.generate(wf, timeout=args.timeout)
    for f in files:
        print(f"  → {f}")


def cmd_stages(args: argparse.Namespace) -> None:
    """Three-stage comparison on a single seed:
       A = photo realistic        (face_only or face_realism)
       B = manhwa stylized        (the chosen stack)
       C = manhwa prompt-only     (no style LoRA)
    """
    plans = [
        ("A_photo",    args.stack_a, "photo"),
        ("B_stylized", args.stack_b, "manhwa_stylized"),
        ("C_prompt",   "face_only",  "manhwa_prompt"),
    ]
    for label, stack_name, prompt_name in plans:
        stack = _stack(stack_name)
        prompt = _prompt(prompt_name, stack_name)
        wf = workflows.txt2img(
            prompt=prompt,
            loras=stack,
            seed=args.seed,
            negative=_negative_for(prompt_name),
            filename_prefix=f"eval/stages_{args.seed}_{label}_{stack_name}",
        )
        print(f"[stages] {label}  stack={stack_name}  prompt={prompt_name}")
        files = comfyui.generate(wf, timeout=args.timeout)
        for f in files:
            print(f"  → {f}")


def cmd_sweep(args: argparse.Namespace) -> None:
    """Run all STACKS (or a filtered subset) for one prompt/seed."""
    stacks = args.stacks.split(",") if args.stacks else list(presets.STACKS)
    for stack_name in stacks:
        stack = _stack(stack_name)
        prompt = _prompt(args.prompt, stack_name)
        wf = workflows.txt2img(
            prompt=prompt,
            loras=stack,
            seed=args.seed,
            negative=_negative_for(args.prompt),
            filename_prefix=f"eval/sweep_{args.prompt}_{args.seed}_{stack_name}",
        )
        print(f"[sweep] stack={stack_name}")
        files = comfyui.generate(wf, timeout=args.timeout)
        for f in files:
            print(f"  → {f}")


def cmd_matrix(args: argparse.Namespace) -> None:
    """Stacks × seeds grid."""
    stacks = args.stacks.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    total = len(stacks) * len(seeds)
    done = 0
    for stack_name in stacks:
        stack = _stack(stack_name)
        prompt = _prompt(args.prompt, stack_name)
        for seed in seeds:
            done += 1
            wf = workflows.txt2img(
                prompt=prompt,
                loras=stack,
                seed=seed,
                negative=_negative_for(args.prompt),
                filename_prefix=f"eval/matrix_{args.prompt}/{stack_name}_s{seed}",
            )
            print(f"[matrix {done}/{total}] {stack_name} seed={seed}")
            files = comfyui.generate(wf, timeout=args.timeout)
            for f in files:
                print(f"  → {f}")


def cmd_checkpoints(args: argparse.Namespace) -> None:
    """Test multiple face-LoRA epochs at fixed prompt/seed."""
    epochs = [int(e) for e in args.epochs.split(",")]
    prompt = _prompt(args.prompt, "face_only")
    for ep in epochs:
        name = f"{args.face_prefix}-{ep:06d}" if args.zero_pad else f"{args.face_prefix}-ep{ep}"
        wf = workflows.txt2img(
            prompt=prompt,
            loras=[(name, args.weight)],
            seed=args.seed,
            negative=_negative_for(args.prompt),
            filename_prefix=f"eval/ckpt_{args.face_prefix}/ep{ep}_w{args.weight}",
        )
        print(f"[ckpt] {name} @ {args.weight}")
        try:
            files = comfyui.generate(wf, timeout=args.timeout)
            for f in files:
                print(f"  → {f}")
        except Exception as e:
            print(f"  ✗ {e}")


def cmd_quicktest(args: argparse.Namespace) -> None:
    """Sanity-check grid — runs before a full training session.

    Uses the four preset keys named in QUICKTEST_PLANS. Each must exist in
    `presets.PROMPTS`. The tracked `presets.py` ships with generic keys
    (id_lock / angle / photo / manhwa_prompt / manhwa_stylized); users who
    want different scenarios should add keys via `eval/presets_local.py`.
    Override the four scenarios here by setting `QUICKTEST_PLANS` in
    presets_local.py.
    """
    face = args.face_lora
    face_weight = args.face_weight
    style = args.style_lora
    style_weight = args.style_weight

    default_plans = [
        ("1_photo",           "photo",            [(face, face_weight)]),
        ("2_id_lock",         "id_lock",          [(face, face_weight)]),
        ("3_manhwa_prompt",   "manhwa_prompt",    [(face, face_weight)]),
        ("4_manhwa_stylized", "manhwa_stylized",  [(face, face_weight), (style, style_weight)]),
    ]
    plans = getattr(presets, "QUICKTEST_PLANS", default_plans)

    # Style trigger prefix for the style LoRA (if any)
    style_prefix = ""
    for k, v in presets.STACK_PROMPT_PREFIX.items():
        if style in [l[0] for l in presets.STACKS.get(k, [])]:
            style_prefix = v
            break
    if style == "flux-manhwa-v5":
        style_prefix = "stylized-manhwa, "
    elif style == "flux-manwha-webtoon":
        style_prefix = "manwha_style, manwha, cartoon, "

    for label, prompt_key, stack in plans:
        uses_style = "manhwa" in prompt_key
        prompt = presets.PROMPTS[prompt_key]
        if uses_style and style_prefix:
            prompt = f"{style_prefix}{prompt}"
        effective_stack = stack if uses_style else [(face, face_weight)]

        wf = workflows.txt2img(
            prompt=prompt,
            loras=effective_stack,
            seed=args.seed,
            negative=_negative_for(prompt_key),
            filename_prefix=f"eval/quicktest_{args.seed}/{label}",
        )
        print(f"[quicktest {label}]  loras={effective_stack}")
        try:
            files = comfyui.generate(wf, timeout=args.timeout)
            for f in files:
                print(f"  → {f}")
        except Exception as e:
            print(f"  ✗ {e}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--timeout", type=int, default=300)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("shot")
    s.add_argument("--prompt", default="id_lock")
    s.add_argument("--stack", default="face_only")
    s.add_argument("--seed", type=int, default=111)
    s.set_defaults(fn=cmd_shot)

    s = sub.add_parser("stages")
    s.add_argument("--seed", type=int, default=111)
    s.add_argument("--stack-a", default="face_realism")
    s.add_argument("--stack-b", default="face_manhwa_v5")
    s.set_defaults(fn=cmd_stages)

    s = sub.add_parser("sweep")
    s.add_argument("--prompt", default="manhwa_stylized")
    s.add_argument("--seed", type=int, default=111)
    s.add_argument("--stacks", default="",
                   help="comma-separated; default = all STACKS")
    s.set_defaults(fn=cmd_sweep)

    s = sub.add_parser("matrix")
    s.add_argument("--prompt", default="manhwa_stylized")
    s.add_argument("--stacks", required=True)
    s.add_argument("--seeds", default="111,222,333,444")
    s.set_defaults(fn=cmd_matrix)

    s = sub.add_parser("quicktest",
                       help="4-scenario sanity check (see cmd_quicktest docstring)")
    s.add_argument("--seed", type=int, default=111)
    s.add_argument("--face-lora", default=presets.FACE_LORA)
    s.add_argument("--face-weight", type=float, default=presets.FACE_WEIGHT)
    s.add_argument("--style-lora", default="flux-manhwa-v5",
                   help="e.g. flux-manhwa-v5 / flux-manwha-webtoon / flux-illustration-alvdansen")
    s.add_argument("--style-weight", type=float, default=0.9)
    s.set_defaults(fn=cmd_quicktest)

    s = sub.add_parser("checkpoints")
    s.add_argument("--face-prefix", required=True,
                   help="output_name prefix from training (without epoch suffix)")
    s.add_argument("--epochs", default="2,4,6,8,10,12")
    s.add_argument("--zero-pad", action="store_true",
                   help="use <prefix>-000004 naming (kohya sd-scripts default)")
    s.add_argument("--weight", type=float, default=0.85)
    s.add_argument("--seed", type=int, default=111)
    s.add_argument("--prompt", default="id_lock")
    s.set_defaults(fn=cmd_checkpoints)

    args = ap.parse_args()
    args.fn(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
