#!/usr/bin/env python3
"""End-to-end eval runner — all subcommands route through the proxy on
:11434 so GPU mode swap is automatic.

Subcommands (see `--help` on each for full options):

  shot         One prompt + one stack
  stages       3-way comparison: photo / stylized / prompt-only
  sweep        All STACKS side-by-side for one prompt
  matrix       Stacks × seeds grid
  checkpoints  Multiple training epochs of one face LoRA
  quicktest    Preset 4-scenario sanity (plans from presets_local.py)
  weights      Face × aux LoRA weight grid on one prompt
  loras        Sweep a list of aux LoRAs at fixed weights
  seeds        Identity robustness — N seeds on one config
  i2i          Img2img denoise sweep from an input image

Examples:

  python3 eval/run.py shot --prompt manhwa_stylized --stack face_manhwa_v5 --seed 111

  python3 eval/run.py weights --prompt photo --face-weights 0.7,0.85,1.0 \\
         --aux flux-realism-xlabs --aux-weights 0,0.3,0.5

  python3 eval/run.py loras --prompt manhwa_stylized \\
         --loras flux-manhwa-v5,flux-manwha-webtoon,flux-illustration-alvdansen

  python3 eval/run.py seeds --prompt photo --seeds 111,222,333,444,555,666 \\
         --face-weight 0.7

  python3 eval/run.py i2i --input my_real.png --prompt manhwa_stylized \\
         --stack face_manhwa_v5 --denoises 0.5,0.65,0.8

  python3 eval/run.py checkpoints --face-prefix my-face-lora \\
         --epochs 2,4,6,8,10,12 --weight 0.85 --zero-pad

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


def _style_prefix_for(lora_name: str) -> str:
    """Trigger-word prefix for common style LoRAs."""
    # Look up via STACKS + STACK_PROMPT_PREFIX first (presets-driven)
    for stack_key, stack in presets.STACKS.items():
        if lora_name in [l[0] for l in stack]:
            prefix = presets.STACK_PROMPT_PREFIX.get(stack_key, "")
            if prefix:
                return prefix
    # Fallback: well-known defaults
    fallback = {
        "flux-manhwa-v5": "stylized-manhwa, ",
        "flux-manwha-webtoon": "manwha_style, manwha, cartoon, ",
    }
    return fallback.get(lora_name, "")


def cmd_weights(args: argparse.Namespace) -> None:
    """Face × aux LoRA weight grid on a single prompt.

    Example:
      run.py weights --prompt photo --face-weights 0.7,0.85,1.0 \\
                     --aux flux-realism-xlabs --aux-weights 0,0.3,0.5

    aux-weight 0 omits the aux LoRA entirely.
    """
    face = args.face_lora
    face_weights = [float(x) for x in args.face_weights.split(",")]
    aux_weights = [float(x) for x in args.aux_weights.split(",")]
    prefix = _style_prefix_for(args.aux) if args.aux else ""

    base_prompt = presets.PROMPTS[args.prompt] if args.prompt in presets.PROMPTS else args.prompt
    prompt = f"{prefix}{base_prompt}" if prefix else base_prompt

    total = len(face_weights) * len(aux_weights)
    done = 0
    for fw in face_weights:
        for aw in aux_weights:
            done += 1
            stack = [(face, fw)]
            if aw > 0 and args.aux:
                stack.append((args.aux, aw))
            label = f"f{fw:.2f}_a{aw:.2f}" if args.aux else f"f{fw:.2f}"
            wf = workflows.txt2img(
                prompt=prompt,
                loras=stack,
                seed=args.seed,
                negative=_negative_for(args.prompt),
                filename_prefix=f"eval/weights_{args.prompt}/{label}",
            )
            print(f"[weights {done}/{total}]  {label}  loras={stack}")
            try:
                files = comfyui.generate(wf, timeout=args.timeout)
                for f in files:
                    print(f"  → {f}")
            except Exception as e:
                print(f"  ✗ {e}")


def cmd_loras(args: argparse.Namespace) -> None:
    """Sweep a list of aux LoRAs stacked on the face LoRA.

    Useful to compare style LoRAs head-to-head at identical settings:
      run.py loras --prompt manhwa_stylized \\
                   --loras flux-manhwa-v5,flux-manwha-webtoon,flux-illustration-alvdansen
    """
    face = args.face_lora
    auxes = args.loras.split(",")
    base_prompt = presets.PROMPTS[args.prompt] if args.prompt in presets.PROMPTS else args.prompt

    for aux in auxes:
        prefix = _style_prefix_for(aux)
        prompt = f"{prefix}{base_prompt}" if prefix else base_prompt
        stack = [(face, args.face_weight), (aux, args.aux_weight)]
        wf = workflows.txt2img(
            prompt=prompt,
            loras=stack,
            seed=args.seed,
            negative=_negative_for(args.prompt),
            filename_prefix=f"eval/loras_{args.prompt}/{aux}",
        )
        print(f"[loras] aux={aux}")
        try:
            files = comfyui.generate(wf, timeout=args.timeout)
            for f in files:
                print(f"  → {f}")
        except Exception as e:
            print(f"  ✗ {e}")


def cmd_seeds(args: argparse.Namespace) -> None:
    """Identity robustness: N seeds on a single config.

    If --stack is set, uses that STACKS entry. Otherwise uses --face-lora
    at --face-weight, no aux LoRA.
    """
    seeds = [int(s) for s in args.seeds.split(",")]
    if args.stack:
        stack = _stack(args.stack)
        prompt = _prompt(args.prompt, args.stack)
        tag = args.stack
    else:
        stack = [(args.face_lora, args.face_weight)]
        base_prompt = presets.PROMPTS[args.prompt] if args.prompt in presets.PROMPTS else args.prompt
        prompt = base_prompt
        tag = f"{args.face_lora}_f{args.face_weight:.2f}"

    for seed in seeds:
        wf = workflows.txt2img(
            prompt=prompt,
            loras=stack,
            seed=seed,
            negative=_negative_for(args.prompt),
            filename_prefix=f"eval/seeds_{args.prompt}_{tag}/s{seed}",
        )
        print(f"[seeds] s={seed}  tag={tag}")
        try:
            files = comfyui.generate(wf, timeout=args.timeout)
            for f in files:
                print(f"  → {f}")
        except Exception as e:
            print(f"  ✗ {e}")


def cmd_i2i(args: argparse.Namespace) -> None:
    """Img2img denoise sweep — preserves source geometry, shifts style.

    Needs an input image already present in ComfyUI's input dir:
      ~/docker-volumes/comfyui/input/<file>

    Example:
      run.py i2i --input sophia_real.png \\
                 --prompt manhwa_stylized --stack face_manhwa_v5 \\
                 --denoises 0.5,0.65,0.8
    """
    denoises = [float(x) for x in args.denoises.split(",")]
    if args.stack:
        stack = _stack(args.stack)
        prompt = _prompt(args.prompt, args.stack)
        tag = args.stack
    else:
        stack = [(args.face_lora, args.face_weight)]
        if args.aux:
            stack.append((args.aux, args.aux_weight))
        base_prompt = presets.PROMPTS[args.prompt] if args.prompt in presets.PROMPTS else args.prompt
        prefix = _style_prefix_for(args.aux) if args.aux else ""
        prompt = f"{prefix}{base_prompt}" if prefix else base_prompt
        tag = f"{args.face_lora}_{args.aux or 'noaux'}"

    for dn in denoises:
        wf = workflows.img2img(
            prompt=prompt,
            input_image=args.input,
            loras=stack,
            seed=args.seed,
            denoise=dn,
            negative=_negative_for(args.prompt),
            filename_prefix=f"eval/i2i_{args.prompt}_{tag}/denoise_{dn:.2f}",
        )
        print(f"[i2i] denoise={dn}  input={args.input}")
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

    s = sub.add_parser("weights",
                       help="face x aux LoRA weight grid on one prompt")
    s.add_argument("--prompt", default="photo")
    s.add_argument("--seed", type=int, default=111)
    s.add_argument("--face-lora", default=presets.FACE_LORA)
    s.add_argument("--face-weights", default="0.7,0.85,1.0",
                   help="comma-separated face weights")
    s.add_argument("--aux", default="",
                   help="aux LoRA name (e.g. flux-realism-xlabs); empty = face only")
    s.add_argument("--aux-weights", default="0,0.5",
                   help="comma-separated aux weights; 0 omits the aux LoRA")
    s.set_defaults(fn=cmd_weights)

    s = sub.add_parser("loras",
                       help="sweep a list of aux LoRAs at fixed weights")
    s.add_argument("--prompt", default="manhwa_stylized")
    s.add_argument("--seed", type=int, default=111)
    s.add_argument("--face-lora", default=presets.FACE_LORA)
    s.add_argument("--face-weight", type=float, default=presets.FACE_WEIGHT)
    s.add_argument("--loras", required=True,
                   help="comma-separated aux LoRA filenames (no .safetensors)")
    s.add_argument("--aux-weight", type=float, default=0.9)
    s.set_defaults(fn=cmd_loras)

    s = sub.add_parser("seeds",
                       help="identity robustness — N seeds on a single config")
    s.add_argument("--prompt", default="photo")
    s.add_argument("--seeds", default="111,222,333,444,555,666")
    s.add_argument("--stack", default="",
                   help="named stack from presets.STACKS; overrides --face-lora/--face-weight")
    s.add_argument("--face-lora", default=presets.FACE_LORA)
    s.add_argument("--face-weight", type=float, default=presets.FACE_WEIGHT)
    s.set_defaults(fn=cmd_seeds)

    s = sub.add_parser("i2i",
                       help="img2img denoise sweep from an input image")
    s.add_argument("--input", required=True,
                   help="filename in ~/docker-volumes/comfyui/input/")
    s.add_argument("--prompt", default="manhwa_stylized")
    s.add_argument("--seed", type=int, default=111)
    s.add_argument("--denoises", default="0.5,0.65,0.8")
    s.add_argument("--stack", default="",
                   help="named stack; overrides --face-lora/--aux")
    s.add_argument("--face-lora", default=presets.FACE_LORA)
    s.add_argument("--face-weight", type=float, default=presets.FACE_WEIGHT)
    s.add_argument("--aux", default="",
                   help="optional style LoRA stacked on face")
    s.add_argument("--aux-weight", type=float, default=0.9)
    s.set_defaults(fn=cmd_i2i)

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
