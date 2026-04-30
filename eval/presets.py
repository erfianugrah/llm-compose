"""Generic prompt presets + LoRA stacks for identity/style eval.

User-specific subject descriptions and extra prompt variants should live
in `eval/presets_local.py` (gitignored). That file can override or extend
SUBJECT, FACE_LORA, PROMPTS, STACKS, and STACK_PROMPT_PREFIX at import
time — see the template in `eval/presets_local.example.py`.
"""

# Generic placeholder subject — override SUBJECT in presets.local.py.
SUBJECT = "subject, portrait"

NEG_REALISTIC = (
    "deformed, blurry, low quality, extra fingers, watermark, text, "
    "bad anatomy, plastic skin, waxy, overly smooth, airbrushed"
)
NEG_MANHWA = (
    "deformed, blurry, low quality, extra fingers, watermark, text, "
    "bad anatomy, 3d render, photorealistic, thick lineart, big anime eyes, "
    "multiple panels"
)

PROMPTS: dict[str, str] = {
    # Identity lock — neutral, clean lighting. Good for comparing face likeness.
    "id_lock": f"{SUBJECT}, headshot, neutral expression, soft lighting, close-up",

    # Angle robustness — 3/4 view tests if LoRA holds geometry.
    "angle": f"{SUBJECT}, three-quarter profile, soft natural light",

    # Realistic photo baseline.
    "photo": (
        f"{SUBJECT}, candid amateur photograph, natural lighting, "
        "film grain, realistic skin texture, pores, subsurface scattering, "
        "shot on iphone"
    ),

    # Manhwa via prompt only (no style LoRA) — sanity baseline.
    "manhwa_prompt": (
        f"{SUBJECT}, korean manhwa illustration, webtoon style, thin brown lineart, "
        "flat cel shading, muted desaturated palette, porcelain skin, 2d illustration, "
        "close-up portrait"
    ),

    # Manhwa with style LoRA (style trigger injected via STACK_PROMPT_PREFIX).
    "manhwa_stylized": (
        f"{SUBJECT}, close-up portrait, thin brown lineart, flat cel shading, "
        "muted palette, 2d illustration"
    ),
}

# Named LoRA stacks. Each entry is a list of (name, strength) pairs.
# Face LoRA name placeholder — override in presets.local.py.
FACE_LORA = "face-lora"
FACE_WEIGHT = 0.85

STACKS: dict[str, list[tuple[str, float]]] = {
    "face_only":       [(FACE_LORA, FACE_WEIGHT)],
    "face_realism":    [(FACE_LORA, FACE_WEIGHT), ("flux-realism-xlabs", 0.5)],
    "face_super":      [(FACE_LORA, FACE_WEIGHT), ("flux-super-realism", 0.5)],
    "face_ultrareal":  [(FACE_LORA, FACE_WEIGHT), ("flux-ultrarealism-canopus", 0.5)],
    "face_manhwa_v5":  [(FACE_LORA, FACE_WEIGHT), ("flux-manhwa-v5", 0.9)],
    "face_manwha_web": [(FACE_LORA, FACE_WEIGHT), ("flux-manwha-webtoon", 0.9)],
    "face_manhwa_a18": [(FACE_LORA, FACE_WEIGHT), ("flux-manhwa-a-18", 0.9)],
    "face_illust":     [(FACE_LORA, FACE_WEIGHT), ("flux-illustration-alvdansen", 0.9)],
    "face_anime":      [(FACE_LORA, FACE_WEIGHT), ("flux-anime-alvdansen", 0.9)],
}

# Trigger-word prefixes for style LoRAs that need them in the prompt.
STACK_PROMPT_PREFIX: dict[str, str] = {
    "face_manhwa_v5":  "stylized-manhwa, ",
    "face_manwha_web": "manwha_style, manwha, cartoon, ",
    "face_manhwa_a18": "",  # no trigger word
}

# Default seeds for eval batches — diverse enough to expose identity drift.
SEEDS = [111, 222, 333, 444, 555, 666, 777, 888]


# ── Local overrides ──────────────────────────────────────────────────
# Import `presets_local` if present — lets users inject subject-specific
# prompts and stacks without tracking them in git.
try:
    from . import presets_local as _local  # type: ignore
except ImportError:
    try:
        import presets_local as _local  # type: ignore
    except ImportError:
        _local = None  # no local overrides

if _local is not None:
    if hasattr(_local, "SUBJECT"):
        SUBJECT = _local.SUBJECT
    if hasattr(_local, "FACE_LORA"):
        FACE_LORA = _local.FACE_LORA
    if hasattr(_local, "PROMPTS"):
        PROMPTS.update(_local.PROMPTS)
    if hasattr(_local, "STACKS"):
        STACKS.update(_local.STACKS)
    if hasattr(_local, "STACK_PROMPT_PREFIX"):
        STACK_PROMPT_PREFIX.update(_local.STACK_PROMPT_PREFIX)
    if hasattr(_local, "NEG_REALISTIC"):
        NEG_REALISTIC = _local.NEG_REALISTIC
    if hasattr(_local, "NEG_MANHWA"):
        NEG_MANHWA = _local.NEG_MANHWA
    if hasattr(_local, "SEEDS"):
        SEEDS = _local.SEEDS
