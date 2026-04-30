"""Example local overrides for eval/presets.py.

Copy to `eval/presets_local.py` (gitignored) and customize.
Imported automatically by presets.py if present — see end of that file.

You can override any of: SUBJECT, FACE_LORA, PROMPTS, STACKS,
STACK_PROMPT_PREFIX, NEG_REALISTIC, NEG_MANHWA, SEEDS, QUICKTEST_PLANS.
"""

# Subject description used in PROMPTS below. Replace with your person/character.
SUBJECT = "subject, portrait"

# Default face LoRA name (filename without .safetensors) — deployed to
# ~/docker-volumes/comfyui/models/loras/.
FACE_LORA = "my-face-lora"
FACE_WEIGHT = 0.85

# Extra or overridden prompts. Keys referenced by the CLI:
#   id_lock, angle, photo, manhwa_prompt, manhwa_stylized
# Any key here replaces the one in presets.py.
PROMPTS = {
    # Override / add your own prompt keys here.
    # "portrait_studio": f"{SUBJECT}, studio portrait, rim lighting",
}

# Extra stacks on top of the defaults in presets.py.
STACKS = {
    # "face_x": [(FACE_LORA, 1.0), ("flux-some-style", 0.8)],
}

# Trigger-word prefixes for custom stacks.
STACK_PROMPT_PREFIX = {
    # "face_x": "some-trigger-word, ",
}

# Custom quicktest plans — list of (label, prompt_key, stack) tuples.
# Defaults in cmd_quicktest are used if this is unset.
# QUICKTEST_PLANS = [
#     ("1_photo",   "photo",            [(FACE_LORA, FACE_WEIGHT)]),
#     ("2_manhwa",  "manhwa_stylized",  [(FACE_LORA, FACE_WEIGHT), ("flux-manhwa-v5", 0.9)]),
# ]
