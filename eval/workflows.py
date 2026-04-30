"""Reusable ComfyUI workflow builders for Flux LoRA testing."""
from __future__ import annotations

from typing import Any

Node = dict[str, Any]
Workflow = dict[str, Node]


def _flux_base(
    prompt: str,
    negative: str,
    seed: int,
    width: int,
    height: int,
    steps: int,
    filename_prefix: str,
    t5xxl_name: str = "t5xxl_fp16.safetensors",
) -> tuple[Workflow, str]:
    """Common Flux plumbing. Returns workflow + name of the 'model' node to
    chain LoRAs onto (caller swaps node refs)."""
    wf: Workflow = {
        "unet": {"class_type": "UNETLoader", "inputs": {
            "unet_name": "flux1-dev.safetensors",
            "weight_dtype": "fp8_e4m3fn",
        }},
        "clip": {"class_type": "DualCLIPLoader", "inputs": {
            "clip_name1": "clip_l.safetensors",
            "clip_name2": t5xxl_name,
            "type": "flux",
        }},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {
            "clip": ["clip", 0], "text": prompt,
        }},
        "neg": {"class_type": "CLIPTextEncode", "inputs": {
            "clip": ["clip", 0], "text": negative,
        }},
        "latent": {"class_type": "EmptyLatentImage", "inputs": {
            "width": width, "height": height, "batch_size": 1,
        }},
        "save": {"class_type": "SaveImage", "inputs": {
            "images": ["decode", 0], "filename_prefix": filename_prefix,
        }},
    }
    return wf, "unet"


def txt2img(
    prompt: str,
    loras: list[tuple[str, float]],
    seed: int = 111,
    width: int = 832,
    height: int = 1216,
    steps: int = 28,
    negative: str = "",
    filename_prefix: str = "eval",
) -> Workflow:
    """Flux txt2img with arbitrary LoRA stack.

    loras: list of (lora_name_without_ext, strength). Applied in order.
           Empty list = no LoRA (pure Flux dev).
    """
    wf, tail = _flux_base(prompt, negative, seed, width, height, steps, filename_prefix)

    # Chain LoRAs onto the UNet output
    for i, (name, strength) in enumerate(loras):
        node_id = f"lora_{i}"
        wf[node_id] = {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": [tail, 0],
            "lora_name": f"{name}.safetensors",
            "strength_model": strength,
        }}
        tail = node_id

    wf["sampler"] = {"class_type": "KSampler", "inputs": {
        "seed": seed, "steps": steps, "cfg": 1.0,
        "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0,
        "model": [tail, 0],
        "positive": ["pos", 0], "negative": ["neg", 0],
        "latent_image": ["latent", 0],
    }}
    wf["decode"] = {"class_type": "VAEDecode", "inputs": {
        "samples": ["sampler", 0], "vae": ["vae", 0],
    }}
    return wf


def img2img(
    prompt: str,
    input_image: str,
    loras: list[tuple[str, float]],
    seed: int = 111,
    denoise: float = 0.65,
    steps: int = 28,
    negative: str = "",
    filename_prefix: str = "eval-i2i",
) -> Workflow:
    """Flux img2img — takes an existing image, re-samples with new prompt.

    input_image: filename relative to ComfyUI input dir (~/docker-volumes/comfyui/input/).
    denoise 0.6-0.7 preserves subject, shifts style. 0.8+ loses identity.
    """
    wf, tail = _flux_base(prompt, negative, seed, 0, 0, steps, filename_prefix)

    # Replace EmptyLatentImage with LoadImage + VAEEncode
    del wf["latent"]
    wf["load_img"] = {"class_type": "LoadImage", "inputs": {"image": input_image}}
    wf["encode"] = {"class_type": "VAEEncode", "inputs": {
        "pixels": ["load_img", 0], "vae": ["vae", 0],
    }}

    for i, (name, strength) in enumerate(loras):
        node_id = f"lora_{i}"
        wf[node_id] = {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": [tail, 0],
            "lora_name": f"{name}.safetensors",
            "strength_model": strength,
        }}
        tail = node_id

    wf["sampler"] = {"class_type": "KSampler", "inputs": {
        "seed": seed, "steps": steps, "cfg": 1.0,
        "sampler_name": "euler", "scheduler": "simple", "denoise": denoise,
        "model": [tail, 0],
        "positive": ["pos", 0], "negative": ["neg", 0],
        "latent_image": ["encode", 0],
    }}
    wf["decode"] = {"class_type": "VAEDecode", "inputs": {
        "samples": ["sampler", 0], "vae": ["vae", 0],
    }}
    return wf
