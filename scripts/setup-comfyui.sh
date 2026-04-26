#!/usr/bin/env bash
# Download ComfyUI models and assets for the llm-compose stack.
# Idempotent — skips already-downloaded files.
#
# Models downloaded:
#   - NoobAI-XL v-pred 1.0 (anime/illustration checkpoint, ~7.1 GB)
#   - SDXL Base 1.0 (general purpose checkpoint, ~6.5 GB)
#   - CLIP-ViT-H-14 (IP-Adapter vision encoder, ~2.4 GB)
#   - IP-Adapter Plus Face SDXL (character face consistency, ~809 MB)
#   - 4x-UltraSharp (upscaler for hires fix, ~64 MB)
#
# Total: ~16.9 GB on first run.

set -euo pipefail

COMFYUI_DIR="${HOME}/docker-volumes/comfyui"

# ── Helper ───────────────────────────────────────────────────────────
download() {
    local dest="$1" url="$2" desc="$3"
    if [ -f "$dest" ]; then
        echo "  ✓ $(basename "$dest") (cached)"
        return
    fi
    echo "  ↓ ${desc} → $(basename "$dest")"
    mkdir -p "$(dirname "$dest")"
    local tmp="${dest}.tmp"
    curl -L --progress-bar -o "$tmp" "$url"
    mv "$tmp" "$dest"
}

echo ""
echo "── ComfyUI Model Setup ──"
echo ""

# ── Checkpoints ──────────────────────────────────────────────────────
echo "Checkpoints:"
download \
    "${COMFYUI_DIR}/models/checkpoints/NoobAI-XL-Vpred-v1.0.safetensors" \
    "https://huggingface.co/Laxhar/noobai-XL-Vpred-1.0/resolve/main/NoobAI-XL-Vpred-v1.0.safetensors" \
    "NoobAI-XL v-pred 1.0 (~7.1 GB, anime/illustration)"

download \
    "${COMFYUI_DIR}/models/checkpoints/sd_xl_base_1.0.safetensors" \
    "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors" \
    "SDXL Base 1.0 (~6.5 GB, general purpose)"

# ── CLIP Vision (for IP-Adapter) ─────────────────────────────────────
echo ""
echo "CLIP Vision:"
download \
    "${COMFYUI_DIR}/models/clip_vision/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors" \
    "https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors" \
    "CLIP-ViT-H-14 (~2.4 GB, IP-Adapter vision encoder)"

# ── IP-Adapter ───────────────────────────────────────────────────────
echo ""
echo "IP-Adapter:"
download \
    "${COMFYUI_DIR}/models/ipadapter/ip-adapter-plus-face_sdxl_vit-h.safetensors" \
    "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors" \
    "IP-Adapter Plus Face SDXL (~809 MB, character consistency)"

# ── Upscaler ─────────────────────────────────────────────────────────
echo ""
echo "Upscalers:"
download \
    "${COMFYUI_DIR}/models/upscale_models/4x-UltraSharp.pth" \
    "https://huggingface.co/Kim2091/UltraSharp/resolve/main/4x-UltraSharp.pth" \
    "4x-UltraSharp (~64 MB, hires fix upscaler)"

# ── Summary ──────────────────────────────────────────────────────────
echo ""
echo "── Summary ──"
echo ""
total=$(du -sh "${COMFYUI_DIR}/models" 2>/dev/null | cut -f1)
echo "Total model storage: ${total:-0}"
echo ""
echo "Checkpoints:  $(ls -1 "${COMFYUI_DIR}/models/checkpoints/" 2>/dev/null | wc -l) files"
echo "CLIP Vision:  $(ls -1 "${COMFYUI_DIR}/models/clip_vision/" 2>/dev/null | wc -l) files"
echo "IP-Adapter:   $(ls -1 "${COMFYUI_DIR}/models/ipadapter/" 2>/dev/null | wc -l) files"
echo "Upscalers:    $(ls -1 "${COMFYUI_DIR}/models/upscale_models/" 2>/dev/null | wc -l) files"
echo ""
echo "✓ ComfyUI models ready."
