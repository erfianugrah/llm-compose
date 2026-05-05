# Blackwell-optimized ComfyUI build
# PyTorch 2.11 + CUDA 12.8 for RTX 5090 (sm_120)
#
# Single-stage build — PyTorch runtime image is already large (~8GB),
# splitting stages saves negligible space and complicates pip package
# sharing. Custom nodes need git at runtime for Manager installs.

FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime

ARG COMFYUI_VERSION=v0.19.5
ARG COMFYUI_MANAGER_VERSION=4.2.1

# Runtime deps: curl (healthcheck), git (Manager node installs), build
# tools for custom nodes that compile C extensions (e.g. insightface)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      curl git ca-certificates build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone ComfyUI at pinned version
RUN git clone --depth 1 --branch ${COMFYUI_VERSION} \
      https://github.com/Comfy-Org/ComfyUI.git

WORKDIR /app/ComfyUI

# Install Python dependencies
# --break-system-packages: container-only Python, no system packages to protect (PEP 668)
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

# ── Custom nodes (baked into image) ──────────────────────────────────
# These are installed into /app/ComfyUI/custom_nodes_builtin/ so they
# survive even if the custom_nodes/ volume mount is empty. ComfyUI
# scans both directories for nodes.

# ComfyUI Manager — UI for installing additional custom nodes and models
RUN git clone --depth 1 --branch ${COMFYUI_MANAGER_VERSION} \
      https://github.com/Comfy-Org/ComfyUI-Manager.git \
      custom_nodes_builtin/ComfyUI-Manager && \
    if [ -f custom_nodes_builtin/ComfyUI-Manager/requirements.txt ]; then \
      pip install --no-cache-dir --break-system-packages \
        -r custom_nodes_builtin/ComfyUI-Manager/requirements.txt; \
    fi

# IP-Adapter Plus — character consistency via reference images
# Pinned to avoid silent breakage from upstream. To update:
# git ls-remote --refs https://github.com/cubiq/ComfyUI_IPAdapter_plus.git main
ARG IPADAPTER_VERSION=a0f451a5113cf9becb0847b92884cb10cbdec0ef
RUN git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus.git \
      custom_nodes_builtin/ComfyUI_IPAdapter_plus && \
    cd custom_nodes_builtin/ComfyUI_IPAdapter_plus && \
    git checkout ${IPADAPTER_VERSION}

# ── Entrypoint ───────────────────────────────────────────────────────
COPY comfyui-entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

VOLUME ["/app/ComfyUI/models", "/app/ComfyUI/output", \
        "/app/ComfyUI/input", "/app/ComfyUI/custom_nodes"]

EXPOSE 8188

ENTRYPOINT ["/app/entrypoint.sh", "--listen", "0.0.0.0", "--port", "8188"]
