# Blackwell-optimized ComfyUI build
# PyTorch 2.11 + CUDA 12.8 for RTX 5090 (sm_120)
#
# Single-stage build — PyTorch runtime image is already large (~8GB),
# splitting stages saves negligible space and complicates pip package
# sharing. ComfyUI-Manager needs git at runtime for node installation.

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

# ComfyUI Manager — UI for installing custom nodes and models
RUN git clone --depth 1 --branch ${COMFYUI_MANAGER_VERSION} \
      https://github.com/Comfy-Org/ComfyUI-Manager.git \
      custom_nodes/ComfyUI-Manager && \
    if [ -f custom_nodes/ComfyUI-Manager/requirements.txt ]; then \
      pip install --no-cache-dir --break-system-packages \
        -r custom_nodes/ComfyUI-Manager/requirements.txt; \
    fi

VOLUME ["/app/ComfyUI/models", "/app/ComfyUI/output", \
        "/app/ComfyUI/input", "/app/ComfyUI/custom_nodes"]

EXPOSE 8188

ENTRYPOINT ["python", "main.py", "--listen", "0.0.0.0", "--port", "8188"]
