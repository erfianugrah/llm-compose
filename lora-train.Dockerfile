FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

# Pin sd-scripts to a known commit for reproducible builds.
# Override with `--build-arg SD_SCRIPTS_REF=<commit-sha>` if needed.
# To update: git ls-remote --refs https://github.com/kohya-ss/sd-scripts.git sd3
ARG SD_SCRIPTS_REF=ae0b0e6be8d0a458bef308944672a1079b162061

RUN apt-get update && apt-get install -y \
      git libgl1-mesa-glx libglib2.0-0 gcc curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Shallow clone at the pinned ref. `git clone --branch` accepts both tags
# and branches but not arbitrary commits — fall back to a fetch-then-reset
# pattern so commit SHAs work too.
RUN git clone https://github.com/kohya-ss/sd-scripts.git /sd-scripts \
    && cd /sd-scripts \
    && git fetch --depth 1 origin "${SD_SCRIPTS_REF}" \
    && git checkout FETCH_HEAD \
    && git log -1 --format='%H %s' > /sd-scripts.commit
WORKDIR /sd-scripts

# Install from sd-scripts requirements + extras we explicitly depend on.
# bitsandbytes is REQUIRED for AdamW8bit (default optimizer for Flux).
# Pin it ourselves rather than trusting upstream requirements.txt to keep it.
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir \
      "accelerate" "bitsandbytes>=0.43" \
      "onnxruntime-gpu" "onnx" "timm" "einops"

# Copy training API server, captioners, and dataset utilities.
# Progress is parsed from sd-scripts stdout in server.py — no tqdm
# patching, no .pth files, no site-packages injection.
COPY train/server.py /train-server.py
COPY train/caption_florence.py /train-hooks/caption_florence.py
COPY train/caption_blip2.py /train-hooks/caption_blip2.py
COPY scripts/audit-dataset.py /audit-dataset.py
COPY scripts/filter-dataset.py /filter-dataset.py
COPY scripts/pick-focus-subset.py /pick-focus-subset.py

WORKDIR /workspace
EXPOSE 8787

# Default: run the HTTP API server
# Override CMD to run one-shot training (e.g. accelerate launch ...)
CMD ["python", "/train-server.py"]
