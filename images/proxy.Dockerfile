# v2 proxy image — Python HTTP server + Docker SDK.
#
# Replaces proxy/Dockerfile, which bundled the docker CLI + compose plugin
# for shellout-style orchestration. The new proxy uses the docker Python
# SDK directly, so the image only needs Python + the docker package + a
# bit of CA bundle for HTTPS asset downloads.

FROM python:3.12-slim

# Minimal runtime deps:
#   ca-certificates — for HTTPS to huggingface.co (mmproj/template downloads)
#   curl            — for healthcheck (used by docker-compose healthcheck spec)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the docker Python SDK. Pinned to a known-good major to avoid
# breaking API changes; bump when needed.
RUN pip install --no-cache-dir 'docker>=7.0,<8.0'

# Copy the llmc package. Source of truth for the proxy logic.
COPY llmc/ /app/llmc/

# Defaults — overridable by docker-compose env_file or `docker run -e`.
ENV LLMC_PROXY_PORT=11434 \
    LLMC_PRESETS_DIR=/presets \
    LLMC_STATE_DIR=/state \
    LLMC_ASSETS_DIR=/assets \
    LLMC_NETWORK=llmc \
    LLMC_HEALTH_TIMEOUT=900 \
    LLMC_VRAM_LIMIT_GB=32 \
    LLMC_VRAM_RESERVE_GB=6 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 11434

# Healthcheck: the proxy returns 200 even during a mode swap (with
# status:switching) so this won't false-positive while we're loading
# a multi-GB GGUF.
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -sf http://localhost:11434/health || exit 1

# Runs as root. The Docker socket bind-mount has a GID that varies per
# host (often 988 or 999), so a fixed non-root UID would need per-host
# `group_add` config. Since docker socket access == effective root anyway,
# non-root in this image would be theater.
ENTRYPOINT ["python", "-m", "llmc.proxy"]
