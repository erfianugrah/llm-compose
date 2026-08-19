# proxy-go image - Go model-routing proxy (proxy-v2, spec R4/R6).
#
# Multi-stage: static Go build, minimal alpine runtime. Replaces the Python
# image (images/proxy.Dockerfile) at cutover; runs side-by-side on port
# 11435 during soak.

FROM golang:1.26-bookworm AS build

WORKDIR /src
COPY proxy-go/go.mod proxy-go/go.sum ./
RUN go mod download
COPY proxy-go/ ./
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /proxy ./cmd/proxy

FROM alpine:3.24

# ca-certificates: HTTPS to huggingface.co (mmproj/template downloads)
# No curl: busybox wget covers the healthcheck, and curl was the image's
# entire CVE surface (Docker Scout 2026-08-19: 2 critical + 4 high).
RUN apk add --no-cache ca-certificates

COPY --from=build /proxy /proxy

ENV LLMC_PROXY_PORT=11434 \
    LLMC_PRESETS_DIR=/presets \
    LLMC_STATE_DIR=/state \
    LLMC_ASSETS_DIR=/assets \
    LLMC_NETWORK=llmc \
    LLMC_HEALTH_TIMEOUT=900 \
    LLMC_VRAM_LIMIT_GB=32 \
    LLMC_VRAM_RESERVE_GB=6 \
    LLMC_DRAIN_GRACE_S=60 \
    LLMC_LOCK_TTL_S=900

EXPOSE 11434

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD wget -q -O /dev/null http://localhost:11434/health || exit 1

# Runs as root: the Docker socket's GID varies per host, and socket access
# is effective root anyway (same reasoning as the Python proxy image).
ENTRYPOINT ["/proxy"]
