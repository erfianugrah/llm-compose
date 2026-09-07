# Re-tag an upstream NInfer build for our own registry, adding the attribution
# that redistribution requires and upstream's own Dockerfile does not carry.
#
# Why this exists: the ninfer:local image is built from a pinned upstream
# checkout, and on 2026-09-07 Docker reclaimed it under disk pressure once the
# spike container was removed - turning a `docker pull` into a 6-minute
# rebuild. A registry copy makes the engine recoverable.
#
# Why a thin layer rather than editing upstream's Dockerfile: the build is
# deliberately UNMODIFIED upstream source at a pinned commit. Patching their
# Dockerfile would make it a derivative work and muddy that claim; a layer on
# top adds only our attribution.
#
# Apache-2.0 section 4 requires recipients get a copy of the license and that
# attribution notices are retained. Upstream's runtime stage copies only the
# two binaries, so neither travels without this.
ARG BASE=ninfer:local
FROM ${BASE}

ARG NINFER_COMMIT
ARG NINFER_REPO=https://github.com/Neroued/ninfer

COPY LICENSE.ninfer /usr/share/licenses/ninfer/LICENSE
COPY NOTICE.ninfer  /usr/share/licenses/ninfer/NOTICE

LABEL org.opencontainers.image.title="ninfer" \
      org.opencontainers.image.description="Unmodified build of Neroued/ninfer, a from-scratch CUDA inference engine for registered Qwen checkpoints. sm_120a only. Repackaged for a private homelab registry; not an official upstream image." \
      org.opencontainers.image.source="${NINFER_REPO}" \
      org.opencontainers.image.revision="${NINFER_COMMIT}" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.vendor="Neroued (upstream); repackaged by erfianugrah" \
      io.erfi.upstream-modified="false"
