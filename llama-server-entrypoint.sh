#!/bin/sh
# llama-server entrypoint — builds the command line from env vars.
#
# Reads:
#   MODEL_REPO       HuggingFace repo (used if /models/$MODEL_FILE missing)
#   MODEL_FILE       GGUF filename. Local if /models/$MODEL_FILE exists,
#                    otherwise downloaded via llama-server's --hf-* flags.
#   MMPROJ_FILE      Optional. Multimodal projection weights filename.
#   TEMPLATE_FILE    Optional. Jinja chat template filename.
#   REASONING        Optional. "on" | "off" — sets --reasoning flag.
#   FLASH_ATTN       Default "on". "on" | "off" | "auto". Pascal (sm_61)
#                    has weak FP16, so the servarr deploy may set "off".
#   CONTEXT_SIZE     Default 65536. Sets both -c and --fit-ctx.
#   PARALLEL_SLOTS   Default 1. -np value.
#   TEMPERATURE      Default 1.0.
#   TOP_P            Default 0.95.
#   TOP_K            Default 64.
#   MIN_P            Optional.
#   PRESENCE_PENALTY Optional.
#   REPEAT_PENALTY   Optional.
#
# All other llama-server flags are hardcoded here (flash-attn, KV quant,
# batch sizes, threads, ngl) — change them by rebuilding the image, not
# by per-preset config. They tune for a single RTX 5090 (32 GB VRAM, sm_120).

set -e

FLASH_ATTN="${FLASH_ATTN:-on}"

if [ -f "/models/${MODEL_FILE}" ]; then
  MODEL_ARGS="-m /models/${MODEL_FILE}"
else
  MODEL_ARGS="--hf-repo ${MODEL_REPO} --hf-file ${MODEL_FILE}"
fi

exec llama-server \
  $MODEL_ARGS \
  ${MMPROJ_FILE:+--mmproj /models/${MMPROJ_FILE}} \
  --jinja \
  ${TEMPLATE_FILE:+--chat-template-file /models/${TEMPLATE_FILE}} \
  ${REASONING:+--reasoning ${REASONING}} \
  --port 8080 \
  --host 0.0.0.0 \
  -ngl 99 \
  --flash-attn "${FLASH_ATTN}" \
  -ctk q8_0 \
  -ctv q8_0 \
  -c "${CONTEXT_SIZE:-65536}" \
  --fit on \
  --fit-ctx "${CONTEXT_SIZE:-65536}" \
  --temp "${TEMPERATURE:-1.0}" \
  --top-p "${TOP_P:-0.95}" \
  --top-k "${TOP_K:-64}" \
  ${MIN_P:+--min-p ${MIN_P}} \
  ${PRESENCE_PENALTY:+--presence-penalty ${PRESENCE_PENALTY}} \
  ${REPEAT_PENALTY:+--repeat-penalty ${REPEAT_PENALTY}} \
  -np "${PARALLEL_SLOTS:-1}" \
  -b 2048 \
  -ub 2048 \
  --threads 8 \
  --threads-batch 8 \
  -v \
  --metrics "$@"
