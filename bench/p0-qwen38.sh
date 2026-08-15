#!/usr/bin/env bash
# p0-qwen38.sh - P0 feasibility spikes for Qwen3.8-27B.
# Plan: docs/plans/2026-08-15-local-model-bench-framework.md section 7.2
#
#   Spike 1: arch support - does the pinned llama.cpp image load the GGUF
#            (managed path: llmc switch qwen38)
#   Spike 2: KV/ctx fit matrix - raw docker runs,
#            ctx {65536,131072,196608} x slots {1,2}, VRAM recorded
#   Spike 3: text completion smoke + vision smoke (mmproj, red-64.png)
#
# SAFETY: waits for GPU idle (no lock, not switching, VRAM below threshold,
# sustained over 3 polls) before touching anything. Safe to leave running
# while the box is in use or while gaming - it parks until the GPU is free.
# Restores the previously-active model at the end (real swap, not a no-op).
#
# Output: bench/results/p0-qwen38-<ts>.log + .jsonl (one record per measurement)
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="$ROOT/bin:$PATH"
RESULTS="$ROOT/bench/results"
TS="$(date +%Y%m%d-%H%M%S)"
LOG="$RESULTS/p0-qwen38-$TS.log"
JSONL="$RESULTS/p0-qwen38-$TS.jsonl"
IMAGE="erfianugrah/llama-server:cuda12.8-sm120"
LLAMA_PIN="$(rg -o 'b[0-9]+' "$ROOT/llama-server.Dockerfile" | head -1)"
MODELS_DIR="${HOME}/docker-volumes/llama-server/models"
BENCH_CONTAINER="p0_qwen38"
BENCH_PORT=9999
MODEL_ID="Qwen3.8-27B-Q4_K_M"
IDLE_VRAM_MIB=8000
POLL_S=60
WAIT_MAX_S=43200  # 12 h

mkdir -p "$RESULTS"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
rec() { echo "$1" >> "$JSONL"; }

gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -dc '0-9'; }
proxy_ok() { curl -sf --max-time 2 http://127.0.0.1:11434/health >/dev/null 2>&1; }
status_field() { llmc status --json 2>/dev/null | jq -r "$1"; }

# ── Wait for GPU idle ─────────────────────────────────────────────────
wait_idle() {
    local waited=0 good=0 used locked switching
    while [ "$waited" -lt "$WAIT_MAX_S" ]; do
        used=$(gpu_used)
        local idle=1
        # P0_ASSUME_IDLE=1 skips the VRAM threshold (operator has confirmed no
        # game/local GPU load is running); lock + switching checks always apply.
        if [ "${P0_ASSUME_IDLE:-0}" != "1" ]; then
            [ "${used:-99999}" -ge "$IDLE_VRAM_MIB" ] && idle=0
        fi
        if [ "$idle" -eq 1 ] && proxy_ok; then
            locked=$(status_field '.mode.locked // empty')
            switching=$(status_field '.mode.switching // false')
            [ -n "$locked" ] && idle=0
            [ "$switching" = "true" ] && idle=0
        fi
        # proxy down + GPU idle is acceptable: we bring the stack up ourselves
        if [ "$idle" -eq 1 ]; then
            good=$((good + 1))
            [ "$good" -ge 3 ] && return 0
        else
            good=0
        fi
        sleep "$POLL_S"; waited=$((waited + POLL_S))
        [ $((waited % 1800)) -eq 0 ] && log "waiting for GPU idle (${waited}s, vram=${used}MiB)"
    done
    return 1
}

# ── Readiness probe: 1-token completion ───────────────────────────────
wait_ready() {  # $1=port $2=timeout_s
    local deadline=$(( $(date +%s) + $2 ))
    while [ "$(date +%s)" -lt "$deadline" ]; do
        if curl -sf --max-time 15 "http://127.0.0.1:$1/v1/chat/completions" \
            -H 'Content-Type: application/json' \
            -d "{\"model\":\"$MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1}" \
            >/dev/null 2>&1; then
            return 0
        fi
        sleep 5
    done
    return 1
}

llama_logs() { docker logs "$1" 2>&1 | tail -40; }

# ── Raw docker run for the ctx matrix ─────────────────────────────────
raw_run() {  # $1=ctx $2=slots
    docker rm -f "$BENCH_CONTAINER" >/dev/null 2>&1
    docker run -d --name "$BENCH_CONTAINER" --gpus all --shm-size 2g \
        -v "$MODELS_DIR:/models" \
        -p "$BENCH_PORT:8080" \
        -e MODEL_REPO=unsloth/Qwen3.8-27B-GGUF \
        -e MODEL_FILE=Qwen3.8-27B-Q4_K_M.gguf \
        -e MMPROJ_FILE=qwen38-mmproj.gguf \
        -e REASONING=on \
        -e CONTEXT_SIZE="$1" \
        -e PARALLEL_SLOTS="$2" \
        "$IMAGE" >/dev/null
}

# ══════════════════════════════════════════════════════════════════════
log "P0 qwen38 spike starting (llama.cpp pin: $LLAMA_PIN, image: $IMAGE)"
rec "{\"ts\":\"$TS\",\"kind\":\"meta\",\"llama_cpp\":\"$LLAMA_PIN\",\"image\":\"$IMAGE\",\"model_file\":\"$MODEL_ID.gguf\"}"

log "phase 0: waiting for GPU idle (lock-free, not switching, vram < ${IDLE_VRAM_MIB} MiB, 3 consecutive polls)"
if ! wait_idle; then
    log "FATAL: GPU did not go idle within ${WAIT_MAX_S}s - aborting, nothing touched"
    rec "{\"ts\":\"$TS\",\"kind\":\"abort\",\"reason\":\"gpu-never-idle\"}"
    exit 3
fi
log "GPU idle (vram=$(gpu_used) MiB)"

if ! proxy_ok; then
    log "proxy down - bringing stack up (make up)"
    make -C "$ROOT" up >>"$LOG" 2>&1 || { log "FATAL: make up failed"; exit 3; }
    for i in $(seq 1 24); do proxy_ok && break; sleep 5; done
    proxy_ok || { log "FATAL: proxy never came up"; exit 3; }
fi

PREV_MODEL="$(status_field '.active_model // empty')"
log "previous active model: ${PREV_MODEL:-none}"

restore() {
    log "restore: switching back to ${PREV_MODEL:-loop}"
    docker rm -f "$BENCH_CONTAINER" >/dev/null 2>&1
    llmc switch "${PREV_MODEL:-loop}" >>"$LOG" 2>&1 || log "WARN: restore switch failed - run 'llmc switch ${PREV_MODEL:-loop}' manually"
}
trap restore EXIT

# ── Spike 1: arch support via managed switch ──────────────────────────
log "spike 1: llmc switch qwen38 (arch support, managed path)"
if llmc switch qwen38 >>"$LOG" 2>&1 && wait_ready 11434 600; then
    VRAM=$(gpu_used)
    log "spike 1 PASS - loaded, vram=${VRAM} MiB"
    rec "{\"ts\":\"$TS\",\"kind\":\"spike1-arch\",\"ok\":true,\"vram_mib\":$VRAM,\"ctx\":131072,\"slots\":1}"
else
    log "spike 1 FAIL - model did not become ready; llama logs:"
    llama_logs llama_server | tee -a "$LOG"
    rec "{\"ts\":\"$TS\",\"kind\":\"spike1-arch\",\"ok\":false}"
    log "VERDICT: pinned image ($LLAMA_PIN) cannot load Qwen3.8-27B - needs llama.cpp bump (plan P0 step 1)"
    exit 2
fi

# ── Spike 3a: text completion smoke (managed, while qwen38 is loaded) ──
log "spike 3a: text completion smoke"
RESP=$(curl -sf --max-time 120 http://127.0.0.1:11434/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly one word: pong\"}],\"max_tokens\":64}" \
    | jq -r '.choices[0].message.content // .choices[0].message.reasoning_content // ""' | head -c 200)
log "text smoke response: ${RESP:-<empty>}"
rec "{\"ts\":\"$TS\",\"kind\":\"spike3-text\",\"ok\":$([ -n "$RESP" ] && echo true || echo false),\"response\":$(jq -Rn --arg s "$RESP" '$s')}"

# ── Spike 3b: vision smoke (managed, mmproj) ──────────────────────────
log "spike 3b: vision smoke (red-64.png)"
B64=$(base64 -w0 "$ROOT/bench/fixtures/red-64.png")
VRESP=$(curl -sf --max-time 180 http://127.0.0.1:11434/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"What colour is this image? Answer with one word.\"},{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,$B64\"}}]}],\"max_tokens\":128}" \
    | jq -r '.choices[0].message.content // ""' | head -c 300)
VOK=false; echo "$VRESP" | rg -qi 'red' && VOK=true
log "vision smoke response: ${VRESP:-<empty>} (pass=$VOK)"
rec "{\"ts\":\"$TS\",\"kind\":\"spike3-vision\",\"ok\":$VOK,\"response\":$(jq -Rn --arg s "$VRESP" '$s')}"

# ── Spike 2: KV/ctx fit matrix (raw docker, direct control) ───────────
log "spike 2: ctx fit matrix (raw docker runs; stopping managed llama first)"
docker rm -f llama_server >/dev/null 2>&1
for CTX in 65536 131072 196608; do
    for SLOTS in 1 2; do
        log "  ctx=$CTX slots=$SLOTS"
        raw_run "$CTX" "$SLOTS"
        if wait_ready "$BENCH_PORT" 600; then
            VRAM=$(gpu_used)
            EFF_CTX=$(docker logs "$BENCH_CONTAINER" 2>&1 | rg -o 'n_ctx = [0-9]+' | tail -1 | rg -o '[0-9]+' || echo "?")
            log "  FIT   ctx=$CTX slots=$SLOTS vram=${VRAM} MiB (effective n_ctx=$EFF_CTX)"
            rec "{\"ts\":\"$TS\",\"kind\":\"spike2-ctxfit\",\"ctx\":$CTX,\"slots\":$SLOTS,\"ok\":true,\"vram_mib\":$VRAM,\"effective_ctx\":${EFF_CTX:-0}}"
        else
            log "  FAIL  ctx=$CTX slots=$SLOTS (timeout/OOM); last logs:"
            llama_logs "$BENCH_CONTAINER" | tail -8 | tee -a "$LOG"
            rec "{\"ts\":\"$TS\",\"kind\":\"spike2-ctxfit\",\"ctx\":$CTX,\"slots\":$SLOTS,\"ok\":false}"
        fi
        docker rm -f "$BENCH_CONTAINER" >/dev/null 2>&1
    done
done

log "P0 complete - results: $JSONL"
exit 0
