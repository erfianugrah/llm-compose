#!/usr/bin/env bash
# p4-mtp.sh - MTP speculative-decoding spike, queued behind the P3 matrix.
# Plan: docs/plans/2026-08-17-mtp-speed-track.md
# S1 perf with draft-mtp, S2 ctx ceiling, S4 accuracy guard. Commits results.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$PWD/bin:$PATH"
MODELS_DIR="${HOME}/docker-volumes/llama-server/models"
IMAGE="erfianugrah/llama-server:cuda12.8-sm120"
log() { echo "[p4 $(date +%H:%M:%S)] $*"; }

# ── 0. wait for the P3 orchestrator ────────────────────────────────────
log "waiting for P3 orchestrator"
while pgrep -f p3-orchestrator >/dev/null 2>&1; do sleep 120; done
while pgrep -f "llmc bench" >/dev/null 2>&1; do sleep 60; done
log "P3 done - starting MTP spike"

# ── 1. qwen38-mtp preset ───────────────────────────────────────────────
# Presets are deduped by model_id (GGUF stem) and the proxy hard-fails on
# duplicates, so the MTP variant needs its own filename - a hardlink shares
# the inode, zero extra disk.
ln -f "$MODELS_DIR/Qwen3.8-27B-Q4_K_M.gguf" "$MODELS_DIR/Qwen3.8-27B-Q4_K_M-mtp.gguf"
python3 - <<'EOF'
from pathlib import Path
src = Path("models/qwen38.toml").read_text()
src = src.replace('name = "Qwen3.8 27B Dense - coding, vision, thinking (eval candidate)"',
                  'name = "Qwen3.8 27B Dense + MTP draft-mtp (speed spike)"')
src = src.replace('file = "Qwen3.8-27B-Q4_K_M.gguf"',
                  'file = "Qwen3.8-27B-Q4_K_M-mtp.gguf"', 1)
src = src.replace('repeat_penalty = 1.0', 'repeat_penalty = 1.0\nspec_type = "draft-mtp"', 1)
Path("models/qwen38-mtp.toml").write_text(src)
EOF
log "preset written"

# ── 2. rebuild images (new SPEC_TYPE support) + restart proxy ──────────
make build >>/tmp/p4-build.log 2>&1 || { log "FATAL: make build failed"; exit 1; }
make restart >>/tmp/p4-build.log 2>&1 || { log "FATAL: make restart failed"; exit 1; }
sleep 5
llmc health >/dev/null 2>&1 || curl -sf http://127.0.0.1:11434/health >/dev/null || { log "FATAL: proxy not healthy"; exit 1; }
log "images rebuilt, proxy restarted"

# ── 3. S1: perf with MTP ───────────────────────────────────────────────
log "S1: llmc bench perf --presets qwen38-mtp"
llmc bench perf --presets qwen38-mtp || log "S1 rc=$?"

# ── 4. S2: ctx ceiling with MTP (raw runs) ─────────────────────────────
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -dc '0-9'; }
wait_ready_raw() {
    local deadline=$(( $(date +%s) + 600 ))
    while [ "$(date +%s)" -lt "$deadline" ]; do
        curl -sf --max-time 15 http://127.0.0.1:9999/v1/chat/completions \
            -H 'Content-Type: application/json' \
            -d '{"model":"m","messages":[{"role":"user","content":"hi"}],"max_tokens":1}' >/dev/null 2>&1 && return 0
        sleep 5
    done
    return 1
}
docker rm -f llama_server >/dev/null 2>&1
for combo in "131072 2" "196608 2" "262144 1"; do
    set -- $combo; CTX=$1; SLOTS=$2
    docker rm -f p4_mtp >/dev/null 2>&1
    docker run -d --name p4_mtp --gpus all --shm-size 2g \
        -v "$MODELS_DIR:/models" -p 9999:8080 \
        -e MODEL_REPO=unsloth/Qwen3.8-27B-GGUF \
        -e MODEL_FILE=Qwen3.8-27B-Q4_K_M.gguf \
        -e MMPROJ_FILE=qwen38-mmproj.gguf \
        -e REASONING=on -e SPEC_TYPE=draft-mtp \
        -e CONTEXT_SIZE="$CTX" -e PARALLEL_SLOTS="$SLOTS" \
        "$IMAGE" >/dev/null
    if wait_ready_raw; then
        log "S2: FIT ctx=$CTX slots=$SLOTS vram=$(gpu_used) MiB (mtp on)"
        docker logs p4_mtp 2>&1 | rg -i "draft|accept" | tail -2 | sed 's/^/    /'
    else
        log "S2: FAIL ctx=$CTX slots=$SLOTS (mtp on)"
        docker logs p4_mtp 2>&1 | tail -5 | sed 's/^/    /'
    fi
    docker rm -f p4_mtp >/dev/null 2>&1
done

# ── 5. S4: accuracy guard ──────────────────────────────────────────────
log "S4: task guard on qwen38-mtp"
llmc bench tasks --presets qwen38-mtp --tasks t1-go-add-truncate,t2-go-fix-palindrome --runs 2 || log "S4 rc=$?"

# ── 6. restore + commit ────────────────────────────────────────────────
llmc switch loop >>/tmp/p4-build.log 2>&1 || true
git add models/qwen38-mtp.toml bench/results/ llmc/ llama-server-entrypoint.sh docs/plans/2026-08-17-mtp-speed-track.md
git commit -q -m "bench: MTP draft-mtp spike results (S1 perf, S2 ctx, S4 guard)" || log "nothing to commit"
git push -q || log "push failed - push manually"
log "=== P4 MTP SPIKE COMPLETE ==="
llmc bench report --compare qwen38 qwen38-mtp || true
