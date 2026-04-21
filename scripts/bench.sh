#!/usr/bin/env bash
# bench.sh — benchmark llama-server flag combinations
#
# Tests different KV cache, slot, batch, and context configurations.
# Reports prompt processing and generation tok/s for each combination.
#
# Usage:
#   ./scripts/bench.sh                    # bench current/default model
#   ./scripts/bench.sh --quick            # fast: old vs new flags only
#   BENCH_MODEL=qwen35 ./scripts/bench.sh # bench a specific preset
#
# Requires: docker, curl, python3, jq (optional)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="erfianugrah/llama-server:cuda12.8-sm120"
VOLUME_DIR="${HOME}/docker-volumes/llama-server"
MODELS_DIR="${VOLUME_DIR}/models"
PORT=9999
CONTAINER="bench_llama"
HEALTH_TIMEOUT=180
QUICK="${1:-}"

# Load model preset
MODEL="${BENCH_MODEL:-gemma4}"
PRESET="${SCRIPT_DIR}/models/${MODEL}.env"
if [ ! -f "$PRESET" ]; then
    echo "Error: preset '$PRESET' not found"
    echo "Available: $(ls -1 "${SCRIPT_DIR}/models/"*.env | xargs -I{} basename {} .env | tr '\n' ' ')"
    exit 1
fi

# Parse preset
get_val() { grep "^${1}=" "$PRESET" | cut -d= -f2; }
MODEL_REPO=$(get_val MODEL_REPO)
MODEL_FILE=$(get_val MODEL_FILE)
MMPROJ_FILE=$(get_val MMPROJ_FILE)
TEMPLATE_FILE=$(get_val TEMPLATE_FILE)
REASONING=$(get_val REASONING)
MODEL_NAME=$(get_val MODEL_NAME)

echo "=============================================="
echo "Benchmark: ${MODEL_NAME}"
echo "=============================================="
echo ""

# Build common flags
COMMON_FLAGS=(
    --hf-repo "$MODEL_REPO"
    --hf-file "$MODEL_FILE"
    --port 8080 --host 0.0.0.0
    -ngl 99 --flash-attn on
    --threads 8 --threads-batch 8
    --no-warmup --jinja
)
[ -n "$MMPROJ_FILE" ] && COMMON_FLAGS+=(--mmproj "/models/${MMPROJ_FILE}")
[ -n "$TEMPLATE_FILE" ] && COMMON_FLAGS+=(--chat-template-file "/models/${TEMPLATE_FILE}")
[ -n "$REASONING" ] && COMMON_FLAGS+=(--reasoning "$REASONING")

# Test prompts
SHORT_PROMPT='{"model":"test","messages":[{"role":"user","content":"What is 2+2? One word."}],"max_tokens":20}'
LONG_PROMPT='{"model":"test","messages":[{"role":"user","content":"Write a Python function that implements binary search on a sorted list. Include docstring and type hints. Then write 5 unit tests for it using pytest."}],"max_tokens":500}'

# Configurations to test
if [ "$QUICK" = "--quick" ]; then
    CONFIGS=(
        "q8_0:q8_0:1:4096:65536:q8 K+V, 1 slot, 4K batch"
        "q8_0:q4_0:1:4096:65536:q8 K + q4 V, 1 slot, 4K batch"
    )
else
    CONFIGS=(
        "q8_0:q8_0:4:2048:65536:old defaults (4 slot, q8 V, 2K batch)"
        "q8_0:q8_0:1:2048:65536:single slot only"
        "q8_0:q4_0:1:2048:65536:+ q4_0 V cache"
        "q8_0:q4_0:1:4096:65536:+ 4K batch"
        "q8_0:q8_0:1:4096:65536:single slot + 4K batch + q8 V"
        "q8_0:q8_0:1:4096:131072:best + 131K context"
    )
fi

cleanup() {
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
    # Ensure port is freed
    while lsof -i ":${PORT}" >/dev/null 2>&1; do sleep 1; done 2>/dev/null || true
}
trap cleanup EXIT

wait_for_health() {
    local elapsed=0
    while ! curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; do
        elapsed=$((elapsed + 2))
        if [ "$elapsed" -ge "$HEALTH_TIMEOUT" ]; then
            echo "  TIMEOUT after ${HEALTH_TIMEOUT}s. Last logs:"
            docker logs "$CONTAINER" 2>&1 | tail -5
            return 1
        fi
        # Show progress every 30s
        if [ $((elapsed % 30)) -eq 0 ] && [ "$elapsed" -gt 0 ]; then
            local status
            status=$(docker logs "$CONTAINER" 2>&1 | tail -1 | head -c 80)
            echo "  ... waiting (${elapsed}s): ${status}"
        fi
        sleep 2
    done
    return 0
}

extract_timings() {
    python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    t = d.get('timings', {})
    print(f'{t.get(\"prompt_per_second\",0):.0f}|{t.get(\"predicted_per_second\",0):.0f}|{t.get(\"predicted_n\",0)}')
except:
    print('0|0|0')
" 2>/dev/null
}

run_test() {
    local ctk="$1" ctv="$2" np="$3" batch="$4" ctx="$5" label="$6"

    cleanup
    sleep 2

    # Start server (no --rm so we can inspect logs on failure)
    if ! docker run -d --name "$CONTAINER" --gpus all \
        -v "${VOLUME_DIR}:/root/.cache" \
        -v "${MODELS_DIR}:/models" \
        -p "${PORT}:8080" \
        "$IMAGE" \
        "${COMMON_FLAGS[@]}" \
        -ctk "$ctk" -ctv "$ctv" -np "$np" \
        -b "$batch" -ub "$batch" -c "$ctx" 2>&1; then
        echo "  FAIL: $label (container failed to start)"
        cleanup
        return
    fi

    # Check container is actually running
    sleep 2
    if ! docker inspect "$CONTAINER" --format='{{.State.Running}}' 2>/dev/null | grep -q true; then
        echo "  FAIL: $label (crashed on startup)"
        docker logs "$CONTAINER" 2>&1 | tail -3
        cleanup
        return
    fi

    # Wait for health with progress
    if ! wait_for_health; then
        cleanup
        return
    fi

    # Warmup (2 requests, discard)
    for _ in 1 2; do
        curl -sf "http://localhost:${PORT}/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d "$SHORT_PROMPT" >/dev/null 2>&1 || true
    done
    sleep 1

    # Short prompt test (3 runs, take best)
    local best_prompt=0 best_gen=0
    for _ in 1 2 3; do
        local result
        result=$(curl -sf "http://localhost:${PORT}/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d "$SHORT_PROMPT" 2>/dev/null)
        IFS='|' read -r pp gp _ <<< "$(echo "$result" | extract_timings)"
        [ "${pp:-0}" -gt "$best_prompt" ] && best_prompt="$pp"
        [ "${gp:-0}" -gt "$best_gen" ] && best_gen="$gp"
    done

    # Long prompt test (1 run)
    local long_result
    long_result=$(curl -sf "http://localhost:${PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "$LONG_PROMPT" 2>/dev/null)
    local long_pp long_gp long_tokens
    IFS='|' read -r long_pp long_gp long_tokens <<< "$(echo "$long_result" | extract_timings)"

    # CPU + VRAM check
    local cpu vram
    cpu=$(docker stats --no-stream "$CONTAINER" 2>/dev/null | tail -1 | awk '{print $3}' || echo "?")
    vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo "?")

    # Actual context from server
    local actual_ctx
    actual_ctx=$(docker logs "$CONTAINER" 2>&1 | grep "n_ctx " | tail -1 | grep -oP '\d+' || echo "?")

    printf "  %-45s | %4s pp %4s gen | %4s pp %4s gen (%3s tok) | ctx:%s CPU:%s VRAM:%sMiB\n" \
        "$label" "${best_prompt}" "${best_gen}" "${long_pp}" "${long_gp}" "${long_tokens}" \
        "${actual_ctx}" "${cpu}" "${vram}"

    cleanup
}

# Header
echo ""
printf "  %-45s | %-15s | %-26s | %s\n" "Configuration" "Short (tok/s)" "Long (tok/s)" "Resources"
printf "  %-45s-+-%-15s-+-%-26s-+-%s\n" "---------------------------------------------" "---------------" "--------------------------" "----------------------------"

# Run all configs
for config in "${CONFIGS[@]}"; do
    IFS=':' read -r ctk ctv np batch ctx label <<< "$config"
    run_test "$ctk" "$ctv" "$np" "$batch" "$ctx" "$label"
done

echo ""
echo "Done."
echo "  pp = prompt processing, gen = generation (higher = better)"
echo "  CPU < 1% = GPU inference (good). CPU > 100% = CPU fallback (bad)."
