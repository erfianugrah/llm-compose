#!/usr/bin/env bash
# bench-perf.sh — per-quant inference efficiency benchmark.
#
# For each quant in bench/quants.txt:
#   1. Start llama-server with the GGUF (via docker run, not compose)
#   2. Wait for /health
#   3. Warm up (2 short requests)
#   4. Measure TTFT via streaming SSE (5 short prompts, take median)
#   5. Measure throughput tok/s via /v1/chat/completions (3 long prompts)
#   6. Capture peak VRAM (nvidia-smi polled every 0.5s during run)
#   7. Capture peak container RAM (docker stats polled)
#   8. Append CSV row to bench/results/perf-<timestamp>.csv
#
# Usage:
#   bench/bench-perf.sh                          # full sweep
#   bench/bench-perf.sh --only Q4_K_M,Q8_0       # subset
#   BENCH_REPO=unsloth/Qwen3.6-27B-GGUF bench/bench-perf.sh
#
# Requires: docker, curl, python3, nvidia-smi
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BENCH_DIR="$ROOT/bench"
RESULTS_DIR="$BENCH_DIR/results"
QUANTS_FILE="${BENCH_QUANTS:-$BENCH_DIR/quants.txt}"
IMAGE="erfianugrah/llama-server:cuda12.8-sm120"
CACHE_DIR="${HOME}/docker-volumes/llama-server"
PORT="${BENCH_PORT:-9999}"
CONTAINER="bench_perf"
HEALTH_TIMEOUT=900
REPO="${BENCH_REPO:-unsloth/Qwen3.6-27B-GGUF}"
CTX="${BENCH_CTX:-32768}"
NGL="${BENCH_NGL:-99}"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
CSV="$RESULTS_DIR/perf-$TIMESTAMP.csv"
ONLY=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --only) ONLY="$2"; shift 2 ;;
        --csv)  CSV="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

mkdir -p "$RESULTS_DIR"
echo "label,file,size_gb,ttft_ms_p50,ttft_ms_p95,gen_tok_s,prompt_tok_s,vram_peak_mib,ram_peak_mib,gen_tokens" > "$CSV"

cleanup() {
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
    pkill -f "vram_poll_$$" 2>/dev/null || true
}
trap cleanup EXIT

wait_health() {
    # The deadline scales with model size: the container downloads the GGUF
    # from HF BEFORE llama-server starts loading, and a 17-29 GB download
    # does not fit in a fixed 900s window on a slow link (observed 2026-08-12:
    # Q4_K_M and UD-Q4_K_XL both FAILed on timeout while still downloading).
    # Budget: 900s floor + 120s per GB.
    local size_int=${3:-0}; size_int=${size_int%.*}  # bash $(( )) is integer-only
    local timeout_s=$(( HEALTH_TIMEOUT + size_int * 120 ))
    local elapsed=0
    while ! curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; do
        elapsed=$((elapsed + 3))
        [ "$elapsed" -ge "$timeout_s" ] && { echo "  TIMEOUT"; docker logs "$CONTAINER" 2>&1 | tail -10; return 1; }
        if [ $((elapsed % 30)) -eq 0 ]; then
            echo "  ... loading/downloading (${elapsed}s / ${timeout_s}s)"
        fi
        sleep 3
    done
}

# Stream a chat completion and print TTFT (ms) and total tokens to stdout
measure_ttft() {
    local prompt="$1"
    python3 - "$PORT" "$prompt" <<'PY'
import json, sys, time, urllib.request
port, prompt = sys.argv[1], sys.argv[2]
body = json.dumps({
    "model": "bench",
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 128,
    "stream": True,
    "temperature": 0.0,
}).encode()
req = urllib.request.Request(
    f"http://localhost:{port}/v1/chat/completions",
    data=body,
    headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
)
t0 = time.perf_counter()
ttft = None
n_tok = 0
try:
    with urllib.request.urlopen(req, timeout=120) as r:
        buf = b""
        while True:
            chunk = r.read(64)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                raw, _, buf = buf.partition(b"\n")
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    raise StopIteration
                try:
                    d = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                delta = (d.get("choices") or [{}])[0].get("delta") or {}
                tok = delta.get("content") or delta.get("reasoning_content") or ""
                if tok:
                    if ttft is None:
                        ttft = (time.perf_counter() - t0) * 1000
                    n_tok += 1
except StopIteration:
    pass
except Exception as e:
    print(f"# ttft err: {e}", file=sys.stderr)
print(f"{ttft:.1f}|{n_tok}" if ttft else "0|0")
PY
}

# Non-streaming: get llama.cpp server timings (prompt_per_second, predicted_per_second)
measure_throughput() {
    local prompt="$1"
    curl -sf "http://localhost:${PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "$(python3 -c "import json,sys; print(json.dumps({'model':'bench','messages':[{'role':'user','content':sys.argv[1]}],'max_tokens':500,'temperature':0.0}))" "$prompt")" \
        | python3 -c "
import json, sys
d = json.load(sys.stdin)
t = d.get('timings', {})
print(f\"{t.get('prompt_per_second',0):.1f}|{t.get('predicted_per_second',0):.1f}|{t.get('predicted_n',0)}\")
" 2>/dev/null || echo "0|0|0"
}

p50() { python3 -c "import sys,statistics; v=[float(x) for x in sys.argv[1:] if x]; print(f'{statistics.median(v):.1f}' if v else '0')" "$@"; }
p95() { python3 -c "import sys; v=sorted(float(x) for x in sys.argv[1:] if x); print(f'{v[int(0.95*(len(v)-1))]:.1f}' if v else '0')" "$@"; }

run_quant() {
    local label="$1" file="$2" size_gb="$3" notes="$4"
    echo ""
    echo "================================================================"
    echo " $label  ($file, ~${size_gb} GB) — $notes"
    echo "================================================================"

    cleanup
    sleep 2

    # Start llama-server with this GGUF. The image entrypoint resolves the
    # model itself from MODEL_FILE/MODEL_REPO env vars (local /models/$MODEL_FILE
    # first, else --hf-repo/--hf-file download into /root/.cache). Do NOT pass
    # -m/--hf-* via argv: the entrypoint always prepends its own model args, and
    # with MODEL_FILE unset that is an empty --hf-repo, which is fatal
    # (observed 2026-08-13: bench_perf exit 1, "invalid HF repo format", then
    # the health wait hangs until timeout).
    if [ -f "${CACHE_DIR}/models/$file" ]; then
        echo "  using local /models/$file"
    fi
    # Entrypoint hardcodes port/host/ngl/flash-attn/ctk/ctv/ctx(np)/threads/
    # jinja/metrics. Appended args below override via llama.cpp last-wins for
    # the few knobs the bench changes (batch size, no-warmup).
    docker run -d --name "$CONTAINER" --gpus all \
        -v "${CACHE_DIR}:/root/.cache" \
        -v "${CACHE_DIR}/models:/models" \
        -e MODEL_REPO="$REPO" \
        -e MODEL_FILE="$file" \
        -e CONTEXT_SIZE="$CTX" \
        -p "${PORT}:8080" \
        --shm-size 2g \
        "$IMAGE" \
        -ngl "$NGL" \
        -b 4096 -ub 4096 \
        --no-warmup >/dev/null

    if ! wait_health "$label" "$file" "$size_gb"; then
        echo "$label,$file,$size_gb,FAIL,FAIL,0,0,0,0,0" >> "$CSV"
        cleanup
        return
    fi

    # Background VRAM/RAM poller
    local poll_log; poll_log="$(mktemp)"
    (
        exec -a "vram_poll_$$" bash -c '
        while sleep 0.5; do
            v=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
            r=$(docker stats --no-stream --format "{{.MemUsage}}" '"$CONTAINER"' 2>/dev/null | awk "{print \$1}")
            echo "$v $r"
        done > '"$poll_log"
    ) &
    local poll_pid=$!

    # Warmup
    for _ in 1 2; do measure_ttft "What is 2+2?" >/dev/null 2>&1 || true; done

    # TTFT — 5 short prompts
    echo "  measuring TTFT..."
    local ttfts=()
    for _ in 1 2 3 4 5; do
        local r; r=$(measure_ttft "Reply with one word: yes")
        local t="${r%|*}"
        ttfts+=("$t")
    done

    # Throughput — 3 long prompts, take best
    echo "  measuring throughput..."
    local best_gen=0 best_pp=0 last_tok=0
    for _ in 1 2 3; do
        local r; r=$(measure_throughput "Write a Python function for binary search with type hints, docstring, and 5 pytest unit tests covering edge cases.")
        local pp="${r%%|*}"; local rest="${r#*|}"; local gp="${rest%|*}"; local tk="${rest##*|}"
        awk -v a="$gp" -v b="$best_gen" 'BEGIN{exit !(a+0>b+0)}' && best_gen="$gp"
        awk -v a="$pp" -v b="$best_pp"  'BEGIN{exit !(a+0>b+0)}' && best_pp="$pp"
        last_tok="$tk"
    done

    # Stop poller, compute peak
    kill "$poll_pid" 2>/dev/null || true
    sleep 1
    local vram_peak=0 ram_peak=0
    if [ -s "$poll_log" ]; then
        vram_peak=$(awk '{if ($1+0>m) m=$1+0} END{print m+0}' "$poll_log")
        ram_peak=$(awk '{r=$2+0; u=$2; if (u~/GiB/) r*=1024; else if (u~/kB/) r/=1024; if (r>m) m=r} END{print m+0}' "$poll_log")
    fi
    rm -f "$poll_log"

    local ttft_p50 ttft_p95
    ttft_p50=$(p50 "${ttfts[@]}")
    ttft_p95=$(p95 "${ttfts[@]}")

    printf "  TTFT p50=%sms p95=%sms | gen=%s tok/s | pp=%s tok/s | VRAM=%s MiB | RAM=%s MiB\n" \
        "$ttft_p50" "$ttft_p95" "$best_gen" "$best_pp" "$vram_peak" "$ram_peak"

    echo "$label,$file,$size_gb,$ttft_p50,$ttft_p95,$best_gen,$best_pp,$vram_peak,$ram_peak,$last_tok" >> "$CSV"
    cleanup
}

# Ensure no other GPU service is hogging the card. whisper-live keeps
# large-v3 resident (measured ~5.6 GB VRAM on 2026-08-13) - with it up,
# Q8_0 (28.6 GB) cannot fit and --fit would silently shrink ctx, producing
# incomparable numbers. NOTE: not restarted by this script - `docker start`
# them after, and beware an active llmc loop will re-grab the GPU via the
# proxy mid-run.
echo "Stopping llama_server, comfyui, lora_train, whisper GPU services..."
for container in llama_server comfyui lora_train whisper-transcribe-whisper-1 whisper-transcribe-whisper-live-1; do
    docker rm -f "$container" >/dev/null 2>&1 || true
done

echo ""
echo "Quant sweep — repo=$REPO ctx=$CTX -> $CSV"
echo ""

while IFS=':' read -r label file size_gb notes; do
    [[ -z "$label" || "$label" =~ ^# ]] && continue
    if [ -n "$ONLY" ] && [[ ",$ONLY," != *",$label,"* ]]; then continue; fi
    run_quant "$label" "$file" "$size_gb" "$notes"
done < "$QUANTS_FILE"

echo ""
echo "Done. Results: $CSV"
column -t -s, "$CSV" | head -30
