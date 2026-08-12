#!/usr/bin/env bash
# bench-quants.sh — full quant sweep: perf + accuracy.
#
# For each quant in bench/quants.txt:
#   1. Boot llama-server with the GGUF (via bench-perf.sh stays-up mode? no —
#      we run the perf script which boots+benches+stops, then re-boot for
#      accuracy. Simpler: boot once, run perf inline, then run evals against
#      the same instance, then stop.)
#   2. Measure perf (TTFT, throughput, peak VRAM, peak RAM)
#   3. Run accuracy evals via bench-eval container (HumanEval / HellaSwag / BFCL)
#   4. Append a combined row to bench/results/sweep-<ts>.csv
#
# Usage:
#   bench/bench-quants.sh                 # full sweep, all evals
#   bench/bench-quants.sh --quick         # smaller subsets, faster
#   bench/bench-quants.sh --perf-only     # skip accuracy
#   bench/bench-quants.sh --only Q4_K_M,Q6_K
#
# Time estimate per quant:
#   - perf:       3-5 min
#   - HumanEval:  10-20 min (164 problems, greedy)
#   - HellaSwag:  5-15 min (subset 1000) | 60+ min (full 10K)
#   - BFCL:       10-30 min (subset)
# Full sweep of 8 quants × full evals = ~6-10 hours.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BENCH_DIR="$ROOT/bench"
RESULTS_DIR="$BENCH_DIR/results"
QUANTS_FILE="${BENCH_QUANTS:-$BENCH_DIR/quants.txt}"
IMAGE="erfianugrah/llama-server:cuda12.8-sm120"
EVAL_IMAGE="erfianugrah/bench-eval:latest"
CACHE_DIR="${HOME}/docker-volumes/llama-server"
BENCH_CACHE="${HOME}/docker-volumes/bench-cache"
PORT=9999
CONTAINER="bench_llama"
HEALTH_TIMEOUT=900
REPO="${BENCH_REPO:-unsloth/Qwen3.6-27B-GGUF}"
CTX="${BENCH_CTX:-32768}"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
CSV="$RESULTS_DIR/sweep-$TIMESTAMP.csv"

# Eval subsets
HUMANEVAL_N=0      # 0 = all 164
HELLASWAG_N=1000   # full = 10042
BFCL_N=100

PERF_ONLY=0
SKIP_PERF=0
ONLY=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick)       HUMANEVAL_N=40; HELLASWAG_N=200; BFCL_N=40; shift ;;
        --perf-only)   PERF_ONLY=1; shift ;;
        --skip-perf)   SKIP_PERF=1; shift ;;
        --only)        ONLY="$2"; shift 2 ;;
        --humaneval-n) HUMANEVAL_N="$2"; shift 2 ;;
        --hellaswag-n) HELLASWAG_N="$2"; shift 2 ;;
        --bfcl-n)      BFCL_N="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

mkdir -p "$RESULTS_DIR" "$BENCH_CACHE"
echo "label,file,size_gb,ttft_ms_p50,ttft_ms_p95,gen_tok_s,prompt_tok_s,vram_peak_mib,ram_peak_mib,humaneval_pass1,humaneval_pass1_plus,hellaswag_acc,hellaswag_acc_norm,bfcl_overall" > "$CSV"

cleanup() {
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
    docker rm -f bench_eval >/dev/null 2>&1 || true
    pkill -f "vram_poll_$$" 2>/dev/null || true
}
trap cleanup EXIT

wait_health() {
    local elapsed=0
    while ! curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; do
        elapsed=$((elapsed + 3))
        [ "$elapsed" -ge "$HEALTH_TIMEOUT" ] && { docker logs "$CONTAINER" 2>&1 | tail -10; return 1; }
        if [ $((elapsed % 30)) -eq 0 ]; then echo "  ... loading (${elapsed}s)"; fi
        sleep 3
    done
}

start_server() {
    local file="$1"
    cleanup
    sleep 2
    docker run -d --name "$CONTAINER" --gpus all \
        -v "${CACHE_DIR}:/root/.cache" \
        -v "${CACHE_DIR}/models:/models" \
        -p "${PORT}:8080" \
        --shm-size 2g \
        "$IMAGE" \
        --hf-repo "$REPO" --hf-file "$file" \
        --port 8080 --host 0.0.0.0 \
        -ngl 99 --flash-attn on \
        -ctk q8_0 -ctv q8_0 \
        -c "$CTX" \
        -np 1 -b 4096 -ub 4096 \
        --threads 8 --threads-batch 8 \
        --jinja --no-warmup --metrics >/dev/null
    wait_health
}

# Stream-based TTFT measurement (Python embedded)
ttft_one() {
    python3 - "$PORT" "$1" <<'PY'
import json, sys, time, urllib.request
port, prompt = sys.argv[1], sys.argv[2]
body = json.dumps({"model":"bench","messages":[{"role":"user","content":prompt}],
                   "max_tokens":64,"stream":True,"temperature":0.0}).encode()
req = urllib.request.Request(
    f"http://localhost:{port}/v1/chat/completions",
    data=body,
    headers={"Content-Type":"application/json","Accept":"text/event-stream"},
)
t0 = time.perf_counter()
ttft = None
try:
    with urllib.request.urlopen(req, timeout=120) as r:
        buf = b""
        while ttft is None:
            chunk = r.read(64)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                raw, _, buf = buf.partition(b"\n")
                line = raw.decode("utf-8","ignore").strip()
                if not line.startswith("data:"):
                    continue
                p = line[5:].strip()
                if p == "[DONE]":
                    ttft = ttft or 0
                    break
                try:
                    d = json.loads(p)
                except json.JSONDecodeError:
                    continue
                ch = (d.get("choices") or [{}])[0]
                delta = ch.get("delta") or {}
                # Treat first token of any kind as TTFT (reasoning or content).
                # Qwen3 thinking models emit reasoning_content first.
                tok = delta.get("content") or delta.get("reasoning_content") or ""
                if tok:
                    ttft = (time.perf_counter()-t0)*1000
                    break
except Exception as e:
    print(f"# ttft err: {e}", file=sys.stderr)
print(f"{ttft:.1f}" if ttft else "0")
PY
}

throughput_one() {
    curl -sf "http://localhost:${PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "$(python3 -c "import json,sys; print(json.dumps({'model':'bench','messages':[{'role':'user','content':sys.argv[1]}],'max_tokens':500,'temperature':0.0}))" "$1")" \
        | python3 -c "
import json,sys
d = json.load(sys.stdin); t = d.get('timings',{})
print(f\"{t.get('prompt_per_second',0):.1f}|{t.get('predicted_per_second',0):.1f}\")
" 2>/dev/null || echo "0|0"
}

run_perf() {
    local poll_log; poll_log="$(mktemp)"
    (
        while sleep 0.5; do
            v=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
            r=$(docker stats --no-stream --format "{{.MemUsage}}" "$CONTAINER" 2>/dev/null | awk '{print $1}')
            echo "$v $r"
        done > "$poll_log"
    ) &
    local poll_pid=$!

    # Warmup
    for _ in 1 2; do ttft_one "warmup" >/dev/null 2>&1 || true; done

    # TTFT × 5
    local ttfts=()
    for _ in 1 2 3 4 5; do ttfts+=("$(ttft_one 'Reply with one word: yes')"); done

    # Throughput × 3 (best)
    local best_gen=0 best_pp=0
    for _ in 1 2 3; do
        local r; r=$(throughput_one "Write a Python binary search function with type hints, docstring, and 5 pytest unit tests covering edge cases.")
        local pp="${r%|*}" gp="${r#*|}"
        awk -v a="$gp" -v b="$best_gen" 'BEGIN{exit !(a+0>b+0)}' && best_gen="$gp"
        awk -v a="$pp" -v b="$best_pp"  'BEGIN{exit !(a+0>b+0)}' && best_pp="$pp"
    done

    kill "$poll_pid" 2>/dev/null || true
    sleep 1
    local vram=0 ram=0
    if [ -s "$poll_log" ]; then
        vram=$(awk '{if ($1+0>m) m=$1+0} END{print m+0}' "$poll_log")
        ram=$(awk '{r=$2+0; u=$2; if (u~/GiB/) r*=1024; else if (u~/kB/) r/=1024; if (r>m) m=r} END{print m+0}' "$poll_log")
    fi
    rm -f "$poll_log"

    local p50 p95
    p50=$(python3 -c "import statistics,sys; v=[float(x) for x in sys.argv[1:] if x and x!='0']; print(f'{statistics.median(v):.1f}' if v else '0')" "${ttfts[@]}")
    p95=$(python3 -c "import sys; v=sorted(float(x) for x in sys.argv[1:] if x and x!='0'); print(f'{v[int(0.95*(len(v)-1))]:.1f}' if v else '0')" "${ttfts[@]}")

    # Stash perf metrics in env vars for caller
    PERF_TTFT_P50="$p50" PERF_TTFT_P95="$p95"
    PERF_GEN="$best_gen" PERF_PP="$best_pp"
    PERF_VRAM="$vram"   PERF_RAM="$ram"
}

run_evals() {
    local label="$1"
    local out="$RESULTS_DIR/eval-$label-$TIMESTAMP.json"
    local flags=("--label" "$label" "--out" "/results/$(basename "$out")"
                 "--base-url" "http://host.docker.internal:${PORT}/v1"
                 "--model" "bench"
                 "--humaneval" "--humaneval-samples" "$HUMANEVAL_N"
                 "--hellaswag" "--hellaswag-subset" "$HELLASWAG_N"
                 "--bfcl"      "--bfcl-subset"      "$BFCL_N")

    docker run --rm --name bench_eval \
        --add-host=host.docker.internal:host-gateway \
        -v "$RESULTS_DIR:/results" \
        -v "$BENCH_CACHE:/cache" \
        -e HF_HOME=/cache/huggingface \
        "$EVAL_IMAGE" "${flags[@]}" || {
            echo "  eval container failed (continuing)"; echo "{}" > "$out"
        }
    echo "$out"
}

parse_evals() {
    local json="$1"
    python3 - "$json" <<'PY'
import json, sys
try:
    d = json.loads(open(sys.argv[1]).read())
except Exception:
    print("||||"); sys.exit(0)
he = d.get("humaneval") or {}
hs = d.get("hellaswag") or {}
bf = d.get("bfcl") or {}
def f(x): return f"{x:.4f}" if isinstance(x,(int,float)) else ""
print(f"{f(he.get('pass@1'))}|{f(he.get('pass@1_plus'))}|{f(hs.get('acc'))}|{f(hs.get('acc_norm'))}|{f(bf.get('overall'))}")
PY
}

# Free GPU
echo "Stopping llama_server, comfyui, lora_train to free GPU..."
for container in llama_server comfyui lora_train; do
    docker rm -f "$container" >/dev/null 2>&1 || true
done

# Build eval image if needed
if [ "$PERF_ONLY" -eq 0 ]; then
    if ! docker image inspect "$EVAL_IMAGE" >/dev/null 2>&1; then
        echo "Building $EVAL_IMAGE (one-time, ~5 min)..."
        docker build -t "$EVAL_IMAGE" -f "$BENCH_DIR/Dockerfile.eval" "$BENCH_DIR"
    fi
fi

echo ""
echo "Quant sweep — repo=$REPO ctx=$CTX -> $CSV"
echo "Evals: HE_n=$HUMANEVAL_N HS_n=$HELLASWAG_N BFCL_n=$BFCL_N (perf_only=$PERF_ONLY)"
echo ""

while IFS=':' read -r label file size_gb notes; do
    [[ -z "$label" || "$label" =~ ^# ]] && continue
    if [ -n "$ONLY" ] && [[ ",$ONLY," != *",$label,"* ]]; then continue; fi

    echo ""
    echo "================================================================"
    echo " $label  ($file, ~${size_gb} GB) — $notes"
    echo "================================================================"

    if ! start_server "$file"; then
        echo "  server failed to start; skipping"
        echo "$label,$file,$size_gb,FAIL,FAIL,0,0,0,0,,,,," >> "$CSV"
        continue
    fi

    PERF_TTFT_P50=0 PERF_TTFT_P95=0 PERF_GEN=0 PERF_PP=0 PERF_VRAM=0 PERF_RAM=0
    if [ "$SKIP_PERF" -eq 0 ]; then
        echo "  measuring perf..."
        run_perf
        printf "  TTFT p50=%sms p95=%sms | gen=%s tok/s | pp=%s tok/s | VRAM=%s MiB | RAM=%s MiB\n" \
            "$PERF_TTFT_P50" "$PERF_TTFT_P95" "$PERF_GEN" "$PERF_PP" "$PERF_VRAM" "$PERF_RAM"
    fi

    eval_metrics="||||"
    if [ "$PERF_ONLY" -eq 0 ]; then
        echo "  running accuracy evals (this is the slow part)..."
        eval_json=$(run_evals "$label")
        eval_metrics=$(parse_evals "$eval_json")
        echo "  evals: $eval_metrics  ->  $eval_json"
    fi

    echo "$label,$file,$size_gb,$PERF_TTFT_P50,$PERF_TTFT_P95,$PERF_GEN,$PERF_PP,$PERF_VRAM,$PERF_RAM,$eval_metrics" >> "$CSV"
done < "$QUANTS_FILE"

cleanup

echo ""
echo "================================================================"
echo " Done. Results: $CSV"
echo "================================================================"
column -t -s, "$CSV"
echo ""
echo "Render report:  python3 bench/bench-report.py $CSV"
