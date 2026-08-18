#!/usr/bin/env bash
# p5-effort-ngram.sh - sequential A/B: reasoning_effort medium, then ngram-mod.
# Background: docs/reference/speculative-decoding.md (ngram-mod mechanics,
# stuck-loop caveat) and the reasoning_effort lever note in models/qwen38.toml.
#
# Phase A (effort):  qwen38 (xhigh) vs qwen38-medium - perf + task suite.
# Phase B (ngram):   qwen38-ngram perf, run twice back-to-back. The ngram
#                    hash pool lives in the server process and dies on any
#                    preset switch, so pass 1 warms and pass 2 measures.
#                    Baseline = phase A's same-session qwen38 perf numbers.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$PWD/bin:$PATH"
log() { echo "[p5 $(date +%H:%M:%S)] $*"; }

# ── 0. canary gate (sensor suite must verify before scoring models) ────
log "canary gate: llmc bench tasks --verify-only"
llmc bench tasks --verify-only --presets qwen38 || { log "FATAL: canary gate failed"; exit 1; }
log "canary gate passed"

# ── 0b. server-health watchdog ─────────────────────────────────────────
# Two failure modes observed 2026-08-18 (b10362; first fixed by the b10472
# pin bump, second present on both builds):
# 1. Slot parking: an abandoned streaming request froze its slot forever
#    (is_processing=true, n_decoded static). Fixed in b10472 (verified:
#    slot now releases within ~20s of client disconnect).
# 2. Decode degradation over server lifetime: after ~10 min of agentic
#    task churn, decode collapses to ~0.5 tok/s with the GPU idling at
#    ~100W, while pp and MTP draft acceptance stay healthy. Upstream
#    reports the same shape (ggml-org/llama.cpp#27151: acceptance decays
#    over time, restart restores; #27296: MTP breaks after long/short
#    prompt mixes on Qwen3.8-27B). Fresh-server synthetic repros of every
#    request-shape variable (long prompt, long decode, stream, tools,
#    cache reuse) are all healthy - only a lived-in server crawls.
# Mitigation here: watch /slots every 60s; a slot that is processing but
# advancing < 100 tokens across two samples (healthy floor measured at
# 41 tok/s even for 2x53k-ctx concurrent) is wedged -> rm the container;
# the proxy respawns on the next request and the client errors/retries
# fast instead of burning a 3600s timeout at 0.5 tok/s.
slots_json() {
    docker exec llama_server curl -s --max-time 10 http://localhost:8080/slots 2>/dev/null \
        | jq -c '[.[] | select(.is_processing==true) | {s:.id, d:.next_token[0].n_decoded, p:.n_prompt_tokens_processed}]' 2>/dev/null \
        || echo "[]"
}
watchdog() {
    local prev="" cur
    while true; do
        cur=$(slots_json)
        if [ -n "$prev" ] && [ "$cur" != "[]" ]; then
            # wedged if identical (parked) or total decode advanced < 100 in 60s
            local delta
            delta=$(jq -n --argjson a "$prev" --argjson b "$cur" \
                '[($b[] | .d // 0)] | add as $nb | [($a[] | .d // 0)] | add as $na | $nb - $na' 2>/dev/null || echo 9999)
            if [ "$cur" = "$prev" ] || { [ "$delta" != "9999" ] && [ "$delta" -lt 100 ] 2>/dev/null; }; then
                log "watchdog: wedged server (delta=${delta} tok/60s, $cur) - respawning llama_server"
                docker rm -f llama_server >/dev/null 2>&1
                sleep 30
                cur=$(slots_json)
            fi
        fi
        prev="$cur"
        sleep 60
    done
}
watchdog &
WD_PID=$!
trap 'kill $WD_PID 2>/dev/null' EXIT

# ── 1. Phase A: reasoning_effort medium ────────────────────────────────
log "A1: perf qwen38 vs qwen38-medium"
llmc bench perf --presets qwen38,qwen38-medium || log "A1 rc=$?"

log "A2: tasks qwen38 vs qwen38-medium (runs=2)"
llmc bench tasks --presets qwen38,qwen38-medium --runs 2 || log "A2 rc=$?"

# ── 2. Phase B: ngram-mod (back-to-back: warm then measured) ───────────
log "B1: perf qwen38-ngram pass 1 (pool warmup - discard for comparison)"
llmc bench perf --presets qwen38-ngram || log "B1 rc=$?"

log "B2: perf qwen38-ngram pass 2 (warm pool - the measurement)"
llmc bench perf --presets qwen38-ngram || log "B2 rc=$?"

# stuck-loop watch (upstream PR #25819): draft reuse after failed verify
log "stuck-loop scan of server logs:"
docker logs llama_server 2>&1 | rg -i 'issue#23268|stuck|ngram' | tail -5 || log "  (no ngram/stuck lines)"

# ── 3. restore + commit results ────────────────────────────────────────
llmc switch qwen38 >/dev/null 2>&1 || true
git add models/qwen38-medium.toml models/qwen38-ngram.toml bench/results/ bench/p5-effort-ngram.sh
git commit -q -m "bench: P5 reasoning_effort medium + ngram-mod A/B results" \
    || log "nothing to commit"

log "=== P5 COMPLETE ==="
llmc bench report --compare qwen38 qwen38-medium || true
llmc bench report --compare qwen38 qwen38-ngram || true
