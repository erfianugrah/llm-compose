#!/usr/bin/env bash
# p5-effort-ngram.sh - v4 (trimmed finish, 2026-08-18 evening)
#
# Prior phases established (all in bench/results/runs.jsonl + commit 8c35091):
# - MTP (draft-mtp) decode degrades to ~0.5 tok/s after ~10 min of agentic
#   churn on b10362 AND b10472; spec-off sustains 55-66 tok/s indefinitely
#   (t3 ran its full 5433s/8-iteration arc at speed in the nospec control).
#   => both effort arms run SPEC-OFF; ngram-mod is evaluated SOLO.
# - A1 perf (spec-off, 2 reps): xhigh 73.2-73.7 vs medium 72.5-74.1 gen
#   tok/s, TTFT ~equal - synthetic perf shows parity; the effort decision
#   rides on task-suite wall time.
# - xhigh ceiling tasks already measured: t3 FAIL 5433s/8iter (same-day
#   control), t6 FAIL 3x historical (3000-4000s). Not re-measured here.
# - xhigh fast tasks measured: t1 42.5/39.0s, t2 19.7/1308.5s (binge!).
#
# This script therefore collects ONLY the missing cells:
#   A2a: nospec arm t4/t5 x2; medium-nospec arm t1/t2/t4/t5 x2
#   A2b: medium-nospec arm t3/t6 x1   <- the decision-driving measurement
#   B:   qwen38-ngramsolo perf, cold-pool then warm-pool passes
#
# POSTSCRIPT 2026-08-19 (p6 validation): ngram-mod ALSO degrades under
# task-suite churn (same ~0.5 tok/s signature as MTP - pp fast, GPU idle,
# restart restores). The adopted qwen38 default is medium + NO
# speculation; the ngram variant presets are deleted. Keep this script
# for the watchdog + phase structure.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$PWD/bin:$PATH"
log() { echo "[p5 $(date +%H:%M:%S)] $*"; }

# ── 0. canary gate (sensor suite must verify before scoring models) ────
log "canary gate: llmc bench tasks --verify-only"
llmc bench tasks --verify-only --presets qwen38 || { log "FATAL: canary gate failed"; exit 1; }
log "canary gate passed"

# ── 0b. server-health watchdog ─────────────────────────────────────────
# b10472 fixed abandoned-stream slot parking; the MTP-lifetime degradation
# is moot here (all arms spec-off / draftless). Kept as a pure backstop:
# same task on the same slot with no decode AND no pp progress across a
# 60s window -> respawn the container (proxy respawns on next request).
slots_json() {
    docker exec llama_server curl -s --max-time 10 http://localhost:8080/slots 2>/dev/null \
        | jq -c '[.[] | select(.is_processing==true) | {s:.id, t:.id_task, d:.next_token[0].n_decoded, p:.n_prompt_tokens_processed}]' 2>/dev/null \
        || echo "[]"
}
watchdog() {
    local prev="" cur frozen
    while true; do
        cur=$(slots_json)
        if [ -n "$prev" ] && [ "$cur" != "[]" ] && [ "$prev" != "[]" ]; then
            frozen=$(jq -n --argjson a "$prev" --argjson b "$cur" '
                [$b[] | . as $c |
                 ([$a[] | select(.s == $c.s and .t == $c.t)] | first) as $p |
                 select($p != null) |
                 select(($c.d // 0) <= ($p.d // 0) and ($c.p // 0) <= ($p.p // 0))
                ] | length' 2>/dev/null || echo 0)
            if [ "$frozen" -gt 0 ]; then
                log "watchdog: $frozen wedged slot(s) ($cur) - respawning llama_server"
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

# ── 1. Phase A: reasoning_effort, missing cells only (spec-off) ────────
log "A2a-i: nospec arm t4/t5 x2"
llmc bench tasks --presets qwen38-nospec --runs 2 \
    --tasks t4-ts-add-camelcase,t5-ts-fix-slugify || log "A2a-i rc=$?"

log "A2a-ii: medium-nospec arm t1/t2/t4/t5 x2"
llmc bench tasks --presets qwen38-medium-nospec --runs 2 \
    --tasks t1-go-add-truncate,t2-go-fix-palindrome,t4-ts-add-camelcase,t5-ts-fix-slugify || log "A2a-ii rc=$?"

log "A2b: medium-nospec arm ceiling tasks t3/t6 x1 (the decision input)"
llmc bench tasks --presets qwen38-medium-nospec --runs 1 \
    --tasks t3-go-write-split-tests,t6-ts-write-slug-tests || log "A2b rc=$?"

# ── 2. Phase B: ngram-mod solo (cold pool, then warm pool) ─────────────
log "B1: perf qwen38-ngramsolo pass 1 (cold pool)"
llmc bench perf --presets qwen38-ngramsolo || log "B1 rc=$?"

log "B2: perf qwen38-ngramsolo pass 2 (warm pool - the measurement)"
llmc bench perf --presets qwen38-ngramsolo || log "B2 rc=$?"

# ngram-mod known issue watch (upstream PR #25819, stuck loop on verify fail)
log "stuck-loop scan of server logs:"
docker logs llama_server 2>&1 | rg -i 'issue#23268|stuck|ngram' | tail -5 || log "  (no ngram/stuck lines)"

# ── 3. restore + commit results ────────────────────────────────────────
llmc switch qwen38 >/dev/null 2>&1 || true
git add models/ bench/results/ bench/p5-effort-ngram.sh
git commit -q -m "bench: P5 v4 - medium-effort task suite (spec-off) + ngram-solo perf" \
    || log "nothing to commit"

log "=== P5 COMPLETE ==="
llmc bench report --compare qwen38-nospec qwen38-medium-nospec || true
llmc bench report --compare qwen38-nospec qwen38-ngramsolo || true
