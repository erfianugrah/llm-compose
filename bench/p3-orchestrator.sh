#!/usr/bin/env bash
# p3-orchestrator.sh - autonomous P3 matrix pipeline (plan section 7.3).
# Phase order: wait for running tasks matrix -> evals (tie-breakers) ->
# gumshoe small-track -> report + commit. Idempotent-ish: each phase appends
# to the shared runs.jsonl; a re-run re-measures rather than corrupting.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$PWD/bin:$PATH"
log() { echo "[p3 $(date +%H:%M:%S)] $*"; }

# ── 1. wait for the tasks matrix already running ───────────────────────
log "phase 1: waiting for tasks matrix to finish"
while pgrep -f "llmc bench tasks" >/dev/null 2>&1; do sleep 60; done
log "phase 1: tasks matrix done"

# ── 2. evals (tie-breaker): humaneval + BFCL on the 3 big presets ──────
log "phase 2: evals (loop,gemma4,qwen38 humaneval+bfcl)"
if ! docker image inspect erfianugrah/bench-eval:latest >/dev/null 2>&1; then
    log "phase 2: building eval image"
    docker build -t erfianugrah/bench-eval:latest -f bench/Dockerfile.eval bench/
fi
llmc bench eval --presets loop,gemma4,qwen38 --humaneval --bfcl || log "phase 2 rc=$?"
log "phase 2: evals done"

# ── 3. gumshoe small track (4 candidates x 18 cases x 3) ───────────────
log "phase 3: gumshoe (qwen35-9b,qwen35-4b,lfm25-8b,gemma4-12b)"
llmc bench gumshoe --presets qwen35-9b,qwen35-4b,lfm25-8b,gemma4-12b --repeats 3 || log "phase 3 rc=$?"
log "phase 3: gumshoe done"

# ── 4. report + commit ─────────────────────────────────────────────────
log "phase 4: report + commit"
{
    echo "# P3 matrix results ($(date -u +%Y-%m-%d))"
    echo
    echo '```'
    llmc bench report --markdown
    echo '```'
    echo
    echo '## loop vs qwen38 (perf)'
    echo
    echo '```'
    llmc bench report --compare loop qwen38
    echo '```'
} > docs/plans/2026-08-16-p3-matrix-results.md
git add bench/results/ docs/plans/2026-08-16-p3-matrix-results.md
git commit -q -m "bench: P3 matrix results (tasks/eval/gumshoe)" || log "nothing to commit"
git push -q || log "push skipped/failed (rc=$?) - push manually"
log "=== P3 COMPLETE ==="
