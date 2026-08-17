#!/usr/bin/env bash
# p3-resume.sh - resume the P3 pipeline from where the 2026-08-17 pause left it.
#
# Completed before the pause (records in bench/results/runs.jsonl):
#   perf: loop, gemma4, qwen38, qwen36-moe
#   tasks: loop/gemma4/qwen38 x 6 tasks x 3 runs (all 18 each)
#   gumshoe: qwen35-9b, qwen35-4b
# Remaining (this script):
#   1. eval image build (BFCL faiss fix) + evals on loop,gemma4,qwen38
#   2. gumshoe on lfm25-8b,gemma4-12b
#   3. report + commit
#   4. P4 MTP spike (bench/p4-mtp.sh, waits for this script's bench runs)
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$PWD/bin:$PATH"
log() { echo "[p3r $(date +%H:%M:%S)] $*"; }

log "phase 1: eval image + evals"
if ! docker image inspect erfianugrah/bench-eval:latest >/dev/null 2>&1; then
    docker build -t erfianugrah/bench-eval:latest -f bench/Dockerfile.eval bench/ || {
        log "FATAL: eval image build failed again - inspect Dockerfile.eval"; exit 1; }
fi
llmc bench eval --presets loop,gemma4,qwen38 --humaneval --bfcl || log "eval rc=$?"

log "phase 2: gumshoe remaining (lfm25-8b, gemma4-12b)"
llmc bench gumshoe --presets lfm25-8b,gemma4-12b --repeats 3 || log "gumshoe rc=$?"

log "phase 3: report + commit"
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
git push -q || log "push failed - push manually"

log "phase 4: handing off to P4 MTP spike"
exec bench/p4-mtp.sh
