"""llmc.bench - the bench framework (plan: docs/plans/2026-08-15-local-model-bench-framework.md).

Subcommands:
    perf    - TTFT/throughput/VRAM measurement through the proxy (real serving path)
    report  - tables + run-over-run comparison from the result store
    watch   - staleness report: which presets lack current-baseline numbers

The result store is bench/results/runs.jsonl (committed to git - the trend
history is the point). One JSON record per measurement.
"""
