"""llmc bench <subcommand> - native bench command dispatch."""
from __future__ import annotations

import argparse
from typing import Optional, Sequence

from llmc.bench import eval as bench_eval
from llmc.bench import gumshoe, perf, report, tasks, watch, context

NATIVE = {"perf", "report", "watch", "eval", "gumint", "gumshoe", "tasks", "context"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="llmc bench")
    sub = p.add_subparsers(dest="bench_command")

    sp = sub.add_parser("perf", help="TTFT/throughput/VRAM through the proxy")
    sp.add_argument("--presets", required=True, help="comma-separated preset names")
    sp.add_argument("--runs", type=int, default=1, help="sample multiplier (default 1 = 5 TTFT + 3 throughput)")
    sp.add_argument("--no-whisper-stop", action="store_true", help="do not stop whisper GPU services first")

    sp = sub.add_parser("report", help="tables + comparison from the result store")
    sp.add_argument("--last", type=int, default=0, help="show last K raw records")
    sp.add_argument("--compare", nargs=2, metavar=("A", "B"), help="compare latest perf of two presets")
    sp.add_argument("--markdown", action="store_true", help="emit a markdown table")

    sp = sub.add_parser("eval", help="HumanEval/HellaSwag/BFCL via the eval container")
    sp.add_argument("--presets", required=True, help="comma-separated preset names")
    sp.add_argument("--humaneval", action="store_true")
    sp.add_argument("--hellaswag", type=int, default=0, metavar="N", help="HellaSwag subset size (needs [bench] tokenizer per preset)")
    sp.add_argument("--bfcl", action="store_true", help="BFCL full non_live collection")

    sp = sub.add_parser("gumshoe", help="research-agent protocol suite (18 cases, stub tools)")
    sp.add_argument("--presets", required=True, help="comma-separated preset names")
    sp.add_argument("--repeats", type=int, default=3, help="runs per case (selection is a rate)")
    sp.add_argument("--cases", help="path to the fixtures JSON (default: gumshoe repo copy)")

    sp = sub.add_parser("tasks", help="sensor-gated loop-task suite (bench/tasks/*.json)")
    sp.add_argument("--presets", required=True, help="comma-separated preset names")
    sp.add_argument("--runs", type=int, default=1, help="runs per task")
    sp.add_argument("--tasks", help="comma-separated task names (default: all)")
    sp.add_argument("--verify-only", action="store_true",
                    help="run loop verify-sensors per task (canary gate), no model scoring")

    sp = sub.add_parser("context", help="context occupancy sweep")
    sp.add_argument("--preset", required=True, help="base preset name")
    sp.add_argument("--ctx", required=True, help="comma-separated context sizes")
    sp.add_argument("--occupancy", required=True, help="comma-separated occupancy fractions (0.0-1.0)")
    sp.add_argument("--gen-tokens", type=int, default=200, help="generation tokens")
    sp.add_argument("--dry-run", action="store_true", help="do not modify environment")
    sp.add_argument("--slots", type=int, default=1, help="parallel slots")

    sub.add_parser("watch", help="staleness report vs llama.cpp pin + preset hashes")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.bench_command == "perf":
        return perf.run_perf(
            [p.strip() for p in args.presets.split(",") if p.strip()],
            runs=args.runs,
            no_whisper_stop=args.no_whisper_stop,
        )
    if args.bench_command == "report":
        pair = tuple(args.compare) if args.compare else None
        return report.run_report(last=args.last, compare_pair=pair, markdown=args.markdown)
    if args.bench_command == "eval":
        return bench_eval.run_eval(
            [p.strip() for p in args.presets.split(",") if p.strip()],
            humaneval=args.humaneval,
            hellaswag=args.hellaswag,
            bfcl=args.bfcl,
        )
    if args.bench_command == "gumshoe":
        return gumshoe.run_gumshoe(
            [p.strip() for p in args.presets.split(",") if p.strip()],
            repeats=args.repeats,
            cases_path=args.cases,
        )
    if args.bench_command == "tasks":
        return tasks.run_tasks(
            [p.strip() for p in args.presets.split(",") if p.strip()],
            runs=args.runs,
            tasks=[t.strip() for t in args.tasks.split(",")] if args.tasks else None,
            verify_only=args.verify_only,
        )
    if args.bench_command == "context":
        return context.run_context_sweep(
            preset_name=args.preset,
            ctx_sizes=[int(x) for x in args.ctx.split(",")],
            slots=args.slots,
            occupancies=[float(x) for x in args.occupancy.split(",")],
            gen_tokens=args.gen_tokens,
            dry_run=args.dry_run,
        )
    if args.bench_command == "watch":
        return watch.run_watch()
    build_parser().print_help()
    return 2
