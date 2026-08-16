"""llmc bench <subcommand> - native bench command dispatch."""
from __future__ import annotations

import argparse
from typing import Optional, Sequence

from llmc.bench import eval as bench_eval
from llmc.bench import gumshoe, perf, report, watch

NATIVE = {"perf", "report", "watch", "eval", "gumshoe"}


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
    sp.add_argument("--bfcl", action="store_true", help="BFCL full ast category")

    sp = sub.add_parser("gumshoe", help="research-agent protocol suite (18 cases, stub tools)")
    sp.add_argument("--presets", required=True, help="comma-separated preset names")
    sp.add_argument("--repeats", type=int, default=3, help="runs per case (selection is a rate)")
    sp.add_argument("--cases", help="path to the fixtures JSON (default: gumshoe repo copy)")

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
    if args.bench_command == "watch":
        return watch.run_watch()
    build_parser().print_help()
    return 2
