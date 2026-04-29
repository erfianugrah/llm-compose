#!/usr/bin/env python3
"""bench-report.py — render quant sweep results.

Reads sweep CSV produced by bench-quants.sh and emits:
  1. Markdown table to stdout (always)
  2. PNG chart matching the Qwen3.6 reference image (if matplotlib installed)

Usage:
    python3 bench/bench-report.py bench/results/sweep-20260428-120000.csv
    python3 bench/bench-report.py latest    # find most recent sweep CSV
    python3 bench/bench-report.py latest --no-chart
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "bench" / "results"

def latest_csv() -> Path:
    files = sorted(RESULTS.glob("sweep-*.csv"))
    if not files:
        sys.exit("no sweep-*.csv found")
    return files[-1]

def load(path: Path) -> list[dict]:
    rows = list(csv.DictReader(path.open()))
    # Drop FAIL rows from numeric processing
    return [r for r in rows if r.get("ttft_ms_p50") not in ("FAIL", "", None)]

def fnum(s: str | None) -> float | None:
    if not s or s in ("FAIL",): return None
    try: return float(s)
    except ValueError: return None

def md_table(rows: list[dict]) -> str:
    cols = [
        ("label",            "Quant"),
        ("size_gb",          "Size GB"),
        ("vram_peak_mib",    "VRAM MiB"),
        ("ttft_ms_p50",      "TTFT p50"),
        ("gen_tok_s",        "Gen tok/s"),
        ("prompt_tok_s",     "PP tok/s"),
        ("humaneval_pass1",  "HumanEval"),
        ("hellaswag_acc_norm", "HellaSwag"),
        ("bfcl_overall",     "BFCL"),
    ]
    out = ["| " + " | ".join(h for _, h in cols) + " |",
           "| " + " | ".join("---" for _ in cols) + " |"]
    for r in rows:
        cells = []
        for k, _ in cols:
            v = r.get(k, "")
            if k in ("humaneval_pass1", "hellaswag_acc_norm", "bfcl_overall"):
                f = fnum(v); cells.append(f"{f*100:.1f}%" if f is not None else "—")
            elif k == "vram_peak_mib":
                f = fnum(v); cells.append(f"{f/1024:.1f}" if f else "—")
            else:
                cells.append(v or "—")
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)

def best_efficiency(rows: list[dict]) -> dict | None:
    """Pick the quant with the best (gen_tok_s × accuracy / vram) score."""
    scored = []
    for r in rows:
        gen = fnum(r["gen_tok_s"]); vram = fnum(r["vram_peak_mib"])
        acc = fnum(r.get("humaneval_pass1")) or 0.5
        if gen and vram:
            scored.append((gen * (acc + 0.5) / (vram / 1024), r))
    return max(scored, key=lambda x: x[0])[1] if scored else None

def render_chart(rows: list[dict], out: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping chart", file=sys.stderr); return

    labels = [r["label"] for r in rows]
    n = len(labels)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Quant Variant Comparison", fontsize=16, weight="bold")

    def bar(ax, key, title, fmt="{:.0f}", scale=1.0, lower_better=False):
        vals = [(fnum(r.get(key)) or 0) * scale for r in rows]
        colors = plt.cm.Blues([0.4 + 0.5 * i / max(n - 1, 1) for i in range(n)])
        bars = ax.bar(labels, vals, color=colors)
        ax.set_title(title + ("  (lower better)" if lower_better else "  (higher better)"), fontsize=11)
        ax.tick_params(axis="x", rotation=30)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v, fmt.format(v),
                    ha="center", va="bottom", fontsize=9)

    bar(axes[0, 0], "humaneval_pass1",   "HumanEval pass@1 (%)", "{:.1f}", 100)
    bar(axes[0, 1], "hellaswag_acc_norm","HellaSwag acc_norm (%)","{:.1f}", 100)
    bar(axes[0, 2], "bfcl_overall",      "BFCL overall (%)",     "{:.1f}", 100)
    bar(axes[1, 0], "gen_tok_s",         "Throughput (tok/s)",   "{:.1f}")
    bar(axes[1, 1], "ttft_ms_p50",       "TTFT p50 (ms)",        "{:.0f}", lower_better=True)
    # VRAM in GB
    for r in rows: r["_vram_gb"] = (fnum(r.get("vram_peak_mib")) or 0) / 1024
    bar(axes[1, 2], "_vram_gb",          "VRAM peak (GB)",       "{:.1f}", lower_better=True)

    plt.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"chart -> {out}", file=sys.stderr)

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("csv", nargs="?", default="latest")
    p.add_argument("--no-chart", action="store_true")
    args = p.parse_args()

    path = latest_csv() if args.csv == "latest" else Path(args.csv)
    rows = load(path)
    if not rows:
        sys.exit("no usable rows in " + str(path))

    print(f"# Quant sweep: {path.name}\n")
    print(md_table(rows))
    print()
    pick = best_efficiency(rows)
    if pick:
        print(f"**Recommended:** `{pick['label']}` "
              f"({fnum(pick['gen_tok_s']):.1f} tok/s, "
              f"{fnum(pick['vram_peak_mib'])/1024:.1f} GB VRAM, "
              f"HumanEval {(fnum(pick.get('humaneval_pass1')) or 0)*100:.1f}%)")

    if not args.no_chart:
        render_chart(rows, path.with_suffix(".png"))
    return 0

if __name__ == "__main__":
    sys.exit(main())
