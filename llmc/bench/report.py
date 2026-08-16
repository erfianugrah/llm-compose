"""llmc bench report - tables + run-over-run comparison from the result store."""
from __future__ import annotations

from typing import Any, Optional

from llmc.bench import store

PERF_COLS = [
    ("ttft_p50_ms", "TTFT p50"),
    ("ttft_p95_ms", "TTFT p95"),
    ("gen_tok_s", "gen tok/s"),
    ("prompt_tok_s", "pp tok/s"),
    ("vram_peak_mib", "VRAM MiB"),
    ("ctx", "ctx"),
    ("slots", "slots"),
]


def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.1f}"
    return str(v)


def perf_table(records: list[dict[str, Any]], markdown: bool = False) -> str:
    """Latest perf record per preset as a table."""
    latest = store.latest_per_preset(records, kind="perf")
    if not latest:
        return "(no perf records in store)"
    header = ["preset", "run", "llama.cpp"] + [label for _, label in PERF_COLS]
    rows = []
    for name in sorted(latest):
        rec = latest[name]
        m = rec.get("metrics", {})
        rows.append([name, rec.get("run", "?"), rec.get("llama_cpp", "?")]
                    + [_fmt(m.get(k)) for k, _ in PERF_COLS])
    if markdown:
        out = ["| " + " | ".join(header) + " |",
               "|" + "---|" * len(header)]
        out += ["| " + " | ".join(r) + " |" for r in rows]
        return "\n".join(out)
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(header)]
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(header))
    body = ["  ".join(v.ljust(widths[i]) for i, v in enumerate(r)) for r in rows]
    return "\n".join([line] + body)


def compare(preset_a: str, preset_b: str, records: list[dict[str, Any]]) -> str:
    """Side-by-side of the latest perf records for two presets, with deltas."""
    latest = store.latest_per_preset(records, kind="perf")
    missing = [p for p in (preset_a, preset_b) if p not in latest]
    if missing:
        return f"no perf record for: {', '.join(missing)}"
    a, b = latest[preset_a], latest[preset_b]
    ma, mb = a.get("metrics", {}), b.get("metrics", {})
    lines = [f"{'metric':<14} {preset_a:>14} {preset_b:>14} {'delta':>12}"]
    for key, label in PERF_COLS:
        va, vb = ma.get(key), mb.get(key)
        delta = "-"
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)) and va:
            pct = (vb - va) / va * 100
            delta = f"{pct:+.1f}%"
        lines.append(f"{label:<14} {_fmt(va):>14} {_fmt(vb):>14} {delta:>12}")
    lines.append(f"(runs: {a.get('run')} vs {b.get('run')}; "
                 f"llama.cpp {a.get('llama_cpp')} vs {b.get('llama_cpp')})")
    return "\n".join(lines)


def last_runs(records: list[dict[str, Any]], k: int) -> str:
    rows = records[-k:] if k > 0 else records
    lines = []
    for rec in rows:
        m = rec.get("metrics", {})
        headline = ", ".join(f"{key}={_fmt(m.get(key))}" for key, _ in PERF_COLS[:4] if m.get(key) is not None)
        lines.append(f"{rec.get('ts', '?')}  {rec.get('kind', '?'):<5} {rec.get('preset', '?'):<12} {headline}")
    return "\n".join(lines) if lines else "(store is empty)"


def run_report(last: int = 0, compare_pair: Optional[tuple[str, str]] = None,
               markdown: bool = False, store_path=None) -> int:
    records = store.load(store_path) if store_path else store.load()
    if compare_pair:
        print(compare(compare_pair[0], compare_pair[1], records))
        return 0
    if last:
        print(last_runs(records, last))
        return 0
    print(perf_table(records, markdown=markdown))
    return 0
