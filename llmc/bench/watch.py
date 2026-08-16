"""llmc bench watch - staleness report.

Compares each preset's current content hash + the current llama.cpp pin
against the result store and reports which presets have no current-baseline
numbers. Not a daemon: run it after a pin bump or preset edit and it tells
you exactly what to re-baseline.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from llmc.bench import store


def staleness(records: list[dict[str, Any]], models_dir: Path,
              pin: Optional[str] = None) -> list[dict[str, Any]]:
    """One row per preset: current hash/pin vs latest perf record."""
    pin = pin or store.llama_pin()
    latest = store.latest_per_preset(records, kind="perf")
    rows = []
    for path in sorted(models_dir.glob("*.toml")):
        name = path.stem
        cur_hash = store.preset_hash(path)
        rec = latest.get(name)
        if rec is None:
            state = "NO-BASELINE"
        elif rec.get("llama_cpp") != pin:
            state = f"STALE-PIN (store {rec.get('llama_cpp')} != {pin})"
        elif rec.get("preset_hash") != cur_hash:
            state = "STALE-PRESET (edited since last run)"
        else:
            state = "current"
        rows.append({
            "preset": name,
            "state": state,
            "last_run": rec.get("run") if rec else "-",
            "current_hash": cur_hash,
        })
    return rows


def run_watch(models_dir: Optional[Path] = None, store_path=None) -> int:
    models_dir = models_dir or (store.REPO_ROOT / "models")
    records = store.load(store_path) if store_path else store.load()
    rows = staleness(records, models_dir)
    stale = 0
    for r in rows:
        marker = " " if r["state"] == "current" else "!"
        if r["state"] != "current":
            stale += 1
        print(f"{marker} {r['preset']:<16} {r['state']:<42} last run: {r['last_run']}")
    if stale:
        print(f"\n{stale} preset(s) need re-baselining: llmc bench perf --presets "
              + ",".join(r["preset"] for r in rows if r["state"] != "current"))
        return 1
    print("\nall presets current")
    return 0
