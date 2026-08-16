"""Result store: bench/results/runs.jsonl + record metadata.

One JSON record per measurement. Records carry enough provenance
(llama.cpp pin, GPU, preset content hash) that a comparison is only ever
made between honestly-equivalent runs - a preset tweak or a llama.cpp bump
invalidates the baseline visibly instead of silently.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "bench" / "results"
STORE = RESULTS_DIR / "runs.jsonl"
DOCKERFILE = REPO_ROOT / "llama-server.Dockerfile"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def preset_hash(preset_path: Path) -> str:
    """sha1 (12 chars) of the preset TOML bytes. Any edit -> new hash -> old
    numbers are visibly stale in `bench watch`."""
    return hashlib.sha1(preset_path.read_bytes()).hexdigest()[:12]


def llama_pin(dockerfile: Path = DOCKERFILE) -> str:
    m = re.search(r"ARG\s+LLAMA_CPP_VERSION=(\S+)", dockerfile.read_text())
    return m.group(1) if m else "unknown"


def gpu_name() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.splitlines()[0].strip() if out.returncode == 0 else "unknown"
    except (OSError, subprocess.TimeoutExpired, IndexError):
        return "unknown"


def make_record(
    kind: str,
    preset: str,
    preset_path: Optional[Path],
    metrics: dict[str, Any],
    rid: str,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "ts": utc_now(),
        "run": rid,
        "kind": kind,
        "preset": preset,
        "llama_cpp": llama_pin(),
        "gpu": gpu_name(),
        "metrics": metrics,
    }
    if preset_path is not None:
        rec["preset_hash"] = preset_hash(preset_path)
    if extra:
        rec.update(extra)
    return rec


def append(record: dict[str, Any], store: Path = STORE) -> None:
    store.parent.mkdir(parents=True, exist_ok=True)
    with store.open("a") as f:
        f.write(json.dumps(record, separators=(",", ":")) + "\n")


def load(store: Path = STORE) -> list[dict[str, Any]]:
    if not store.exists():
        return []
    out = []
    for line in store.read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def latest_per_preset(
    records: Iterable[dict[str, Any]], kind: Optional[str] = None
) -> dict[str, dict[str, Any]]:
    """Latest record per preset (optionally filtered by kind)."""
    out: dict[str, dict[str, Any]] = {}
    for rec in records:
        if kind and rec.get("kind") != kind:
            continue
        out[rec.get("preset", "?")] = rec  # store is append-only chronological
    return out
