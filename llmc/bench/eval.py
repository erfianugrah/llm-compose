"""llmc bench eval - HumanEval / HellaSwag / BFCL via the bench-eval container.

Drives bench/run-evals.py inside the erfianugrah/bench-eval image against
each preset served through the proxy. Improvements over the shell driver:
  - per-preset tokenizer from the preset TOML [bench] section (HellaSwag is a
    loglikelihood task - the tokenizer must match the GGUF family)
  - BFCL subset flag removed honestly: BFCL has no working subset mechanism
    (the old --bfcl-subset mapped to a no-op --num-gpus 1 and the FULL
    category always ran). Now it runs the full ast category and says so.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional

from llmc.bench import store
from llmc.bench.perf import wait_ready
from llmc.presets import load_all

EVAL_IMAGE = "erfianugrah/bench-eval:latest"
RESULTS_DIR = store.REPO_ROOT / "bench" / "results"
BENCH_CACHE = Path.home() / "docker-volumes" / "bench-cache"

LogFn = Callable[[str], None]


def _docker(args: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(["docker"] + args, **kw)


def ensure_image(log: LogFn) -> bool:
    if _docker(["image", "inspect", EVAL_IMAGE], capture_output=True).returncode == 0:
        return True
    log(f"building {EVAL_IMAGE} (one-time, ~5 min)...")
    r = _docker(["build", "-t", EVAL_IMAGE,
                 "-f", str(store.REPO_ROOT / "bench" / "Dockerfile.eval"),
                 str(store.REPO_ROOT / "bench")])
    return r.returncode == 0


def build_eval_flags(preset_name: str, model_id: str, out_name: str,
                     humaneval: bool, hellaswag: int, bfcl: bool,
                     tokenizer: Optional[str]) -> list[str]:
    flags = ["--label", preset_name, "--out", f"/results/{out_name}",
             "--base-url", "http://host.docker.internal:11434/v1",
             "--model", model_id]
    if humaneval:
        flags.append("--humaneval")
    if hellaswag > 0:
        flags += ["--hellaswag", "--hellaswag-subset", str(hellaswag),
                  "--hellaswag-tokenizer", tokenizer or ""]
    if bfcl:
        flags.append("--bfcl")
    return flags


def parse_eval_json(path: Path) -> dict[str, Any]:
    """run-evals.py output -> flat metrics for the store."""
    d = json.loads(path.read_text())
    metrics: dict[str, Any] = {}
    he = d.get("humaneval") or {}
    if he.get("pass@1") is not None:
        metrics["humaneval_pass1"] = he["pass@1"]
        if he.get("pass@1_plus") is not None:
            metrics["humaneval_pass1_plus"] = he["pass@1_plus"]
    hs = d.get("hellaswag") or {}
    if hs.get("acc_norm") is not None:
        metrics["hellaswag_acc_norm"] = hs["acc_norm"]
    bf = d.get("bfcl") or {}
    if bf.get("overall") is not None:
        metrics["bfcl_overall"] = bf["overall"]
    return metrics


def run_eval(
    preset_names: list[str],
    humaneval: bool = False,
    hellaswag: int = 0,
    bfcl: bool = False,
    rid: Optional[str] = None,
    log: LogFn = print,
) -> int:
    from llmc.cli import ProxyClient

    if not (humaneval or hellaswag or bfcl):
        log("nothing to do: pass --humaneval, --hellaswag N and/or --bfcl")
        return 2

    presets = {p.name: p for p in load_all(store.REPO_ROOT / "models").values()}
    unknown = [p for p in preset_names if p not in presets]
    if unknown:
        log(f"unknown preset(s): {', '.join(unknown)}")
        return 2
    if hellaswag:
        missing_tok = [n for n in preset_names if not presets[n].bench.get("tokenizer")]
        if missing_tok:
            log(f"HellaSwag needs a tokenizer per preset ([bench] tokenizer = ...); "
                f"missing on: {', '.join(missing_tok)}")
            return 2
    if bfcl:
        log("note: BFCL runs the FULL ast category (no working subset mechanism)")
    if not ensure_image(log):
        log("eval image build failed")
        return 1

    client = ProxyClient()
    rid = rid or store.run_id()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    BENCH_CACHE.mkdir(parents=True, exist_ok=True)
    rc = 0
    for name in preset_names:
        preset = presets[name]
        log(f"\n=== {name} ===")
        status, payload = client.set_lock(name, owner="bench")
        if status != 200:
            log(f"  lock failed ({status}): {payload.get('error', payload)}")
            rc = 1
            continue
        status, payload = client.set_mode("llm", model=name)
        if status != 200:
            log(f"  switch failed ({status}): {payload.get('error', payload)}")
            client.set_lock(False, owner="bench")
            rc = 1
            continue
        if not wait_ready(preset.model_id):
            log("  FAIL: model did not become ready")
            client.set_lock(False, owner="bench")
            rc = 1
            continue

        out_name = f"eval-{name}-{rid}.json"
        flags = build_eval_flags(name, preset.model_id, out_name,
                                 humaneval, hellaswag, bfcl,
                                 preset.bench.get("tokenizer"))
        r = _docker([
            "run", "--rm", "--name", f"bench_eval_{name}",
            "--add-host=host.docker.internal:host-gateway",
            "-v", f"{RESULTS_DIR}:/results",
            "-v", f"{BENCH_CACHE}:/cache",
            "-e", "HF_HOME=/cache/huggingface",
            EVAL_IMAGE, *flags,
        ])
        out_path = RESULTS_DIR / out_name
        if r.returncode != 0 or not out_path.exists():
            log(f"  eval container failed for {name}")
            rc = 1
            client.set_lock(False, owner="bench")
            continue
        metrics = parse_eval_json(out_path)
        rec = store.make_record(
            "eval", name, store.REPO_ROOT / "models" / f"{name}.toml", metrics, rid,
            extra={"model_file": preset.model.file},
        )
        store.append(rec)
        log(f"  {json.dumps(metrics)}")
        client.set_lock(False, owner="bench")
    return rc
