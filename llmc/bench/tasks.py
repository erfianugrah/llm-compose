"""llmc bench tasks - sensor-gated loop-task suite (the decisive metric).

Runs each task manifest (bench/tasks/*.json) against a fixture snapshot
(bench/fixtures/...) through the loop CLI with llama-server/<preset> as the
ONLY model rung. Sensors-only by design (no LLM judge).

Trust protocol (per the self-correcting-loop skill): every probe has a
canary and the suite passes `loop verify-sensors` before any model is
scored on it (--verify-only runs just that gate).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Optional

from llmc.bench import store
from llmc.bench.perf import wait_ready
from llmc.presets import load_all

TASKS_DIR = store.REPO_ROOT / "bench" / "tasks"
FIXTURES_DIR = store.REPO_ROOT / "bench" / "fixtures"
SOLUTIONS_DIR = store.REPO_ROOT / "bench" / "solutions"
LOOP_TIMEOUT_S = 7200

LogFn = Callable[[str], None]


def load_manifests(tasks_dir: Path = TASKS_DIR) -> list[dict[str, Any]]:
    return [json.loads(p.read_text()) for p in sorted(tasks_dir.glob("*.json"))]


def materialize_harness(manifest: dict[str, Any], model_id: str) -> dict[str, Any]:
    """Task manifest -> full loop harness.json (model rung + resolved paths)."""
    h = {k: v for k, v in manifest.items() if k not in ("name", "fixture", "probe")}
    h["models"] = [f"llama-server/{model_id}"]
    sensors = []
    for s in manifest.get("sensors", []):
        s = dict(s)
        if "canary" in s:
            s["canary"] = s["canary"].replace("{SOLUTIONS}", str(SOLUTIONS_DIR))
        sensors.append(s)
    h["sensors"] = sensors
    return h


def setup_workdir(fixture: str, probe: str, harness: dict[str, Any]) -> Path:
    """Copy the fixture snapshot, keep ONLY this task's probe file, init git,
    write .pi/harness.json."""
    src = FIXTURES_DIR / fixture
    if not src.is_dir():
        raise ValueError(f"fixture not found: {src}")
    tmp = Path(tempfile.mkdtemp(prefix="llmc-bench-task-"))
    shutil.copytree(src, tmp)
    # Each task sees only its own probe; the other fixtures' probes would
    # pollute `go test` / `bun test` with out-of-scope failures.
    for p in tmp.rglob("probe_*"):
        if probe and p.name == Path(probe).name:
            continue
        p.unlink()
    env = {"GIT_AUTHOR_NAME": "bench", "GIT_AUTHOR_EMAIL": "bench@local",
           "GIT_COMMITTER_NAME": "bench", "GIT_COMMITTER_EMAIL": "bench@local"}
    for args in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "baseline"]):
        subprocess.run(["git"] + args, cwd=tmp, check=True, capture_output=True, env=env)
    (tmp / ".pi").mkdir(exist_ok=True)
    (tmp / ".pi" / "harness.json").write_text(json.dumps(harness, indent=2))
    return tmp


def parse_report(workdir: Path) -> dict[str, Any]:
    """Best-effort metrics extraction from .pi/harness-report.json."""
    p = workdir / ".pi" / "harness-report.json"
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text())
    except json.JSONDecodeError:
        return {}
    out: dict[str, Any] = {}
    iters = data.get("iterations") if isinstance(data, dict) else data
    if isinstance(iters, list):
        out["iterations"] = len(iters)
        out["rolled_back"] = sum(1 for i in iters if isinstance(i, dict) and not i.get("kept", True))
        out["escalations"] = sum(1 for i in iters if isinstance(i, dict) and i.get("escalated"))
    return out


def run_task(manifest: dict[str, Any], model_id: str, verify_only: bool,
             log: LogFn) -> dict[str, Any]:
    harness = materialize_harness(manifest, model_id)
    workdir = setup_workdir(manifest["fixture"], manifest.get("probe", ""), harness)
    try:
        full_env = {**os.environ, "PI_COMPACT_FRACTION": "0.95"}
        if verify_only:
            t0 = time.monotonic()
            r = subprocess.run(["loop", "verify-sensors"], cwd=workdir,
                               capture_output=True, text=True, env=full_env,
                               timeout=LOOP_TIMEOUT_S)
            return {"task": manifest.get("name"), "verify_ok": r.returncode == 0,
                    "verify_output": (r.stdout + r.stderr)[-2000:],
                    "wall_s": round(time.monotonic() - t0, 1)}
        t0 = time.monotonic()
        r = subprocess.run(["loop", "run"], cwd=workdir, capture_output=True,
                           text=True, env=full_env, timeout=LOOP_TIMEOUT_S)
        metrics = parse_report(workdir)
        metrics.update({
            "task": manifest.get("name"),
            "pass": r.returncode == 0,
            "exit_code": r.returncode,
            "wall_s": round(time.monotonic() - t0, 1),
            "tail": (r.stdout + r.stderr)[-1500:],
        })
        return metrics
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def run_tasks(preset_names: list[str], runs: int = 1,
              tasks: Optional[list[str]] = None, verify_only: bool = False,
              rid: Optional[str] = None, log: LogFn = print) -> int:
    from llmc.cli import ProxyClient

    manifests = load_manifests()
    if tasks:
        wanted = set(tasks)
        manifests = [m for m in manifests if m.get("name") in wanted]
        if not manifests:
            log(f"no manifests matched: {', '.join(tasks)}")
            return 2

    presets = {p.name: p for p in load_all(store.REPO_ROOT / "models").values()}
    unknown = [p for p in preset_names if p not in presets]
    if unknown:
        log(f"unknown preset(s): {', '.join(unknown)}")
        return 2

    client = ProxyClient()
    rid = rid or store.run_id()
    rc = 0
    for name in preset_names:
        preset = presets[name]
        model_id = preset.model_id
        log(f"\n=== {name} ({len(manifests)} tasks x {runs}) ===")

        if verify_only:
            results = []
            for m in manifests:
                res = run_task(m, model_id, verify_only=True, log=log)
                results.append({"task": res["task"], "ok": res["verify_ok"]})
                log(f"  {res['task']}: verify {'OK' if res['verify_ok'] else 'STUCK'}")
                if not res["verify_ok"]:
                    rc = 1
                    log(res.get("verify_output", ""))
            store.append(store.make_record(
                "verify", name, store.REPO_ROOT / "models" / f"{name}.toml",
                {"tasks": results}, rid, extra={"model_file": preset.model.file}))
            continue

        client.set_lock(name, owner="bench")
        status, payload = client.set_mode("llm", model=name)
        if status != 200:
            log(f"  switch failed ({status}): {payload.get('error', payload)}")
            client.set_lock(False, owner="bench")
            rc = 1
            continue
        if not wait_ready(model_id):
            log("  FAIL: model did not become ready")
            client.set_lock(False, owner="bench")
            rc = 1
            continue

        for m in manifests:
            for i in range(runs):
                res = run_task(m, model_id, verify_only=False, log=log)
                tp = res.get("pass")
                log(f"  {res['task']} run {i + 1}: {'PASS' if tp else 'FAIL'} "
                    f"in {res.get('wall_s')}s, {res.get('iterations', '?')} iterations")
                store.append(store.make_record(
                    "task", name, store.REPO_ROOT / "models" / f"{name}.toml",
                    {k: v for k, v in res.items() if k != "tail"}, rid,
                    extra={"model_file": preset.model.file}))
                if not tp:
                    log(f"    tail: {res.get('tail', '')[-400:]}")
        client.set_lock(False, owner="bench")
    return rc
