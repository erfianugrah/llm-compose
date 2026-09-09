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


def materialize_harness(manifest: dict[str, Any], model_id: str,
                        rung_prefix: str = "llmc") -> dict[str, Any]:
    """Task manifest -> full loop harness.json (model rung + resolved paths)."""
    h = {k: v for k, v in manifest.items() if k not in ("name", "fixture", "probe")}
    h["models"] = [f"{rung_prefix}/{model_id}"]
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
    shutil.copytree(src, tmp, dirs_exist_ok=True)
    # Each task sees only its own probe; the other fixtures' probes would
    # pollute `go test` / `bun test` with out-of-scope failures.
    for p in tmp.rglob("probe_*"):
        if probe and p.name == Path(probe).name:
            continue
        p.unlink()
    env = {"GIT_AUTHOR_NAME": "bench", "GIT_AUTHOR_EMAIL": "bench@local",
           "GIT_COMMITTER_NAME": "bench", "GIT_COMMITTER_EMAIL": "bench@local"}
    (tmp / ".pi").mkdir(exist_ok=True)
    (tmp / ".pi" / "harness.json").write_text(json.dumps(harness, indent=2))
    for args in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "baseline"]):
        subprocess.run(["git"] + args, cwd=tmp, check=True, capture_output=True, env=env)
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
        # pi-loop records the agent process exit per iteration (agentExit).
        # A non-zero exit means pi died before or during the request - an
        # unresolvable rung, a proxy 422 under someone else's lock, a crash -
        # and the iteration says nothing about the model. The 2026-09-08
        # "t4/t5 regression" was eight of these in a row per run, scored as
        # eight rolled-back iterations; without this count they were
        # indistinguishable from the model writing bad code eight times.
        out["agent_errors"] = sum(
            1 for i in iters
            if isinstance(i, dict) and i.get("agentExit") not in (None, 0)
            and not i.get("agentTimedOut"))
    return out


# A run whose agent died this many times or more is infrastructure, not a
# score. It is stored with invalid=True (pass stays False) and the suite stops.
INVALID_AGENT_ERRORS = 2

PI_MODELS_JSON = Path(os.environ.get("PI_MODELS_JSON",
                                     Path.home() / ".pi" / "agent" / "models.json"))


def preflight_rung(model_id: str, rung_prefix: str, advertised: list[str],
                   models_json: Path = PI_MODELS_JSON) -> Optional[str]:
    """Return None when `<rung_prefix>/<model_id>` can resolve in pi, else the
    reason it cannot. Two checks: the provider key exists in pi's models.json
    (the llama-server -> llmc rename broke this and produced 48 Model-not-found
    rows on 2026-09-08), and the proxy advertises the model id (pi's
    llmc-dynamic extension registers models from GET /v1/models, so an id the
    proxy does not list is unreachable no matter what the preset says)."""
    try:
        providers = json.loads(models_json.read_text()).get("providers", {})
    except (OSError, json.JSONDecodeError) as exc:
        return f"cannot read pi models.json at {models_json}: {exc}"
    if rung_prefix not in providers:
        return (f"pi has no provider {rung_prefix!r} in {models_json} "
                f"(has: {', '.join(sorted(providers))})")
    if model_id not in advertised:
        return (f"proxy does not advertise {model_id!r} as a model id or preset "
                f"(GET /v1/models lists: {', '.join(sorted(advertised))})")
    return None


def advertised_ids(models_payload: dict[str, Any]) -> list[str]:
    """Every name a request may carry: raw model ids plus preset names (pi
    registers presets; the proxy resolves both)."""
    out: list[str] = []
    for m in models_payload.get("data", []):
        if m.get("id"):
            out.append(m["id"])
        preset = (m.get("meta") or {}).get("preset")
        if preset:
            out.append(preset)
    return out


def run_task(manifest: dict[str, Any], model_id: str, verify_only: bool,
             log: LogFn, rung_prefix: str = "llmc",
             rung_id: Optional[str] = None) -> dict[str, Any]:
    """rung_id is the id pi knows the model by. For the llmc provider that is
    the PRESET NAME: llmc-dynamic.ts registers `id: meta.preset`, so a rung of
    `llmc/<model file id>` never matched and pi fell back to a "custom model
    id" that inherits the metadata of whichever model the proxy happened to
    list first (GET /v1/models was unordered) - a random reasoning flag,
    context window and output cap per leg. Every task row before 2026-09-09
    ran that way. External endpoints keep rung_id == model_id."""
    harness = materialize_harness(manifest, rung_id or model_id, rung_prefix=rung_prefix)
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
        out = r.stdout + r.stderr
        agent_errors = metrics.get("agent_errors")
        if agent_errors is None:
            # No report (loop died before writing one): fall back to pi-loop's
            # own stderr marker so a rung failure still counts as agent errors.
            agent_errors = out.count("(agent exited ")
            metrics["agent_errors"] = agent_errors
        metrics.update({
            "task": manifest.get("name"),
            "pass": r.returncode == 0,
            "exit_code": r.returncode,
            "wall_s": round(time.monotonic() - t0, 1),
            # Kept in the store now. It was stripped before, which is why the
            # 2026-09-08 lock-refusal stubs could not be told from real fails.
            "tail": out[-1500:],
            "invalid": agent_errors >= INVALID_AGENT_ERRORS,
        })
        return metrics
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def run_tasks_external(url: str, model_id: str, label: str,
                       runs: int = 1, tasks: Optional[list[str]] = None,
                       verify_only: bool = False,
                       rid: Optional[str] = None, log: LogFn = print) -> int:
    """Task suite against an external OpenAI endpoint (no proxy lock/switch).

    The loop harness rung becomes `external/<model_id>`, resolved via the
    `external` provider in pi's models.json (baseUrl 127.0.0.1:8001) -
    external services must bind that port. verify_only uses the same gate.
    """
    manifests = load_manifests()
    if tasks:
        wanted = set(tasks)
        manifests = [m for m in manifests if m.get("name") in wanted]
        if not manifests:
            log(f"no manifests matched: {', '.join(tasks)}")
            return 2
    rid = rid or store.run_id()
    rc = 0
    log(f"\n=== {label} (external {url}, {len(manifests)} tasks x {runs}) ===")
    if not verify_only and not wait_ready(model_id, proxy=url):
        log("  FAIL: endpoint did not answer")
        return 1
    if verify_only:
        results = []
        for m in manifests:
            res = run_task(m, model_id, verify_only=True, log=log, rung_prefix="external")
            results.append({"task": res["task"], "ok": res["verify_ok"]})
            log(f"  {res['task']}: verify {'OK' if res['verify_ok'] else 'STUCK'}")
            if not res["verify_ok"]:
                rc = 1
                log(res.get("verify_output", ""))
        store.append(store.make_record(
            "verify", label, None, {"tasks": results}, rid,
            extra={"model_file": model_id, "external": url}))
        return rc
    for m in manifests:
        for i in range(runs):
            res = run_task(m, model_id, verify_only=False, log=log, rung_prefix="external")
            tp = res.get("pass")
            log(f"  {res['task']} run {i + 1}: {'PASS' if tp else 'FAIL'} "
                f"in {res.get('wall_s')}s, {res.get('iterations', '?')} iterations")
            store.append(store.make_record(
                "task", label, None,
                res, rid,
                extra={"model_file": model_id, "external": url}))
            if not tp:
                rc = 1
                log(f"    tail: {res.get('tail', '')[-400:]}")
    return rc


def _acquire(client, name: str, model_id: str, owner: str, log: LogFn) -> bool:
    """Lock + switch + wait for one leg. False (with the reason logged) when
    the leg cannot start; the caller decides whether that aborts the suite."""
    status, payload = client.set_lock(name, owner=owner)
    if status != 200:
        log(f"  lock failed ({status}): {payload.get('error', payload)}")
        return False
    status, payload = client.set_mode("llm", model=name)
    if status != 200:
        log(f"  switch failed ({status}): {payload.get('error', payload)}")
        client.set_lock(False, owner=owner)
        return False
    if not wait_ready(model_id):
        log("  FAIL: model did not become ready")
        client.set_lock(False, owner=owner)
        return False
    return True


def _foreign_lock(client) -> Optional[str]:
    """Describe a lock someone else holds, or None when the proxy is free."""
    status, payload = client.status()
    if status != 200:
        return f"proxy status {status}"
    if payload.get("locked") or payload.get("lock_owners") or payload.get("lock_queue"):
        return (f"{payload.get('locked')!r} held by "
                f"{', '.join(payload.get('lock_owners') or []) or '?'}"
                + (f", queue {payload.get('lock_queue')}" if payload.get("lock_queue") else ""))
    return None


def run_tasks(preset_names: list[str], runs: int = 1,
              tasks: Optional[list[str]] = None, verify_only: bool = False,
              rid: Optional[str] = None, log: LogFn = print,
              interleave: Optional[bool] = None) -> int:
    """Score presets on the task suite through the proxy.

    Lock owner is `bench-<run id>`, unique per invocation: two suites can no
    longer release each other's lock (both were "bench" until 2026-09-08, and
    a second suite's unlock handed the GPU away under the first one). The
    suite refuses to start while anyone else holds or queues for the lock;
    a bench must run alone on the GPU or its numbers mean nothing.

    interleave (default: on when comparing 2+ presets) swaps the engine per
    leg - for each (task, run) every preset runs once, order alternating
    (ABAB, then BABA on the next run), so engine state and wall-clock drift
    land on both sides. Off = the old per-preset blocks.

    A run with >= INVALID_AGENT_ERRORS agent deaths is stored invalid and the
    suite stops (exit 3): eight fast rollbacks are a broken rung, not a score.
    """
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
    if interleave is None:
        interleave = len(preset_names) > 1

    client = ProxyClient()
    rid = rid or store.run_id()
    owner = f"bench-{rid}"

    if verify_only:
        rc = 0
        for name in preset_names:
            preset = presets[name]
            results = []
            log(f"\n=== {name} (verify {len(manifests)} tasks) ===")
            for m in manifests:
                res = run_task(m, preset.model_id, verify_only=True, log=log)
                results.append({"task": res["task"], "ok": res["verify_ok"]})
                log(f"  {res['task']}: verify {'OK' if res['verify_ok'] else 'STUCK'}")
                if not res["verify_ok"]:
                    rc = 1
                    log(res.get("verify_output", ""))
            store.append(store.make_record(
                "verify", name, store.REPO_ROOT / "models" / f"{name}.toml",
                {"tasks": results}, rid, extra={"model_file": preset.model.file}))
        return rc

    foreign = _foreign_lock(client)
    if foreign:
        log(f"refusing to start: model lock {foreign}. A bench must run alone; "
            f"wait for the owner to finish or `llmc unlock --owner <id>`.")
        return 3
    _, models_payload = client.models()
    advertised = advertised_ids(models_payload)
    for name in preset_names:
        # pi resolves llmc/<preset name>; the file-derived model id is only
        # for the readiness probe and the store's model_file column.
        reason = preflight_rung(name, "llmc", advertised)
        if reason:
            log(f"preflight failed for {name}: {reason}")
            return 3

    def leg(name: str, m: dict[str, Any], i: int) -> Optional[bool]:
        """One (preset, task, run). None = leg could not start / run invalid."""
        preset = presets[name]
        if not _acquire(client, name, preset.model_id, owner, log):
            return None
        try:
            res = run_task(m, preset.model_id, verify_only=False, log=log, rung_id=name)
        finally:
            client.set_lock(False, owner=owner)
        tp = res.get("pass")
        flag = "INVALID" if res.get("invalid") else ("PASS" if tp else "FAIL")
        log(f"  [{name}] {res['task']} run {i + 1}: {flag} "
            f"in {res.get('wall_s')}s, {res.get('iterations', '?')} iterations"
            + (f", {res['agent_errors']} agent errors" if res.get("agent_errors") else ""))
        store.append(store.make_record(
            "task", name, store.REPO_ROOT / "models" / f"{name}.toml",
            res, rid, extra={"model_file": preset.model.file, "rung": f"llmc/{name}",
                             "interleaved": interleave, "owner": owner}))
        if res.get("invalid"):
            log(f"    agent died {res['agent_errors']}x - infrastructure, not a score. "
                f"Stopping the suite.\n    tail: {res.get('tail', '')[-600:]}")
            return None
        if not tp:
            log(f"    tail: {res.get('tail', '')[-400:]}")
        return bool(tp)

    rc = 0
    if interleave:
        log(f"\n=== interleaved A/B: {' vs '.join(preset_names)} "
            f"({len(manifests)} tasks x {runs} runs, engine swap per leg, owner {owner}) ===")
        for ti, m in enumerate(manifests):
            for i in range(runs):
                order = preset_names if (ti + i) % 2 == 0 else list(reversed(preset_names))
                for name in order:
                    ok = leg(name, m, i)
                    if ok is None:
                        return 3
                    if not ok:
                        rc = 1
        return rc

    for name in preset_names:
        log(f"\n=== {name} ({len(manifests)} tasks x {runs}, owner {owner}) ===")
        for m in manifests:
            for i in range(runs):
                ok = leg(name, m, i)
                if ok is None:
                    return 3
                if not ok:
                    rc = 1
    return rc
