"""Tests for llmc.bench: store, perf pure helpers, report, watch."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from llmc.bench import perf, report, store, watch


# ── store ──────────────────────────────────────────────────────────────

def test_preset_hash_changes_on_edit(tmp_path: Path):
    f = tmp_path / "x.toml"
    f.write_text("a = 1\n")
    h1 = store.preset_hash(f)
    f.write_text("a = 2\n")
    h2 = store.preset_hash(f)
    assert h1 != h2 and len(h1) == 12


def test_llama_pin_reads_dockerfile(tmp_path: Path):
    df = tmp_path / "Dockerfile"
    df.write_text("FROM x\nARG LLAMA_CPP_VERSION=b99999\nRUN echo hi\n")
    assert store.llama_pin(df) == "b99999"


def test_append_load_roundtrip(tmp_path: Path):
    s = tmp_path / "runs.jsonl"
    rec = {"ts": "t", "run": "r1", "kind": "perf", "preset": "a", "metrics": {"gen_tok_s": 42.0}}
    store.append(rec, s)
    store.append({**rec, "preset": "b"}, s)
    loaded = store.load(s)
    assert [r["preset"] for r in loaded] == ["a", "b"]
    assert loaded[0]["metrics"]["gen_tok_s"] == 42.0


def test_latest_per_preset_keeps_last(tmp_path: Path):
    recs = [
        {"kind": "perf", "preset": "a", "run": "r1"},
        {"kind": "perf", "preset": "a", "run": "r2"},
        {"kind": "task", "preset": "a", "run": "r3"},
    ]
    latest = store.latest_per_preset(recs, kind="perf")
    assert latest["a"]["run"] == "r2"


# ── perf pure helpers ──────────────────────────────────────────────────

def test_p50_p95():
    assert perf.p50([1, 2, 3, 4, 5]) == 3
    # nearest-rank (lower), not interpolated: index int(0.95 * (n - 1)). Every
    # p95 quoted in the scorecards since 2026-08-16 was computed this way.
    assert perf.p95([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) == 9
    assert perf.p95(list(range(1, 101))) == 95
    assert perf.p95([]) == 0.0
    assert perf.p50([]) == 0.0 and perf.p95([]) == 0.0


def test_parse_sse_data():
    assert perf.parse_sse_data("data: [DONE]") == {"done": True}
    assert perf.parse_sse_data(": keep-alive") is None
    assert perf.parse_sse_data("data: {not json") is None
    d = perf.parse_sse_data('data: {"choices": [{"delta": {"content": "hi"}}]}')
    assert d["choices"][0]["delta"]["content"] == "hi"


def test_token_of_content_and_reasoning():
    assert perf.token_of({"choices": [{"delta": {"content": "x"}}]}) == "x"
    assert perf.token_of({"choices": [{"delta": {"reasoning_content": "r"}}]}) == "r"
    assert perf.token_of({"choices": [{"delta": {}}]}) == ""


# ── report ─────────────────────────────────────────────────────────────

def _perf_rec(preset: str, run: str, gen: float, pin: str = "b1", ph: str = "h") -> dict:
    return {"ts": "t", "run": run, "kind": "perf", "preset": preset,
            "llama_cpp": pin, "preset_hash": ph,
            "metrics": {"ttft_p50_ms": 100.0, "ttft_p95_ms": 150.0,
                        "gen_tok_s": gen, "prompt_tok_s": 4000.0,
                        "vram_peak_mib": 20000, "ctx": 131072, "slots": 1}}


def test_perf_table_renders_latest():
    out = report.perf_table([_perf_rec("a", "r1", 40.0), _perf_rec("a", "r2", 45.0)])
    assert "r2" in out and "45.0" in out and "r1" not in out


def test_perf_table_markdown():
    out = report.perf_table([_perf_rec("a", "r1", 40.0)], markdown=True)
    assert out.startswith("| preset") and "| a |" in out


def test_compare_deltas():
    out = report.compare("a", "b", [_perf_rec("a", "r1", 40.0), _perf_rec("b", "r1", 60.0)])
    assert "+50.0%" in out  # gen tok/s 40 -> 60


def test_compare_missing():
    assert "no perf record" in report.compare("a", "zzz", [_perf_rec("a", "r1", 40.0)])


# ── watch ──────────────────────────────────────────────────────────────

def test_staleness_states(tmp_path: Path):
    (tmp_path / "a.toml").write_text("x = 1\n")
    (tmp_path / "b.toml").write_text("x = 2\n")
    (tmp_path / "c.toml").write_text("x = 3\n")
    hash_a = store.preset_hash(tmp_path / "a.toml")
    recs = [
        _perf_rec("a", "r1", 40.0, pin="PIN", ph=hash_a),   # current
        _perf_rec("b", "r1", 40.0, pin="OLD", ph="whatever"),  # stale pin
        # c: no record -> NO-BASELINE
    ]
    rows = {r["preset"]: r["state"] for r in watch.staleness(recs, tmp_path, pin="PIN")}
    assert rows["a"] == "current"
    assert rows["b"].startswith("STALE-PIN")
    assert rows["c"] == "NO-BASELINE"


def test_staleness_preset_edit_detected(tmp_path: Path):
    f = tmp_path / "a.toml"
    f.write_text("x = 1\n")
    recs = [_perf_rec("a", "r1", 40.0, pin="PIN", ph=store.preset_hash(f))]
    f.write_text("x = 2\n")  # edit after the run
    rows = watch.staleness(recs, tmp_path, pin="PIN")
    assert rows[0]["state"].startswith("STALE-PRESET")


# ── eval ───────────────────────────────────────────────────────────────

from llmc.bench import eval as bench_eval


def test_build_eval_flags_tokenizer_and_subsets():
    flags = bench_eval.build_eval_flags("p", "model-id", "out.json",
                                        humaneval=True, hellaswag=500, bfcl=True,
                                        tokenizer="unsloth/X-GGUF")
    s = " ".join(flags)
    assert "--humaneval" in flags
    assert "--hellaswag-subset 500" in s
    assert "--hellaswag-tokenizer unsloth/X-GGUF" in s
    assert "--bfcl" in flags
    assert "--num-gpus" not in s  # the silent no-op is gone


def test_build_eval_flags_none_requested():
    flags = bench_eval.build_eval_flags("p", "m", "o", False, 0, False, None)
    assert "--humaneval" not in flags and "--bfcl" not in flags and "--hellaswag" not in flags


def test_parse_eval_json(tmp_path: Path):
    p = tmp_path / "r.json"
    p.write_text(json.dumps({
        "label": "x",
        "humaneval": {"pass@1": 0.5, "pass@1_plus": 0.45},
        "hellaswag": {"acc": 0.8, "acc_norm": 0.85},
        "bfcl": {"overall": 0.6},
    }))
    m = bench_eval.parse_eval_json(p)
    assert m == {"humaneval_pass1": 0.5, "humaneval_pass1_plus": 0.45,
                 "hellaswag_acc_norm": 0.85, "bfcl_overall": 0.6}


def test_parse_eval_json_partial_failures(tmp_path: Path):
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"label": "x", "humaneval": {"pass@1": None, "error": "boom"}}))
    assert bench_eval.parse_eval_json(p) == {}


# ── gumshoe ────────────────────────────────────────────────────────────

from llmc.bench import gumshoe


def test_parse_action_variants():
    assert gumshoe.parse_action('{"tool": "web_search", "args": {}}')["tool"] == "web_search"
    assert gumshoe.parse_action('```json\n{"final": "ok"}\n```')["final"] == "ok"
    assert gumshoe.parse_action("prose then {\"tool\": \"fetch\", \"args\": {\"url\": \"x\"}}")["tool"] == "fetch"
    assert gumshoe.parse_action("no json here") is None
    assert gumshoe.parse_action('{"other": 1}') is None


def test_ordered_subsequence_with_alts():
    assert gumshoe._ordered_subsequence(["a", "x", "b"], [{"a"}, {"b"}])
    assert gumshoe._ordered_subsequence(["a", "b"], [{"a"}, {"b"}, {"c"}]) is False
    # {"any": [...]} is the CASE-FILE spelling; check_trace expands it to a set
    # via _expand_alts before matching, so the matcher only ever sees sets.
    alts = gumshoe._expand_alts([{"any": ["osint_domain", "osint_url"]}])
    assert alts == [{"osint_domain", "osint_url"}]
    assert gumshoe._ordered_subsequence(["osint_url"], alts)
    assert gumshoe._ordered_subsequence(["web_search"], [{"any": ["osint_domain"]}]) is False


def _case(exp):
    return {"id": "t", "prompt": "q", "expect": exp}


def test_check_trace_sequence_args_answer():
    case = _case({"tools": ["web_search"], "args": {"q": "nginx"}})
    good = {"steps": [{"tool": "web_search", "args": {"query": "nginx"}}], "answer": "done"}
    bad_seq = {"steps": [{"tool": "fetch", "args": {}}], "answer": "x"}
    no_final = {"steps": [{"tool": "web_search", "args": {"q": "nginx"}}], "answer": ""}
    assert gumshoe.check_trace(case, good)
    assert not gumshoe.check_trace(case, bad_seq)
    assert not gumshoe.check_trace(case, no_final)


def test_run_case_with_scripted_model():
    responses = iter([
        '{"tool": "osint_domain", "args": {"domain": "google.com"}}',
        '{"tool": "osint_ip", "args": {"ip": "142.250.185.78"}}',
        '{"final": "Google mail infrastructure summary"}',
    ])
    def fake_llm(messages, max_tokens=1200, temperature=0.2):
        return next(responses)
    case = _case({"tools": ["osint_domain", "osint_ip"]})
    trace = gumshoe.run_case(fake_llm, case)
    assert [s["tool"] for s in trace["steps"]] == ["osint_domain", "osint_ip"]
    assert trace["json_invalid"] == 0
    assert gumshoe.check_trace(case, trace)


def test_run_case_invalid_json_nudges_then_recovers():
    responses = iter([
        "sorry, let me think about that",   # not JSON -> nudge
        '{"tool": "web_search", "args": {"query": "x"}}',
        '{"final": "answer"}',
    ])
    def fake_llm(messages, max_tokens=1200, temperature=0.2):
        return next(responses)
    trace = gumshoe.run_case(fake_llm, _case({"tools": ["web_search"]}))
    assert trace["json_invalid"] == 1
    assert [s["tool"] for s in trace["steps"]] == ["web_search"]


# ── tasks ──────────────────────────────────────────────────────────────

from llmc.bench import tasks as bench_tasks


def test_materialize_harness_strips_meta_and_resolves_solutions():
    m = {"name": "t", "fixture": "fx", "probe": "p_test.go", "task": "do x",
         "sensors": [{"name": "probe", "cmd": "go test ./...",
                      "canary": "cp {SOLUTIONS}/t-x.go x.go"}]}
    h = bench_tasks.materialize_harness(m, "model-id")
    assert h["models"] == ["llmc/model-id"]
    assert "name" not in h and "fixture" not in h and "probe" not in h
    assert "{SOLUTIONS}" not in h["sensors"][0]["canary"]
    assert h["sensors"][0]["canary"].endswith("/t-x.go x.go")


def test_setup_workdir_keeps_only_this_probe(tmp_path: Path, monkeypatch):
    fx = tmp_path / "fixtures" / "fx"
    fx.mkdir(parents=True)
    (fx / "main.go").write_text("package main\n")
    (fx / "probe_a_test.go").write_text("package main\n")
    (fx / "probe_b_test.go").write_text("package main\n")
    monkeypatch.setattr(bench_tasks, "FIXTURES_DIR", tmp_path / "fixtures")
    wd = bench_tasks.setup_workdir("fx", "probe_a_test.go", {"task": "x"})
    try:
        assert (wd / "probe_a_test.go").exists()
        assert not (wd / "probe_b_test.go").exists()
        assert (wd / ".pi" / "harness.json").exists()
        # git baseline committed
        import subprocess
        r = subprocess.run(["git", "log", "--oneline"], cwd=wd,
                           capture_output=True, text=True)
        assert "baseline" in r.stdout
    finally:
        import shutil
        shutil.rmtree(wd, ignore_errors=True)


def test_parse_report_shapes(tmp_path: Path):
    (tmp_path / ".pi").mkdir()
    (tmp_path / ".pi" / "harness-report.json").write_text(
        '{"iterations": [{"kept": true}, {"kept": false}, {"kept": true, "escalated": true}]}')
    out = bench_tasks.parse_report(tmp_path)
    assert out == {"iterations": 3, "rolled_back": 1, "escalations": 1, "agent_errors": 0}
    assert bench_tasks.parse_report(tmp_path / "nope") == {}


def test_parse_report_counts_agent_errors(tmp_path: Path):
    (tmp_path / ".pi").mkdir()
    (tmp_path / ".pi" / "harness-report.json").write_text(json.dumps({"iterations": [
        {"kept": False, "agentExit": 1},
        {"kept": False, "agentExit": 1},
        {"kept": True, "agentExit": 0},
        {"kept": False, "agentExit": 124, "agentTimedOut": True},  # timeout is not an agent error
        {"kept": False},  # older report without the field
    ]}))
    out = bench_tasks.parse_report(tmp_path)
    assert out["agent_errors"] == 2
    assert out["iterations"] == 5 and out["rolled_back"] == 4


def test_preflight_rung_checks_provider_and_advertised(tmp_path: Path):
    mj = tmp_path / "models.json"
    mj.write_text(json.dumps({"providers": {"llmc": {}, "openrouter": {}}}))
    ok = bench_tasks.preflight_rung("qwen3.8-27b-nvfp4", "llmc", ["qwen3.8-27b-nvfp4"], models_json=mj)
    assert ok is None
    bad_provider = bench_tasks.preflight_rung("x", "llama-server", ["x"], models_json=mj)
    assert bad_provider and "no provider 'llama-server'" in bad_provider
    bad_model = bench_tasks.preflight_rung("gone", "llmc", ["x"], models_json=mj)
    assert bad_model and "does not advertise" in bad_model
    missing = bench_tasks.preflight_rung("x", "llmc", ["x"], models_json=tmp_path / "nope.json")
    assert missing and "cannot read" in missing


class _FakeClient:
    """Scripted proxy: records lock/switch calls, answers status from `state`."""

    def __init__(self, state):
        self.state = state
        self.calls = []

    def status(self):
        return 200, self.state

    def models(self):
        return 200, {"data": [{"id": "A-id", "meta": {"preset": "A"}},
                              {"id": "B-id", "meta": {"preset": "B"}}]}

    def set_lock(self, lock, owner=None, wait=False):
        self.calls.append(("lock", lock, owner))
        return 200, {}

    def set_mode(self, mode, model=None, owner=None):
        self.calls.append(("switch", model))
        return 200, {}


def _fake_presets():
    class P:
        def __init__(self, name):
            self.name = name
            self.model_id = f"{name}-id"
            self.model = type("M", (), {"file": f"{name}.gguf"})()
    return {"A": P("A"), "B": P("B")}


def _patch_run_tasks(monkeypatch, client, results):
    import llmc.cli as cli_mod
    monkeypatch.setattr(cli_mod, "ProxyClient", lambda: client)
    monkeypatch.setattr(bench_tasks, "load_all", lambda _d: _fake_presets())
    monkeypatch.setattr(bench_tasks, "wait_ready", lambda _m: True)
    monkeypatch.setattr(bench_tasks, "load_manifests", lambda: [
        {"name": "t1", "fixture": "fx"}, {"name": "t2", "fixture": "fx"}])
    monkeypatch.setattr(bench_tasks.store, "append", lambda rec: results.append(rec))
    monkeypatch.setattr(bench_tasks.store, "make_record",
                        lambda kind, preset, path, metrics, rid, extra=None:
                        {"kind": kind, "preset": preset, "metrics": metrics, **(extra or {})})


def test_run_tasks_refuses_to_start_under_a_foreign_lock(monkeypatch):
    client = _FakeClient({"locked": "qwen38", "lock_owners": ["pi-1234"], "lock_queue": []})
    results = []
    _patch_run_tasks(monkeypatch, client, results)
    logs = []
    rc = bench_tasks.run_tasks(["A", "B"], runs=1, log=logs.append)
    assert rc == 3
    assert any("refusing to start" in l and "pi-1234" in l for l in logs)
    assert client.calls == [] and results == []


def test_run_tasks_interleaves_and_uses_unique_owner(monkeypatch):
    client = _FakeClient({"locked": None, "lock_owners": [], "lock_queue": []})
    results = []
    _patch_run_tasks(monkeypatch, client, results)
    rungs = []
    def fake_run_task(m, model_id, verify_only, log, rung_id=None):
        rungs.append(rung_id)
        return {"task": m["name"], "pass": True, "iterations": 1,
                "agent_errors": 0, "invalid": False, "wall_s": 1.0, "tail": ""}
    monkeypatch.setattr(bench_tasks, "run_task", fake_run_task)
    rc = bench_tasks.run_tasks(["A", "B"], runs=2, rid="RID", log=lambda _s: None)
    assert rc == 0
    # the rung is the preset name pi registered, never the file-derived model id
    assert set(rungs) == {"A", "B"}
    assert all(r["rung"] in ("llmc/A", "llmc/B") for r in results)
    # (t1,run1)=AB (t1,run2)=BA (t2,run1)=BA (t2,run2)=AB -> engine swap every leg
    switches = [c[1] for c in client.calls if c[0] == "switch"]
    assert switches == ["A", "B", "B", "A", "B", "A", "A", "B"]
    owners = {c[2] for c in client.calls if c[0] == "lock"}
    assert owners == {"bench-RID"}
    assert all(r["owner"] == "bench-RID" and r["interleaved"] is True for r in results)
    assert len(results) == 8


def test_run_tasks_stops_on_invalid_run_and_keeps_tail(monkeypatch):
    client = _FakeClient({"locked": None, "lock_owners": [], "lock_queue": []})
    results = []
    _patch_run_tasks(monkeypatch, client, results)
    monkeypatch.setattr(bench_tasks, "run_task",
                        lambda m, model_id, verify_only, log, rung_id=None: {
                            "task": m["name"], "pass": False, "iterations": 8, "rolled_back": 8,
                            "agent_errors": 8, "invalid": True, "wall_s": 60.0,
                            "tail": "Error: model lock active on qwen38"})
    logs = []
    rc = bench_tasks.run_tasks(["A", "B"], runs=5, log=logs.append)
    assert rc == 3
    assert len(results) == 1 and results[0]["metrics"]["invalid"] is True
    assert "lock active" in results[0]["metrics"]["tail"]
    assert any("Stopping the suite" in l for l in logs)
    # the lock was released even though the suite aborted
    assert client.calls[-1] == ("lock", False, client.calls[0][2])


def test_run_tasks_preflight_blocks_bad_rung(monkeypatch, tmp_path: Path):
    client = _FakeClient({"locked": None, "lock_owners": [], "lock_queue": []})
    results = []
    _patch_run_tasks(monkeypatch, client, results)
    client.models = lambda: (200, {"data": [{"id": "A-id", "meta": {"preset": "A"}}]})  # B not advertised
    logs = []
    rc = bench_tasks.run_tasks(["A", "B"], runs=1, log=logs.append)
    assert rc == 3
    assert any("preflight failed for B" in l for l in logs)
    assert not any(c[0] == "switch" for c in client.calls)


def test_advertised_ids_includes_preset_names():
    payload = {"data": [{"id": "qwen3.8-27b-nvfp4", "meta": {"preset": "qwen38-ninfer"}},
                        {"id": "auto", "meta": {"alias": True}}]}
    assert bench_tasks.advertised_ids(payload) == ["qwen3.8-27b-nvfp4", "qwen38-ninfer", "auto"]


def test_run_task_rung_uses_rung_id(monkeypatch, tmp_path: Path):
    seen = {}
    def fake_setup(fixture, probe, harness):
        seen["models"] = harness["models"]
        raise RuntimeError("stop here")
    monkeypatch.setattr(bench_tasks, "setup_workdir", fake_setup)
    m = {"name": "t", "fixture": "fx", "task": "x"}
    try:
        bench_tasks.run_task(m, "file-derived-id", verify_only=True, log=print, rung_id="qwen38")
    except RuntimeError:
        pass
    assert seen["models"] == ["llmc/qwen38"]
