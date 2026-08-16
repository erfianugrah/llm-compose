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
    assert perf.p95([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) == 9.5
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
    assert gumshoe._ordered_subsequence(["osint_url"], [{"any": ["osint_domain", "osint_url"]}])
    assert gumshoe._ordered_subsequence(["web_search"], [{"any": ["osint_domain"]}]) is False


def _case(exp):
    return {"id": "t", "prompt": "q", "expect": exp}


def test_check_trace_sequence_args_answer():
    case = _case({"tools": ["web_search"], "args": {"q": "nginx"}})
    good = {"steps": [{"tool": "web_search", "args": {"query": "nginx"}}], "answer": "done"}
    bad_seq = {"steps": [{"tool": "fetch", "args": {}}], "answer": "x"}
    no_final = {"steps": [{"tool": "web_search", "args": {"nginx"}}], "answer": ""}
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
    assert h["models"] == ["llama-server/model-id"]
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
    assert out == {"iterations": 3, "rolled_back": 1, "escalations": 1}
    assert bench_tasks.parse_report(tmp_path / "nope") == {}
