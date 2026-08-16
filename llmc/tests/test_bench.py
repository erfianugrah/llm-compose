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
