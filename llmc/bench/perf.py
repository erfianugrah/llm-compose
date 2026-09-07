"""llmc bench perf - TTFT / throughput / VRAM measurement through the proxy.

Port of bench/bench-perf.sh's methodology, but driven through the proxy
(POST /mode) using preset TOMLs instead of raw `docker run` - presets are
the single source of truth for model config, and the proxy path is the
serving path loops actually use.

Protocol per preset (mirrors bench-perf.sh so numbers are comparable):
  - stop whisper GPU services first (whisper-live holds ~5.6 GB; restarted after)
  - llmc lock <preset> --owner bench; llmc switch <preset>
  - 2 warmup requests
  - 5*runs TTFT samples (SSE, "Reply with one word: yes", max_tokens 128, temp 0)
  - 3*runs throughput samples (non-stream, 500 tokens, best of)
  - VRAM/RAM peak via 0.5s poller
"""
from __future__ import annotations

import json
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable, Optional

from llmc.bench import store
from llmc.presets import load_all

PROXY = "http://127.0.0.1:11434"
TTFT_PROMPT = "Reply with one word: yes"
THROUGHPUT_PROMPT = ("Write a Python function for binary search with type hints, "
                     "docstring, and 5 pytest unit tests covering edge cases.")
WHISPER_SERVICES = ["whisper-transcribe-whisper-1", "whisper-transcribe-whisper-live-1"]

LogFn = Callable[[str], None]


# ── Pure helpers (unit-tested) ─────────────────────────────────────────

def p50(values: list[float]) -> float:
    return round(statistics.median(values), 1) if values else 0.0


def p95(values: list[float]) -> float:
    if not values:
        return 0.0
    v = sorted(values)
    return round(v[int(0.95 * (len(v) - 1))], 1)


def parse_sse_data(line: str) -> Optional[dict[str, Any]]:
    """One SSE line -> parsed payload, None for keep-alive/[DONE]/junk."""
    line = line.strip()
    if not line.startswith("data:"):
        return None
    payload = line[5:].strip()
    if payload == "[DONE]":
        return {"done": True}
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


def token_of(payload: dict[str, Any]) -> str:
    delta = (payload.get("choices") or [{}])[0].get("delta") or {}
    return delta.get("content") or delta.get("reasoning_content") or ""


# ── Measurement primitives ─────────────────────────────────────────────

def measure_ttft(model_id: str, proxy: str = PROXY, timeout: int = 300) -> tuple[float, int]:
    """SSE request -> (ttft_ms, n_tokens). (0, 0) on failure."""
    body = json.dumps({
        "model": model_id,
        "messages": [{"role": "user", "content": TTFT_PROMPT}],
        "max_tokens": 128, "stream": True, "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(
        f"{proxy}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    t0 = time.perf_counter()
    ttft = 0.0
    n_tok = 0
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            buf = b""
            while True:
                chunk = r.read(64)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    raw, _, buf = buf.partition(b"\n")
                    payload = parse_sse_data(raw.decode("utf-8", "ignore"))
                    if payload is None:
                        continue
                    if payload.get("done"):
                        return ttft, n_tok
                    if token_of(payload):
                        if not ttft:
                            ttft = round((time.perf_counter() - t0) * 1000, 1)
                        n_tok += 1
    except Exception as e:  # a failed sample is data, not a crash
        print(f"# ttft err: {e}", file=sys.stderr)
    return ttft, n_tok


# Prefill rate needs a prompt big enough that fixed per-request overhead is
# noise, and it must not be served from the KV cache. The short
# THROUGHPUT_PROMPT above measured ~27 tok/s against a real 2450 tok/s
# (2026-09-02) - that number was overhead, not prefill.
PREFILL_PROMPT = ("Analyse the following log excerpt and summarise it.\n\n"
                  + "2026-09-02T00:00:00Z INFO worker=7 stage=ingest "
                    "records=1284 latency_ms=37 status=ok\n" * 400)


def measure_prefill(model_id: str, proxy: str = PROXY, timeout: int = 600) -> dict[str, float]:
    """Prompt-processing rate on a large, uncached prompt."""
    body = json.dumps({
        "model": model_id,
        "messages": [{"role": "user", "content": PREFILL_PROMPT}],
        "max_tokens": 1, "temperature": 0.0, "cache_prompt": False,
    }).encode()
    req = urllib.request.Request(
        f"{proxy}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read())
        wall = time.perf_counter() - t0
        t = d.get("timings", {})
        if t:
            return {"prompt_per_s": round(t.get("prompt_per_second", 0), 1),
                    "prompt_n": t.get("prompt_n", 0)}
        # external engines (ninfer) have no llama.cpp timings: wall-clock + usage
        usage = d.get("usage", {})
        n = usage.get("prompt_tokens", 0)
        return {"prompt_per_s": round(n / wall, 1) if n and wall > 0 else 0.0,
                "prompt_n": n}
    except Exception as e:
        print(f"# prefill err: {e}", file=sys.stderr)
        return {"prompt_per_s": 0.0, "prompt_n": 0}


def measure_throughput(model_id: str, proxy: str = PROXY, timeout: int = 600) -> dict[str, float]:
    """Non-stream request -> llama.cpp server timings."""
    body = json.dumps({
        "model": model_id,
        "messages": [{"role": "user", "content": THROUGHPUT_PROMPT}],
        "max_tokens": 500, "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(
        f"{proxy}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read())
        wall = time.perf_counter() - t0
        t = d.get("timings", {})
        if t:
            return {
                "prompt_per_s": round(t.get("prompt_per_second", 0), 1),
                "gen_per_s": round(t.get("predicted_per_second", 0), 1),
                "gen_n": t.get("predicted_n", 0),
            }
        # external engines (ninfer) have no llama.cpp timings: wall-clock + usage.
        # gen rate includes prefill time - conservative (understates gen).
        usage = d.get("usage", {})
        gen_n = usage.get("completion_tokens", 0)
        return {
            "prompt_per_s": 0.0,
            "gen_per_s": round(gen_n / wall, 1) if gen_n and wall > 0 else 0.0,
            "gen_n": gen_n,
        }
    except Exception as e:
        print(f"# throughput err: {e}", file=sys.stderr)
        return {"prompt_per_s": 0.0, "gen_per_s": 0.0, "gen_n": 0}


class VramPoller:
    """Background 0.5s nvidia-smi sampler; .peak_mib after .stop()."""

    def __init__(self) -> None:
        self.peak_mib = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                )
                if out.returncode == 0:
                    self.peak_mib = max(self.peak_mib, int(out.stdout.splitlines()[0].strip()))
            except (OSError, subprocess.TimeoutExpired, ValueError, IndexError):
                pass
            self._stop.wait(0.5)

    def __enter__(self) -> "VramPoller":
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=3)


# ── Docker helpers ─────────────────────────────────────────────────────

def _docker(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(["docker"] + args, capture_output=True, text=True, timeout=60)


def stop_whisper(log: LogFn) -> list[str]:
    """Stop compose-managed whisper GPU services (they hold ~5.6 GB and would
    skew --fit / VRAM numbers). Returns the ones we stopped (to restart)."""
    stopped = []
    for c in WHISPER_SERVICES:
        r = _docker(["inspect", "-f", "{{.State.Running}}", c])
        if r.returncode == 0 and r.stdout.strip() == "true":
            if _docker(["stop", c]).returncode == 0:
                stopped.append(c)
                log(f"  stopped {c} (restart after)")
    return stopped


def restart_whisper(stopped: list[str], log: LogFn) -> None:
    for c in stopped:
        _docker(["start", c])
        log(f"  restarted {c}")


# ── The runner ─────────────────────────────────────────────────────────

def run_perf_external(
    url: str,
    model_id: str,
    label: str,
    ctx: int,
    slots: int,
    runs: int = 1,
    rid: Optional[str] = None,
    log: LogFn = print,
    no_whisper_stop: bool = False,
) -> int:
    """Measure an external OpenAI-compatible endpoint (no proxy lock/switch).

    For engine spikes (e.g. NInfer) where the service is not proxy-managed.
    Same measurement protocol as run_perf so numbers are comparable.
    """
    rid = rid or store.run_id()
    stopped = [] if no_whisper_stop else stop_whisper(log)
    try:
        log(f"\n=== {label} (external {url}, model {model_id}, ctx {ctx} x {slots}) ===")
        if not wait_ready(model_id, proxy=url):
            log("  FAIL: endpoint did not answer in 900s")
            return 1
        with VramPoller() as poller:
            for _ in range(2):
                measure_ttft(model_id, proxy=url)  # warmup
            ttfts = [measure_ttft(model_id, proxy=url)[0] for _ in range(5 * runs)]
            ttfts = [t for t in ttfts if t > 0]
            best = {"prompt_per_s": 0.0, "gen_per_s": 0.0, "gen_n": 0}
            for _ in range(3 * runs):
                t = measure_throughput(model_id, proxy=url)
                if t["gen_per_s"] > best["gen_per_s"]:
                    best = t
            prefill = {"prompt_per_s": 0.0, "prompt_n": 0}
            for _ in range(runs):
                t = measure_prefill(model_id, proxy=url)
                if t["prompt_per_s"] > prefill["prompt_per_s"]:
                    prefill = t
        metrics = {
            "ttft_p50_ms": p50(ttfts),
            "ttft_p95_ms": p95(ttfts),
            "gen_tok_s": best["gen_per_s"],
            "prompt_tok_s": prefill["prompt_per_s"],
            "prompt_n": prefill["prompt_n"],
            "vram_peak_mib": poller.peak_mib,
            "ctx": ctx,
            "slots": slots,
        }
        rec = store.make_record("perf", label, None, metrics, rid,
                                extra={"model_file": model_id, "external": url})
        store.append(rec)
        log(f"  TTFT p50={metrics['ttft_p50_ms']}ms p95={metrics['ttft_p95_ms']}ms | "
            f"gen={metrics['gen_tok_s']} tok/s | pp={metrics['prompt_tok_s']} tok/s "
            f"({metrics['prompt_n']} tok) | "
            f"VRAM peak={metrics['vram_peak_mib']} MiB")
        return 0
    finally:
        restart_whisper(stopped, log)


def wait_ready(model_id: str, timeout_s: int = 900, proxy: str = PROXY) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        ttft, _ = measure_ttft(model_id, proxy=proxy, timeout=30)
        if ttft > 0:
            return True
        time.sleep(5)
    return False


def run_perf(
    preset_names: list[str],
    runs: int = 1,
    rid: Optional[str] = None,
    proxy: str = PROXY,
    log: LogFn = print,
    no_whisper_stop: bool = False,
) -> int:
    """Measure each preset through the proxy. Returns process exit code."""
    from llmc.cli import ProxyClient  # late import: keeps store/perf importable without cli

    presets = {p.name: p for p in load_all(store.REPO_ROOT / "models").values()}
    unknown = [p for p in preset_names if p not in presets]
    if unknown:
        log(f"unknown preset(s): {', '.join(unknown)} (have: {', '.join(sorted(presets))})")
        return 2

    client = ProxyClient()
    rid = rid or store.run_id()
    prev_model = None
    try:
        status, payload = client.status()
        if status == 200:
            prev_model = (payload.get("mode") or {}).get("model")
    except Exception:
        pass

    stopped = [] if no_whisper_stop else stop_whisper(log)
    rc = 0
    try:
        for name in preset_names:
            preset = presets[name]
            model_id = preset.model_id
            log(f"\n=== {name} ({preset.model.file}, ctx {preset.runtime.context_size} x {preset.runtime.parallel_slots}) ===")

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
            if not wait_ready(model_id, proxy=proxy):
                log("  FAIL: model did not become ready in 900s")
                client.set_lock(False, owner="bench")
                rc = 1
                continue

            with VramPoller() as poller:
                for _ in range(2):
                    measure_ttft(model_id, proxy=proxy)  # warmup
                ttfts = [measure_ttft(model_id, proxy=proxy)[0] for _ in range(5 * runs)]
                ttfts = [t for t in ttfts if t > 0]
                best = {"prompt_per_s": 0.0, "gen_per_s": 0.0, "gen_n": 0}
                for _ in range(3 * runs):
                    t = measure_throughput(model_id, proxy=proxy)
                    if t["gen_per_s"] > best["gen_per_s"]:
                        best = t
                prefill = {"prompt_per_s": 0.0, "prompt_n": 0}
                for _ in range(runs):
                    t = measure_prefill(model_id, proxy=proxy)
                    if t["prompt_per_s"] > prefill["prompt_per_s"]:
                        prefill = t

            metrics = {
                "ttft_p50_ms": p50(ttfts),
                "ttft_p95_ms": p95(ttfts),
                "gen_tok_s": best["gen_per_s"],
                "prompt_tok_s": prefill["prompt_per_s"],
                "prompt_n": prefill["prompt_n"],
                "vram_peak_mib": poller.peak_mib,
                "ctx": preset.runtime.context_size,
                "slots": preset.runtime.parallel_slots,
            }
            rec = store.make_record(
                "perf", name, store.REPO_ROOT / "models" / f"{name}.toml", metrics, rid,
                extra={"model_file": preset.model.file},
            )
            store.append(rec)
            log(f"  TTFT p50={metrics['ttft_p50_ms']}ms p95={metrics['ttft_p95_ms']}ms | "
                f"gen={metrics['gen_tok_s']} tok/s | pp={metrics['prompt_tok_s']} tok/s "
                f"({metrics['prompt_n']} tok) | "
                f"VRAM peak={metrics['vram_peak_mib']} MiB")
            client.set_lock(False, owner="bench")
    finally:
        restart_whisper(stopped, log)
        if prev_model:
            log(f"restoring previous model: {prev_model}")
            client.set_mode("llm", model=prev_model)
    return rc
