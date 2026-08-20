"""llmc bench context - occupancy sweep driver.

Measures generation throughput (tok/s) as a function of real KV occupancy for
a preset at candidate context sizes. Answers: what is the largest usable
context_size before tg collapses?

Design (docs/plans/2026-08-19-context-occupancy-suite.md):
- For each candidate ctx, materialize a THROWAWAY preset whose GGUF is a
  symlink to the base preset's GGUF (unique file stem => unique model_id; a
  plain TOML copy duplicates the model ID and the store rejects it). The TOML
  + symlink go into the MAIN repo's models/ dir (the dir the proxy mounts and
  live-reloads), not the worktree.
- Occupancy is achieved by putting a filler corpus sized to
  int(ctx*frac) - gen_tokens - HEADROOM tokens directly into the measurement
  request (llama.cpp is stateless per request). Sized via POST /tokenize.
- Generation speed from the response's timings.predicted_per_second.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Callable, Optional

from llmc.bench import store
from llmc.presets import load_all
from llmc.cli import ProxyClient
from llmc import volumes as volumes_mod

# Main repo models dir (proxy mounts this, live-reloads). The worktree's own
# models/ is NOT what the proxy sees - throwaway presets must land here.
MAIN_MODELS_DIR = store.REPO_ROOT / "models"
MAIN_VOLUMES_TOML = store.REPO_ROOT / "volumes.toml"

HEADROOM_TOKENS = 64
TOKENIZE_TIMEOUT = 300  # s; a near-full-ctx filler tokenizes slowly
GEN_TIMEOUT = 3600      # s; prefill of a 200k-token prompt takes minutes
SWEEP_STORE = store.RESULTS_DIR / "context-runs.jsonl"


def run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


# ── Pure helpers (unit-tested) ─────────────────────────────────────────

def fill_to_tokens(target: int, source_text: str, tokenize_fn: Callable[[str], list]) -> str:
    """Return a string drawn from source_text with exactly `target` tokens.

    source_text must tokenize to >= target (caller grows the corpus first).
    Greedy-append with binary search on the tail chunk.
    """
    if target <= 0:
        return ""
    if len(tokenize_fn(source_text)) < target:
        raise ValueError("source_text too small for target tokens")
    # Binary search on a character prefix (token count is monotonic enough for
    # whitespace-bounded prose; we verify at the end).
    lo, hi, best = 0, len(source_text), ""
    while lo <= hi:
        mid = (lo + hi) // 2
        cand = source_text[:mid]
        n = len(tokenize_fn(cand))
        if n <= target:
            best = cand
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def occupancy_target(ctx: int, frac: float, gen_tokens: int) -> int:
    """Filler tokens for a sweep point; <=0 means skip (won't fit)."""
    return int(ctx * frac) - gen_tokens - HEADROOM_TOKENS


# ── Corpus ─────────────────────────────────────────────────────────────

def build_corpus(tokenize_fn: Callable[[str], list], min_tokens: int, log) -> str:
    """Rotating prose from docs/*.md, grown until it tokenizes to >= min_tokens.

    Real prose (not repeated single chars) so prefill exercises real attention.
    """
    paragraphs: list[str] = []
    for doc in sorted(MAIN_MODELS_DIR.parent.glob("docs/**/*.md")):
        try:
            paragraphs += [p.strip() for p in doc.read_text().split("\n\n") if p.strip()]
        except OSError:
            continue
    if not paragraphs:
        paragraphs = ["The quick brown fox jumps over the lazy dog."] * 100
    rng = random.Random(42)
    corpus = ""
    while len(tokenize_fn(corpus)) < min_tokens:
        corpus += rng.choice(paragraphs) + "\n\n"
    return corpus


# ── Proxy / tokenizer plumbing ─────────────────────────────────────────

def make_tokenizer(proxy: str) -> Callable[[str], list]:
    def tok(text: str) -> list:
        body = json.dumps({"content": text}).encode()
        req = urllib.request.Request(
            f"{proxy}/tokenize", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=TOKENIZE_TIMEOUT) as r:
            return json.loads(r.read())["tokens"]
    return tok


def _chat(proxy: str, model: str, prompt: str, max_tokens: int, timeout: int) -> dict:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(
        f"{proxy}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _vram_mib() -> int:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        if out.returncode == 0 and out.stdout.strip():
            return int(out.stdout.splitlines()[0].strip())
    except Exception:
        pass
    return 0


def _model_dir() -> Path:
    """Resolve the GGUF volume dir (llmc-llama-models) via volumes.toml."""
    reg = volumes_mod.load(MAIN_VOLUMES_TOML)
    return reg.device_for("llmc-llama-models")


def _register_ephemeral(proxy: str, preset_name: str, base, ctx: int, slots: int) -> None:
    """Register a throwaway preset with the proxy's ephemeral registry
    (POST /v1/presets). In-memory only - never touches the live models/ dir."""
    body = json.dumps({
        "name": preset_name,
        "display_name": preset_name,
        "vram_gb": base.vram_gb,
        "model": {"repo": base.model.repo, "file": f"{preset_name}.gguf"},
        "runtime": {"context_size": ctx, "parallel_slots": int(slots)},
    }).encode()
    req = urllib.request.Request(
        f"{proxy}/v1/presets", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        r.read()


def _delete_ephemeral(proxy: str, preset_name: str) -> None:
    req = urllib.request.Request(f"{proxy}/v1/presets/{preset_name}", method="DELETE")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            r.read()
    except Exception:
        pass


# ── The sweep ──────────────────────────────────────────────────────────

def run_context_sweep(
    preset_name: str,
    ctx_sizes: list[int],
    slots: int,
    occupancies: list[float],
    gen_tokens: int,
    dry_run: bool = False,
    proxy: str = "http://127.0.0.1:11434",
    log: Callable[[str], None] = print,
    tokenize_fn: Optional[Callable[[str], list]] = None,
    preset_path_fn: Optional[Callable[[Path], Optional[Path]]] = None,
) -> int:
    presets = load_all(MAIN_MODELS_DIR)
    base = next((p for p in presets.values() if p.name == preset_name), None)
    if base is None:
        log(f"error: base preset {preset_name!r} not found")
        return 1

    # Dry-run: print the plan, touch nothing.
    log(f"Sweep plan: preset={preset_name} ctx={ctx_sizes} slots={slots} occ={occupancies} gen={gen_tokens}")
    for ctx in ctx_sizes:
        for frac in occupancies:
            tgt = occupancy_target(ctx, frac, gen_tokens)
            status = "skip" if tgt <= 0 else f"filler={tgt} tokens"
            log(f"  ctx={ctx} occ={frac}: {status}")
    if dry_run:
        log("[dry-run] no docker/proxy/preset changes made")
        return 0

    tokenize_fn = tokenize_fn or make_tokenizer(proxy)
    client = ProxyClient()
    rid = run_id()
    max_fill = max((occupancy_target(c, f, gen_tokens) for c in ctx_sizes for f in occupancies), default=0)
    corpus = build_corpus(tokenize_fn, max_fill, log) if max_fill > 0 else ""

    created: list[Path] = []
    try:
        model_dir = _model_dir()
        for ctx in ctx_sizes:
            sweep_id = f"ctx-sweep-{ctx}"
            gguf_link = model_dir / f"{sweep_id}.gguf"

            # 1. Symlink the GGUF in the models VOLUME (unique stem => unique
            #    model_id). No TOML is written - the preset is registered with
            #    the proxy's ephemeral registry instead (no live-dir hazard).
            if gguf_link.exists() or gguf_link.is_symlink():
                gguf_link.unlink()
            gguf_link.symlink_to(model_dir / base.model.file)
            created.append(gguf_link)

            # 2. Register the ephemeral preset with the proxy.
            _register_ephemeral(proxy, sweep_id, base, ctx, slots)

            # 3. Lock + switch to the throwaway preset.
            client.set_lock(sweep_id, owner="bench-context", wait=True)
            client.set_mode("llm", model=sweep_id, owner="bench-context")
            log(f"switched to {sweep_id} (ctx={ctx})")

            # 4. Warm-up trigger (drives the swap + load).
            _chat(proxy, sweep_id, "hi", max_tokens=1, timeout=900)

            for frac in occupancies:
                tgt = occupancy_target(ctx, frac, gen_tokens)
                if tgt <= 0:
                    log(f"  occ={frac}: skipped (no headroom)")
                    continue
                filler = fill_to_tokens(tgt, corpus, tokenize_fn)
                try:
                    resp = _chat(proxy, sweep_id, filler + "\n\n?", gen_tokens, GEN_TIMEOUT)
                    tg = resp.get("timings", {}).get("predicted_per_second", 0.0)
                    usage = resp.get("usage", {})
                    metrics = {
                        "gen_tok_s": tg, "vram_mib": _vram_mib(),
                        "ctx": ctx, "occupancy": frac, "slots": slots,
                        "prompt_tokens": usage.get("prompt_tokens"),
                        "completion_tokens": usage.get("completion_tokens"),
                    }
                    rec = store.make_record("context-occupancy", sweep_id, None, metrics, rid,
                                            extra={"gen_tokens": gen_tokens})
                    store.append(rec, store=SWEEP_STORE)
                    log(f"  occ={frac}: tg={tg} tok/s vram={metrics['vram_mib']} MiB prompt={metrics['prompt_tokens']}")
                except Exception as e:
                    log(f"  occ={frac}: error {e}")
            client.set_lock(False, owner="bench-context")
            _delete_ephemeral(proxy, sweep_id)
        return 0
    except Exception as e:
        log(f"error: {e}")
        return 1
    finally:
        for f in created:
            try:
                if f.exists() or f.is_symlink():
                    f.unlink()
            except OSError:
                pass
        if not dry_run:
            try:
                client.set_lock(False, owner="bench-context")
                client.set_mode("llm", model=preset_name)  # restore
            except Exception:
                pass


def main() -> int:
    p = argparse.ArgumentParser(prog="llmc bench context")
    p.add_argument("--preset", required=True)
    p.add_argument("--ctx", required=True, help="comma-separated candidate context sizes")
    p.add_argument("--occupancy", required=True, help="comma-separated occupancy fractions (0..1)")
    p.add_argument("--gen-tokens", type=int, default=200)
    p.add_argument("--slots", type=int, default=1)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    ctx_sizes = [int(x) for x in args.ctx.split(",")]
    occupancies = [float(x) for x in args.occupancy.split(",")]
    return run_context_sweep(args.preset, ctx_sizes, args.slots, occupancies, args.gen_tokens, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
