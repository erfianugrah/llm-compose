"""llmc bench needle - needle-in-context accuracy probe.

Measures whether a model still ANSWERS correctly with N tokens of
distractor context in front of the question. Speed is not the point -
`llmc bench context` measures throughput under occupancy; this measures
context QUALITY.

Design:
- For each candidate ctx, register a THROWAWAY preset at that context via
  the same ephemeral-preset mechanism `llmc/bench/context.py` uses
  (_register_ephemeral) and lock+switch to it.
- Per (ctx, depth) cell: fill a prompt to ctx - 512 - gen_tokens tokens,
  splice a one-line needle sentence at `depth * target` tokens, and ask at
  the end for the needle's codeword. Score = word-boundary, case-insensitive
  match of the expected word in the response.
- Results land in the bench store (kind "needle"); a hit=0 on every run of
  a cell is a FAIL for that cell and for the exit code.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from typing import Callable, Optional

from llmc.bench import context as ctx_mod
from llmc.bench import store
from llmc.presets import load_all
from llmc.cli import ProxyClient

NEEDLE_HEADROOM = 512  # tokens reserved for needle + question at the end
NEEDLE_STORE = store.RESULTS_DIR / "needle-runs.jsonl"

# 8 uncommon-but-pronounceable codewords, one per cell (deterministic).
NEEDLE_WORDS = [
    "brazos", "cinder", "dovetail", "fennel",
    "halcyon", "mirthful", "pinnacle", "quartz",
]


# ── Pure helpers (unit-tested) ─────────────────────────────────────────

def needle_sentence(ctx: int, depth: float, word: str) -> str:
    return f"The secret codeword for level {ctx}-{depth} is {word}."


def needle_question(ctx: int, depth: float) -> str:
    return f"What is the secret codeword for level {ctx}-{depth}? Reply with the codeword only."


def needle_level(ctx: int, depth: float) -> str:
    return f"{ctx}-{depth}"


def needle_offset(target: int, depth: float) -> int:
    """Insertion token offset for a needle at `depth` of the filler budget."""
    return int(target * depth)


def cell_filler_target(ctx: int, depth: float, gen_tokens: int,
                       tokenize_fn: Callable[[str], list]) -> int:
    """Filler tokens for a cell: target minus the needle+question tail.

    Returns <=0 when the ctx leaves no room.
    """
    target = ctx - gen_tokens - NEEDLE_HEADROOM
    tail = len(tokenize_fn(needle_sentence(ctx, depth, "word") + "\n" + needle_question(ctx, depth)))
    return target - tail


def score_hit(expected: str, response: str) -> bool:
    """Case-insensitive, word-boundary match of expected in response."""
    return re.search(r"\b" + re.escape(expected) + r"\b", response, re.IGNORECASE) is not None


def expand_cells(ctx_sizes: list[int], depths: list[float], runs: int) -> list[tuple[int, float, int]]:
    """(ctx, depth, run) cells, run index 0-based, ctx-major."""
    return [(ctx, depth, run)
            for ctx in ctx_sizes for depth in depths for run in range(runs)]


def ephemeral_name(ctx: int) -> str:
    return f"needle-{ctx}"


# ── Probe + recording ──────────────────────────────────────────────────

def _probe(proxy: str, model: str, filler: str, needle: str, question: str,
           gen_tokens: int) -> dict:
    prompt = filler + "\n" + needle + "\n" + question
    t0 = time.monotonic()
    resp = ctx_mod._chat(proxy, model, prompt, gen_tokens, ctx_mod.GEN_TIMEOUT)
    latency = time.monotonic() - t0
    content = ""
    try:
        content = resp["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        pass
    return {"latency_s": round(latency, 3), "response_excerpt": content.strip()[:120]}


def _print_grid(log, ctx_sizes: list[int], depths: list[float],
                results: list[dict], runs: int) -> None:
    hdr = ["ctx"] + [f"{d:.2f}" for d in depths]
    widths = [max(len("ctx"), max((len(str(c)) for c in ctx_sizes), default=0))]
    widths += [4] * len(depths)  # 4 = len("3/4") upper bound per depth column
    log("  " + "  ".join(h.rjust(w) for h, w in zip(hdr, widths)))
    log("  " + "-" * (sum(widths) + 2 * len(widths)))
    rate: dict[tuple[int, float], list[int]] = {}
    for r in results:
        rate.setdefault((r["ctx"], r["depth"]), []).append(r["hit"])
    for c in ctx_sizes:
        row = [str(c)]
        for d in depths:
            hits = rate.get((c, d), [0] * runs)
            row.append(f"{sum(hits)}/{len(hits)}")
        log("  " + "  ".join(v.rjust(w) for v, w in zip(row, widths)))


# ── The probe sweep ────────────────────────────────────────────────────

def run_needle(
    preset_name: str,
    depths: list[float],
    ctx_sizes: list[int],
    runs: int = 1,
    gen_tokens: int = 64,
    proxy: str = "http://127.0.0.1:11434",
    log: Callable[[str], None] = print,
    tokenize_fn: Optional[Callable[[str], list]] = None,
) -> int:
    presets = load_all(ctx_mod.MAIN_MODELS_DIR)
    base = next((p for p in presets.values() if p.name == preset_name), None)
    if base is None:
        log(f"error: base preset {preset_name!r} not found")
        return 1

    tokenize_fn = tokenize_fn or ctx_mod.make_tokenizer(proxy, hf_repo=base.bench.get("tokenizer"))
    cells = expand_cells(ctx_sizes, depths, runs)
    max_fill = max((cell_filler_target(c, d, gen_tokens, tokenize_fn) for c, d, _ in cells), default=0)
    if max_fill <= 0:
        log(f"error: no cell fits (smallest ctx={min(ctx_sizes)}, gen={gen_tokens})")
        return 1
    corpus = ctx_mod.build_corpus(tokenize_fn, max_fill, log)

    rid = store.run_id()
    results: list[dict] = []
    client = ProxyClient()
    try:
        for ctx in ctx_sizes:
            sweep_id = ephemeral_name(ctx)
            # Same ephemeral mechanism as the context sweep (in-memory only).
            ctx_mod._register_ephemeral(proxy, sweep_id, base, ctx, 1)
            client.set_lock(sweep_id, owner="bench-needle", wait=True)
            client.set_mode("llm", model=sweep_id, owner="bench-needle")
            log(f"switched to {sweep_id} (ctx={ctx})")
            ctx_mod._chat(proxy, sweep_id, "hi", max_tokens=1, timeout=900)  # warm-up

            for depth, run in [cd for c, cd in ((c, (d, r)) for c, d, r in cells) if c == ctx]:
                word = NEEDLE_WORDS[(ctx_sizes.index(ctx) * len(depths) + depths.index(depth)) % len(NEEDLE_WORDS)]
                filler = ctx_mod.fill_to_tokens(cell_filler_target(ctx, depth, gen_tokens, tokenize_fn),
                                                corpus, tokenize_fn)
                try:
                    probe = _probe(proxy, sweep_id, filler,
                                   needle_sentence(ctx, depth, word),
                                   needle_question(ctx, depth), gen_tokens)
                    hit = int(score_hit(word, probe["response_excerpt"]))
                except Exception as e:
                    probe = {"latency_s": 0.0, "response_excerpt": f"error: {e}"}
                    hit = 0
                metrics = {"ctx": ctx, "depth": depth, "run": run, "hit": hit,
                           "latency_s": probe["latency_s"],
                           "response_excerpt": probe["response_excerpt"],
                           "word": word, "gen_tokens": gen_tokens}
                rec = store.make_record("needle", preset_name, None, metrics, rid)
                store.append(rec, store=NEEDLE_STORE)
                results.append(metrics)
                log(f"  ctx={ctx} depth={depth} run={run}: {'HIT' if hit else 'MISS'} "
                    f"({probe['latency_s']}s) {probe['response_excerpt'][:60]!r}")
            client.set_lock(False, owner="bench-needle")
            ctx_mod._delete_ephemeral(proxy, sweep_id)
    except Exception as e:
        log(f"error: {e}")
        return 1
    finally:
        try:
            client.set_lock(False, owner="bench-needle")
            client.set_mode("llm", model=preset_name)  # restore
        except Exception:
            pass

    _print_grid(log, ctx_sizes, depths, results, runs)
    failed = {("ctx", r["ctx"], "depth", r["depth"]) for r in results if r["hit"] == 0}
    if failed:
        log(f"FAIL: {len(failed)} cell(s) missed on every run")
        return 1
    return 0


def main() -> int:
    p = argparse.ArgumentParser(prog="llmc bench needle")
    p.add_argument("--preset", required=True, help="base preset name")
    p.add_argument("--depths", required=True, help="comma-separated needle depths (fractions of ctx, e.g. 0.05,0.5,0.95)")
    p.add_argument("--ctxs", required=True, help="comma-separated context sizes")
    p.add_argument("--runs", type=int, default=1, help="repetitions per (ctx, depth) cell")
    p.add_argument("--gen-tokens", type=int, default=64, help="max generation tokens per probe")
    args = p.parse_args()
    return run_needle(
        args.preset,
        depths=[float(x) for x in args.depths.split(",") if x.strip()],
        ctx_sizes=[int(x) for x in args.ctxs.split(",") if x.strip()],
        runs=args.runs,
        gen_tokens=args.gen_tokens,
    )


if __name__ == "__main__":
    sys.exit(main())
