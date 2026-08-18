# Speculative decoding: ngram-mod + draft-mtp

Reference for combining llama.cpp's `ngram-mod` self-speculation with the
MTP heads already used by the qwen38 preset (`spec_type = "draft-mtp"`).
Researched 2026-08-18 against upstream master docs + source, cross-checked
against the pinned build (b10362). Trigger: a LocalLLaMA config thread
reporting +6-8 t/s on JS generation from adding `ngram-mod` next to MTP.

## What ngram-mod is

A draftless speculative decoding implementation: no draft model, no extra
VRAM. For each n-gram it computes a hash (LCG) and stores the next token in
a hash pool; during speculation it rolls the hash over the last n tokens and
drafts from the pool.

Characteristics (upstream `docs/speculative.md`):

- Lightweight: ~16 MB, constant memory and complexity.
- Variable draft lengths (m is not fixed, unlike ngram-map).
- The hash pool is shared across all server slots, so parallel requests
  benefit from each other's history. The pool accumulates accepted tokens
  over time - it gets better the longer the server runs and the more
  repetitive the workload.

Upstream-listed applications: iterating over a block of text/code (llama.vim
style), reasoning models repeating their thinking in the final answer,
summarization. Agentic coding hits all three: tool-call outputs recur across
turns, code patterns repeat, and thinking models restate their reasoning.

## How it combines with draft-mtp

`--spec-type` takes a comma-separated list. Implementations have a fixed
priority order, set in `common/speculative.cpp` (`common_speculative_init`):

1. ngram-simple, ngram-map-k, ngram-map-k4v, **ngram-mod**, ngram-cache
2. draft-simple, draft-eagle3, **draft-mtp**, draft-dflash, draft-dspark

`common_speculative_draft()` walks the list in order; each implementation
drafts only for sequences still marked `drafting`, and a sequence with a
non-empty draft is done. So with `ngram-mod,draft-mtp`:

- ngram-mod drafts first wherever it has a hash-pool hit (repetitive
  content: tool outputs, repeated code, restated thinking).
- MTP drafts for everything else (novel generation).

They are complementary, not competing. MTP's cost (draft-context VRAM +
draft compute) is unchanged; ngram-mod adds ~16 MB RAM and CPU hash work.

Upstream doc phrasing: "If a draft model is combined with a draftless
decoding the draftless decoding has higher precedence."

## Parameters and defaults

| Flag | Default | Notes |
|---|---|---|
| `--spec-ngram-mod-n-match` | 24 | lookup n-gram length |
| `--spec-ngram-mod-n-min` | 48 | min draft tokens |
| `--spec-ngram-mod-n-max` | 64 | max draft tokens |

Upstream guidance: small n not recommended; MoEs require long drafts; dense
models can reduce n-min/n-max. The values quoted in the Reddit thread
(24/48/64) are just the defaults.

`--spec-default` enables ngram-mod alone.

## Adoption path in this stack

The entrypoint passes `SPEC_TYPE` straight to `--spec-type`, so no proxy
schema change is needed:

```toml
# models/qwen38-ngram.toml (A/B variant)
spec_type = "ngram-mod,draft-mtp"
```

A/B mechanics (same pattern as `bench/p4-mtp.sh`):

1. Presets dedup by model_id (GGUF stem) - the variant needs its own
   filename: `ln -f Qwen3.8-27B-Q4_K_M.gguf Qwen3.8-27B-Q4_K_M-ngram.gguf`
   in `~/docker-volumes/llama-server/models/`.
2. `llmc bench perf` across the two presets (lock+switch per preset, results
   land in `bench/results/runs.jsonl` with preset_hash provenance).
3. Warm-up caveat: the ngram pool starts empty and builds over the session.
   A cold-start benchmark understates it; either warm the server with a
   representative workload before measuring, or compare steady-state rates.
4. Server logs print per-implementation stats -
   `statistics ngram_mod: #calls = ..., #gen drafts = ..., #acc tokens = ...` -
   which shows whether ngram-mod is actually firing on the workload.

Availability: `ngram-mod` is present in the pinned b10362 build (flag +
docs verified against the b10362 tag). No pin bump required.

## Caveats and conflicting reports

- **Conflicting benchmarks.** The thread OP measured +6-8 t/s on JS
  generation (5060 Ti); another user measured -4 t/s on an R9700. The hash
  work is CPU-side, so the sign of the effect is hardware- and
  workload-dependent. A/B on the 5090 before adopting.
- **Known issue: stuck-loop on failed verification** (upstream PR #25819,
  open, WIP mitigation). When an ngram-mod draft fails verification, the
  draft can be reused and fail again in a loop with a non-deterministic end
  condition. Root cause unclear as of 2026-08-18. If the server hangs mid-
  generation with ngram-mod on, this is the first suspect.
- **Acceptance rate is a vanity metric.** Do not tune `--spec-draft-p-min`
  to raise it: one scripted test showed p-min 0.85 lifting acceptance from
  76.5% to 92.5% while dropping throughput 7.4% - higher p-min suppresses
  drafts rather than improving them. Accepted-token count tracks tok/s;
  optimize for tok/s.
- **Mac/bandwidth-constrained slowdowns don't apply here.** Reports of MTP
  hurting on M4 Max / M1 Ultra are unified-memory bandwidth effects; MTP is
  already benched positive on the 5090 (gen +15.1%, pp +38.1%, TTFT -20%).

## Related thread claims checked and rejected

From the same thread, deliberately NOT adopted:

- Sampling temp 0.4 / top_p 0.90 / top_k 15 / min_p 0.02 labelled "Official /
  Recommended" - hallucinated (OP sourced it from ChatGPT). Actual model-card
  thinking-mode values (temp 1.0, top_p 0.95, top_k 20, min_p 0) are already
  in `models/qwen38.toml`.
- KV cache q4_1 to stretch context - a 16 GB VRAM workaround; pointless at
  32 GB where 196k x 2 slots already fit, and q4 KV measurably degrades
  long-context recall vs q8.

## Sources

- Upstream spec doc (HTTP 200 verified): https://raw.githubusercontent.com/ggml-org/llama.cpp/master/docs/speculative.md
- Priority order + draft orchestration: `common/speculative.cpp`
  (`common_speculative_init`, `common_speculative_draft`) at master;
  flag list cross-checked at tag b10362 (README + docs/speculative.md both
  present in the pinned build).
- ngram-mod introduction: llama.cpp PR #19164; score-based pruning PR
  #19294 (open); stuck-loop mitigation PR #25819 (open, WIP).
- Community reports: r/LocalLLaMA Qwen3.8-27B config thread (2026-08),
  incl. the p-min debunk and the R9700 regression report.
