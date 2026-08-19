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

## Adoption in this stack - REVERSED (2026-08-19)

First adoption was `ngram-mod` as the default `spec_type`. The p6
validation suite then caught ngram-mod degrading identically to MTP
under agentic churn (0.6-0.7 tok/s, GPU idle ~106W, restart restores).
Final state: `models/qwen38.toml` runs NO speculation - the only config
proven stable under agentic task churn. `SPEC_NGRAM_N_{MIN,MAX,MATCH}`
plumbing remains in the entrypoint + schema for future re-tests on
newer pins. `qwen38-xhigh` keeps MTP for babysat interactive use.

## Measured results (2026-08-19, b10472, RTX 5090)

Perf suite vs no-spec baseline (73.2 tok/s gen), cold pool then warm pool:

| variant | cold gen | warm gen | TTFT p50 |
|---|---|---|---|
| defaults (n-match 24, n-min 48, n-max 64) | 141.1 (+93%) | 462.1 (+531%) | 351.0-361.8ms |
| small-n (n-min 4, n-max 8, n-match 32) | 146.3 (+100%) | 262.6 (+259%) | 252.7-259.2ms |

Defaults won on warm-pool peak (long drafts on verbatim repeats), small-n
won TTFT by ~100ms. Caveat on warm numbers: the perf harness reuses
canned prompts, so a warm pool is maximally matched - the 462.1 tok/s is
the mechanism's ceiling, not a realistic agentic-workload figure.
Defaults adopted (bigger upside, TTFT is a wash at ~100ms).

Why nothing speculative runs by default - the degradation applies to
BOTH speculator types. MTP evidence first:

1. MTP (draft-mtp) decode degrades to ~0.5 tok/s within ~10 min of
   agentic task churn, on both b10362 and b10472. pp stays fast
   (~1500 tok/s), draft acceptance stays healthy (0.8+), GPU idles
   (~100W). Restart restores; re-degrades.
2. No-spec control: full task suite at a sustained 55-66 tok/s, incl.
   t3's 5433s/8-iteration arc. Flawless.
3. Matches upstream: #27151 (MTP acceptance decays over time, restart
   restores) and #27296 (MTP corrupts draft context across long/short
   prompt mixes on Qwen3.8-27B; reproduced by them on b10344+b10472).
4. kept as `qwen38-xhigh` preset for babysat interactive use only.
5. ngram-mod (draftless): identical degradation caught by the p6
   validation suite on 2026-08-19 - the hash-pool warmup stays fine,
   but under sustained loop-task churn decode collapses the same way.
   Shared-mechanism hypothesis: rejected-draft rollback of the hybrid
   Gated DeltaNet recurrent state, expensive at 24k+ context. No-spec
   never rolls back and never degrades.

b10472 pin bump (was b10362) also fixed a separate abandoned-stream
slot-parking bug (frozen slot, dec stop, GPU idle, slot never frees;
verified fixed: slot releases ~20s after client disconnect).

## Caveats and conflicting reports

- **Conflicting benchmarks resolved here.** The thread OP measured +6-8
  t/s on JS generation (5060 Ti); another user measured -4 t/s on an
  R9700; one Vulkan user reports ngram consistently slower. The hash
  work is CPU-side, so the sign is hardware-dependent. On this 5090
  stack: strongly positive (cold +93-100%).
- **Known issue: stuck-loop on failed verification** (upstream PR #25819,
  open, WIP mitigation). When an ngram-mod draft fails verification, the
  draft can be reused and fail again in a loop with a non-deterministic
  end condition. If the server hangs mid-generation with ngram-mod on,
  this is the first suspect. (Not observed in our suite so far.)
- **Acceptance rate is a vanity metric.** Do not tune `--spec-draft-p-min`
  to raise it: one scripted test showed p-min 0.85 lifting acceptance
  from 76.5% to 92.5% while dropping throughput 7.4% - higher p-min
  suppresses drafts rather than improving them. Accepted-token count
  tracks tok/s; optimize for tok/s.
- **Draft-KV quant debunked** (community follow-on comments): quantizing
  the MTP drafter's KV only lowers acceptance (fewer accepted tokens =
  slower), quality untouched, and it barely saves VRAM. We do not
  quantize draft KV. One Vulkan user reports q4 draft best on his
  backend - backend-specific; on CUDA/sm120 unquantized is right.
- **Mac/bandwidth-constrained slowdowns don't apply here.** Reports of
  MTP hurting on M4 Max / M1 Ultra are unified-memory bandwidth effects.

## Related thread claims checked and rejected

From the same thread, deliberately NOT adopted:

- Sampling temp 0.4 / top_p 0.90 / top_k 15 / min_p 0.02 labelled
  "Official / Recommended" - hallucinated (the thread OP sourced it from
  ChatGPT). Actual model-card thinking-mode values (temp 1.0, top_p 0.95,
  top_k 20, min_p 0) remain in `models/qwen38.toml`.
- KV cache q4_1 on the MAIN cache to stretch context - a 16 GB VRAM
  workaround; pointless at 32 GB, and q4 main KV measurably degrades
  long-context recall vs q8. (Draft-KV quant is a separate no-op we
  also skip; see caveats.)

## Sources

- Upstream spec doc (HTTP 200 verified): https://raw.githubusercontent.com/ggml-org/llama.cpp/master/docs/speculative.md
- Priority order + draft orchestration: `common/speculative.cpp`
  (`common_speculative_init`, `common_speculative_draft`) at master;
  flag list cross-checked at tag b10362 (README + docs/speculative.md
  both present in the pinned build).
- ngram-mod introduction: llama.cpp PR #19164; score-based pruning PR
  #19294 (open); stuck-loop mitigation PR #25819 (open, WIP).
- MTP degradation upstream issues: #27151, #27296 (both open).
- Community reports: r/LocalLLaMA Qwen3.8-27B config thread (2026-08),
  incl. the p-min debunk and R9700 regression; OVERBRING Labs follow-on
  article + comments (2026-08-17), incl. small-n guidance
  (4/8/32 vs defaults) and the draft-KV-quant debunk.
