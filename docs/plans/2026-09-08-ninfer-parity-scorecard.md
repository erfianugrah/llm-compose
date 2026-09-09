# NInfer parity + quality scorecard (2026-09-08)

The follow-up to the speed spike (`2026-09-06-ninfer-nvfp4-spike.md`): the
engine is fast, this doc answers "but is it accurate/good". Everything here
is reproducible - each row names the command that produced it.

## What "good" means for a coding-agent engine

Not benchmark maxima. The bar is: does it drive a real agent loop to a
correct result, at medium effort, without tool-call corruption, context
collapse, or silent truncation. Speed is already proven; this is the
quality side.

## Task parity (the decisive metric) - MEASURED

`llmc bench tasks` = six sensor-gated loop tasks (t1-t6), each a real
edit-the-code-until-the-probe-test-passes loop with a red baseline and a
canary. Run against `qwen38-ninfer` on 2026-09-08:

| task | kind | result | wall |
|---|---|---|---|
| t1-go-add-truncate | Go, add a fn | PASS | 81s, 1 iter |
| t2-go-fix-palindrome | Go, fix a bug | PASS | 36s, 1 iter |
| t3-go-write-split-tests | Go, write tests | FAIL | 1898s, 8 iter |
| t4-ts-add-camelcase | TS, add a fn | PASS | 35s, 1 iter |
| t5-ts-fix-slugify | TS, fix a bug | PASS | 37s, 1 iter |
| t6-ts-write-slug-tests | TS, write tests | FAIL | 441s, 8 iter |

**4/6.** Reproduce: `llmc bench tasks --presets qwen38-ninfer --runs 1`.

### Interleaved A/B vs llama.cpp (2026-09-09) - the controlled number

The single-run table above and the 2026-09-08 "t4/t5 regression" scare were
both uncontrolled (docs/plans/2026-09-08-ninfer-ab-prep.md has the forensics:
the regression was a lock-refusal artifact). This run had both engines on
the same tasks in the same session, engine swap per leg (ABAB, order
alternating per run), one suite alone on the GPU, unique lock owner, and -
new that day - pi resolving the REGISTERED preset ids (`llmc/qwen38-ninfer`,
`llmc/qwen38`). Every earlier task row used file-derived ids pi had never
registered, so pi fell back to a "custom model id" that copied the metadata
of whichever model the proxy listed first, and that order was random per
call. Run `20260908-232622`, 40 rows, 0 invalid, 0 agent errors, 0 fallback
warnings:

| task | llama.cpp UD-Q4_K_M | NInfer NVFP4 | wall p50 llama / ninfer |
|---|---|---|---|
| t1-go-add-truncate | 5/5 | 5/5 | 63.5s / 36.6s |
| t2-go-fix-palindrome | 5/5 | 5/5 | 33.4s / 34.8s |
| t4-ts-add-camelcase | 5/5 | 5/5 | 31.9s / 31.8s |
| t5-ts-fix-slugify | 5/5 | 5/5 | 30.4s / 31.8s |

Over 20 runs each: llama.cpp 20 iterations, all first-try; NInfer 24
iterations, 17 first-try. Wall p50 35.0s vs 34.1s. The four extra NInfer
iterations are three runs that needed a second or third attempt (t1 x1,
t4 x2); at n=20 that is not a separable difference. The same suite under
the old (random-metadata) wiring, run `20260908-213625`, was also 20/20 vs
20/20 with 25 vs 27 iterations.

Reproduce: `llmc bench tasks --presets qwen38-ninfer,qwen38 --runs 5
--tasks t1-go-add-truncate,t2-go-fix-palindrome,t4-ts-add-camelcase,t5-ts-fix-slugify`
(interleave is the default with two presets).

t3/t6, the test-writing tasks, were started as a second interleaved pass
and stopped after one pair: 0/1 vs 0/1, NInfer losing in 885s, llama.cpp in
3128s. They fail for every model ever benched here and cannot separate the
engines; the remaining legs would have cost hours for nothing.

### Read on the two failures

t3 and t6 are the two test-WRITING tasks, and they are the hardest in the
suite for every model and engine ever benched here - not a NInfer
regression:

| model / engine | t3 | t6 |
|---|---|---|
| qwen38 Q4_K_M (llama.cpp) | 1/3 | 0/3 |
| qwen38 medium-nospec (llama.cpp) | 0/1 | 0/1 |
| gemma-4-31B (llama.cpp) | 0/3 | 0/3 |
| loop-gemma-4-26B (llama.cpp) | 0/6 | 0/6 |
| **qwen38-nvfp4 (NInfer)** | **0/1** | **0/1** |

The llama.cpp baseline also goes 4/6 with the same two misses. NInfer at
medium matches its own-weights llama.cpp baseline on the suite that most
resembles the work it was adopted for. One t3 failure's run output carried
`inference request expired while waiting for admission` - the needle loop
was sharing the engine at the time (max_concurrency=1), so that pass was
contention, not a clean quality signal; but t3 is hard enough that it fails
uncontended too (every prior model shows this).

## Tool calls - MEASURED (BFCL non_live, 2026-09-08)

`llmc bench eval --presets qwen38-ninfer --bfcl` (full non_live collection,
image bench-eval, against the live proxy). Per-category AST accuracy:

| category | accuracy |
|---|---|
| simple_python | 68.25% |
| simple_javascript | 48.00% |
| simple_java | 40.00% |
| parallel | 76.50% |
| parallel_multiple | 73.00% |
| multiple | 76.00% |

The `overall` aggregate is null in the record because BFCL's leaderboard
aggregator crashed on a None score from the `irrelevance` category
(TypeError: NoneType * int) - a harness display bug, not a model failure.
Function-calling is clearly functional (the multi/parallel categories the
agent loop actually exercises are 73-76%); simple_java/js are the weak
categories.

Indirect corroboration: the lockstep v1 build-out and this session's t1-t6
+ needle loop drove tool calls through the OpenAI-compatible endpoint with
zero parse failures logged (336 requests, 0 errors in the spike run).

## Long-context - PARTIAL

- Speed under occupancy: proven flat to 262K (spike).
- Quality under occupancy: the probe exists - `llmc bench needle`
  splices a codeword at a depth fraction of a filled context and scores
  retrieval. Tokenizer blocker FIXED (776fa17: local-HF-tokenizer fallback
  for engines with no `/tokenize`). Structural blocker FIXED (`--no-swap`
  mode, a8bbaa5 + later: probe the resident model at its configured ctx, no
  ephemeral preset/lock/switch).
- **NEW finding, 2026-09-08: the effective serving ceiling is below the
  configured 262144, and it is load-dependent.** The preset now advertises
  252928 - NInfer's own published eval value for this exact artifact
  ("to fit the RTX 5090 after weights"). The KV pool fits at startup
  (~324-633 MiB free), but a single maxed-out request 400s when the draft
  reserve + fp8 page rounding + prefix-cache state push the total over, and
  the usable window shrinks further under desktop GPU load (browsers,
  terminals on the same 5090). The engine returns 400 and the client
  retries smaller - that IS the behaviour, not a bug. Speed flat to 262K is
  a capacity claim; a single maxed-out request tops out at whatever the
  current free KV allows. The needle retrieval probe at the ceiling has not
  completed a clean run (every attempt 400'd at a different point as load
  varied); it needs a quiet GPU.

## Churn stability - NOT YET MEASURED

The spike's "no churn decay" bar (p6) is not closed by this session.
t1-t6 is short-horizon. The 6.2% sub-100 tok/s tail (worst 33.9, clustered
66-72K ctx) is still unexplained and is the strongest known quality/
robustness caveat - see the preset description.

## Standardized accuracy - MEASURED (HumanEval, 2026-09-08)

`llmc bench eval --presets qwen38-ninfer --humaneval` (evalplus, greedy,
164 problems, through the live proxy):

- **HumanEval pass@1 = 0.598** (n=164), HumanEval+ pass@1 = 0.591.
- This is the FIRST working eval number this stack has produced - the two
  2026-08-17 llama.cpp rows in runs.jsonl errored at parse time
  (`no eval_results.json`), so there is no same-stack llama.cpp baseline to
  compare against. Treat 0.598 as the reference point future presets/
  engines run against, not as a delta.
- HellaSwag (language-modeling, loglikelihood) not yet run: needs the
  `[bench] tokenizer` field, which is set, so `--hellaswag 1000` is
  runnable next time the GPU is up.

## Frontier anchor - NOT YET MEASURED

No absolute ceiling is set. There is no local GPT-5 endpoint (pi's
providers are `llama-server`, `openrouter`, `external`; the `:4141` URL in
an earlier draft of this section was wrong - that port is not pi). The
anchor path is OpenRouter: point `bench/run-evals.py --base-url
https://openrouter.ai/api/v1 --model <frontier-model>` at it with the
OpenRouter key for the HumanEval/BFCL ceiling, plus a t1-t6 subset.
DEFERRED.

## Ops hardening landed this session

- `runtime.max_output_tokens` per preset -> `meta.max_output` in
  /v1/models -> pi registers `maxTokens` from it (plan step 1). Verified
  at the engine: `max output 65,536`.
- `make ship-proxy` no longer restarts the stack (it ships the Python
  rollback-lane image but the restart was recreating the LIVE Go proxy
  from a stale image; a new preset key crash-looped it). `ship-proxy-go`
  is the daily flow.
- Per-effort NInfer presets: `qwen38-ninfer-low` / `-xhigh`. The template
  enum is low|medium|xhigh - no "high" exists on this engine.
- `make check-ninfer-drift` gates `build-ninfer`: fails when the pinned
  upstream checkout moved off the reviewed commit in `NINFER_PIN`.

## Verdict

Quality parity with its own-weights llama.cpp baseline on the decisive
suite (4/6, same two hard misses). The unanswered questions are the ones
that need the GPU: standardized accuracy (eval), long-context QUALITY
(needle, blocked on a NInfer tokenizer path), and churn stability. None
are blockers to using the preset; all three are scheduled.
