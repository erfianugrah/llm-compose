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
resembles the work it was adopted for. One t3 failure carried
`inference request expired while waiting for admission` - the needle loop
was sharing the engine at the time (max_concurrency=1), so that iteration
is contention, not a clean quality signal; the t3 task is hard enough that
it fails uncontended too (every prior model).

## Tool calls - MEASURED

- BFCL is wired (`llmc bench eval --presets qwen38-ninfer --bfcl`) but NOT
  yet run on this engine. DEFERRED (GPU released).
- Indirect: the lockstep v1 build-out and this session's t1-t6 + needle
  loop all drove tool calls through the OpenAI-compatible endpoint with
  zero parse failures logged. 336 requests, 0 errors in the spike run.

## Long-context - PARTIAL

- Speed under occupancy: proven flat to 262K (spike).
- Quality under occupancy: the probe now EXISTS - `llmc bench needle`
  (this session) splices a codeword at a depth fraction of a filled
  context and scores retrieval. **But it is blocked on NInfer**: the probe
  (like the context sweep) tokenizes via llama.cpp's `/tokenize` endpoint,
  and NInfer has no tokenizer endpoint (404). Needle currently runs on
  llama presets only. A NInfer tokenizer path is the follow-up.

## Churn stability - NOT YET MEASURED

The spike's "no churn decay" bar (p6) is not closed by this session.
t1-t6 is short-horizon. The 6.2% sub-100 tok/s tail (worst 33.9, clustered
66-72K ctx) is still unexplained and is the strongest known quality/
robustness caveat - see the preset description.

## Standardized accuracy - NOT YET MEASURED

`llmc bench eval --presets qwen38-ninfer --humaneval --bfcl --hellaswag N`
is wired and the eval image builds; it has not been run on this engine.
DEFERRED (GPU released). This is the row that compares against published
Qwen3.8 numbers.

## Frontier anchor - NOT YET MEASURED

No absolute ceiling is set. Plan: `run-evals.py --base-url
http://127.0.0.1:4141/v1 --model gpt-5-high` (pi's own endpoint) for the
HumanEval/BFCL anchor, plus a t1-t6 subset. DEFERRED.

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
