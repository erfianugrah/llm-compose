# 2026-09-08 - NInfer vs llama.cpp A/B: prep (not running yet)

The question from the session: "is the test flawed?" and "is qwen dumber on
NInfer than on llama.cpp?" The data so far is contaminated by three things:
infrastructure stubs (Model-not-found, 10s exit-1), the provider rename
breaking the suite mid-run, and the engine being swapped under load. This doc
is the prep for a clean run. Nothing here touches the GPU.

## What the current data actually shows

Per-task pass rates, real runs only (wall > 30s = the model engaged):

| task | llama.cpp | ninfer |
|---|---|---|
| t1 go-add-truncate | 36/36 | 5/5 |
| t2 go-fix-palindrome | 27/27 | 5/5 |
| t3 go-write-split-tests | 1/14 | 0/3 |
| t4 ts-add-camelcase | 23/23 | 1/3 |
| t5 ts-fix-slugify | 17/17 | 1/3 |
| t6 ts-write-slug-tests | 0/13 | 0/3 |

t4/t5 are the gap. But the ninfer t4/t5 numbers mix two failure modes:
- STUB runs (10-11s, exit 1): the model never loaded - the provider rename
  left the suite pointing at `llama-server/<id>` when the provider was
  already `llmc`. Infrastructure, not quality.
- REAL runs (61-65s, 8 iterations, 8 rollbacks): the model engaged and
  failed the probe 8 times. Quality signal.

The two REAL ninfer t4/t5 failures are the only clean data points, and both
came AFTER the two REAL passes (34.5s, 36.7s, 1 iteration, the first runs
after the fix). So the honest read: 2/4 on t4/t5 real runs, not 1/3.

## The five things to fix before the A/B means anything

### 1. Purge the stub rows
`bench/results/runs.jsonl` has the Model-not-found stubs mixed in with real
runs. The pass-rate math above already filters by wall>30s, but the store
should not carry them. One-off: filter out task records with wall_s < 30.

### 2. Unique lock owner + refuse to start on contention
The bench suite locks the preset it is running. If a second run (or a pi
session) grabs the lock mid-suite, the suite's swap fails and the task fails
with a stub. Fix: `llmc bench tasks` takes the lock with a unique owner
(the run id), and REFUSES to start if the lock is held by someone else -
no silent fallback. `--force` unlocks for the operator.

### 3. Agent exit code + tail in the record, abort after 2 agent errors
The run records carry `exit_code` and `iterations` but not the agent's
stderr tail or the loop's exit code. When a task fails you cannot tell
"model wrote bad code" from "the loop crashed". Add `agent_tail` (last 400
chars of the agent's stdout+stderr) and `loop_exit` to the metrics. Abort
the suite after 2 consecutive agent-level errors (not task failures) - a
stub storm should stop the run, not fill the store.

### 4. Rung preflight + engine-log capture
Before each leg, verify the rung resolves (a 1-token probe against the
proxy). If the rung is dead, fail fast with a clear error, not 8 iterations
of stubs. Capture the engine log lines for the leg's requests into the
record so a failure is diagnosable after the fact.

### 5. Interleaved ABAB, not AABB
`run_tasks` currently iterates preset-then-task (AABB: all ninfer, then all
llama). That confounds engine state (a warm engine vs a cold swap) with
quality. The fix: outer loop over (run, task), swap engine per leg - ABAB
so each task sees both engines in the same session state. One restructure
in `run_tasks`. The CLI flag is proposed (`--interleave`), not yet
implemented.

## The confounds to name before running

The two presets are NOT the same model under two engines. They are two
different serving stacks. Verified from the preset files and proxy source:

- **Quantization**: NVFP4 (ninfer) vs unsloth UD-Q4_K_M (llama.cpp). The
  variable you actually want to measure.
- **KV precision**: ninfer serves fp8 KV; llama.cpp runs the default f16.
  A quality confound outside the quant.
- **Chat template**: llama.cpp uses the fixed v22.4 template (adopted for
  tool-call reliability); ninfer uses the artifact's baked-in template. In
  an agentic loop, template quality IS tool-call reliability, and tool-call
  failures cost iterations. If ninfer loses on t4/t5, this is the first
  suspect, not the quant.
- **Sampling**: the llama.cpp preset pins temperature/top_p/top_k/min_p/
  presence penalty as serve flags; the ninfer preset has no sampling fields
  and the proxy only injects reasoning effort. ninfer runs pi's defaults.
  Unpinned sampling is the cheapest confound to remove.
- **Speculative decoding**: ninfer runs MTP draft=3 + lm-head-draft;
  llama.cpp runs no speculation (both speculators degraded under churn).
  Correct speculation is output-equivalent, so this should be speed only -
  the d1 preset exists to test that assumption.
- **Output cap**: ninfer publishes 65536; llama.cpp presets publish none,
  so pi registers them at 16384. Medium effort rarely hits either, but it
  is asymmetric.
- **Effort**: both legs land at medium (proxy coerces ninfer, llama.cpp
  preset serves medium). Already equal.

## How to run it (when the GPU is quiet)

Decide the question first:
- **"Which stack daily?"** - run the as-deployed A/B, do not equalize. The
  template and KV choices are part of each product.
- **"Does NVFP4 lose quality?"** - only if the first A/B shows a gap. Then
  isolate one variable at a time: pin sampling on ninfer to match, d1 preset
  to rule out MTP, f16 KV if ninfer offers it.

The command (after the five fixes land):
```
llmc bench tasks --presets qwen38-ninfer,qwen38 --runs 5 --interleave
```
(`--interleave` is the proposed new flag from item 5 - not yet implemented.)

One existing data point cuts against the "dumber" hypothesis: HumanEval
pass@1 was 0.598 on ninfer vs 0.451 on llama.cpp - but the llama.cpp number
is from 2026-08-17 on the older Q4_K_M + template, so it needs a rerun
before it counts.

## Order

1. Purge the stub rows (one-off).
2. Items 2 + 4 (lock owner, rung preflight) - small, ship first.
3. Item 5 (interleave) - the method fix.
4. Items 3 (exit code + tail) - makes the next failure diagnosable, not a
   blocker for a valid number.
5. Then the ABAB run on a quiet GPU.
