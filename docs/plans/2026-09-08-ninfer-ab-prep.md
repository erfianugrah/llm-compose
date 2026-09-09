# 2026-09-08 - NInfer vs llama.cpp A/B: prep, fixes, and the clean run

The question from the session: "is the test flawed?" and "is qwen dumber on
NInfer than on llama.cpp?" The data so far is contaminated by three things:
infrastructure stubs (Model-not-found, 10s exit-1), the provider rename
breaking the suite mid-run, and the engine being swapped under load. This doc
is the prep for a clean run. Nothing here touches the GPU.

## What the 2026-09-08 data actually was (corrected 2026-09-09)

The first draft of this section split the ninfer t4/t5 rows into STUB
(10s, Model-not-found) and REAL (61-65s, 8 iterations, 8 rollbacks) and
read the REAL rows as "the model engaged and failed the probe 8 times". The
proxy log says otherwise. Timeline (UTC):

- 11:39 the suite was launched twice at once (a `&` and a bg task) with the
  stale `llama-server/` rung: 48 Model-not-found rows, both engines.
- 11:52 the corrected suite started in the background under lock owner
  `bench`. ninfer t1 2/2, t2 2/2 clean; t3 run 1 failed at 209s.
- 12:13:35 while that suite was still in its ninfer leg, a bare `llmc unlock`
  cleared its lock and `llmc switch qwen38` swapped the engine. t3 run 2
  (1005s) straddles the swap.
- 12:14:48 a second suite started in the foreground for qwen38 under the SAME
  owner `bench`, locked it, and swapped the GPU to llama.cpp.
- 12:14:53 onward: every request from the first suite's ninfer leg was
  rejected by the scheduler with 422 "model lock active on qwen38: refusing
  to swap" BEFORE forwarding (no `req start` line in the proxy log - exactly
  one nvfp4 request completed after this point). pi exited in ~7s, pi-loop
  logged "(agent exited N; continuing)" and counted a rolled-back iteration;
  eight of those is a 50-65s FAIL that looks like a real run.
- 12:21:11 the first suite released `bench`, which also released the second
  suite's lock (same owner).

So the ninfer t4/t5/t6 rows from 12:15 on were lock refusals, not model
output, and the "2/4 on real runs" read was wrong too: the two REAL passes
were historical rows from the parity run, not from that day. Valid ninfer
data from 2026-09-08: t1 2/2, t2 2/2, t3 0/1. The t4/t5 question was OPEN
until the interleaved run below.

The store was purged 2026-09-09: 63 uncommitted rows dropped (the 48 stubs,
the 7 post-swap ninfer rows, and 8 llama.cpp rows from the two overlapping
suites sharing one slot), the 5 valid ninfer rows kept.

## The five things to fix before the A/B means anything (1, 2, 3, 4-preflight, 5 landed 2026-09-09)

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
(`--interleave` is the default whenever 2+ presets are compared; `--no-interleave` restores AABB.)

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

## Result of the clean runs (2026-09-09)

Two interleaved passes of the four discriminating tasks, 5 runs each. The
first (run `20260908-213625`) exposed a wiring bug, the second (run
`20260908-232622`) is the number.

**The wiring bug.** The harness handed pi `llmc/<model file id>`
(`qwen3.8-27b-nvfp4`, `Qwen3.8-27B-UD-Q4_K_M`). pi's llmc provider registers
models by PRESET NAME (`pi --list-models` shows `llmc qwen38-ninfer`), so
every leg logged `Model "..." not found for provider "llmc". Using custom
model id.` pi's fallback copies the first listed model's metadata, and the
proxy's GET /v1/models was map-ordered, i.e. random per call - so each leg
ran with a random reasoning flag, context window and output cap. Effort was
still equal on both sides (the proxy pins medium on NInfer; llama-server
b10472 has no `reasoning_effort` string in its binary, so it ignores the
per-request field and takes medium from its template kwargs), and the wall
times rule out any high-effort binge. Fixed: the rung is now the preset
name, the row stores it (`rung`), the preflight accepts preset names, and
the proxy sorts /v1/models.

| task | llama.cpp | NInfer | (old wiring, run 213625) |
|---|---|---|---|
| t1-go-add-truncate | 5/5 | 5/5 | 5/5 vs 5/5 |
| t2-go-fix-palindrome | 5/5 | 5/5 | 5/5 vs 5/5 |
| t4-ts-add-camelcase | 5/5 | 5/5 | 5/5 vs 5/5 |
| t5-ts-fix-slugify | 5/5 | 5/5 | 5/5 vs 5/5 |

Corrected run, 20 runs per engine: iterations 20 (llama.cpp, all first-try)
vs 24 (NInfer, 17 first-try); wall p50 35.0s vs 34.1s; 0 agent errors, 0
fallback warnings. Parity. The "NInfer is dumber" hypothesis has no support
on this suite; the Radarr pagination mistake that started it was one
anecdote on one session and is the model being the model, not the engine.

Phase 2 (t3, t6, interleaved, run `20260908-220907`) was stopped after one
pair: 0/1 vs 0/1 (885s vs 3128s, the llama.cpp run had one aborted pi
iteration, recorded as `agent_errors: 1`, below the invalid threshold).
These tasks fail for every model benched here and cannot separate engines.

What was NOT tested: sampling parity (the NInfer preset still has no
sampling fields), KV precision parity (fp8 vs f16), HumanEval on the current
llama.cpp preset (the 0.451 is from the old quant), and anything at long
context. None is needed to close the regression question; they are the
isolation steps if a gap ever shows up.

