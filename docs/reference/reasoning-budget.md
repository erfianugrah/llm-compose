# Thinking budget - the knob, and why nothing sets it

`runtime.reasoning_budget` in a preset maps to llama.cpp's
`--reasoning-budget`: `-1` unrestricted (the default), `0` ends thinking
immediately, `N > 0` caps thinking at N tokens. The plumbing exists in both
schemas (`llmc/presets.py` and `proxy-go/internal/proxy/presets.go`) and the
entrypoint.

**No preset sets it.** This document is why.

## What it does when it fires

Budget exhaustion is not a truncated response. The server injects the
end-of-thinking tag (plus `--reasoning-budget-message` if set) and the model
answers from a reasoning chain that was cut mid-sentence. So the failure mode
is not "shorter answer", it is "answer derived from incomplete reasoning".

## Why it was proposed, and why that reason did not survive

On 2026-09-02 a loop task took 1165.4s inside ONE iteration - no retry storm,
one very long turn. The obvious reading was a thinking binge, and a budget is
the only thing that bounds that tail (`reasoning_effort` lowers the average
and bounds nothing).

Then the generation lengths were actually measured, per request, from the
llama-server timing lines:

| workload | requests | total gen | p50 | p90 | p95 | p99 | max |
|---|---|---|---|---|---|---|---|
| 4 fast loop tasks, effort=medium | 18 | 3,902 | 198 | 407 | 412 | 412 | 412 |
| same 4 tasks, effort=xhigh | 12 | 5,368 | 307 | 890 | 1134 | 1134 | 1134 |
| pylon Phase 0 (real greenfield build) | 202 | 490,818 | 1367 | 6237 | 7819 | 9201 | 9564 |

Two conclusions:

1. **The micro-benchmarks cannot answer this question.** At a 412-token
   maximum, any budget worth setting is inert - the knob would have measured
   as "no effect" and been adopted on a false negative. Even at xhigh the
   ceiling is 1134 tokens. A real task generates 20x that.
2. **The 1165s outlier was not a thinking binge.** The largest single
   generation ever observed here is 9564 tokens, which is ~127s at this
   model's ~75 tok/s. A 1165s wall is therefore many turns or a stall, not
   one long thought - and a thinking budget would not have touched it. The
   original justification was wrong.

## What a budget would actually cost

On the real task, 9 of 202 requests (4.5%) exceeded 8000 tokens - and that
run passed **on iteration 1**. Those long generations were part of a
successful trajectory, so an 8000-token cap would have cut into work that was
converging, not into a spiral. Nothing observed exceeded 9564 tokens, so even
a generous runaway guard (16000, say) would never have fired in any run
recorded here. Setting one would be speculative config.

Note the measurement's own limit: `n_gen` counts thinking AND answer tokens,
while the budget caps thinking only. So the table is an upper bound on how
often a budget of a given size would bind.

## When to revisit

Set a budget when there is evidence of a genuine runaway - a generation that
does not terminate, or one that blows past the ~9.5k ceiling observed here -
and set it above the p99 of *successful* runs, never below. A cap tuned to
make the suite faster will make it solve less; that is a regression wearing a
speedup's clothes.

## Measuring it yourself

```bash
docker logs llama_server 2>&1 | grep -o 'n_gen = *[0-9]*' | grep -o '[0-9]*'
```

One line per request, thinking + answer combined. Prompt sizes come from the
same log:

```bash
docker logs llama_server 2>&1 | grep -oP 'prompt eval time =.*?/\s+\K\d+(?= tokens)'
```
