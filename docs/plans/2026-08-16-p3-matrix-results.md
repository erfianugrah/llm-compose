# P3 matrix results (2026-08-17)

Final state of the local-model bench matrix. Supersedes the partial commit from
2026-08-16 (which had empty eval cells - the eval harness was silently broken,
see docs/reference/eval-harness.md).

## Perf (llama.cpp b10362, RTX 5090)

| preset | TTFT p50 | TTFT p95 | gen tok/s | pp tok/s | VRAM MiB | ctx | slots |
|---|---|---|---|---|---|---|---|
| loop (Gemma 26B-A4B MoE) | 160.7 | 164.4 | 199.2 | 178.7 | 22536 | 196608 | 2 |
| gemma4 (Gemma 31B dense) | 363.2 | 366.3 | 63.0 | 306.8 | 26689 | 65536 | 1 |
| qwen38 (Qwen3.8 27B dense) | 373.2 | 395.7 | 74.0 | 19.7 | 27259 | 196608 | 2 |
| qwen38-mtp (spike) | 297.1 | 298.6 | 85.2 | 27.2 | 30072 | 196608 | 2 |
| qwen36-moe (eliminated) | 202.1 | 210.4 | 196.0 | 57.9 | 27078 | 163840 | 1 |

MTP delta on qwen38: gen +15.1%, pp +38.1%, TTFT -20%, VRAM +10.3%. Adopted
into the qwen38 preset (`spec_type = "draft-mtp"`); the separate qwen38-mtp
preset was removed after the spike.

## Task suite (sensor-gated loop tasks, 6 tasks x 3 runs)

| preset | t1 go-add | t2 go-fix | t3 go-write-tests | t4 ts-add | t5 ts-fix | t6 ts-write-tests | total |
|---|---|---|---|---|---|---|---|
| loop | 3/3 | 3/3 | 0/3 | 3/3 | 3/3 | 0/3 | 12/18 |
| gemma4 | 3/3 | 3/3 | 0/3 | 3/3 | 3/3 | 0/3 | 12/18 |
| qwen38 | 3/3 | 3/3 | 1/3 | 3/3 | 3/3 | 0/3 | 13/18 |

All three one-shot scoped edits; all three stall on write-new-tests (the
suite's ceiling case). Loop-engine tiebreak is economic: 2.7x decode speed,
doomed tasks cost 10-45 min on loop vs up to 75 min on qwen38.

## HumanEval (evalplus, greedy, harness-relative)

| preset | pass@1 |
|---|---|
| loop | 0.116 |
| gemma4 | 0.293 |
| qwen38 | 0.451 |

Absolute values run well below published numbers for these model classes
(greedy through the proxy's sampling params; no chat-template tuning) - treat
as RELATIVE ranking within this harness only, not model-card numbers.

## Gumshoe small track (18 cases x 3 repeats)

| preset | hit | json | steps | forced |
|---|---|---|---|---|
| qwen35-9b (incumbent) | 0.944 | 0.983 | 2.30 | 0.111 |
| gemma4-12b | 0.944 | 0.917 | 1.67 | 0.093 |
| qwen35-4b | 0.870 | 0.930 | 2.43 | 0.259 |
| lfm25-8b | 0.722 | 0.854 | 3.02 | 0.389 |

g15-chain is 0/3 for every preset - the suite's discriminator case.

## BFCL - dropped

Not completed: bfcl-eval 2026.3.23 needed five harness fixes (see
docs/reference/eval-harness.md) and the remaining one crashes inside bfcl's
own leaderboard CSV formatter. Marginal value was judged too low for the
remaining GPU hours: the open decisions were already answered, and the
gumshoe suite is a better tool-use proxy for how these models are actually
driven. The image + runner fixes are committed, so `--bfcl` works up to the
scoring step if revisited.

## Decisions adopted

- **loop engine**: loop preset (Gemma 26B-A4B) stays. Unattended-loop
  economics beat qwen38's +1 task and better HumanEval.
- **interactive coding/thinking**: qwen38, now with MTP on by default.
- **small track (servarr 1070/3080 Ti)**: gemma4-12b ties the qwen35-9b
  incumbent on hit rate with better efficiency (1.67 vs 2.30 mean steps,
  0.093 vs 0.111 forced-final). Deploy-fit check on the 3080 Ti is the
  remaining gate before swapping gumshoe's model.
- **qwen38-mtp preset**: merged into qwen38; spike preset deleted.
