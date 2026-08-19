# 2026-08-19 - qwen38: reasoning_effort A/B + speculative-decoding findings (P5)

Session outcome doc. Raw data: `bench/results/runs.jsonl` (kind=task/perf,
preset_hash provenance). Reference: `docs/reference/speculative-decoding.md`
(ngram-mod mechanics + measured results + MTP disqualification detail).

## Decisions adopted

`models/qwen38.toml` (daily driver) now:

(amended same-day after p6 validation: speculation removed entirely - see
the Speculative decoding section.)

- `reasoning_effort = "medium"` - adopted from the A/B below.
- `spec_type` unset (no speculation) - see the correction below.

Variants kept:

- `qwen38-xhigh` - xhigh effort + MTP, for babysat interactive use only
  (MTP degrades under agentic churn; xhigh is the community's presumed
  quality ceiling worth keeping reachable for hard, babysat work).
- `qwen38-nospec` - no-spec control, kept for future spec A/B baselines.

## Reasoning-effort A/B (both arms spec-off, llama.cpp b10472)

Quality gate (6-task sensor suite): **neutral.** Fast tasks all PASS at
both efforts; ceiling tasks (t3-go-write-split-tests, t6-ts-write-slug-tests)
FAIL at both efforts - t3: 5433s (xhigh) vs 5108s (medium), t6: 3000-4079s
historical (xhigh) vs 1852s (medium). Same outcomes, cheaper failures.

Speed gate (wall_s per run, PASS everywhere):

| task | xhigh | medium |
|---|---|---|
| t1-go-add-truncate | 42.5, 39.0, 44.0, 25.4 | 31.4, 19.3, 20.2, 12.7 |
| t2-go-fix-palindrome | 19.7, **1308.5**, 35.9, 11.5 | 12.5, 19.6, 10.8, 17.2, 25.3, 17.7 |
| t4-ts-add-camelcase | 49.1, 14.2, 38.8, 19.6 | 13.1, 10.8, 11.2, 10.6, 9.8, 10.7 |
| t5-ts-fix-slugify | 11.3, 18.7, 13.6, 12.4 | 10.5, 10.3, 10.6, 11.5, 9.6, 11.3 |

Medium's headline: no thinking binges. xhigh burned 1308s on a 19.7s
task (one pi turn thinking ~80k tokens at ~65 tok/s). For unattended
agentic loops, predictability is throughput.

Perf suite (synthetic): parity spec-off (~73 tok/s both efforts). With
MTP (short-burst only), medium showed gen 110.5-111.6 vs 85.9-86.5 (+29%);
that arm was abandoned when the MTP degradation surfaced.

Community cross-check (OVERBRING article + comments, 2026-08-17): the
medium-vs-xhigh quality debate is unresolved there - one measurement
claims medium ~= Qwen3.6 default, several users run xhigh unattended
overnight with zero issues, several others report endless-thinking doom
loops even on medium. Our suite shows identical failure outcomes and
dramatically fewer wasted tokens; adopt medium, keep `qwen38-xhigh`
reachable for hard babysat work.

## Speculative decoding

Full detail in `docs/reference/speculative-decoding.md`. Summary:

- MTP draft-mtp: +15% gen on short bursts, but degrades to ~0.5 tok/s
  after ~10 min of agentic churn (reproduced on b10362 + b10472; upstream
  #27151/#27296). Disqualified for loops/benches.
- ngram-mod (draftless): +93-100% gen cold pool on synthetic perf, BUT
  CORRECTION (p6 validation, same day): ngram-mod degrades identically
  under agentic task churn (0.6-0.7 tok/s, GPU idle ~106W, fresh-server
  probes fine, restart restores). Both speculators require rolling back
  the hybrid Gated DeltaNet recurrent state on rejected drafts; only
  no-spec (no rollbacks) sustained 55-66 tok/s through a 5433s task.
  Adopted default is therefore NO speculation. The perf numbers stand as
  short-burst evidence only. SPEC_NGRAM_N_{MIN,MAX,MATCH} plumbing kept
  in the entrypoint + schema for future re-tests on newer pins.
- llama.cpp pin b10362 -> b10472 (commit 8c35091): fixed a separate
  abandoned-stream slot-parking bug that froze slots under the loop
  harness (verified fixed upstream on b10472).

## Process notes (bench hardening)

- bench/p5-effort-ngram.sh embeds a /slots watchdog (respawns the server
  on a frozen slot) - the pattern from the degradation hunt; kept as a
  backstop.
- Perf-suite pp/gen metrics are noisy across repetitions (same preset:
  46-96 tok/s across days) - only trust task-suite wall times and
  same-session compares.
- Preset variants via hardlinked GGUF filenames (presets dedup by
  model_id stem). All A/B hardlinks removed post-adoption except
  -nospec/-xhigh.
