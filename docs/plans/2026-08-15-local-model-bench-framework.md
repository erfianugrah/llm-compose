# Plan: Qwen3.8 / GLM-5.3 local-model evaluation + real bench framework

Date: 2026-08-15
Status: plan only, no code yet

## 1. What actually shipped (verified 2026-08-15)

| Model | Weights | Local-runnable here? |
|---|---|---|
| GLM-5.3 | ~2 weeks after launch (staged, safety review). 743B base, same as GLM-5.2 | **No.** Even a 1-bit GGUF is ~200 GB+. Not a 5090/1070 candidate. Relevant only as an API rung (openrouter) for loop ladders/judges. Watch for a hypothetical GLM-5.3-Air class release - none announced. |
| Qwen3.8-2.4T-A95B | Live (Aug 12), custom qwen3.8-max license, text-only | **No.** 397 GB at 1-bit. Datacenter model. |
| Qwen3.8-27B | Live, Apache 2.0, dense multimodal (text+image+video), 262K native ctx (1M via YaRN), thinking on by default with xhigh/medium/low | **Yes.** ~16-17 GB at Q4_K_M. unsloth/Qwen3.8-27B-GGUF exists. This is the only locally relevant release. |

Self-reported (unreplicated) Qwen3.8-27B number: 61.7% SWE-bench Pro. Treat as marketing until our own bench runs.

**Conclusion up front:** the local question is "Qwen3.8-27B vs Gemma 4 (31B dense `gemma4` preset, 26B-A4B MoE `loop` preset)". GLM-5.3 changes nothing locally; it only matters as a loop-ladder/judge rung via openrouter once weights + providers land.

## 2. Why the current bench setup is insufficient

Current state in `bench/` + `scripts/bench.sh` (~1150 lines):

- `bench-perf.sh` - perf sweep (TTFT p50/p95, gen/prompt tok/s, peak VRAM/RAM) over `quants.txt`, raw `docker run`, CSV output. Works but: shell glue, CSV-only, bypasses the proxy/preset system (duplicates model config that TOML presets already own), no model-vs-model comparison, no trend tracking.
- `bench-quants.sh` - quant quality sweep driver.
- `run-evals.py` - HumanEval (evalplus) / HellaSwag (lm-eval) / BFCL in a container. Known bug: BFCL `limit` silently dropped (maps to no-op `--num-gpus 1`), full category always runs. HellaSwag tokenizer is a CLI flag instead of a per-preset property.
- `bench-report.py` - CSV to table.

Fundamental gaps for the decision we need to make:

1. **No task-quality measurement of the actual workload.** "Better for local work" means: pi-loop engine tasks, bg_task subagents, WebUI chat, summarizer. HumanEval/HellaSwag/BFCL are generic; the decisive metric is loop-task pass rate + iterations + wall time, which we already know how to measure (the loop harness with sensor-gated tasks).
2. **No persistent structured result store.** CSVs with ad-hoc columns; can't answer "did the llama.cpp bump from b10362 to bXXXX regress tok/s on loop".
3. **Config duplication.** bench scripts re-specify model repo/ctx/NGL that presets already encode; they drift.

## 3. Decision criteria (what "better than Gemma 4" means)

In priority order, measured on the 5090 (32 GB):

1. **Loop-task competence**: pass rate and mean iterations on a fixed sensor-gated task suite (section 4.3), run through the real serving path (proxy + preset + loop CLI). Gemma 4 26B-A4B is the incumbent.
2. **Ctx headroom**: max usable context within 32 GB at the quant we'd actually run. The loop workload wants 196608 ctx x 2 slots (current `loop` preset). A dense 27B has a very different KV footprint than the 26B-A4B MoE - if Qwen3.8-27B can't hold ~2x98K at Q4_K_M it loses for loop work regardless of quality. Must be measured, not assumed.
3. **Speed**: TTFT p50, gen tok/s at representative ctx (32K and 128K filled), prompt-processing tok/s.
4. **Multimodal parity**: Qwen3.8-27B is natively multimodal; `gemma4` is our vision preset. If Qwen3.8-27B wins text AND carries vision, it could consolidate two presets into one. Needs mmproj wiring + a vision smoke test.
5. **Generic quality** (tie-breaker only): HumanEval+/BFCL via the existing eval container.

## 4. Bench framework design

Replace the shell scripts with a Python bench package. Decision: **make it part of `llmc`** (`llmc bench ...`) rather than a third CLI - it reuses preset TOML loading, proxy API client, state handling, and the repo's existing test patterns. `llmc` is already the schema+HTTP boundary per AGENTS.md.

### 4.1 Commands

```
llmc bench perf [--presets loop,qwen38] [--ctx 32768,131072] [--runs N]
llmc bench eval [--presets ...] [--humaneval] [--hellaswag N] [--bfcl N]
llmc bench tasks [--presets ...] [--suite bench/tasks/*.json] [--runs N]
llmc bench report [--last K] [--compare runA runB] [--markdown]
```

- `perf` - port of bench-perf.sh methodology (warmup, SSE TTFT median, throughput, nvidia-smi/docker-stats polling) but drives models **through the proxy** (`POST /mode`) using preset TOMLs. Bench-specific overrides (ctx, NGL) expressed as a temporary generated preset, not CLI soup. Retires bench-perf.sh, bench-quants.sh, scripts/bench.sh.
- `eval` - keeps the eval container (Dockerfile.eval, evalplus/lm-eval/BFCL), fixes the BFCL limit bug, moves the HellaSwag tokenizer into preset TOML (`[bench] tokenizer = "..."`).
- `tasks` - the new piece, section 4.3.
- `report` - reads the result store, prints per-model tables and run-over-run deltas.

### 4.2 Result store

JSONL, one record per measurement, in `bench/results/runs.jsonl` (gitignored or committed - decide; committed gives trend history in git, ~KBs per run):

```json
{"ts": "...", "kind": "perf|eval|task", "preset": "qwen38", "model_file": "...gguf",
 "quant": "Q4_K_M", "ctx": 131072, "llama_cpp": "b10362", "gpu": "RTX 5090",
 "preset_hash": "sha1", "metrics": {...}, "duration_s": ...}
```

`preset_hash` over the rendered TOML so a preset tweak invalidates comparisons honestly. This is what enables "did the llama.cpp bump regress us" - the question the CSVs can't answer.

### 4.3 Task suite (the differentiator)

Fixed set of ~6-10 sensor-gated mini-tasks under `bench/tasks/`, each a manifest like the loop slices: scoped multi-hunk edit on a fixture repo (a pinned snapshot of a small Go repo + a small TS repo), with operator-owned acceptance probes as sensors. Run via the loop CLI with `llama-server/<preset>` as the only rung, sensors-only (no LLM judge - free, deterministic, and judges can't be gamed by the judged model family).

Metrics per task: pass/fail, iterations to green, wall time, malformed-output count (the peg-gemma4 failure mode - count parse-error loops explicitly since they don't trip stallPatience), tokens consumed (from proxy /metrics if available).

Guardrails from loop experience baked in: `llmc lock <preset> --owner bench` for the whole run, 196608 ctx / PI_COMPACT_FRACTION=0.95 equivalents, agentTimeoutMs 3600000, tasks sliced to ~3-hunk scope, sensors never rebuild the serving stack, runs sequential (one GPU).

Suite must include at least one task known to be at the edge of Gemma's ability (e.g. writing NEW test code - the task C repetition-loop failure) or the suite can't discriminate upward.

### 4.4 Non-goals

- No CI workflow (solo repo, Makefile is the pipeline).
- No GGUF download manager - presets already declare repo/file; first run downloads via the normal path.
- No cross-machine benchmarking (1070/servarr) in v1.

### 4.5 Watch mode (decided: in scope)

`llmc bench watch` - compares the current llama.cpp pin (LLAMA_CPP_VERSION in llama-server.Dockerfile) + each preset's `preset_hash` against the result store and reports which presets have no current-baseline numbers. Intended use: after a pin bump or preset edit, run `llmc bench watch` and it tells you exactly what to re-baseline (`llmc bench perf --presets <stale>`). Not a daemon - a staleness report, so it stays a Makefile-era tool, not a service.

## 5. Execution phases

### P0 - Feasibility spikes (do first, cheap)

1. **llama.cpp support**: our pin is b10362 (2026-08-12). Qwen3.8-27B GGUF exists on unsloth, but confirm the pinned build loads it; if the arch landed after b10362, decide the bump (rebuild ~10 min, affects all presets - rerun `llmc bench perf` on `loop`+`gemma4` after bump to re-baseline; this is exactly what the result store is for).
2. **KV/ctx fit**: load Qwen3.8-27B Q4_K_M, measure VRAM at 65536 / 131072 / 196608 ctx, 1 and 2 slots. Determines whether it can even compete for the loop role.
3. **Vision path**: identify the mmproj for Qwen3.8-27B (native ViT, check unsloth repo for a mmproj GGUF) - needed if it's to replace `gemma4`.

### P1 - Framework skeleton

- `llmc/bench/` package: result store + `perf` ported from bench-perf.sh, driven via proxy.
- Preset TOML gains optional `[bench]` section (tokenizer, tags).
- Delete bench-perf.sh / bench-quants.sh / scripts/bench.sh after parity verified against an existing perf CSV (same preset, tolerances).

### P2 - Quality + task runners

- `eval` subcommand (fix BFCL limit, per-preset tokenizer).
- `tasks` subcommand + initial 6-task suite + fixture repos.
- `report` with run comparison.

### P3 - The actual evaluation

Candidates: `gemma4` (31B dense), `loop` (26B-A4B), `qwen38` (new preset, Q4_K_M; optionally UD-Q4_K_XL), `qwen36-moe` (already on disk, MoE reference, lost the last A/B - cheap to include since perf numbers exist).

Matrix: perf (2 ctx points) + tasks (2-3 runs per task per model) + humaneval/bfcl tie-breakers. Sequential, locked, overnight-capable. Output: `llmc bench report --markdown` table + a decision: keep/switch loop engine, keep/switch gemma4 preset, recorded in AGENTS.md + memory.

## 6. Decisions (2026-08-15, user)

1. **Commit `runs.jsonl` to git.** Public repo, small, trend history in VCS is the point.
2. **Vendor the task-suite fixtures.** Pinned snapshots of a small Go repo + a small TS repo under `bench/fixtures/`. Reproducible; per-run generation would poison cross-model comparison.
3. **Watch mode is in scope** - see section 4.5.
4. GLM-5.3 API rung: add `openrouter/z-ai/glm-5.3` to the loop ladder + judge rotation once providers list it. Separate small change (dotfiles ladder), not part of this bench work, noted here so it isn't lost.

## 7. Qwen3.8-27B test plan (the concrete run)

### 7.1 New preset

`models/qwen38.toml` (exact GGUF filename + mmproj confirmed in P0):

- repo `unsloth/Qwen3.8-27B-GGUF`, quant Q4_K_M (~16-17 GB); optional second pass at UD-Q4_K_XL if P0 shows VRAM headroom.
- `[mmproj]` from the same unsloth repo if published (native ViT - required for the gemma4-replacement question).
- `[runtime]` reasoning on (thinking is default-on for this model; effort medium as the loop-appropriate setting, xhigh only for a quality-ceiling probe), temperature/top_p per model card, context_size staged per P0 results (target: 196608 x 2 slots if it fits, else the max that does).
- `[bench] tokenizer = "Qwen/Qwen3.8-27B"` for loglikelihood evals.

### 7.2 P0 spikes (sequential, roughly 1-2 h)

1. **Arch support**: `docker run` the pinned llama-server image with the Qwen3.8-27B GGUF. If it fails to load, check the unsloth model card for the minimum llama.cpp build, bump LLAMA_CPP_VERSION, `make rebuild-llama`, and re-baseline `loop` + `gemma4` perf immediately (the result store's reason for existing).
2. **KV/ctx fit**: for ctx in {65536, 131072, 196608} x slots in {1, 2}: load, measure peak VRAM, record fit/OOM in the store. Output is the max config inside 32 GB with ~2 GB spare.
3. **Vision smoke**: with mmproj wired, one image-understanding request through the proxy. Pass/fail only.

### 7.3 Full matrix (P3, overnight, sequential, `llmc lock --owner bench` throughout)

| Preset | Role | perf @32K/128K | tasks (6 x 3 runs) | humaneval + bfcl-subset | vision smoke |
|---|---|---|---|---|---|
| `loop` (Gemma 4 26B-A4B) | incumbent loop engine | yes | yes | yes | n/a |
| `gemma4` (31B dense) | incumbent chat/vision | yes | yes | yes | yes |
| `qwen38` (27B dense) | challenger | yes | yes | yes | yes |
| `qwen36-moe` (35B-A3B) | MoE reference (on disk) | yes | no (lost last A/B) | no | n/a |

Task suite: 6 sensor-gated tasks from bench/fixtures, 3 runs each, sensors-only, malformed-output counted. Estimated: perf ~1 h, tasks ~4-6 h (dominant cost), evals ~90 min.

### 7.4 Decision rules (pre-registered, so the result isn't argued after the fact)

- **qwen38 replaces `loop`** if: task pass rate >= loop's AND mean iterations <= loop's AND it holds 2x98K ctx at the chosen quant AND no malformed-output storms (>10% of iterations).
- **qwen38 additionally replaces `gemma4`** if: the above AND vision smoke passes AND gen tok/s at 32K is not >15% slower.
- **If ctx fit fails at 2x98K but quality wins**: split decision - qwen38 for single-shot/subagent work, keep `loop` for loops. The framework supports this; it is a legitimate outcome, not a failure.
- **Rollback**: presets are additive; reverting = `llmc switch loop` + pi model id unchanged. No state to unwind.

### 7.5 Evidence output

`llmc bench report --markdown` table committed next to this plan as `2026-08-1X-qwen38-eval-results.md`, decision recorded in AGENTS.md + pi memory.
