# Plan: Qwen3.8 / GLM-5.3 local-model evaluation + real bench framework

Date: 2026-08-15
Status: P0 DONE (spike results in 7.2); qwen38 preset + p0-qwen38.sh committed; gumshoe fixtures + runner committed in the gumshoe repo; framework (P1+) not started

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

Sensor trust protocol (from the self-correcting-loop skill, mandatory per task): every acceptance probe gets a `canary` (a command that plants the fault it catches) and the suite passes `loop verify-sensors` before any model is scored on it; feature sensors carry `expect: "fail"` and are confirmed red at baseline; each task manifest gets a `loop run --trial` before the full matrix. An unsatisfiable sensor looks identical to a hard task and would silently score every model 0 - this is the cheapest check in the whole plan and the one that catches the most expensive failure.

Suite must include at least one task known to be at the edge of Gemma's ability (e.g. writing NEW test code - the task C repetition-loop failure) or the suite can't discriminate upward.

### 4.4 Non-goals

- No CI workflow (solo repo, Makefile is the pipeline).
- No GGUF download manager - presets already declare repo/file; first run downloads via the normal path.
- No cross-machine benchmarking (1070/servarr) in v1, EXCEPT the single deploy-fit check for the small-model track winner (section 8.4).

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

**DONE 2026-08-15** (bench/results/p0-qwen38-20260815-122704.jsonl): arch support PASS on b10362 (no llama.cpp bump needed), text + vision smokes PASS, full ctx matrix FIT including 196608 x 2 slots at 31153 MiB (2x98304 effective/slot). Preset set to 196608 x 2. Details below kept as the record of method.

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

## 8. Small-model track: gumshoe research-agent (servarr 1070)

Separate from the 5090 track. The gumshoe research-agent runs on the 1070 (8 GB VRAM, sm_61, flash-attn off, shares the card with jellyfin NVENC bursts - real LLM headroom ~6-6.5 GB) with thinking FORCED OFF and a 12-tool JSON action protocol. Incumbent: Qwen3.5-9B. Generic benchmark numbers (Luxand BFCL-agentic, July 2026) were measured with reasoning ON, so they are directional only for this role.

HARDWARE NOTE (2026-08-16, user): a 3080 Ti (12 GB) is planned for servarr eventually. Until it lands, candidates must fit the 1070; the Gemma 4 12B row is scored on quality but its deploy-fit is DEFERRED to the 3080 Ti.

### 8.1 Candidates

| Preset | Model | Size @Q4 | Fits 1070 now? | Why |
|---|---|---|---|---|
| `qwen35-9b` (incumbent baseline) | Qwen3.5-9B | ~5.5 GB | yes | already deployed, thinking-off tolerant |
| `qwen35-4b` | Qwen3.5-4B | 2.7 GB | yes | Luxand 67.0% agentic, best score-per-GB; big speed win on Pascal |
| `lfm25-8b` | LFM2.5-8B-A1B | 5.3 GB | yes | 1.5B active = fastest decode; LFM Open license, not Apache |
| `gemma4-12b` | Gemma 4 12B Unified | 7.1 GB (Q4_K_M) / 6.4 GB (IQ4_XS) | **no** (too tight with KV + jellyfin bursts) | quality ceiling; Tau2 69.0; MTP drafter; deploy-fit deferred to the 3080 Ti |

All four GGUFs (+ 12B MTP drafter + IQ4_XS fallback) already downloaded and verified on the dev box. No Qwen3.8 small exists (27B is the smallest 3.8 so far).

### 8.2 Workload-shaped suite (`llmc bench tasks --suite gumshoe`)

Scripted research prompts scored on the things the role actually does:

1. **Tool-sequence correctness**: did it pick the right tool (osint_ip vs web_search vs fetch vs direct-answer) per prompt; oracle defined per fixture.
2. **JSON protocol validity**: parseable action JSON every turn (the peg failure mode), thinking-off.
3. **Steps-to-answer** within the 6-step bound, forced-final fallback rate.
4. **Final synthesis quality**: rubric-scored by a fixed judge (paid rung, one model for all candidates so the judge is not a variable).

~15-20 prompts across the 4 gumshoe tool families (web_search, fetch, osint_*, direct). CANONICAL HOME of the fixtures + runner is the gumshoe repo (scripts/gumshoe-fixtures-draft.json + scripts/gumshoe-eval-runner.py, committed 2026-08-16 as 26716d4; 18 cases with per-case oracles, repeats-as-rate, {any} alternatives, args checks, multi-turn history). llmc bench CONSUMES that file - do not re-vendor a copy into bench/fixtures (two sources of truth for the same oracle is how baselines rot). The bench-side work is the raw-llama-server runner variant (same cases, but pointed at the candidate on the 5090 without the gateway, which also exposes raw JSON-protocol validity that the gateway's parse-and-reamit hides).

### 8.3 Method

- **Quality A/B on the 5090** (fast iteration, one GPU, sequential): all candidates through the gumshoe suite at matched ctx (32K). Perf measured too but 5090 tok/s is NOT the deploy number - it ranks, it does not predict Pascal.
- **Deploy-fit check on the 1070** for the winner only (section 8.4).

### 8.4 Deploy-fit check (servarr)

For the quality winner AMONG THE 1070-FIT CANDIDATES (9B / 4B / LFM2.5): load on the gumshoe llama container (Pascal flags: flash-attn off), measure VRAM at 32K ctx, tok/s on a representative research turn, and one live end-to-end research-agent call. Pass = fits with ~1 GB headroom AND tok/s not worse than the incumbent by >20%.

`gemma4-12b` deploy-fit is DEFERRED to the 3080 Ti (12 GB): Q4_K_M 7.1 GB + MTP drafter, or IQ4_XS 6.4 GB if the card arrives with other residents. If the 12B wins quality by a wide margin, that is the evidence for accelerating the card swap.

### 8.5 Decision rules

- **gemma4-12b replaces qwen35-9b WHEN THE 3080 Ti LANDS** if tool-sequence accuracy >= incumbent AND JSON-valid rate >= 99%. It cannot deploy to the 1070; a quality win here is a card-swap justification, not an immediate change.
- **qwen35-4b or lfm25-8b replaces the incumbent NOW** if tool-sequence accuracy is within 5 points of the incumbent AND JSON-valid rate >= 99% AND 1070 deploy-fit passes - the trade is quality for speed, and it only wins if it is nearly free.
- Otherwise keep Qwen3.5-9B and record why.

### 8.6 Sequencing

Slot into P2 (suite build, gumshoe fixtures alongside loop-task fixtures) and P3 (run after the 5090 matrix; total added time ~2-3 h on the 5090 + 1 h on servarr). The 1070 check needs servarr up and no gumshoe traffic - coordinate, don't surprise the stack.
