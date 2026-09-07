# 2026-09-06 - NInfer + NVFP4 spike: the parked S3 engine lever

Status: SPIKE PARKED 2026-09-07 - N3/N4/N6 PASS, N5 open (churn soak
not yet run; attempt 1 stopped for GPU reclaim). Engine + artifact kept
in place: `ninfer-spike` container **stopped, not removed**, image
`ninfer:local`, artifact bind-mounted from `.ninfer/models/`.

Resume: take `llama_server` down first (the GPU holds one workload), then
`docker start ninfer-spike`. There is NO ninfer service in `compose.yaml`
- section 2's compose sketch was never implemented; the spike ran as a
bare `docker run`. The exact serving config behind every number in
section 6, recovered from `docker inspect`:

```
ninfer-serve /models/qwen3_8_27b_nvfp4.ninfer
  --model-id qwen3.8-27b-nvfp4 --host 0.0.0.0 --port 8080
  --max-context 262144 --max-concurrency 1 --kv-dtype fp8
  --host-state-slots 0 --host-kv-mib 0 --device-state-slots 0
  --spec mtp --draft-tokens 3 --lm-head-draft --preserve-thinking --vision
```

Port mapping `127.0.0.1:8001 -> 8080`; mount
`.ninfer/models:/models:ro`. pi reaches it as
`external/qwen3.8-27b-nvfp4` via the provider block in
`~/dotfiles/.pi/agent/models.json` (262144 ctx, text+image, zero cost).

Trigger: r/unsloth thread (5090 + Qwen3.8-27B
speed) where the consensus boost is NInfer + NVFP4, with multiple
independent reporters at 120-170 tok/s sustained agentic coding on a 5090
at 150-262K ctx. That is the exact wall our llama.cpp stack sits at.

## 0. Why now (prior data)

- llama.cpp no-spec baseline on the 5090: **73.2 tok/s** gen (b10472,
  196K ctx). Every in-engine lever is exhausted:
  - MTP draft-mtp: disqualified (upstream #27151/#27296, degrades under
    agentic churn; plan `2026-08-17-mtp-speed-track.md`).
  - ngram-mod: +93-100% on synthetic perf, then p6 showed identical
    degradation under churn. Speculation removed entirely
    (`2026-08-19-qwen38-p5-effort-spec.md`).
  - 2026-08-23 megathread note: llama.cpp decode decays 122 -> 69 t/s at
    long ctx while vLLM/NInfer hold >100. Our workload IS long-ctx
    agentic.
- S3 (NVFP4) in the MTP plan was never executed: "stays manual - wants a
  human call on the quant family switch". This spike is that call,
  widened to engine+quant (the GGUF NVFP4 variant does not exist for
  NInfer's format; NInfer uses its own `.ninfer` artifact).

## 1. What NInfer is (verified against github.com/Neroued/ninfer README)

- From-scratch C++/CUDA inference engine, **sm_120a only** (RTX 5090 -
  the build rejects anything else). One GPU, one resident model, 1-8
  startup-fixed request lanes.
- No packaged binary: build from source (cmake + ninja, CUDA 13.1
  toolkit) or `docker build --tag ninfer:local .` from the repo's own
  Dockerfile. We take the Docker path.
- Model is a single `.ninfer` file (weights + tokenizer + chat template
  + MTP head + vision tower). Qwen3.8-27B NVFP4 artifact:
  `neroued/Qwen3.8-27B-nvfp4-NInfer`, 22.08 GB, sha256
  `552c374c685dce302603b95fbe940fb04243c0cd44c083efc644ad3d980d462c`
  (from the repo's own SHA256SUMS, verified 2026-09-06; the hash in the
  engine README was stale).
  Derived from unsloth's Qwen3.8-27B-NVFP4.
- OpenAI-compatible `/v1/chat/completions` server (`ninfer-serve`).
  MTP is built in but OFF by default (`--spec mtp --draft-tokens 3
  --lm-head-draft`); vision off by default (`--vision`).
- Known counter-signal from the thread: one report of tool-call parsing
  breakage (disputed; claimed fixed). N6 tests this directly.

## 2. Integration shape (spike-only, non-invasive)

- **Standalone compose service, profile-gated**, NOT proxy-managed:
  ```yaml
  ninfer:
    build: <external checkout of Neroued/ninfer>   # pinned commit
    profiles: ["ninfer"]
    ports: ["127.0.0.1:8001:8080"]
    volumes: ["./docker-volumes/ninfer-models:/models:ro"]
    deploy: { resources: { reservations: { devices: [gpu] } } }
  ```
- GPU exclusivity is OUR responsibility during the spike: `llama_server`
  must be down (`llmc mode`-swap or `docker stop llama_server`) before
  `docker compose --profile ninfer up`. The proxy knows nothing about
  ninfer; adoption (if it happens) is a separate proxy/preset design.
- Bench harness already parameterises on `proxy=` (perf.py, gumshoe.py
  take a base URL + model_id). `run_perf` is preset-coupled (lock +
  set_mode), so add a minimal `--external URL --model-id ID` path (new
  flags, proposed here, do not exist yet) that
  skips lock/switch and reuses measure_ttft/throughput/prefill +
  VramPoller + store unchanged. Same flag on `bench tasks` / gumshoe
  runner. This keeps every result in `bench/results/runs.jsonl` with
  preset_hash provenance (hash the flags dict for the external arm).

## 3. Spike ladder

- **N0 - environment**: DONE 2026-09-06 - host CUDA UMD 13.3 >= 13.1
  required. GPU currently busy (29.2/32.6 GB, llama_server up): N1+
  needs a GPU-exclusive window.
- **N1 - build + fetch**: DONE 2026-09-06 - `ninfer:local` image built
  (BUILD_RC=0, 10m26s) from upstream commit 487f897; artifact downloaded
  to `.ninfer/models/` and verified against upstream SHA256SUMS
  (`sha256sum -c` OK).
- **N2 - smoke**: serve at 131K ctx, 1 lane, spec off; curl one
  chat completion; confirm OpenAI shape + finish_reason. Then enable
  `--spec mtp --draft-tokens 3 --lm-head-draft` (thread config) and
  `--vision` only if VRAM headroom allows at target ctx.
- **N3 - perf**: `llmc bench perf --external http://127.0.0.1:8001
  --model-id qwen3.8-27b-nvfp4` (proposed new flags), ctx matrix
  {131072, 196608} x {1, 2}
  lanes (restart per cell; capacity is startup-fixed), cold + warm
  passes. Baseline to beat: 73.2 tok/s at 196K, and the p6 churn-decay
  floor.
- **N4 - quality guard**: `llmc bench tasks --tasks
  t1-go-add-truncate,t2-go-fix-palindrome --runs 2` against the external
  endpoint; require the same pass profile as qwen38 baseline (t1/t2 both
  3/3). Plus one gumshoe repeat block (fragile JSON protocol - a bad
  engine interaction shows there first).
- **N5 - churn validation (the test that killed MTP and ngram-mod)**:
  p6-style: cold pool, warm pool, then interleave long agentic-style
  sessions and re-measure. Adoption requires NO decode decay as ctx
  fills (the 122 -> 69 llama.cpp failure mode is the thing we are
  buying our way out of).
- **N6 - tool calls**: drive a tool-calling loop (pi harness or a raw
  tools=[...] completion) through the endpoint; verify arguments parse
  cleanly across >=20 calls. Thread counter-signal lives here.

## 4. Decision rules

- ADOPT as a candidate daily driver if: N3 >= **1.5x** decode vs 73.2
  tok/s at 196K (>= ~110 tok/s), N4 passes, N5 shows no churn decay, N6
  clean. Adoption work (proxy engine support, preset schema, GPU
  lifecycle handoff) is a follow-up plan, not this spike.
  N3-N6 gate state: N3 2.2x PASS, N4 PASS, N6 PASS, N5 interim PASS -
  adoption decision pending the churn soak (see the N5 correction in
  section 6).
- **Adoption is per-preset, never a fleet-wide llama.cpp replacement.**
  Verified against the pinned checkout (487f897): ninfer serves five
  explicitly registered artifact identities, all Qwen - Qwen3.6-27B
  (groupwise-int, nvfp4), Qwen3.8-27B (groupwise-int, nvfp4),
  Qwen3.6-35B-A3B (groupwise-int) - with "no runtime model discovery or
  unregistered checkpoint fallback", per-family converters only
  (`tools/convert/qwen3_{6,6_27b,6_35b_a3b,8_27b}`), and an sm_120a-only
  build. Against our 8 presets: `qwen38` + `qwen38-xhigh` map directly;
  `loop` + `erfi` are local Qwen3.8-derived artifacts that would need the
  converter run on merged weights (untested); `gemma4`, `gemma4-12b`,
  `summarizer` (gemma-4-26b-a4b) and `lfm25-8b` can never move. llama.cpp
  stays as the multi-model engine regardless of the N5 outcome. One
  resident model + no runtime discovery also means the proxy's hot-swap
  cannot live inside ninfer: adoption is a ninfer container per artifact
  under GPU exclusivity.
- **Unmeasured lever (README claim, not ours):** Qwen3.8-27B artifacts
  with DFlash2 companion weights accept `--spec dflash2 --draft-tokens 7`
  (draft counts 1..15) where the spike ran `--spec mtp --draft-tokens 3`.
  Worth an N3 cell if N5 passes.
- PARTIAL (keep llama.cpp daily, ninfer for babysat/interactive like
  qwen38-xhigh): N3 fast but N5 or N6 fails.
- REJECT: N3 < 1.5x, or quality regression on N4, or VRAM cannot hold
  131K+ ctx with headroom. Record numbers, delete profile + image,
  llama.cpp untouched either way.
- KV dtype: start `--kv-dtype fp8` (thread config; maintainer notes
  bf16-vs-quant cache difference is task-dependent). If N4 marginal,
  retry N4 once at bf16 KV before rejecting on quality.

## 6. Spike results (2026-09-06)

Environment: upstream commit 487f897, `ninfer:local` image, artifact
verified against upstream SHA256SUMS. Config: NVFP4 weights, fp8 KV,
`--spec mtp --draft-tokens 3 --lm-head-draft --preserve-thinking`,
`--host-state-slots 0 --host-kv-mib 0` (pinned-host cudaMallocHost OOMs
on WSL2 Docker Desktop - the plan's anticipated risk; no decode cost
observed at 1-2 lanes).

N3 perf matrix (llmc bench perf --external, runs.jsonl):

| ctx x lanes | gen tok/s | pp tok/s | TTFT p50 | VRAM peak |
|---|---|---|---|---|
| 131K x 1, spec off | 74.3 | 7716 | 69.5ms | 25.4 GB |
| 131K x 1, mtp3 | 158.3 | 7664 | 70.4ms | 26.4 GB |
| 196K x 1, mtp3 | 158.3 | 7662 | 70.9ms | 28.6 GB |
| 131K x 2, mtp3 | 156.6 | 7539 | 33.7ms | 26.7 GB |
| 196K x 2, mtp3 | 158.7 | 7655 | 33.5ms | 28.9 GB |
| 262K x 1, mtp3 + vision | 159.8 | 7573 | 70.4ms | 31.7 GB |

vs llama.cpp baseline: 73.2 tok/s cold-start gen at 196K (and the known
122 -> 69 decay under churn). **2.2x decode, ~3x prefill, flat across
ctx.** 262K + vision fits with ~900 MiB to spare.

N4 quality: t1/t2 2/2 PASS each (43-56s, 1 iteration); gumshoe 18 cases
x 3: hit=1.0, json_valid=1.0, steps=1.46, forced=0.0 - the best gumshoe
result in the store.

N6 tool calls: 25/25 OpenAI tools calls parsed clean over multi-round
loops (weather + calc schemas), plus a live pi session (126 tools in
schema) driving read/edit/bash against the endpoint.

N5 churn soak (INTERIM, 2026-09-06): lockstep task via bg pi on the
external rung - 22 requests, 67-70K prompt ctx, decode 138-174 tok/s,
MTP draft acceptance 56-80% (llama.cpp's MTP degraded to 0.5 tok/s
under this load; ninfer's holds), 99% prefix-cache hits, VRAM steady
31.4 GB. Task outcome: 3 drift tests added, 18/18 cargo tests pass in
2m43s.

**N5 correction (2026-09-07): the day-long soak did not happen.** The
session ended at 00:14; the engine's last request was req#35 at
00:01:49 and the container then sat idle 8h20m. So the whole of N5 is
35 requests over ~9 minutes of wall clock - enough to show MTP holding
where llama.cpp's collapsed, not enough to close the decay gate that
disqualified draft-mtp and ngram-mod. Engine health over that window is
clean: 0 restarts, 0 error/panic/CUDA-failure lines in the full
container log.

N5 (real) was to be the lockstep v1 build-out driven by the
self-correcting loop on the `external/qwen3.8-27b-nvfp4` rung:
`~/infra/lockstep/.pi/harness.json`, 40-iteration budget, 10 green
guards + 5 red feature sensors + an opus-5 judge, ladder
`external/qwen3.8-27b-nvfp4 -> claude-sonnet-5` with stallPatience 4.
Harness design and canary evidence: `~/infra/lockstep/docs/loop-harness.md`.

**N5 attempt 1 (2026-09-07) - STOPPED at iteration 2/2 of the trial, GPU
reclaimed for other work.** Not a verdict either way. What it produced:

- Engine health over the container's full life (2026-09-06T23:52:41Z to
  2026-09-07T08:47Z, ~16 min of actual serving across two segments):
  70 completed requests, **0 error / panic / CUDA-failure lines**,
  0 restarts.
- Per-request decode across all 70: min **103.7**, p50 **147.7**,
  p95 183.0, max 221.6 tok/s. No request fell below 100 tok/s. That is
  the anti-decay signal in miniature - llama.cpp's failure mode is
  122 -> 69 as ctx fills - but 70 requests over 16 minutes is nowhere
  near the hours of churn the gate needs.
- Prompt contexts: min 58, p50 48,450, max 88,985 tokens. Prefix-cache
  hit rate p50 96.6%. MTP draft acceptance p50 64.0% (range 45.5-94.4%).

The blocker is on the harness side, not the engine: **trial iteration 1
ran 6.6 min on the ninfer rung, exited 0, and changed zero files**
(`changedFiles: []`, `scopeViolations: []`, `kept: true`). ~35 requests
at 47K ctx with `tools 4`, one tool call per request - so read/edit/write/
bash were all in schema and being called. The repo is `--bind` (rw) in
the jail, so the sandbox did not eat the writes; the agent explored and
never wrote. Which of "explored instead of acting" vs "writes failed
silently" is unresolved, because bwrap puts `~/.pi/agent` on a
`--tmp-overlay` and the iteration's pi transcript is discarded at exit.

Next step for N5, in order: one iteration with `LOOP_SANDBOX=off` so the
transcript survives and the tool sequence is readable; if the model was
merely exploring, slice the manifest to one milestone (ws dispatch + auth
only) per the local-rung working-window rules in the loop skill's
docs/models.md, which already say the local rung stalls on multi-file
work. Then re-run the trial before spending the 40-iteration budget.
Killed runs are not journaled, so `loop history` shows nothing for this
attempt.

- VRAM budget at 196K x 2 lanes with fp8 KV + MTP + vision: unmeasured.
  20 GiB weights + KV; 32 GB card. N2 smoke logs actual usage; drop
  vision or a lane first, ctx last (ctx is the point).
## 5. Risks / open questions

(WSL2 pinned-host risk CONFIRMED and worked around - see section 6.)
- VRAM budget at 262K + vision: 31727 MiB peak measured (section 6) -
  usable but no room for a second lane at that ctx; watch for creep in N5.
- Artifact provenance: third-party repack of unsloth's quant. sha256
  pinned above; `make audit` does not cover it - manual verify in N1.
- Upstream is young (repo created ~2026-09-05 per GitHub metadata);
  pin the commit, do not track master.
