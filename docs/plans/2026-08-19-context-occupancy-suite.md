# Context-occupancy suite + proxy-v2 followups Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or implement this plan task-by-task in-session. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** empirically determine the maximum usable per-slot context for qwen38 (and any future preset) by measuring generation throughput at real KV occupancy - not at empty-prompt allocation - and land the remaining proxy-v2 cutover followups.

**Architecture:** a new `llmc bench context` subcommand (Python, `llmc/bench/context.py`) drives occupancy sweeps through the Go proxy: stuff the KV to a target fraction of `context_size` with a deterministic filler corpus (sized via llama-server's `/tokenize`), then measure generation `predicted_per_second` from the response's `timings` object. Results append to `bench/results/runs.jsonl` under tag `context-occupancy`, following the bench-store conventions.

**Tech Stack:** Python (llmc package), Go proxy (`:11434`), llama-server `/tokenize` + `timings`, llmc bench result store.

## Why (evidence from 2026-08-19 spikes)

All measurements on qwen38 (Qwen3.8-27B Q4_K_M, MTP on, RTX 5090 32GB), fresh containers, ~20-token prompts:

| context_size x slots | per-conversation | VRAM | tg | verdict |
|---|---|---|---|---|
| 196608 x 2 | 98304 | 31948 MiB | 55 t/s | old config |
| 262144 x 1 | 262144 | 31718 MiB | **0.37 t/s** | pathological - prefill 0.88 t/s too; slow from EMPTY KV, so the cliff is structural (ctx size), not occupancy |
| 229376 x 1 | 229376 | ~30.0 GB | fast on tiny prompts | unverified under occupancy |
| 196608 x 1 | 196608 | 29425 MiB | 67 t/s | parked config (current) |

Open questions the suite answers:
1. Where is the throughput cliff between 229376 and 262144 - and is it VRAM pressure (llama.cpp falls off CUDA graphs near full VRAM) or a kernel/graph limit at 256k?
2. Does tg hold at high occupancy (e.g. 180k tokens resident) for the parked config?
3. Does MTP draft acceptance collapse at large ctx (would explain the cliff)?

**Parked state:** `models/qwen38.toml` is at `context_size = 196608`, `parallel_slots = 1` (proven fast). The suite re-tests candidates; do not bump the preset without suite evidence.

## File structure

- Create: `llmc/bench/context.py` - the sweep driver (argparse, corpus builder, measurement, result append).
- Modify: `llmc/bench/__init__.py` or `llmc/cli.py` - register `llmc bench context`.
- Create: `llmc/tests/test_bench_context.py` - unit tests (corpus sizing math, config validation, result schema; HTTP mocked).
- Modify: `compose.yaml` - `model-proxy` behind a `rollback` profile.
- Modify: `README.md`, `AGENTS.md`, `docs/specs/2026-08-19-model-proxy-v2.md`, dotfiles `llm-compose` skill - final ctx number + suite results.
- Lexicanum: update the llm-compose-related doc (find via `rg -l 'llm-compose|model.proxy' ~/lexicanum/src/content/docs`).

## Conventions the implementer must follow

- Bench store: `bench/results/runs.jsonl`, one JSON object per run, must carry preset, preset_hash, llama.cpp pin, GPU name (see `llmc/bench/` existing modules for the exact envelope - read `llmc/bench/perf.py` first and mirror it).
- `llmc bench` subcommands lock the model via the proxy before running (`llmc lock <preset> --owner bench-context --wait`) and unlock after. This suite MUST lock: an eviction mid-sweep ruins the numbers.
- llama-server `/tokenize`: `POST /tokenize {"content": "..."}` -> `{"tokens": [...]}`; count = len(tokens). Available through the proxy (`/v1/*` is not the only llm route - check `classify()` in `proxy-go/internal/proxy/server.go`; if `/tokenize` is not routed, hit `http://llama-server:8080` from inside the `llmc` network via `docker run --rm --network llmc curlimages/curl` or add the route to the Go proxy as a task step).
- Generation speed: request with `"max_tokens": 200`, read `timings.predicted_per_second` from the llama-server response (present on OpenAI-compat responses when requested; if absent, compute `completion_tokens / elapsed`).

### Task 1: Revert guard - confirm parked config is live

**Files:**
- Modify: none (verification only)

- [ ] **Step 1: Confirm `models/qwen38.toml` has `context_size = 196608` and `parallel_slots = 1`**

Run: `rg -n 'context_size|parallel_slots' ~/infra/ai/llm-compose/models/qwen38.toml`
Expected: `context_size = 196608`, `parallel_slots = 1`

- [ ] **Step 2: Trigger respawn and verify**

```bash
curl -s --max-time 880 http://127.0.0.1:11434/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen38","messages":[{"role":"user","content":"hi"}],"max_tokens":5,"stream":false}'
docker exec model_proxy_go wget -q -O - http://llama-server:8080/props | python3 -c 'import json,sys; print(json.load(sys.stdin)["default_generation_settings"]["n_ctx"])'
```
Expected: n_ctx = 196608.

### Task 2: `/tokenize` reachability

**Files:**
- Modify: `proxy-go/internal/proxy/server.go` (only if needed)

- [ ] **Step 1: Probe through the proxy**

```bash
curl -s -X POST http://127.0.0.1:11434/tokenize -H 'Content-Type: application/json' -d '{"content":"hello world"}' | head -c 200
```
Expected: either a token list (routed) or 404 unknown route.

- [ ] **Step 2: If 404, add `/tokenize` + `/detokenize` to the llm routes in `classify()` in `proxy-go/internal/proxy/server.go`:**

```go
if path == "/tokenize" || path == "/detokenize" {
    return "llm", path, true
}
```

Add a unit test in `proxy-go/internal/proxy/server_test.go`: `POST /tokenize` with llm mode inactive must go through the acquire path (POST = swap trigger) and, with the fake orchestrator succeeding, end at a 502 (forward attempted, no upstream) - NOT a 404 (route missing) and NOT a 503 (that is the read-only-GET path). Run `cd proxy-go && go test ./... -race -count=1`. Then `make build-proxy-go && docker compose up -d --force-recreate model-proxy-go`.

### Task 3: Corpus builder + tokenizer sizing

**Files:**
- Create: `llmc/bench/context.py`

- [ ] **Step 1: Write the failing test** (`llmc/tests/test_bench_context.py`):

```python
def test_fill_to_tokens_sizes_exactly():
    # fake tokenize: 1 token per 4 chars
    tok = lambda text: {"tokens": list(range(len(text) // 4))}
    out = fill_to_tokens(target=1000, source="abcd" * 10000, tokenize=tok)
    assert len(tok(out)["tokens"]) == 1000
```

- [ ] **Step 2: Implement `fill_to_tokens(target, source, tokenize)`** in `llmc/bench/context.py`: greedily append chunks of `source` (binary-search the final chunk) until the tokenized length == target. Filler source: a repeated, non-degenerate paragraph (rotate a few paragraphs from the repo's own docs to avoid the model collapsing into repetition loops).

### Task 4: The sweep driver

**Files:**
- Create: `llmc/bench/context.py` (continue)

Behavior:

```text
llmc bench context --preset qwen38 --ctx 196608,229376,245760,262144 --slots 1 \
  --occupancy 0.25,0.5,0.75,0.9,0.98 --gen-tokens 200
```

For each ctx value, create a THROWAWAY preset so the proxy can spawn the
variant. A plain TOML copy does NOT work: model_id derives from the GGUF
filename and the store rejects duplicate model IDs (this is why loop.toml
exists - same pattern). So per candidate:
1. Symlink the GGUF: `ln -s Qwen3.8-27B-Q4_K_M.gguf ~/docker-volumes/llama-server/models/ctx-sweep-<n>.gguf`
2. Write `models/ctx-sweep-<n>.toml` - copy of the source preset with
   `name`, `context_size`, `parallel_slots` overridden and `[model] file =
   "ctx-sweep-<n>.gguf"` (visible name, not dotfile: Go's filepath.Glob
   matches dotfiles but Python's glob does not - keep the loaders consistent)
3. `llmc lock ctx-sweep-<n> --owner bench-context --wait`, then one chat
   request to drive the swap + healthy wait

Then for each occupancy fraction: llama.cpp is stateless per request, so
occupancy is achieved by putting the filler IN the measurement request:
`messages = [user: filler + question]` where filler is exactly
`int(ctx * frac) - gen_tokens - 64` tokens (leave headroom: prompt plus
gen tokens must fit under n_ctx or llama-server truncates/errors).
Measure `predicted_per_second` on a 200-token generation.

Record per point: `{tag: "context-occupancy", preset, ctx, slots, occupancy, prompt_tokens, completion_tokens, tg, vram_mib (nvidia-smi), llama_cpp pin, gpu, ts}` appended to `bench/results/runs.jsonl`.
Cleanup per candidate: unlock owner `bench-context`, delete the TOML +
symlink. After the whole sweep: `llmc switch qwen38` to restore the parked
config (deleting the active throwaway preset without switching back leaves
state.model dangling).

- [ ] **Step 1: failing test** - result envelope matches the bench store schema (mirror `test_bench.py` conventions; read it first).
- [ ] **Step 2: implement**; run `python3 -m unittest llmc.tests.test_bench_context`.
- [ ] **Step 3: dry run with one point** (`--ctx 196608 --occupancy 0.25`) to validate end-to-end before the full sweep.

### Task 5: Run the full sweep + decide

- [ ] **Step 1:** full sweep as a bg task, not interactively. Realistic estimate 2-4 hours: 4 ctx x 5 occupancies, where each high-occupancy point pays its filler prefill every time (224k tokens at ~400-2000 t/s prefill = 2-9 min per deep point) plus 4 model reloads (~3 min each).
- [ ] **Step 2:** decide final `context_size`: the largest ctx whose tg stays >= 20 t/s at 0.90 occupancy AND >= 40 t/s at 0.50 occupancy (floors from the 196608 baseline of 67 t/s; adjust if the baseline itself degrades at occupancy - that is a finding too).
- [ ] **Step 3:** if 262144 fails, capture `docker logs llama_server` for the failing run (look for CUDA graph capture failures / MTP acceptance rates) and record the mechanism in the TOML comment.
- [ ] **Step 4:** set `models/qwen38.toml` to the winner with the spike table in the comment; commit.

### Task 6: compose rollback profile

**Files:**
- Modify: `compose.yaml` (model-proxy service)

- [ ] **Step 1:** add `profiles: ["rollback"]` to `model-proxy` so `make up` / `docker compose up -d` never starts the Python proxy. Rollback procedure becomes: `docker compose --profile rollback up -d model-proxy` + swap the two published ports (11434 <-> 11436) + restart webui.
- [ ] **Step 2:** verify `docker compose config --quiet` and `docker compose config --services` lists model-proxy-go + open-webui by default, model-proxy only with `--profile rollback`.

### Task 7: Push the Go proxy image

- [ ] **Step 1:** `cd ~/infra/ai/llm-compose && make push-proxy-go` (pushes `erfianugrah/llmc-proxy-go:v1` to Docker Hub; requires `docker login` already done - verify with `docker info | rg Username`).

### Task 8: Proxy liveness recovery (known gap from cutover day)

The Go scheduler trusts `state.Model`; if `llama_server` dies out-of-band
(OOM-kill, manual `docker rm`), acquires keep granting and forwarding
502-loops until a proxy restart reconciles. The Python proxy checked
`current_mode()` live per request and would respawn. Fix:

**Files:**
- Modify: `proxy-go/internal/proxy/scheduler.go` (new event), `proxy-go/internal/proxy/server.go` (notify on connection failure)
- Test: `proxy-go/internal/proxy/scheduler_test.go`

- [ ] **Step 1: failing test** - grant resident acquire, `NoteUpstreamDead("llm")`, next acquire for the same model must trigger a spawn (fake orchestrator records SpawnLlama).
- [ ] **Step 2: implement** - add `NoteUpstreamDead(mode string)` to Scheduler (new loop event): sets `st.Mode = "idle"` (keep `st.Model`), persists, logs. In `server.go` `forwardTo`, call it when `upstreamClient.Do` fails with a connection error (not on upstream 5xx - the container answered then).
- [ ] **Step 3:** `go test ./... -race -count=1`, rebuild + recreate `model-proxy-go`, commit.

### Task 8b: Lock renewal + expiry visibility (from the dispatch-run postmortem)

Verified by inspection (2026-08-19): `LLMC_LOCK_TTL_S` defaults to 900s in
`cmd/proxy/main.go` and `llmc lock --help` shows no renew/heartbeat verb.
The postmortem's scenario follows: a leg running longer than the TTL has the
lock expire mid-leg and the queue drains unprotected. The owner-refresh on
granted requests only helps tenants that request continuously.

**Files:**
- Modify: `proxy-go/internal/proxy/scheduler.go` (renewLock event),
  `proxy-go/internal/proxy/server.go` (route), `llmc/cli.py` (verb),
  `llmc/state.py` (expose expires_at in status payload mapping if filtered)
- Test: `proxy-go/internal/proxy/scheduler_test.go`, `llmc/tests/test_proxy.py`

- [ ] **Step 1: failing tests** - scheduler: lock with TTL 1s, renew at 0.5s, still locked at 1.2s; renew by a non-owner -> 404/409 (decide: 409). CLI test: `lock --renew` maps to POST /mode {"renew": true, "owner": ...}.
- [ ] **Step 2: implement renew** - scheduler event `evRenew{owner}`: owner in LockOwners (or holding a queue entry) -> LockExpiresAt = now+TTL, persist, 200; else 409. Route: `POST /mode {"renew": true, "owner": X}`. CLI: `llmc lock --renew [--owner id]`.
- [ ] **Step 3: expiry visibility** - `GET /mode` payload gains `lock_expires_at` (unix) + `lock_ttl_seconds`; `llmc status` prints `Locked: <model> (expires in Ns)`.
- [ ] **Step 4: hurl smoke entry** - renew flow in `tests/hurl/proxy-go-smoke.hurl` (lock, renew, verify expires_at moved).
- [ ] **Step 5:** gates + commit.

### Task 8c: Driver hardening lessons (recorded, not llm-compose code)

From the postmortem - apply to the llmc loop harness when next touched
(NOT this repo's scope today):
- Treat `exit 0 + empty output` from `pi -p` as failure + one retry (observed:
  silent NOOP dispatch, empty log, zero changes - mechanism undiagnosed;
  candidate upstream report: pi -p should non-zero on an empty assistant
  turn).
- Lock renewal belongs IN the driver loop (heartbeat per iteration), not a
  sidecar watchdog - Task 8b provides the verb.
- Observation to validate in the context suite (Task 5), reported but not
  measured this session: dispatch throughput per doc was several-fold faster
  on a fresh llama-server container than a hot one (same model, same prompt
  shape). No KV/slot metrics captured, so mechanism unknown - add one sweep
  row measuring tg at container age T vs T+60min under identical occupancy
  to confirm/deny before adding any periodic recycle knob.
- Cosmetic: `llmc up` prints a compose "no configuration file provided"
  error when run outside the repo dir; pipe it through or cd first.
- Host-level flag (outside this repo): dmesg showed a JBD2 I/O error on sde
  at boot + repeated loop0 read errors against the Docker Desktop VHDX
  (pre-dating the resume). One occurrence may be a hard shutdown; if loop0
  errors recur, decide chkdsk / Docker Desktop reset before the model cache
  is at risk. Cannot be diagnosed from inside WSL.

### Task 9: Client verification against the Go proxy

- [ ] **Step 1: bench harness** - `cd ~/infra/ai/llm-compose && export PATH="$PWD/bin:$PATH" && llmc bench perf --presets qwen38 --runs 1`. Verifies the lock/swap path the loop engine depends on. Expected: run completes, result row in `bench/results/runs.jsonl`.
- [ ] **Step 2: real Claude Code session** - `ANTHROPIC_BASE_URL=http://127.0.0.1:11434 claude -p "read the file /etc/hostname and tell me its content"` (forces a tool_use round-trip + streaming). Expected: completes with the hostname; check `docker logs model_proxy_go` for clean translation (no dropped-block warnings beyond acceptable ones).

### Task 10: Docs sweep (after Task 5 decides the final number)

- [ ] `README.md` - make-target table has `ship-proxy-go`; architecture line mentions proxy-go.
- [ ] `AGENTS.md` - proxy-go section: final ctx number, rollback-profile procedure; **fix the architecture diagram** (still says `llmc-proxy :11434 --- Python proxy`).
- [ ] `docs/specs/2026-08-19-model-proxy-v2.md` - status section: suite results + final ctx.
- [ ] dotfiles skill `.pi/agent/skills/llm-compose/SKILL.md` - final ctx number + suite one-liner.
- [ ] Lexicanum: `rg -l 'llm-compose|model.proxy' ~/lexicanum/src/content/docs`, update the proxy/stack doc with the v2 rewrite (drain-before-swap, capability routing, lock TTL, Anthropic shim) - follow `~/lexicanum/AGENTS.md` conventions (frontmatter unchanged, sentence-case headings, `bun run build` must pass).

**Deferred (not a task now):** retire `llmc/proxy.py` + the `model-proxy` rollback service after 2-4 weeks of stable Go-proxy operation. Note it in the spec status when Task 10 lands.

### Task 11: Final gates + commit

- [ ] `cd proxy-go && go test ./... -race -count=1`
- [ ] `hurl --variable base=http://127.0.0.1:11434 --test tests/hurl/proxy-go-smoke.hurl` (11/11)
- [ ] `python3 -m unittest discover llmc.tests`
- [ ] Commit llm-compose + dotfiles; commit messages per repo conventions (no AI attribution).
