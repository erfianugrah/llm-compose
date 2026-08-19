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

Add a unit test in `proxy-go/internal/proxy/server_test.go` asserting `POST /tokenize` with inactive llm mode returns 503 (read-path: it is a POST, so it triggers acquire - assert it reaches forwarding, i.e. 502 with no upstream in tests). Run `cd proxy-go && go test ./... -race -count=1`. Then `make build-proxy-go && docker compose up -d --force-recreate model-proxy-go`.

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

For each ctx value: rewrite a THROWAWAY preset copy (`models/.bench-ctx-<n>.toml`, same content as the source preset but name/context_size/parallel_slots overridden), wait for spawn+healthy via the proxy, then for each occupancy fraction: fill KV to `int(ctx*frac)` tokens with a single user message of filler + "Reply with exactly: READY", then send the measurement request (fresh short prompt in the same conversation? NO - llama.cpp is stateless per request; occupancy is achieved by putting the filler IN the measurement request itself: `messages = [user: filler(ctx*frac tokens) + question]`, measure `predicted_per_second` on a 200-token generation).
Record per point: `{tag: "context-occupancy", preset, ctx, slots, occupancy, prompt_tokens, completion_tokens, tg, vram_mib (nvidia-smi), llama_cpp pin, gpu, ts}` appended to `bench/results/runs.jsonl`.
Lock/unlock around the whole sweep (`--owner bench-context --wait`).
Delete throwaway presets after; restore the source preset untouched.

- [ ] **Step 1: failing test** - result envelope matches the bench store schema (mirror `test_bench.py` conventions; read it first).
- [ ] **Step 2: implement**; run `python3 -m unittest llmc.tests.test_bench_context`.
- [ ] **Step 3: dry run with one point** (`--ctx 196608 --occupancy 0.25`) to validate end-to-end before the full sweep.

### Task 5: Run the full sweep + decide

- [ ] **Step 1:** full sweep (est. 60-90 min; 4 ctx x 5 occupancies x ~30-60s each + reloads). Run as a bg task, not interactively.
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

### Task 8: Docs sweep (after Task 5 decides the final number)

- [ ] `README.md` - make-target table has `ship-proxy-go`; architecture line mentions proxy-go.
- [ ] `AGENTS.md` - proxy-go section: final ctx number, rollback-profile procedure.
- [ ] `docs/specs/2026-08-19-model-proxy-v2.md` - status section: suite results + final ctx.
- [ ] dotfiles skill `.pi/agent/skills/llm-compose/SKILL.md` - final ctx number + suite one-liner.
- [ ] Lexicanum: `rg -l 'llm-compose|model.proxy' ~/lexicanum/src/content/docs`, update the proxy/stack doc with the v2 rewrite (drain-before-swap, capability routing, lock TTL, Anthropic shim) - follow `~/lexicanum/AGENTS.md` conventions (frontmatter unchanged, sentence-case headings, `bun run build` must pass).

### Task 9: Final gates + commit

- [ ] `cd proxy-go && go test ./... -race -count=1`
- [ ] `hurl --variable base=http://127.0.0.1:11434 --test tests/hurl/proxy-go-smoke.hurl` (11/11)
- [ ] `python3 -m unittest discover llmc.tests`
- [ ] Commit llm-compose + dotfiles; commit messages per repo conventions (no AI attribution).
