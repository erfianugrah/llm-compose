# AGENTS.md

Local LLM + image/video inference + LoRA training stack: llama.cpp (GPU)
+ ComfyUI (GPU) + lora-train (GPU) + model-switching proxy + Open WebUI.
All Docker. Only one GPU workload runs at a time — the proxy manages
container lifecycle via the Docker SDK.

## Commands

Tool boundary: **make** for pure docker/curl/nvidia-smi (~30 ms),
**llmc** for proxy state + schema + HTTP (~120 ms). Don't go through
the CLI for things make does fine — Python startup is real for tight
loops.

```bash
# Add the llmc wrapper to PATH (one-time)
export PATH="\/infra/ai/llm-compose/bin:\"   # then: llmc --help

# Stack lifecycle (pure shell)
make setup              # generate .env + create named volumes
make up                 # docker compose up + pre-flight check
make down               # stop GPU services + docker compose down
make restart            # force-recreate proxy + webui
make status             # via llmc (proxy state + table render)
make deploy             # setup + build + up (full bootstrap)
make clean              # down + remove llmc-* volumes (bind data preserved)

# Logs (direct docker, no Python startup)
make logs-{proxy,webui,llama,comfyui,train}

# Quick checks
make gpu                # nvidia-smi
make health             # curl /health
make metrics            # curl /metrics
make audit              # preset GGUFs vs upstream HF (drift/orphans)
make install-timer      # enable the weekly audit timer (systemd user)

# Mode + model switching (CLI)
llmc switch <preset>    # hot-swap LLM (POST /mode {mode:llm, model:X})
llmc lock [preset] [--owner id] [--wait]  # pin a preset: refuse GPU-evicting swaps.
#   Contended (another preset pinned): 409 fail-fast, or --wait joins the FIFO
#   queue and polls until the current owners drain (swap to your preset is lazy,
#   on first request after the grant). A contended lock NEVER hijacks the
#   running model (pre-2026-08-17 it did - that bug killed a loop mid-iteration).
llmc unlock [--owner id]        # release one owner (also drops its queue entry); no owner = force-clear all
llmc lock --renew [--owner id]  # heartbeat the lock TTL (LLMC_LOCK_TTL_S, 900s). Grants under the
#   lock refresh the TTL on their own; a leg that makes NO requests for >TTL (long local
#   thinking, waiting in the FIFO queue) lapses the lock without this. `llmc status` and
#   GET /mode show lock_expires_at.
# Concurrent loops: loops can share one preset concurrently (lock with a distinct --owner per session, e.g. the pi session id); loops on DIFFERENT presets queue with --wait. When looping the same repo use a separate git worktree per loop; loop sensors must never rebuild/restart the stack that serves them.
llmc mode <m>           # llm | comfyui | train
llmc models             # list TOML presets

# Training (proxies /train/*; needs train mode active)
llmc train status / logs / cancel / list / cleanup / deploy <name>
llmc dataset audit / filter / focus / caption / caption-status / ...

# Volumes
llmc volumes ls / create / refresh / shell
#   refresh = drop+recreate every volume (Docker Desktop bind-mount fix)

# WebUI + ComfyUI helpers
llmc webui configure / reset --yes
llmc comfyui open       # print direct URL, auto-swap to comfyui mode

# Images
make build              # all 4 images (cache-aware, ~5s if unchanged)
make build-proxy        # just the proxy (daily flow for llmc/ changes)
make rebuild-llama      # --no-cache full rebuild (slow, ~10 min)
make pull               # pull from registry instead of building
make ship-proxy         # build-proxy + push-proxy + restart (daily ship loop)
make ship               # full release: build all + push all + restart stack
make push-{proxy,llama,comfyui,train}  # per-image push

# Tests
make test               # unit + schema (~1s, no Docker)
make test-docker        # + daemon integration (~30s)
make test-integration   # + GPU end-to-end (~90s, stack up + GPU)
```

## Architecture (v2)

```
OpenCode / Open WebUI / pi / Claude Code
       |
  model-proxy-go :11434  --- Go proxy, scheduler event loop (proxy-go/)
       |
       |-- /v1/*         -> llama-server :8080   (LLM, GPU exclusive)
       |-- /v1/messages  -> Anthropic shim (Claude Code)
       |-- /v1/presets   -> ephemeral preset registry (context sweep)
       |-- /comfyui/*    -> comfyui :8188        (image/video, GPU exclusive)
       |-- /train/*      -> lora-train :8787     (LoRA training, GPU exclusive)

  Only ONE of llama-server, comfyui, lora-train runs at a time.
  Proxy spawns GPU services via Docker Engine API (no compose for them).
  Containers labelled llmc.mode=<mode> so a proxy restart can recover.
  Open WebUI also exposed directly on :3000. ComfyUI UI on :8188 when active.
```

Two services live in compose.yaml:
- `model-proxy-go` (port 11434, static IP 172.29.0.4)
- `open-webui` (port 3000)

`model-proxy` (the legacy Python proxy) is behind the `rollback` profile -
not started by `make up`. Rollback: swap the published ports
(11434 <-> 11436), `docker compose --profile rollback up -d model-proxy`,
restart webui.

GPU services live OUTSIDE compose. The proxy spawns them via Docker SDK
in `llmc/orchestrator.py` and labels them with `llmc.mode` so it can
find them again after a restart.

## proxy-go (v2 rewrite, in soak)

`proxy-go/` is the Go rewrite of the proxy (spec:
`docs/specs/2026-08-19-model-proxy-v2.md`). It adds: drain-before-swap
(in-flight requests finish before a model swap kills the container,
`LLMC_DRAIN_GRACE_S` deadline), capability serve-in-place routing
(`X-LLM-Capability` header or `cap:<name>` model form skips the swap
when the resident model advertises the capability in its TOML
`capabilities` list), lock TTL + durable FIFO queue in active.toml,
and an Anthropic `/v1/messages` shim so Claude Code can point
`ANTHROPIC_BASE_URL` at it.

`model-proxy-go` owns 127.0.0.1:11434 (authoritative since 2026-08-19)
with its own state dir (`~/docker-volumes/state-go`). The Python proxy
(`model-proxy`) is stopped and kept on :11436 as the rollback lane
(swap the published ports back to revert).

```bash
make build-proxy-go   # build the Go image
make test-proxy-go    # go test -race (host toolchain)
make smoke-proxy-go   # live hurl suite (edit base var if testing a non-default port)
```

Cutover done (2026-08-19): pi's provider, the llmc CLI default port, and
Open WebUI (env: OPENAI_API_BASE_URL/COMFYUI_BASE_URL -> model-proxy-go)
all land on the Go proxy. `llmc/proxy.py` remains for rollback only.

Architecture: single-goroutine scheduler event loop
(`internal/proxy/scheduler.go`) owns lock/queue/in-flight state; swaps
are fire-and-forget goroutines; handlers are thin. Stdlib + BurntSushi/toml
only. Docker Engine API via a hand-rolled unix-socket client
(`internal/proxy/docker.go`).

## Source of truth

| Concern               | Lives in                                              |
|-----------------------|-------------------------------------------------------|
| Model presets         | `models/*.toml` (validated by `llmc.presets`)         |
| Volume registry       | `volumes.toml` (created by `llmc volumes create`)     |
| Active mode + model   | `/state/active.toml` (in `llmc-state` named volume)   |
| WEBUI_SECRET_KEY      | `.env` (generated by `llmc setup`, never rewritten)   |

The proxy never rewrites `.env` (which was a major source of v1 jank).
Container env vars are passed directly to `docker run` via the SDK.

**A preset's `repo`/`file` pair is a promise that decays.** It is the
entrypoint's download fallback, and upstream repos delete files: unsloth
wiped every plain K-quant of Qwen3.8-27B on 2026-08-19, and ggml-org did the
same to both gemma-4 repos, orphaning four of our GGUFs while the presets
kept naming them. `make audit` checks every preset file against its repo and
backs up anything with no upstream copy; the weekly
`llmc-model-audit.timer` runs it with backup enabled. Details and status
semantics: `docs/reference/model-audit.md`. When changing a preset's quant,
remember the GGUF stem IS the advertised `model_id`.

## Route-based GPU switching

| Route          | Target                       | Mode      | Notes                                                |
|----------------|------------------------------|-----------|------------------------------------------------------|
| `/v1/*`        | llama-server:8080            | `llm`     | POST auto-swaps; upstream read timeout 3600s         |
| `/metrics`     | llama-server:8080            | `llm`     | read-only; 503 when llm mode not active              |
| `/comfyui/*`   | comfyui:8188 (stripped)      | `comfyui` | POST auto-swaps                                      |
| `/train/*`     | lora-train:8787 (stripped)   | `train`   | POST auto-swaps                                      |
| `/health`      | proxy self                   | any       | 200 even mid-swap (status: switching)                |
| `/v1/models`   | proxy self                   | any       | TOML presets live-reloaded each call                 |
| `GET /mode`    | proxy self                   | any       | current mode + switching flag + active model         |
| `POST /mode`   | proxy self                   | (action)  | switch mode (`{mode, model}`) or manage the model lock (`{lock: preset\|true\|false}`) |

**Read-only methods (GET/HEAD/OPTIONS) do NOT trigger auto-swap.** They
503 cleanly if the target backend isn't active. This prevents status
polls from accidentally stopping the running service.

**Model lock** (`llmc lock [preset] [--owner id]` / `POST /mode {"lock": ...}`):
while locked, the proxy refuses anything that would evict the pinned preset -
model swaps, comfyui/train mode swaps, and unknown-model passthrough.
Use it for unattended multi-hour consumers (a self-correcting loop
worker on the `loop` preset); without it any client POST (Open WebUI
re-POSTs the previously selected model) silently evicts the running
model mid-generation. The lock persists via the state file
(`locked` + `lock_owners`, restored on proxy start; commit a566af5) -
a proxy restart no longer clears it. Consequence: a loop that exits
without unlocking leaves the pinned model RESIDENT, holding VRAM
indefinitely (observed 2026-08-13: Gemma 26B squatting 22.5 GiB hours
after the loop ended). `llmc unlock` to release and free the GPU.
The lock survives deletion of the locked preset's TOML (the running
model stays servable by name).

The lock is SHARED with named owners: each consumer locks with a
distinct `--owner` (e.g. the pi session id) and releases only itself;
the preset stays pinned until the last owner releases. Ownerless unlock
force-clears everything (admin escape hatch). `GET /mode` and
`llmc status` show `lock_owners`. Concurrent loops: share ONE preset
(`loop` runs `parallel_slots = 1`, 262144 ctx - Qwen3.8 Dense KV is
45.1 KiB/token so 2 wide slots no longer fit the 32 GB card), one git
worktree per loop when looping the same repo, and never let a loop's
sensors rebuild/restart the stack that serves them.

The lock has a TTL (`LLMC_LOCK_TTL_S`, default 900s) so a crashed
consumer can't pin the GPU forever: every granted request under the
lock refreshes it, and `llmc lock --renew` (POST /mode `{"renew": true,
"owner": X}`) extends it explicitly - heartbeat this from any leg that
goes >TTL without a request (2026-08-19 postmortem: a 30-min leg found
the lock silently lapsed). Renewing a queued wait works too (keeps the
FIFO entry alive). `GET /mode` / `llmc status` show `lock_expires_at`.

**Liveness recovery (2026-08-21):** a connection-level upstream death
(container crash/OOM/kill out-of-band) flips the proxy to `idle`
(keeping the model name); the next acquire respawns instead of
502-looping. Reports from grants on a since-swapped-away model are
ignored as stale. Verified live: `docker kill llama_server` -> 502 +
mode idle -> next request respawned + served in ~7s.

Contended lock requests QUEUE instead of hijacking (2026-08-17): a
`lock M` while another preset is pinned fails fast with 409, or with
`--wait` / `{"wait": true}` joins an in-memory FIFO queue (202 +
position). The head waiter acquires the lock on its next poll once the
last owner releases; the swap to its preset is lazy (first request
after grant). Unlocking also drops the owner's queue entry. The queue
is in-memory only - a proxy restart drops it and polling waiters
re-enqueue on their next cycle (the lock itself stays restart-safe via
the state file). `GET /mode` and `llmc status` show `lock_queue`.
Bench modules lock with owner `bench` and FAIL FAST on contention (no
queue) - rerun when `llmc status` shows the GPU free.

POST to a route in a different mode auto-swaps:
1. Stop current GPU service (`stop_gpu_services` finds containers by label)
2. Spawn target via Docker SDK (`spawn_llama` / `spawn_comfyui` / `spawn_train`)
3. Wait for healthcheck (default 900s for LLM; 120s for ComfyUI/train)
4. Update `/state/active.toml`
5. Forward the original request

## Adding a model

1. Create `models/<name>.toml` (copy an existing preset)
2. Required: `name`, `vram_gb`, `[model]` with `repo` + `file`
3. Optional: `[mmproj]`, `[template]`, `[runtime]`
4. Schema is strict — unknown keys / wrong types rejected at load time
5. `llmc models` confirms it parses, `llmc switch <name>` loads it
6. No image rebuild, no proxy restart (presets live-reload on `/v1/models`,
   on switch, and on lock - `_ensure_model` reloads per request)

`vram_gb` must be <= LIMIT - RESERVE (default 32 - 6 = 26 GB).

## Benchmarking (quant sweeps)

`bench/bench-perf.sh` measures per-quant TTFT / gen tok/s / prompt tok/s /
peak VRAM+RAM into `bench/results/perf-<ts>.csv`; `bench/bench-quants.sh`
is the full sweep (perf + HumanEval/HellaSwag/BFCL accuracy, ~6-10 h).
`--only Q4_K_M,Q8_0` for a subset; matrix lives in `bench/quants.txt`.

Hard-won operating rules (all observed 2026-08-12/13):

- **Model selection is env-only.** The llama-server image entrypoint
  builds its own model args from `MODEL_FILE`/`MODEL_REPO` env (local
  `/models/$MODEL_FILE` preferred, else HF download into the cache).
  Passing `-m`/`--hf-repo` via argv leaves the entrypoint's empty
  `--hf-repo` in front - fatal ("invalid HF repo format"), and the
  health wait then hangs until timeout. Both bench scripts pass env now.
- **Stop whisper GPU services first.** whisper-live keeps large-v3
  resident (~5.6 GiB VRAM); with it up, Q8_0 (28.6 GB) cannot fit and
  `--fit on` would silently shrink ctx, producing incomparable numbers.
  The scripts `docker stop` (never `docker rm`) the two whisper
  containers - they are compose-managed, rm destroys them. Recover
  with `cd ~/infra/ai/whisper-transcribe && make up`.
- **An active llmc loop re-grabs the GPU mid-bench** (lock + first
  request re-spawns llama_server). Bench only when no loop is running,
  or accept skewed/OOM'd rows.
- **HF downloads inside llama.cpp are fragile** (3 retries then exit).
  If a quant download dies mid-blob, resume it with curl into the hub
  cache (`blobs/<oid>.downloadInProgress`, then mv + snapshot symlink)
  from a root container - the cache dir is root-owned.

Qwen3.6-27B, ctx=32K, RTX 5090 (2026-08-13): UD-Q4_K_XL 73.8 tok/s gen
/ 20.6 GiB peak; Q4_K_M 75.1 / 25.6 GiB; Q8_0 52.1 / 31.6 GiB. Verdict:
UD-Q4_K_XL stays the default (Q4_K_M speed at -5 GiB); Q8_0 is 30%
slower with ~1 GiB headroom - not viable as a daily driver.

## Image builds

| Image                                           | Source                       | Notes |
|-------------------------------------------------|------------------------------|-------|
| `erfianugrah/llmc-proxy:v2`                     | `images/proxy.Dockerfile`    | <30s build. Python 3.12 + docker SDK + llmc package. |
| `erfianugrah/llama-server:cuda12.8-sm120`       | `llama-server.Dockerfile`    | ~10 min. Pinned via `LLAMA_CPP_VERSION=b10362`. sm_120 (Blackwell). |
| `erfianugrah/comfyui:cuda12.8-sm120`            | `comfyui.Dockerfile`         | ~5 min. PyTorch 2.11 + CUDA 12.8. ComfyUI v0.19.5 + Manager. |
| `erfianugrah/lora-train:latest`                 | `lora-train.Dockerfile`      | ~5 min. PyTorch 2.7 + kohya sd-scripts. HTTP API on :8787. |
| Open WebUI                                      | `ghcr.io/open-webui/...`     | Pinned to `v0.10.2`. |

## Named volumes

All bind paths declared in `volumes.toml`. `llmc volumes create` uses
`local` driver + `o=bind` so volume names map to host paths without
`${VAR}` interpolation in compose. Existing data preserved across
migrations.

| Volume                       | Default host path                                      | Purpose                          |
|------------------------------|--------------------------------------------------------|----------------------------------|
| `llmc-state`                 | `~/docker-volumes/state`                              | proxy state + secrets            |
| `llmc-llama-cache`           | `~/docker-volumes/llama-server`                       | HuggingFace cache                |
| `llmc-llama-models`          | `~/docker-volumes/llama-server/models`                | GGUFs + mmproj + templates       |
| `llmc-comfyui-models`        | `~/docker-volumes/comfyui/models`                     | diffusion checkpoints            |
| `llmc-comfyui-output`        | `~/docker-volumes/comfyui/output`                     | generated images/videos          |
| `llmc-comfyui-input`         | `~/docker-volumes/comfyui/input`                      | uploaded inputs                  |
| `llmc-comfyui-custom-nodes`  | `~/docker-volumes/comfyui/custom_nodes`               | ComfyUI extensions               |
| `llmc-comfyui-user`          | `~/docker-volumes/comfyui/user`                       | saved workflows                  |
| `llmc-comfyui-loras`         | `~/docker-volumes/comfyui/models/loras`               | LoRA destination (rw for train)  |
| `llmc-training-data`         | `~/docker-volumes/training-data`                      | datasets/, configs/, output/     |
| `llmc-webui-data`            | `~/docker-volumes/webui`                              | Open WebUI DB                    |
| `llmc-bench-cache`           | `~/docker-volumes/bench-cache`                        | benchmark HF cache               |

## llmc package layout

```
llmc/
├── __init__.py
├── __main__.py             # `python -m llmc` entry
├── cli.py                  # argparse subcommands + HTTP/Docker client
├── presets.py              # TOML loader + schema validation
├── volumes.py              # named volume registry (subprocess to `docker`)
├── state.py                # atomic state file r/w
├── orchestrator.py         # Docker SDK lifecycle (lazy-imported docker dep)
├── proxy.py                # HTTP server + mode swap logic
└── tests/
    ├── test_presets.py        # 17 tests
    ├── test_volumes.py        # 13 tests (Docker integration gated)
    ├── test_state.py          # 15 tests
    ├── test_orchestrator.py   # 17 tests (mock + Docker integration)
    ├── test_proxy.py          # 24 tests (HTTP + routing)
    ├── test_cli.py            # 34 tests (stubbed proxy HTTP)
    └── test_swap_integration.py  # 5 tests (LLMC_TEST_INTEGRATION=1)
```

The proxy is the only place that needs the `docker` Python SDK
(installed in `images/proxy.Dockerfile`). Everything else is stdlib —
the CLI uses subprocess + http.client.

## Evaluation workflow (LoRA testing)

`eval/` holds the ComfyUI-based LoRA evaluation stack. Routes through
the proxy on :11434 so GPU swap is automatic.

```bash
llmc eval quicktest        # 4-scenario sanity (~3 min, page-cache warm)
llmc eval stages           # 3-stage: photo / stylized / prompt-only
llmc eval sweep            # all named stacks side-by-side
llmc eval matrix           # seeds × stacks grid
llmc eval checkpoints      # compare training checkpoints (ep2/4/6/...)
llmc eval weights          # face × aux LoRA weight grid
llmc eval loras            # sweep a list of aux LoRAs
llmc eval seeds            # identity robustness — N seeds, one config
llmc eval i2i              # img2img denoise sweep
```

Output lands in `~/docker-volumes/comfyui/output/eval/<run>/` with
deterministic filenames encoding prompt/stack/seed/epoch.

Named stacks (see `eval/presets.py`): `face_only`, `face_realism`,
`face_super`, `face_ultrareal`, `face_manhwa_v5`, `face_manwha_web`,
`face_manhwa_a18`, `face_illust`, `face_anime`.

User-specific overrides live in `eval/presets_local.py` (gitignored).

## Training workflow

```bash
llmc mode train                                 # swap to train mode
llmc dataset audit <name>                       # WD14 audit
llmc dataset filter <src> <dst>                 # exclude rejects
llmc dataset focus <src> <dst> --n 40           # pick best N images
llmc dataset caption <name> --engine blip2      # async captioning
llmc dataset caption-status                     # poll captioning
# (configure dataset TOML in ~/docker-volumes/training-data/configs/)
curl -X POST http://localhost:11434/train/train \
    -H 'Content-Type: application/json' \
    -d '{"dataset":"<n>", "output_name":"my-lora", "epochs":4, ...}'
llmc train status                               # poll
llmc train deploy <name>                        # copy to ComfyUI loras volume
```

SDXL: `model_type=sdxl`, batch_size=10-16, dim=32, alpha=dim, lr=1e-4.
Flux: `model_type=flux`, batch_size=2-4, dim=16, alpha=16, fp8_base=true.

## MCP servers (OpenCode integration)

| Server                    | Tools                                                          |
|---------------------------|----------------------------------------------------------------|
| `mcp/whisper-server.py`   | `yt_transcribe`, `yt_transcribe_playlist`, `whisper_transcribe` |
| `mcp/comfyui-server.py`   | `comfyui_generate`, `comfyui_status`, `comfyui_history`         |
| `mcp/train-server.py`     | `train_start/status/logs/cancel/list/deploy`, `caption_*`       |

**Harness note (works for both OpenCode and pi):** these are registered in
`~/.config/opencode/opencode.json` under `mcp` and load natively in OpenCode.
The `pi` harness has no built-in MCP, so a bridge extension at
`~/.pi/agent/extensions/mcp-bridge/` reads that SAME `mcp` block, registers
every `type:"local"` enabled stdio server's tools as native pi tools, and
spawns the `.py` server lazily on first call. One shared registry, one set of
`.py` files — no duplication. pi commands: `/mcp-status` (list bridged
servers + tools), `/mcp-refresh` (re-discover + rewrite the tools cache after
editing a server). Tool names are namespaced already (`whisper_*`,
`comfyui_*`, `train_*`); on a genuine cross-server collision the bridge
prefixes with the server name (e.g. research's `wait_job` → `research_wait_job`
since whisper claims `wait_job` first).

Whisper is a separate stack (`whisper-transcribe`, port 7860). It now runs
two GPU services: the batch `whisper` service (turbo/large-v3, idle-unloads
after ~300s) and a separate `whisper-live` service (`LIVE_MODEL=large-v3`,
port 7861) that powers Discord voice/live transcription. During a voice
call large-v3 stays resident alongside llama-server, so the VRAM budget is
tighter than the old single-turbo-model picture.

ComfyUI + train MCP servers go through the proxy at :11434, which means
calling them triggers a GPU mode swap.

## Sister stacks on the same machine

`whisper-transcribe` is a separate compose stack (different repo,
~/whisper-transcribe) that needs to call the LLM for TL;DW
summarization. It joins the `llmc` network as an external network in
its own compose.yaml:

```yaml
networks:
  llmc:
    external: true
    name: llmc
```

The bot service then sets `LLM_API_URL=http://model_proxy_go:11434/v1`
(plus `LLM_VISION_API_URL` / `LLM_TEXT_API_URL`) and reaches the proxy by
hostname over the shared network. Whisper uses the `model_proxy_go` form (the
`container_name`); the compose service name `model-proxy` also resolves via
Docker DNS, so either works.

**Cross-stack model-name contract.** Whisper's `LLM_VISION_MODEL` /
`LLM_SYNTHESIS_MODEL` env values must match a preset's `model_id` — the
GGUF filename minus `.gguf` (`presets.py:59`), which is what the proxy lists
in `/v1/models`. `gemma-4-26B-A4B-it-Q4_K_M` is the stem of
`summarizer.toml`'s `file = "gemma-4-26B-A4B-it-Q4_K_M.gguf"`, NOT the
human `name` title. (The proxy's `preset_by_name` also accepts the `name`
title as an alias, but whisper uses the stem.) Change that `file` and
whisper's vision/synthesis calls silently 404 — nothing on the llm-compose
side flags it. Confirm advertised IDs with `llmc models` after any edit.

**Teardown footgun.** The `llmc` network is compose-owned *here* (no
`external:` on this side; whisper marks it external). `make down` runs
`docker compose down`, which tries to delete the network whisper is still
attached to — the delete errors with "active endpoints" and the proxy
container disappears regardless, so whisper loses `model_proxy_go` resolution.
Stop whisper first, or expect its bot to throw connection errors until the
proxy is back up.

If you ever rename the network here, every external compose stack
that declares it as external needs the same rename in its own
compose.yaml — otherwise `docker compose up` errors with "network not
found". Run `docker network ls` to confirm both repos see the same
name.

## Gotchas

- **POST /mode model swap**: routes through `_ensure_model`, not just
  `_ensure_mode`. The latter short-circuits if already in target mode
  and would silently not swap. Caught by the `test_swap_integration`
  battery — verified by inspecting container env after every swap.
- **Docker SDK command shlex split**: `containers.run(command=str)`
  splits at whitespace, which mangles multi-line bash scripts. Always
  pass as `[script]` (single-element list).
- **GET requests do NOT auto-swap mode**. Status polls return 503 if
  the backend isn't active. Use POST /mode or `llmc mode <m>` first.
- **llama-server `-v` is off by default** (spawned-container logs are
  capped at 50m x 3 via json-file rotation; verbose request logging
  would churn through them on multi-hour agentic runs). Set
  `LLAMA_VERBOSE=1` in the spawn env to re-enable.
- **Mode swap latency**: 5-10s page-cache warm, 30-60s cold storage.
  First-time GGUF load can take up to 12 min for a 22 GB model.
- **ComfyUI WebSocket** (live preview) is NOT proxied. Use direct
  `:8188` when in comfyui mode. Open WebUI uses HTTP polling which
  works through the proxy.
- **Open WebUI image gen** triggers a mode swap: clicking generate
  image hits `COMFYUI_BASE_URL` → proxy swaps to ComfyUI → llama-server
  stops → chat unavailable until swap back.
- **Training takes 10-60+ min**. The LLM is unavailable that whole
  time. Use `llmc train status` to monitor.
- **LoRA compatibility**: LoRAs trained on one base only work on that
  same base. eps-pred LoRAs on v-pred models = broken output.
- **Flux face LoRA stacking**: dim=16 face LoRAs don't survive style
  LoRA stacking. Use weak style LoRAs (a18, webtoon) at low weight
  (~0.4) to preserve identity.
- **Rosacea on pale Flux subjects**: tokens like `realistic skin
  texture, pores, film grain` cause red blotches. Add to negatives:
  `rosacea, acne, red blotches, blotchy skin`.

## No tests for application code

`llm-compose` itself has no application code beyond the llmc package.
There's a test suite (`make test`) for the orchestration logic. There's
no separate frontend / API / DB stack to lint or build. Verification is
unit tests + Docker build + runtime behavior.
