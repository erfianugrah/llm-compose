# llm-compose

Local LLM + image/video inference + LoRA training stack for a single
NVIDIA GPU. A reverse proxy routes OpenAI-compatible chat to llama.cpp,
image generation to ComfyUI, and training to a kohya sd-scripts service
— auto-swapping GPU workloads so only one runs at a time.

Built for an **RTX 5090** (32 GB VRAM) on **WSL2**. All inside Docker.

```
                Open WebUI / OpenCode / curl
                         │
              ┌──────────▼──────────┐
              │   llmc proxy :11434  │ ── routes by URL prefix
              └──────────┬──────────┘
                         │
            ┌────────────┼────────────┐
            │            │            │
       /v1/*        /comfyui/*    /train/*
       (LLM)       (image gen)   (LoRA training)
            │            │            │
        llama-server    comfyui    lora-train
            │            │            │
            └────────────┼────────────┘
                  GPU exclusive — only one at a time
```

## Quick start

```bash
git clone https://github.com/erfianugrah/llm-compose.git
cd llm-compose

# Add the llmc wrapper to your PATH (one-time)
echo 'export PATH="$HOME/llm-compose/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc

# Bootstrap (~10 min on first run for image pulls)
make deploy   # = setup + build + up
```

Or step-by-step:

```bash
make setup    # generate .env, create named volumes
make pull     # pull images from Docker Hub (~17 GB) or `make build` to compile
make up       # start proxy + Open WebUI
```

Chat UI: <http://localhost:3000>
LLM API: <http://localhost:11434/v1>
ComfyUI: <http://localhost:8188> (when ComfyUI mode is active)

Switch model on the fly:

```bash
llmc switch gemma4         # or any other preset
llmc models                # list available presets
llmc status                # show what's running
llmc lock loop             # pin a preset against GPU-evicting swaps
llmc lock qwen38 --wait    # another preset pinned? queue FIFO until it drains (never hijacks)
# Concurrent loops: loops can share one preset concurrently (lock with a distinct --owner per session, e.g. the pi session id); loops on different presets queue with --wait; when looping the same repo use a separate git worktree per loop; loop sensors must never rebuild/restart the stack that serves them.
llmc unlock                # clear the lock (also drops your queue entry)
```

### Tool boundary

The `Makefile` covers anything that benefits from being a direct
docker/curl/nvidia-smi call — fast targets that don't need Python's
~100 ms interpreter startup:

| make target | does |
|---|---|
| `up` / `down` / `restart` | pre-flight check + docker compose |
| `logs-{proxy,webui,llama,comfyui,train}` | `docker logs -f` |
| `gpu` / `health` / `metrics` | nvidia-smi + curl |
| `build` / `build-X` / `push-X` | docker build / docker push |
| `build-proxy-go` / `test-proxy-go` / `smoke-proxy-go` | Go proxy (proxy-go/, soak on :11435): build / go test -race / live hurl suite |
| `rebuild-X` | docker build `--no-cache` (slow - for base bumps) |
| `ship` / `ship-proxy` | build + push + restart stack (all four / proxy only) |
| `deploy` | full bootstrap: setup + build + up |

The `llmc` CLI covers anything that needs proxy state, schema
validation, or HTTP coordination:

| llmc command | does |
|---|---|
| `switch <preset>` / `mode <m>` | hot-swap LLM / change GPU mode |
| `models` | list TOML presets (live from proxy or local) |
| `status` | full status table (mode, health, presets, active model) |
| `train *` / `dataset *` | training + caption job lifecycle |
| `eval *` / `bench *` | pass-through to eval/run.py / bench scripts |
| `volumes ls / create / refresh / shell` | named volume admin |
| `webui configure / reset` | Open WebUI workspace setup |

`make help` for the make surface, `llmc --help` for the CLI.

## Concepts

### Presets

Each LLM is defined in `models/<name>.toml`. Schema is strict and
validated at load time:

```toml
name = "Qwen3.6 27B Dense — coding, vision, thinking"
description = "Flagship coding model. 27B dense params, SWE-bench 77.2."
vram_gb = 17.5

[model]
repo = "unsloth/Qwen3.6-27B-GGUF"
file = "Qwen3.6-27B-UD-Q4_K_XL.gguf"

[mmproj]                          # optional, for multimodal
url = "https://huggingface.co/.../mmproj-BF16.gguf"

[template]                        # optional, custom chat template
file = "qwen38-fixed.jinja"

[runtime]
context_size = 163840
reasoning = "on"
temperature = 1.0
```

Drop a TOML file in `models/`, the proxy live-reloads on next
`/v1/models` call. No image rebuild, no proxy restart.

### GPU modes

The proxy keeps track of which GPU service is active:

| Mode      | What runs              | Started by                            |
|-----------|------------------------|---------------------------------------|
| `idle`    | Nothing                | Initial state                         |
| `llm`     | llama-server           | `llmc switch <preset>` or hitting /v1/|
| `comfyui` | ComfyUI                | `llmc mode comfyui` or hitting /comfyui/|
| `train`   | lora-train             | `llmc mode train` or hitting POST /train/|

Switching is automatic — sending a POST to a route in a different mode
stops the current GPU service and starts the new one. State is persisted
to `llmc-state` named volume so a proxy restart recovers cleanly.

GET requests do NOT trigger auto-swap (read-only status polls can't
accidentally stop the running service).

If the active GPU service dies out-of-band (crash/OOM/`docker kill`), the
proxy flips to `idle` on the first failed request and the next request
respawns it automatically (no 502-loop, no proxy restart needed).

### Named volumes

All data lives in named Docker volumes backed by host bind paths. See
`volumes.toml` for the registry and `~/docker-volumes/` for the default
paths. The `local` driver with `o=bind` gives:

- compose.yaml refers to volumes by name only — no `${VAR}` interpolation
- existing host data preserved (no copying or symlinks during migration)
- `ls ~/docker-volumes/comfyui/output/` still works for inspection
- portable: each machine creates volumes pointing to its own paths

`llmc volumes ls` shows everything; `llmc volumes shell` opens a
busybox with every volume mounted at `/vol/<name>` for poking around.

## CLI reference

```
llmc status                show stack, GPU mode, active model
llmc health                proxy health check
llmc mode <m>              get / set GPU mode (llm | comfyui | train)
llmc switch <preset>       hot-swap LLM model
llmc models                list TOML presets

llmc lock <preset> --owner <id> [--wait]   pin a preset against evicting swaps
                           (--wait joins the FIFO queue on contention instead of 409)
llmc lock --renew [--owner id]             heartbeat the lock TTL (900s; a leg
                           that makes no requests for >TTL lapses without this)
llmc unlock [--owner id]   release one owner; ownerless = force-clear all

llmc up / down             stack lifecycle (or use `make up`/`down` for speed)
llmc setup                 first-time: generate .env, create volumes

llmc volumes ls            list named volumes + verify bind paths
llmc volumes create        create all volumes from volumes.toml
llmc volumes refresh       drop+recreate volumes (Docker Desktop snapshot fix)
llmc volumes shell         busybox with all volumes mounted at /vol/<name>

llmc webui configure       import workspace models from webui/models.json
llmc webui reset --yes     nuke webui data (accounts, chats)
llmc comfyui open          print direct ComfyUI URL (auto-swaps mode)

llmc train status / logs / cancel / list / cleanup / deploy <name>
llmc dataset audit / filter / focus / caption / caption-status / ...

llmc eval <subcommand> [args...]   pass-through to eval/run.py
llmc bench <subcommand> [args...]  pass-through to bench scripts
```

## Available models

12 presets in `models/`:

| Preset                  | Model                     | Active | VRAM   | Vision | Best for                                |
|-------------------------|---------------------------|--------|--------|--------|-----------------------------------------|
| `gemma4`                | Gemma 4 31B Dense         | 31B    | 20.2GB | yes    | Coding + image input, agentic           |
| `qwen36`                | Qwen3.6 27B Dense         | 27B    | 17.5GB | yes    | Agentic coding, SWE-bench 77.2          |
| `qwen36-moe`            | Qwen3.6 35B-A3B MoE       | 3B     | 22.0GB | yes    | Fast agentic coding                     |
| `qwen36-moe-uncensored` | Qwen3.6 35B Uncensored    | 3B     | 23.0GB | yes    | Creative writing, RP                    |
| `qwen3`                 | Qwen3 32B Dense           | 32B    | 20.0GB | no     | Research, reasoning, tools              |
| `qwen3-coder`           | Qwen3 Coder 30B-A3B MoE   | 3.3B   | 18.6GB | no     | Fast code gen                           |
| `qwen3-vl`              | Qwen3-VL 2B               | 2B     | 4.5GB  | yes    | Lightweight image/video description     |
| `qwen38`                | Qwen3.8 27B Dense         | 27B    | 21.0GB | yes    | Daily driver: coding, vision, thinking  |
| `qwen38-xhigh`          | Qwen3.8 27B + MTP + xhigh | 27B    | 21.0GB | yes    | Interactive quality ceiling             |
| `summarizer`            | Gemma 4 26B-A4B MoE       | 4B     | 24.0GB | yes    | TL;DW bot, 256K context summarization   |
| `loop`                  | Gemma 4 26B-A4B MoE       | 4B     | 24.0GB | no     | Loop-engine worker (agentic coding, 131072 ctx, text-only; same weights as summarizer via symlink) |
| `erfi`                  | Qwen3-4B fine-tune        | 4B     | 3.5GB  | no     | Personal-voice bot (local GGUF)         |

All fit in 32GB VRAM at Q4 quantization. VRAM column includes mmproj.

## Prerequisites

### Hardware

NVIDIA GPU with 24–32 GB VRAM. The VRAM budget defaults to
`LIMIT=32 − RESERVE=6 = 26 GB` for model weights; the proxy rejects
presets that don't fit.

### Software

- WSL2 + Ubuntu (tested on 24.04)
- Docker Engine (not Docker Desktop)
- NVIDIA driver ≥ 550 on the Windows host
- NVIDIA Container Toolkit
- Python 3.11+ on the host (for `llmc` CLI)

```bash
# Verify GPU access from Docker
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
```

## Architecture

### Services declared in compose.yaml

Only two long-lived services. GPU services are spawned by the proxy via
the Docker SDK and labelled `llmc.mode=<mode>` so they live outside
compose.

| Service | Port | Network IP | Notes |
|---------|------|------------|-------|
| `model-proxy` | 11434 | 172.29.0.4 | Routes + spawns GPU services |
| `open-webui`  | 3000  | dynamic    | Chat UI                       |

### GPU services (spawned by the proxy)

| Image                                       | Profile  | Trigger                  |
|---------------------------------------------|----------|--------------------------|
| `erfianugrah/llama-server:cuda12.8-sm120`   | llm      | `llmc switch <preset>`   |
| `erfianugrah/comfyui:cuda12.8-sm120`        | comfyui  | `llmc mode comfyui`      |
| `erfianugrah/lora-train:latest`             | train    | `llmc mode train`        |
| `erfianugrah/llmc-proxy:v2`                 | (proxy)  | `make up`                |

### Volumes (declared in volumes.toml)

| Volume                       | Bind path (default)                                    |
|------------------------------|--------------------------------------------------------|
| `llmc-state`                 | `~/docker-volumes/state` — proxy state + secrets       |
| `llmc-llama-cache`           | `~/docker-volumes/llama-server` — HF cache             |
| `llmc-llama-models`          | `~/docker-volumes/llama-server/models` — GGUFs         |
| `llmc-comfyui-models`        | `~/docker-volumes/comfyui/models`                      |
| `llmc-comfyui-output`        | `~/docker-volumes/comfyui/output`                      |
| `llmc-comfyui-input`         | `~/docker-volumes/comfyui/input`                       |
| `llmc-comfyui-custom-nodes`  | `~/docker-volumes/comfyui/custom_nodes`                |
| `llmc-comfyui-user`          | `~/docker-volumes/comfyui/user`                        |
| `llmc-comfyui-loras`         | `~/docker-volumes/comfyui/models/loras`                |
| `llmc-training-data`         | `~/docker-volumes/training-data`                       |
| `llmc-webui-data`            | `~/docker-volumes/webui`                               |
| `llmc-bench-cache`           | `~/docker-volumes/bench-cache`                         |

Edit `volumes.toml` to point any volume at a different host path before
running `llmc volumes create`.

### Mode-switching latency

Cold storage (first load): 30–60 s per swap (GGUF read from disk).
Page-cache warm: 5–10 s per swap.

`make test-integration` measures this end-to-end with an upper bound of
300 s; regression alarms catch only catastrophic failures.

## Development

### Tests

```bash
make test              # unit + schema (no Docker, ~1s)
make test-docker       # + Docker daemon integration (~30s)
make test-integration  # + end-to-end GPU swap correctness (~90s, stack must be up)
```

125 tests across 6 modules:
- `test_presets` — TOML schema, migration fidelity vs legacy `.env`
- `test_volumes` — named volume registry, Docker integration
- `test_state` — atomic state file r/w
- `test_orchestrator` — Docker SDK lifecycle (mocked + real)
- `test_proxy` — HTTP server, routing, mode swap correctness
- `test_cli` — CLI dispatch + stubbed proxy HTTP
- `test_swap_integration` — end-to-end model swap with real GPU

### Building images

```bash
make build              # all 4 (docker build, cache-aware, ~5s if unchanged)
make build-proxy        # just the proxy — daily flow for llmc/ changes
make rebuild-llama      # --no-cache full rebuild (slow, ~10 min)
make pull               # pull instead of building
make ship-proxy         # build-proxy + push-proxy + restart (daily ship loop)
make ship               # full release: build all + push all + restart stack
```

Both `ship-proxy` and `ship` end with a `docker compose up -d --force-recreate
model-proxy open-webui` so the running stack picks up the new proxy image.
GPU services (llama-server, comfyui, lora-train) aren't restarted — they're
spawned on demand and the next mode swap will use the freshly-pushed image
automatically.

`make build` always invokes `docker build` — there's no "skip if image
already exists" guard. Docker's layer cache handles incrementality: when
nothing changed in the Dockerfile or build context, each image resolves
in under a second. When a single file in the context changes (`llmc/cli.py`,
`llama-server-entrypoint.sh`, etc.), only the affected layers rebuild.

`make rebuild-X` adds `--no-cache` for the rare case where you need a
totally fresh build (base image bump, CUDA arch change, etc.). For
"I changed Python code, rebuild the proxy", just `make build-proxy`.

The proxy image (`erfianugrah/llmc-proxy:v2`, ~216 MB) is the only image
llmc itself owns — it bundles the `docker` Python SDK + the llmc package.
The other three (llama-server, comfyui, lora-train) are larger
purpose-built images that change rarely. `make ship-proxy` is the
common-case shortcut for daily code changes: rebuild proxy, push, restart
the running container.

### Adding a model

1. Drop a TOML file in `models/`
2. `llmc models` to verify it parses
3. `llmc switch <name>` to load it

That's it. No image rebuild, no proxy restart. Schema is strict — typos
in unknown keys are rejected at load time.

### Migrating from v1

The v1 .env-based presets are removed in v2. If you have a checkout that
predates v2, replace your `.env` and `docker-compose.yml` with:

```bash
git pull
rm .env  # the old format conflicts with the v2 expectation
make setup
make up
```

Existing data (`~/docker-volumes/...`) is preserved — named volumes are
backed by the same bind paths, no copying needed.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `proxy not reachable` | `make up`, then `llmc status` |
| `train service not active` | `llmc mode train` first |
| `image not found: erfianugrah/...` | `make pull` (or `make build` to compile locally) |
| Mode swap hangs >2 min on first model | First-time GGUF load (~17–22 GB) — wait |
| `VRAM exceeded` error | Preset's vram_gb > LIMIT−RESERVE budget |
| Open WebUI port 3000 in use | `WEBUI_PORT=3001` in `.env`, restart |
| ComfyUI UI not reachable on :8188 | Only bound when comfyui mode is active |
| Container start fails with `no such file or directory` for a path under `/run/desktop/mnt/.../docker-desktop-bind-mounts/` | Docker Desktop's bind-mount snapshot is stale (volumes were created against a path that's since been reorganised). `llmc down && llmc volumes refresh && llmc up`. Data is preserved — only Docker's volume metadata is rewritten. |

`docker logs model_proxy_go --tail 50` is the first thing to check on any
weirdness — the proxy logs every mode swap and error.

## License

MIT.
