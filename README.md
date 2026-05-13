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
make setup    # generate .env, create named volumes
make build    # build / pull all images (or `make pull` to use the registry)
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
```

All commands live behind `python3 -m llmc <subcommand>`. Run
`python3 -m llmc --help` for the full surface, or `make help` for the
operational shortcuts that go through `make`.

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

llmc up / down / logs      stack lifecycle
llmc setup                 first-time: generate .env, create volumes

llmc volumes ls            list named volumes + verify bind paths
llmc volumes create        create all volumes from volumes.toml
llmc volumes shell         busybox with all volumes mounted at /vol/<name>

llmc webui configure       import workspace models from webui/models.json
llmc webui reset --yes     nuke webui data (accounts, chats)
llmc comfyui open          print direct ComfyUI URL (auto-swaps mode)

llmc train status / logs / cancel / list / cleanup / deploy <name>
llmc dataset audit / filter / focus / caption / caption-status / ...

llmc eval <subcommand> [args...]   pass-through to eval/run.py
llmc bench <subcommand> [args...]  pass-through to bench scripts
```

`make` shortcuts are in `Makefile` (limited to setup/up/down/status/
build/test/release — everything else is `llmc`).

## Available models

8 presets in `models/`:

| Preset                  | Model                     | Active | VRAM   | Vision | Best for                                |
|-------------------------|---------------------------|--------|--------|--------|-----------------------------------------|
| `gemma4`                | Gemma 4 31B Dense         | 31B    | 20.2GB | yes    | Coding + image input, agentic           |
| `qwen36`                | Qwen3.6 27B Dense         | 27B    | 17.5GB | yes    | Agentic coding, SWE-bench 77.2          |
| `qwen36-moe`            | Qwen3.6 35B-A3B MoE       | 3B     | 22.0GB | yes    | Fast agentic coding                     |
| `qwen36-moe-uncensored` | Qwen3.6 35B Uncensored    | 3B     | 23.0GB | yes    | Creative writing, RP                    |
| `qwen3`                 | Qwen3 32B Dense           | 32B    | 20.0GB | no     | Research, reasoning, tools              |
| `qwen3-coder`           | Qwen3 Coder 30B-A3B MoE   | 3.3B   | 18.6GB | no     | Fast code gen                           |
| `qwen3-vl`              | Qwen3-VL 2B               | 2B     | 4.5GB  | yes    | Lightweight image/video description     |
| `summarizer`            | Gemma 4 26B-A4B MoE       | 4B     | 24.0GB | yes    | TL;DW bot, 128K context summarization   |

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
make build            # build any missing images
make rebuild-proxy    # force-rebuild proxy
make pull             # pull from registry instead
make release          # build all + push to Docker Hub
```

The proxy image (`erfianugrah/llmc-proxy:v2`) is the only image llmc
itself owns — it bundles the `docker` Python SDK + the llmc package and
is ~216 MB. The other three (llama-server, comfyui, lora-train) are
larger purpose-built images that haven't changed in v2.

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

`docker logs model_proxy --tail 50` is the first thing to check on any
weirdness — the proxy logs every mode swap and error.

## License

MIT.
