# llm-compose

Local LLM inference stack (llama.cpp + Open WebUI) for WSL2 with NVIDIA GPU.

Multi-model local LLM inference on an RTX 5090 with flash attention,
quantized KV cache, and 64k context — all inside Docker. Swap models without
rebuilding the image.

## Quick start

```bash
make setup          # one-time: .env, volumes, model assets, image build
make download-all   # pre-download all model GGUFs (~57 GB total, recommended)
make up             # start the stack
```

Open the UI at [http://localhost:3000](http://localhost:3000).

Or do everything in one shot (build, push to registry, start):

```bash
make deploy
```

> **First start without pre-download:** if you skip `make download-all`, the
> model GGUF (~19 GB) downloads inside the container on first boot. The health
> check allows up to **12.5 minutes** for this (`start_period: 600s` + 5
> retries at 30s). Subsequent starts load from cache in ~60-90s.

Run `make help` for all available targets.

## Available models

| Preset | Model | Type | Size | Vision | Thinking | Best for |
|---|---|---|---|---|---|---|
| `gemma4` | Gemma 4 31B Dense | Dense | ~19 GB | Yes | Yes | Multimodal, agentic coding |
| `qwen3-coder` | Qwen3 Coder 30B A3B | MoE | ~19 GB | No | No | Fast code generation |
| `qwen3` | Qwen3 32B | Dense | ~20 GB | No | Yes | Research, science, tool use |

All models fit in 32 GB VRAM at Q4_K_M with 64k context.

### Switching models

**From OpenCode** (recommended): just select a different model in `/models`.
The proxy detects the mismatch and auto-swaps (~60-90s on first request).

**From terminal:**

```bash
make run MODEL=qwen3-coder   # switch + restart in one shot
```

List available presets with `make models`. Pre-download all with `make download-all`.

## Prerequisites

### NVIDIA Container Toolkit (WSL2)

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```

Verify GPU access:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
```

## Setup

The recommended way is `make setup` (see Quick start above). This will:

1. Generate `.env` with a random `WEBUI_SECRET_KEY`
2. Create volume directories (handling root-owned Docker volumes)
3. Load the default model preset (`gemma4`) and download its assets (mmproj, template)
4. Build the model-proxy and llama-server Docker images (skips if already present)

The text model GGUF (~19 GB) auto-downloads from HuggingFace on first container start.
To pre-download all models: `make download-all` (strongly recommended).

**Note:** The model-proxy container requires access to the Docker socket
(`/var/run/docker.sock`) to recreate llama-server during model swaps.

## Architecture

```
OpenCode / Open WebUI
        │
        ▼
  model-proxy :11434    ← auto-swaps models based on request
        │
        ▼
  llama-server :8080    ← GPU inference (internal, not exposed)
```

| Service | Address | Purpose |
|---|---|---|
| model-proxy | `127.0.0.1:11434` | Reverse proxy, auto model switching |
| llama-server | internal only | LLM inference (GPU) |
| Open WebUI | `127.0.0.1:3000` | Browser-based chat interface |

All host ports are bound to `127.0.0.1` only — not exposed to the network.

### Network

All services run on a dedicated Docker bridge network (`172.28.0.0/24`) with
static IPs:

| Service | IP | Port |
|---|---|---|
| llama-server | `172.28.0.2` | 8080 (internal only) |
| Open WebUI | `172.28.0.3` | 8080 (mapped to host :3000) |
| model-proxy | `172.28.0.4` | 11434 (mapped to host :11434) |
| Gateway | `172.28.0.1` | — |

Open WebUI connects to the proxy via its static IP (`http://172.28.0.4:11434/v1`)
so container name resolution isn't required.

### Volumes and data directories

| Host path | Container mount | Purpose |
|---|---|---|
| `~/docker-volumes/llama-server/` | `/root/.cache` | HuggingFace model cache (GGUFs) |
| `~/docker-volumes/llama-server/models/` | `/models` | mmproj files, jinja templates |
| `~/docker-volumes/webui/` | `/app/backend/data` | Open WebUI user data, chat history |

`make dirs` creates these directories and fixes ownership if Docker created
them as root. All `make` targets that need volumes call `dirs` automatically.

Downloaded model GGUFs live in the HuggingFace cache under
`~/docker-volumes/llama-server/huggingface/`. `make clean` removes Docker
volumes but **preserves** downloaded models.

### How model switching works

The proxy intercepts the `model` field in each request. If it differs from
what's currently loaded, the proxy automatically:

1. Updates `.env` with the new model preset
2. Recreates the llama-server container via Docker socket
3. Waits for the model to load (~60-90s, unavoidable VRAM physics)
4. Forwards the original request

From OpenCode: just select a different model in `/models`. The first request
takes ~90s while the model loads; subsequent requests are instant.

To pre-download all models so the swap only costs the load time:

```bash
make download-all
```

### Health checks

Each service has a Docker health check that gates dependent services:

| Service | Endpoint | start_period | Retries | Interval | Total window |
|---|---|---|---|---|---|
| llama-server | `localhost:8080/health` | 600s | 5 | 30s | **750s** (~12.5 min) |
| model-proxy | `localhost:11434/health` | 10s | 3 | 30s | 100s |
| Open WebUI | `localhost:8080/health` | 30s | 3 | 30s | 120s |

**Startup dependency chain:** llama-server (healthy) → model-proxy (healthy) → Open WebUI.

The llama-server `start_period` of 600s accommodates first-time model downloads
(~19 GB GGUF from HuggingFace). Once models are cached, startup takes ~60-90s
(VRAM load only).

During model switching, the proxy returns `{"status": "switching"}` (HTTP 200)
on its health endpoint so Docker doesn't kill it mid-swap. The proxy's
`HEALTH_TIMEOUT` (180s) controls how long it waits for llama-server after
recreating the container.

### Security

All services are hardened with:

- **`no-new-privileges`** — prevents privilege escalation via setuid/setgid
- **`cap_drop: ALL`** — drops all Linux capabilities (llama-server, Open WebUI)
- **Localhost-only ports** — `127.0.0.1` binding, not exposed to LAN
- **Read-only mounts** — model presets and Docker socket mounted `:ro`
- **Minimal runtime images** — only essential runtime deps in final stage
- **tmpfs for Open WebUI** — `/tmp` mounted noexec, nosuid, 256 MB limit
- **JSON logging with rotation** — 50 MB × 3 files (llama-server, Open WebUI), 10 MB × 3 (proxy)

The model-proxy requires the Docker socket (`/var/run/docker.sock:ro`) to
recreate llama-server during model swaps. This is the only container with
Docker socket access. It does **not** have `cap_drop: ALL` because it needs
network capabilities for proxying.

### Monitoring

- **Health:** `make health` or `curl http://localhost:11434/health`
- **Metrics:** `make metrics` or `curl http://localhost:11434/metrics` (Prometheus format)
- **GPU:** `make gpu` or `nvidia-smi --query-gpu=utilization.gpu,power.draw,memory.used --format=csv`
- **Status:** `make status` — container status + health check

## Makefile reference

### Getting started

| Target | Description |
|---|---|
| `make setup` | First-time setup: .env, volumes, model assets, image build |
| `make deploy` | Full deploy: setup + push images to registry + start stack |
| `make up` | Start the stack in background |
| `make down` | Stop the stack |

### Model switching

| Target | Description |
|---|---|
| `make models` | List available model presets |
| `make switch MODEL=name` | Switch preset (updates .env + downloads assets) |
| `make run MODEL=name` | Switch + restart in one shot |
| `make download-all` | Pre-download all model GGUFs (~57 GB total) |
| `make assets` | Download assets (mmproj, template) for current model |

### Image management

| Target | Description |
|---|---|
| `make build` | Build all images (skips llama-server if present) |
| `make pull` | Pull all custom images from registry |
| `make push` | Push all custom images to registry |
| `make rebuild` | Force rebuild llama-server from source + restart |
| `make release` | Rebuild + push + restart |

### Operations

| Target | Description |
|---|---|
| `make restart` | Restart all services |
| `make logs` | Follow logs for all services |
| `make logs-llama` | Follow logs for llama-server only |
| `make status` | Show container status, active model, health |
| `make clean` | Stop stack + remove Docker volumes (keeps models) |

### Monitoring

| Target | Description |
|---|---|
| `make gpu` | GPU utilization, power draw, VRAM usage |
| `make metrics` | Fetch Prometheus metrics from llama-server |
| `make health` | Check llama-server health endpoint |

## Environment variables

### Model preset variables (in `models/*.env`)

| Variable | Description | Example |
|---|---|---|
| `MODEL_REPO` | HuggingFace repo for GGUF | `ggml-org/gemma-4-31B-it-GGUF` |
| `MODEL_FILE` | GGUF filename | `gemma-4-31B-it-Q4_K_M.gguf` |
| `MODEL_NAME` | Display name (shown in proxy /v1/models) | `Gemma 4 31B Dense (local)` |
| `MMPROJ_FILE` | Multimodal projector filename (empty = text-only) | `mmproj-gemma-4-31B-it-f16.gguf` |
| `MMPROJ_URL` | Download URL for mmproj | HuggingFace URL |
| `TEMPLATE_FILE` | Jinja chat template filename (empty = GGUF default) | `google-gemma-4-interleaved.jinja` |
| `TEMPLATE_URL` | Download URL for template | GitHub raw URL |
| `REASONING` | Enable thinking mode (`on` or empty) | `on` |
| `CONTEXT_SIZE` | Context window size in tokens | `65536` |
| `TEMPERATURE` | Sampling temperature | `1.0` |
| `TOP_P` | Nucleus sampling threshold | `0.95` |
| `TOP_K` | Top-k sampling | `64` |
| `MIN_P` | Min-p sampling (0 = disabled) | `0.05` |

### Stack variables (in `.env`)

All model preset variables above, plus:

| Variable | Description |
|---|---|
| `WEBUI_SECRET_KEY` | Open WebUI session secret (auto-generated by `make setup`) |

### Proxy environment (set in `docker-compose.yml`)

| Variable | Default | Description |
|---|---|---|
| `LLAMA_HOST` | `llama-server` | Hostname of llama-server container |
| `LLAMA_PORT` | `8080` | Port of llama-server |
| `PROXY_PORT` | `11434` | Port the proxy listens on |
| `PRESETS_DIR` | `/presets` | Container path to model preset files |
| `PROJECT_DIR` | `/project` | Container path to project dir (.env, compose file) |
| `HEALTH_TIMEOUT` | `180` | Seconds to wait for llama-server after model swap |
| `HOST_HOME` | `${HOME}` | Host HOME path (for `~` resolution in compose volumes) |

## Why llama.cpp instead of Ollama?

We switched from Ollama to llama-server (llama.cpp) for three reasons:

1. **Ollama 0.20.x has a bug** where flash attention causes Gemma 4 to run on CPU
   despite reporting 100% GPU ([ollama#15237](https://github.com/ollama/ollama/issues/15237)).
2. **llama.cpp is faster** — 167 tok/s generation vs ~35 tok/s on Ollama (same model,
   same hardware) because FA + quantized KV cache work correctly.
3. **No abstraction overhead** — llama.cpp is the engine Ollama wraps. Direct access
   means fewer bugs and more control.

### Performance comparison (RTX 5090, Gemma 4 26B MoE Q4_K_M)

| | Ollama 0.20.2 (FA off) | llama-server (FA on) |
|---|---|---|
| Generation | ~35 tok/s | **167 tok/s** |
| Prompt eval | ~40 tok/s | **190 tok/s** |
| GPU utilization | 28% | **85%** |
| Power draw | 126W | **228W** |
| KV cache | f16 (FA required for q8_0) | **q8_0** |

## Model details

### Gemma 4 31B Dense (default)

The flagship Gemma 4 model. Fits on 32 GB VRAM at Q4_K_M thanks to hybrid
sliding-window attention (50 local layers + 10 global layers):

- **18.7 GB** on disk (Q4_K_M quantization)
- **All 31B parameters active** per token — no routing, no sparsity
- **256K native context** (we use 64k — only the 10 global layers scale with context)
- Top benchmarks in its class (LiveCodeBench: 80%, AIME 2026: 89.2%)
- Fits **100% on GPU** with flash attention + q8_0 KV cache at 64k context (~24.6 GB total)
- Thinking mode via the interleaved template (PR #21418)
- **Vision support** via multimodal projector (mmproj)

### Qwen3 Coder 30B A3B

MoE coding specialist — only 3.3B active params per token for fast inference:

- **18.6 GB** on disk (Q4_K_M quantization)
- **Non-thinking mode** — fast, direct code outputs
- **262K native context** (we use 64k)
- Optimized for code generation, refactoring, and agentic coding
- Tool calling support for OpenCode

### Qwen3 32B

Dense general-purpose model with strong reasoning and tool calling:

- **~20 GB** on disk (Q4_K_M quantization)
- **Thinking mode** — adaptive chain-of-thought reasoning
- **131K native context** (we use 64k)
- Excellent at research, science, daily questions, web search, tool use
- Strong multilingual support

## Custom Docker image

The `llama-server.Dockerfile` builds llama.cpp from source with:

- **CUDA 12.8** with native **sm_120** (Blackwell) kernels — no PTX JIT overhead
- Only the `llama-server` binary + shared libs in the runtime image (~5.7 GB)
- Multi-stage build: `nvidia/cuda:12.8.1-devel` → `nvidia/cuda:12.8.1-runtime`

To rebuild after a llama.cpp update:

```bash
# Update LLAMA_CPP_VERSION in llama-server.Dockerfile, then:
make rebuild        # local only
make release        # rebuild + push to registry
```

### Docker images

| Image | Registry | Description |
|---|---|---|
| `erfianugrah/llama-server:cuda12.8-sm120` | Docker Hub | llama.cpp with CUDA 12.8 / sm_120 |
| `erfianugrah/model-proxy:latest` | Docker Hub | Python reverse proxy with Docker CLI |
| `ghcr.io/open-webui/open-webui:v0.8.12` | GHCR | Third-party chat UI (not pushed) |

`make push` pushes both custom images. `make pull` pulls both.

## Vision (image) support

Gemma 4 includes a multimodal projector (mmproj) for image understanding.
`make switch MODEL=gemma4` auto-downloads the mmproj (~1.1 GB). Text-only
models (qwen3-coder, qwen3) skip the mmproj — the `--mmproj` flag is
conditionally included only when `MMPROJ_FILE` is set in the model preset.

The `libmtmd.so` library that powers multimodal inference is already compiled
into the Docker image. No rebuild needed.

## Using with OpenCode

Add the provider to `~/.config/opencode/opencode.json`. Register all models
you want to use — only the one loaded by llama-server will respond:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "llama-server": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "llama.cpp (local)",
      "options": {
        "baseURL": "http://localhost:11434/v1"
      },
      "models": {
        "gemma-4-31B-it-Q4_K_M": {
          "name": "Gemma 4 31B Dense (local)",
          "attachment": true
        },
        "qwen3-coder-30b-a3b-instruct-q4_k_m": {
          "name": "Qwen3 Coder 30B MoE (local)"
        },
        "Qwen3-32B-Q4_K_M": {
          "name": "Qwen3 32B (local)"
        }
      }
    }
  }
}
```

Then in OpenCode, run `/models` and select the local model.

**Switching models:** just select a different model in OpenCode's `/models`
menu. The proxy auto-swaps (~60-90s). Or from terminal: `make run MODEL=<name>`.

## Adding custom models

Create a new file in `models/` following the preset format:

```bash
# models/my-model.env
MODEL_REPO=username/My-Model-GGUF
MODEL_FILE=my-model-Q4_K_M.gguf
MODEL_NAME=My Model (local)
MMPROJ_FILE=                    # leave empty for text-only
MMPROJ_URL=
TEMPLATE_FILE=                  # leave empty to use GGUF default
TEMPLATE_URL=
REASONING=                      # "on" for thinking models, empty to disable
CONTEXT_SIZE=65536
TEMPERATURE=0.7
TOP_P=0.95
TOP_K=40
MIN_P=0
```

Then: `make switch MODEL=my-model && make up`

The Docker image is model-agnostic — it runs any GGUF that llama.cpp supports.
No rebuild required to add new models.

### Gemma 4 thinking mode

Thinking is enabled via `--reasoning on` and the interleaved template
(`google-gemma-4-interleaved.jinja` from [PR #21418](https://github.com/ggml-org/llama.cpp/pull/21418)).
The interleaved template preserves reasoning between tool calls, which Google's
[prompting guide](https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4) recommends
for agentic workflows.

The 31B Dense uses **adaptive thinking** — it decides whether and how deeply to think
based on prompt complexity. Simple prompts may produce empty thinking blocks; harder
problems trigger extended reasoning. This is by design.

**Known limitation:** `--reasoning-budget N` is broken for Gemma 4
([llama.cpp #21487](https://github.com/ggml-org/llama.cpp/issues/21487)).
[PR #21697](https://github.com/ggml-org/llama.cpp/pull/21697) (approved, pending merge)
fixes this by adding the missing `thinking_start_tag` / `thinking_end_tag` to the
Gemma 4 parser. Once merged, bump `LLAMA_CPP_VERSION` and rebuild to get budget control.

**Important:** Do **not** set `--chat-template gemma` — that forces the legacy Gemma 1/2
template and breaks tool calling. The Gemma 4 template is auto-detected from the GGUF metadata.

## Troubleshooting

### `dependency failed to start: container llama_server is unhealthy`

The model GGUF is still downloading or loading. Check progress:

```bash
docker logs llama_server --tail 20    # look for "downloadInProgress"
make status                           # check container health state
```

If models aren't pre-downloaded, the first start can take 5-10+ minutes
depending on network speed (~19 GB GGUF). The health check allows up to 12.5
minutes. To avoid this entirely:

```bash
make download-all   # pre-cache all GGUFs, then restart
make up
```

### Container starts but health check keeps failing

```bash
docker inspect llama_server --format='{{json .State.Health}}' | python3 -m json.tool
```

Common causes:
- **Model too large for VRAM** — check `docker logs llama_server` for OOM errors
- **Missing .env** — run `make setup` or `make switch MODEL=gemma4`
- **Corrupt download** — delete `~/docker-volumes/llama-server/huggingface/` and restart

### Model switching fails or times out

The proxy has a 180s timeout for model swaps (`HEALTH_TIMEOUT`). If the model
isn't cached, it must download first. Pre-download all models:

```bash
make download-all
```

Check proxy logs: `docker logs model_proxy --tail 30`

### GPU not detected

```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
```

If this fails, install the [NVIDIA Container Toolkit](#nvidia-container-toolkit-wsl2).

### Open WebUI won't start

Open WebUI depends on model-proxy being healthy, which depends on llama-server.
Check the chain:

```bash
docker compose ps                     # all three containers should be "healthy"
curl -s http://localhost:11434/health  # proxy health
docker exec llama_server curl -sf http://localhost:8080/health  # direct health check
```

## Known issues

### First start is slow

Two scenarios:

1. **First-ever start** (model not cached): GGUF downloads from HuggingFace
   (~19 GB), then loads into VRAM. Total: 5-10+ minutes depending on network.
   Health check window: 750s (12.5 min).
2. **Subsequent starts** (model cached): VRAM load only. ~60-90s.

Pre-download with `make download-all` to always get scenario 2.

### Model switching takes ~60-90 seconds

When the proxy detects a model mismatch, it recreates llama-server with the new model.
The delay is the GGUF load time into VRAM — there's no way around this with 32 GB VRAM
(only one ~20 GB model fits at a time). The proxy blocks the triggering request until
the new model is healthy, then forwards it. Pre-download all models with
`make download-all` to eliminate network wait on first switch.
