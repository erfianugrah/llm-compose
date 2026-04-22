# llm-compose

Local LLM inference stack running llama.cpp + Open WebUI on WSL2 with an
NVIDIA GPU. A model-switching reverse proxy auto-swaps GGUF models on
demand — no image rebuild, no terminal needed.

Built for an **RTX 5090** (32 GB VRAM) with flash attention, quantized
KV cache, and auto-fit context. All inside Docker.

## Quick start

```bash
make setup          # one-time: .env, volumes, model assets, image build
make download-all   # pre-download all model GGUFs (~98 GB total)
make up             # start the stack
```

Open the UI at [http://localhost:3000](http://localhost:3000).

Or do everything in one shot (build, push to registry, start):

```bash
make deploy
```

> **First start without pre-download:** if you skip `make download-all`, the
> active model GGUF (~17-21 GB) downloads inside the container on first boot.
> The health check allows up to **12.5 minutes** for this (`start_period:
> 600s` + 5 retries at 30s). Subsequent starts load from cache in ~60-90s.

Run `make help` for all available targets.

## Available models

| Preset | Model | Type | Size | Active | Vision | Thinking | Best for |
|---|---|---|---|---|---|---|---|
| `qwen35` | Qwen 3.5 27B | Dense | ~18 GB | 27B | Yes | Yes | Multimodal reasoning, agentic coding |
| `qwen36-moe` | Qwen3.6 35B-A3B | MoE | ~22.1 GB | 3B | Yes | Yes | Agentic coding, frontend, repo-level reasoning |
| `gemma4` | Gemma 4 31B | Dense | ~20.2 GB | 31B | Yes | Yes | Multimodal, agentic coding |
| `qwen3-coder` | Qwen3 Coder 30B A3B | MoE | ~18.6 GB | 3.3B | No | No | Fast code generation |
| `qwen3` | Qwen3 32B | Dense | ~20 GB | 32B | No | Yes | Research, science, tool use |

All models fit in 32 GB VRAM at Q4 quantization with auto-fit context (typically 130-260K depending on model size). The "Size"
column includes the mmproj file for multimodal models. The "Active" column
shows parameters evaluated per token — MoE models activate a small subset of
total parameters, trading quality for speed.

### Switching models

**From OpenCode** (recommended): select a different model in `/models`.
The proxy detects the mismatch and auto-swaps (~60-90s on first request).

**From terminal:**

```bash
make run MODEL=qwen35       # switch + restart in one shot
```

List available presets with `make models`. Pre-download all with `make download-all`.

## Prerequisites

### Hardware

- NVIDIA GPU with 24-32 GB VRAM (tested on RTX 5090)
- The VRAM budget enforces a hard limit: model weight size must be
  ≤ `VRAM_LIMIT - VRAM_RESERVE` (default: 32 - 10 = **22 GB**). The
  remaining 10 GB covers KV cache, CUDA context, and flash attention
  workspace. Context auto-scales to fit available VRAM.

### Software

- **WSL2** with an Ubuntu distro (tested on Ubuntu 24.04)
- **Docker Engine** (not Docker Desktop) — install via
  [docker.com/engine/install](https://docs.docker.com/engine/install/ubuntu/)
- **NVIDIA driver** ≥ 550 on the Windows host
- **NVIDIA Container Toolkit** (see below)
- **GNU Make**, **curl**, **openssl**, **bc** (pre-installed on most distros)

### NVIDIA Container Toolkit (WSL2)

This lets Docker containers access the GPU:

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

If this shows your GPU, you're ready. If not, check the driver and toolkit
installation.

## Setup (step by step)

`make setup` automates everything below. This section explains what it does
so you can troubleshoot or customize.

### 1. Generate `.env`

The `.env` file holds the active model configuration plus the Open WebUI
session secret. `make setup` creates it automatically:

```bash
# Manual equivalent:
echo "WEBUI_SECRET_KEY=$(openssl rand -hex 32)" > .env
```

### 2. Create volume directories

Three host directories store persistent data across container restarts:

```bash
# make setup calls `make dirs` which creates:
~/docker-volumes/llama-server/          # HuggingFace model cache (GGUFs)
~/docker-volumes/llama-server/models/   # mmproj files, jinja templates
~/docker-volumes/webui/                 # Open WebUI user data, chat history
```

If Docker previously created these as root, `make dirs` fixes ownership.

### 3. Load a model preset

```bash
make switch MODEL=qwen35    # or gemma4, qwen3-coder, etc.
```

This copies `models/qwen35.env` into `.env` (preserving `WEBUI_SECRET_KEY`),
then downloads any model-specific assets (mmproj files, jinja templates) into
`~/docker-volumes/llama-server/models/`.

The GGUF itself is NOT downloaded here — it's fetched by llama-server on first
container start via the `--hf-repo` / `--hf-file` flags, and cached in the
HuggingFace cache directory.

### 4. Build Docker images

```bash
make build
```

See [Docker images](#docker-images) below for the full build process.

### 5. Start the stack

```bash
make up
```

Docker Compose starts three services in dependency order:
llama-server → model-proxy → Open WebUI.

### 6. (Recommended) Pre-download all models

```bash
make download-all
```

This runs each model's GGUF through llama-server in a temporary container
(CPU-only, no GPU needed) to populate the HuggingFace cache. Takes a while
(~98 GB total) but means model switching later only costs VRAM load time
(~60-90s) instead of download + load (5-10+ min).

## Architecture

```
OpenCode / Open WebUI
        |
        v
  model-proxy :11434    <-- auto-swaps models based on request
        |
        v
  llama-server :8080    <-- GPU inference (internal, not exposed)
```

### Services

| Service | Address | Purpose |
|---|---|---|
| model-proxy | `127.0.0.1:11434` | Reverse proxy with auto model switching |
| llama-server | internal only | LLM inference engine (GPU) |
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
| Gateway | `172.28.0.1` | -- |

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

The proxy intercepts the `model` field in each `/v1/chat/completions` request.
If it differs from what's currently loaded, the proxy:

1. Validates the VRAM budget (rejects if model weights exceed 22 GB)
2. Updates `.env` with the new model preset (preserves `WEBUI_SECRET_KEY`)
3. Runs `docker compose -p llm-compose up -d --force-recreate llama-server`
4. Polls llama-server's `/health` endpoint until `{"status": "ok"}`
   (up to `HEALTH_TIMEOUT` seconds, default 900)
5. Forwards the original request to the now-loaded model

From OpenCode: just select a different model in `/models`. The first request
takes ~60-90s while the model loads; subsequent requests are instant.

The proxy requires Docker socket access (`/var/run/docker.sock:ro`) to
recreate the llama-server container. It's the only container with socket
access.

### Health checks

Each service has a Docker health check that gates dependent services:

| Service | Endpoint | start_period | Retries | Interval | Total window |
|---|---|---|---|---|---|
| llama-server | `localhost:8080/health` | 600s | 5 | 30s | **750s** (~12.5 min) |
| model-proxy | `localhost:11434/health` | 10s | 3 | 30s | 100s |
| Open WebUI | `localhost:8080/health` | 30s | 3 | 30s | 120s |

**Startup dependency chain:** llama-server (healthy) -> model-proxy (healthy) -> Open WebUI.

The llama-server `start_period` of 600s accommodates first-time model downloads
(~20 GB GGUF from HuggingFace). Once models are cached, startup takes ~60-90s
(VRAM load only).

During model switching, the proxy returns `{"status": "switching"}` (HTTP 200)
on its health endpoint so Docker doesn't kill it mid-swap.

## Docker images

The stack uses three Docker images. Two are built locally (or pulled from
Docker Hub), one is third-party:

| Image | Registry | Description |
|---|---|---|
| `erfianugrah/llama-server:cuda12.8-sm120` | Docker Hub | llama.cpp with CUDA 12.8 / sm_120 |
| `erfianugrah/model-proxy:latest` | Docker Hub | Python reverse proxy with Docker CLI |
| `ghcr.io/open-webui/open-webui:v0.8.12` | GHCR | Third-party chat UI (not built) |

### Building from source vs pulling

`make build` checks if the llama-server image exists locally. If not, it
builds from source (~10 min). The proxy image always builds (fast, <30s).

```bash
make build          # build locally (skips llama-server if image exists)
make rebuild        # force rebuild llama-server from source
make pull           # pull pre-built images from Docker Hub (skip build)
make push           # push locally-built images to Docker Hub
make release        # rebuild + push + restart
```

For a fresh machine, either:

- `make pull` — fastest, uses pre-built images from Docker Hub
- `make build` — builds from source, required if you changed the Dockerfile
  or need a different llama.cpp version

### llama-server image (`llama-server.Dockerfile`)

Multi-stage build that compiles llama.cpp from source with GPU-specific
optimizations. This is the only image that takes significant build time
(~10 min on a modern CPU).

**Stage 1 — Build** (`nvidia/cuda:12.8.1-devel-ubuntu24.04`):

1. Installs build tools: `git`, `cmake`, `build-essential`, `curl`,
   `ca-certificates`, `libssl-dev`
   (TLS for HuggingFace downloads)
2. Clones llama.cpp at a pinned version (`LLAMA_CPP_VERSION=b8799`)
3. Runs CMake with CUDA enabled and **`CMAKE_CUDA_ARCHITECTURES=120`**
   (sm_120 = native Blackwell kernels for RTX 5090 — no PTX JIT overhead)
4. Builds only the `llama-server` target (not the full suite)
5. Collects all `.so` shared libraries (libggml-cuda.so, libllama.so,
   libmtmd.so for multimodal, etc.)

The `--allow-shlib-undefined` linker flag lets the build succeed without
`libcuda.so` present — the real library is injected by the NVIDIA container
runtime at startup.

**Stage 2 — Runtime** (`nvidia/cuda:12.8.1-runtime-ubuntu24.04`):

1. Installs minimal runtime deps: `curl` (health checks), `ca-certificates`
   (TLS), `libgomp1` (OpenMP)
2. Copies shared libraries and the `llama-server` binary from stage 1
3. Runs `ldconfig` to register the shared libraries

The runtime image uses `nvidia/cuda:runtime` (not `base`) because it needs
`libcublas` and other CUDA shared libraries for GPU inference. Final image
size: ~5.7 GB.

**Key CMake flags:**

| Flag | Purpose |
|---|---|
| `GGML_CUDA=ON` | Enable CUDA backend |
| `CMAKE_CUDA_ARCHITECTURES=120` | Native sm_120 kernels (RTX 5090 Blackwell) |
| `GGML_CUDA_FORCE_CUBLAS=OFF` | Use custom CUDA kernels where faster than cuBLAS |
| `CMAKE_BUILD_TYPE=Release` | Optimization level -O3 |
| `GGML_NATIVE=OFF` | Don't use host CPU features (portability) |

**To update llama.cpp version:**

1. Edit `LLAMA_CPP_VERSION` in `llama-server.Dockerfile`
2. Run `make rebuild` (local) or `make release` (rebuild + push)

**To target a different GPU architecture:**

Edit `CMAKE_CUDA_ARCHITECTURES` in the Dockerfile. Common values:

| Architecture | GPUs |
|---|---|
| `86` | RTX 3090, 3080 (Ampere) |
| `89` | RTX 4090, 4080 (Ada Lovelace) |
| `100` | RTX 5090 (Blackwell, compute only) |
| `120` | RTX 5090 (Blackwell, sm_120 with full features) |

### model-proxy image (`proxy/Dockerfile`)

Lightweight Python proxy with Docker CLI for model swapping. Single-stage
build, takes <30s.

1. Base: `python:3.12-slim`
2. Installs Docker CE CLI + Compose plugin (no Docker daemon — it uses the
   host's Docker via the mounted socket)
3. Copies `proxy.py` (single-file, stdlib-only Python — no pip dependencies)

The proxy needs the Docker CLI to run
`docker compose up -d --force-recreate llama-server` when swapping models.

### Open WebUI

Third-party image pulled from GHCR. Not built locally. Pinned to a specific
version (`v0.8.12`) to avoid breaking changes.

## Performance tuning

The stack is tuned to get the most out of a single GPU (RTX 5090, 32 GB)
without sacrificing model quality. Here's what each setting does and why.

### How GPU memory is used

When you run a language model, your GPU memory (VRAM) is split between
three things:

1. **Model weights** (~17-21 GB) — the actual brain of the model, loaded
   once and stays in VRAM. Larger models are smarter but use more space.
2. **KV cache** (~2-10 GB) — the model's "short-term memory" of your
   conversation. Grows with context length (how much text the model can
   see at once). Longer context = more memory.
3. **Compute workspace** (~1-2 GB) — scratch space for the math the GPU
   does during generation. Relatively fixed.

On a 32 GB GPU, after the model weights there's ~11-15 GB left for
everything else. The settings below maximize what you get from that space.

### Context window and auto-fit (`--fit`)

Each model preset specifies a context size (default 65K tokens). This is the
amount of conversation the model can "see" at once — roughly 50,000 words.
65K is a good balance of memory and speed for coding tasks.

**Why not maximize context?** Larger context uses more VRAM for the KV cache
and can trigger performance issues. At very high context sizes (>130K), some
model architectures cause CUDA graph compilation failures that silently push
compute to the CPU, dropping throughput from ~170 tok/s to ~5 tok/s.

**The `--fit` safety net:** llama.cpp's `--fit` mechanism (on by default)
acts as a guard rail. If the model + context don't fit in VRAM, it
**automatically reduces the context** until everything fits:

1. Calculates how much VRAM the model + requested context would need
2. If it exceeds available VRAM, reduces context until it fits
3. `--fit-ctx 32768` sets the floor — it won't go below 32K tokens

This protects against OOM on smaller GPUs or if you manually increase
`CONTEXT_SIZE` in a preset beyond what fits. Example log output:
```
llama_params_fit_impl: context size reduced from 131072 to 98304
llama_params_fit_impl: entire model can be fit by reducing context
```

To increase context for a specific model, edit its preset and restart:
```bash
# In models/qwen35.env, change:
CONTEXT_SIZE=131072
# Then:
make run MODEL=qwen35
```

If the new size doesn't fit, `--fit` reduces it automatically. Check the
actual context in the logs: `docker logs llama_server | grep n_ctx`.

### Flash attention (`--flash-attn on`)

Flash attention is a faster algorithm for the attention computation (the
core operation in transformer models). It's not an approximation — it
produces identical results but uses **less memory and runs faster**.

Without flash attention, the KV cache must be stored in full precision
(f16). With flash attention enabled, the KV cache can be quantized (see
below), cutting its memory usage significantly.

### Quantized KV cache (`-ctk q8_0 -ctv q8_0`)

The KV cache (the model's conversation memory) is normally stored in 16-bit
floating point. Quantization compresses it:

- **K cache at q8_0** (8-bit): Keys are used for attention scoring, where
  precision matters. 8-bit preserves quality while halving memory vs f16.
- **V cache at q8_0** (8-bit): Values are weighted-summed. 8-bit is the
  sweet spot — still halves memory vs f16 with no quality loss.

Combined, this cuts KV cache memory by ~50% compared to f16.

**Why not q4_0 for V cache?** Benchmarking showed q4_0 V cache halves
throughput on every tested model (57 → 19 tok/s on Gemma 4, 185 → 118
tok/s on Qwen 3.5 MoE). The VRAM savings (~1 GB) aren't worth the 2x
speed penalty. This appears to be a CUDA kernel issue in llama.cpp with
q4_0 V quantization on sm_120 (Blackwell).

### Single slot (`-np 1`)

llama-server can serve multiple users simultaneously using "slots" — each
slot gets its own KV cache. The default (`auto`) allocates 4 slots, meaning
4x the KV cache memory.

Since this is a single-user setup (one person using OpenCode or the web UI),
we set `-np 1`. This frees ~75% of the KV cache memory that would be wasted
on unused slots.

### Batch sizes (`-b 4096 -ub 4096`)

Batch size controls how many tokens are processed at once during "prompt
processing" (when the model reads your input before generating a response).
Larger batches = faster prompt processing on GPUs with enough compute.

The RTX 5090 has massive compute capacity. `-b 4096 -ub 4096` (up from
the default 2048) roughly doubles prompt processing speed at minimal
extra memory cost.

### GPU offloading (`-ngl 99`)

Offloads all model layers to the GPU. On a 32 GB card with ~20 GB models,
everything fits in VRAM with room to spare. No CPU fallback needed.

### llama-server command reference

The `docker-compose.yml` command block configures all of this. Values come
from `.env` (set by model presets):

```
llama-server \
  --hf-repo ${MODEL_REPO}              # HuggingFace repo (auto-downloads GGUF)
  --hf-file ${MODEL_FILE}              # GGUF filename within the repo
  --mmproj /models/${MMPROJ_FILE}      # multimodal projector (conditional)
  --jinja                              # use Jinja2 chat template from GGUF
  --chat-template-file /models/${TEMPLATE_FILE}  # override template (conditional)
  --reasoning ${REASONING}             # thinking mode: on/off (conditional)
  --port 8080 --host 0.0.0.0
  -ngl 99                              # all layers on GPU
  --flash-attn on                      # flash attention
  -ctk q8_0 -ctv q8_0                  # quantized KV cache (8-bit K and V)
  -c ${CONTEXT_SIZE}                   # context window (0 = auto from model)
  --fit on --fit-ctx 32768             # auto-scale context to fit VRAM (min 32K)
  --temp ${TEMPERATURE}                # sampling temperature
  --top-p ${TOP_P}                     # nucleus sampling
  --top-k ${TOP_K}                     # top-k sampling
  --min-p ${MIN_P}                     # min-p sampling (conditional)
  -np 1                                # single slot (single-user)
  -b 4096 -ub 4096                     # batch sizes (prompt processing speed)
  --threads 8 --threads-batch 8        # CPU threads (for non-GPU ops)
  -v --metrics                         # verbose + Prometheus metrics
```

Conditional flags (e.g., `--mmproj`, `--chat-template-file`, `--reasoning`,
`--min-p`) are only included when their corresponding env var is non-empty.
This is handled by shell parameter expansion: `${VAR:+--flag ${VAR}}`.

### Tuning for different GPUs

| GPU | VRAM | What to change |
|---|---|---|
| RTX 4090 (24 GB) | 24 GB | Use Q3_K_M quants (~14 GB), `--fit` auto-adjusts context |
| RTX 3090 (24 GB) | 24 GB | Same as 4090, change `CMAKE_CUDA_ARCHITECTURES=86` |
| RTX 4080 (16 GB) | 16 GB | Use 7-14B models only, `--fit-ctx 8192` |
| 2x GPUs | varies | Add `-ts auto` for tensor split, increase `-np` |

The `--fit` mechanism handles most of this automatically. Just change the
model preset to a smaller quant and restart.

## Security

All services are hardened with:

- **`no-new-privileges`** — prevents privilege escalation via setuid/setgid
- **`cap_drop: ALL`** — drops all Linux capabilities (llama-server, Open WebUI)
- **`user: "1000:1000"`** — Open WebUI runs as non-root, matching host user UID
- **Localhost-only ports** — `127.0.0.1` binding, not exposed to LAN
- **Read-only mounts** — model presets and Docker socket mounted `:ro`
- **Minimal runtime images** — only essential runtime deps in final stage
- **tmpfs for Open WebUI** — `/tmp` mounted noexec, nosuid, 256 MB limit
- **JSON logging with rotation** — 50 MB x 3 files (llama-server, Open WebUI), 10 MB x 3 (proxy)

The model-proxy requires the Docker socket (`/var/run/docker.sock:ro`) to
recreate llama-server during model swaps. This is the only container with
Docker socket access. It does **not** have `cap_drop: ALL` because it needs
network capabilities for proxying.

## Using with OpenCode

Add the provider to `~/.config/opencode/opencode.json`. Register all models
you want to use — the proxy auto-swaps to whichever one you select:

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
        "Qwen3.5-27B-Q4_K_M": {
          "name": "Qwen 3.5 27B Dense (local)",
          "attachment": true,
          "reasoning": true,
          "tool_call": true,
          "modalities": { "input": ["text", "image"], "output": ["text"] },
          "limit": { "context": 65536, "output": 32768 }
        },
        "Qwen3.6-35B-A3B-UD-Q4_K_M": {
          "name": "Qwen3.6 35B MoE (local)",
          "attachment": true,
          "reasoning": true,
          "tool_call": true,
          "modalities": { "input": ["text", "image"], "output": ["text"] },
          "limit": { "context": 65536, "output": 32768 }
        },
        "gemma-4-31B-it-Q4_K_M": {
          "name": "Gemma 4 31B Dense (local)",
          "attachment": true,
          "reasoning": true,
          "tool_call": true,
          "modalities": { "input": ["text", "image"], "output": ["text"] },
          "limit": { "context": 65536, "output": 32768 }
        },
        "qwen3-coder-30b-a3b-instruct-q4_k_m": {
          "name": "Qwen3 Coder 30B MoE (local)",
          "tool_call": true,
          "limit": { "context": 65536, "output": 32768 }
        },
        "Qwen3-32B-Q4_K_M": {
          "name": "Qwen3 32B (local)",
          "reasoning": true,
          "tool_call": true,
          "limit": { "context": 65536, "output": 32768 }
        }
      }
    }
  }
}
```

The `model` key is the GGUF filename without `.gguf` — this is the model ID
the proxy uses to match requests to presets.

| Field | Purpose |
|---|---|
| `attachment` | Enables image/file uploads (multimodal models only) |
| `reasoning` | Model supports thinking mode (`<think>` blocks) |
| `tool_call` | Model supports tool/function calling |
| `modalities` | Input/output types (set for vision models) |
| `limit` | Context window and max output tokens |

Then in OpenCode, run `/models` and select the local model. Switching models
in the `/models` menu triggers an automatic swap (~60-90s).

## Open WebUI configuration

The browser-based chat UI at [http://localhost:3000](http://localhost:3000)
is pre-configured with sensible defaults via environment variables. After
first startup, import workspace models with per-model system prompts and
parameters:

```bash
make configure-webui
```

This imports 5 workspace models from `webui/models.json`, each with:

- **Terse system prompt** — direct, no filler, code-first (caveman-lite style)
- **Per-model sampling parameters** — temperature, top_p matched to each model's preset
- **Native function calling** — `function_calling: "native"` for tool support
- **Vision capabilities** — auto-detected from proxy metadata for multimodal models

### Authentication

The init script (`scripts/init-webui.sh`) needs an admin API token. Three
methods (checked in order):

1. `WEBUI_API_KEY` env var — if you have an existing API key
2. `WEBUI_ADMIN_EMAIL` + `WEBUI_ADMIN_PASSWORD` in `.env` — headless admin
   auto-created on first startup (set these before `make up`)
3. Interactive prompt — script asks for email/password at runtime

For fully automated (no-touch) setup:

```bash
# Add to .env before first make up
WEBUI_ADMIN_EMAIL=admin@localhost
WEBUI_ADMIN_PASSWORD=$(openssl rand -hex 16)
```

The init script also accepts `WEBUI_URL` (default `http://localhost:3000`)
and `MODELS_FILE` (default `webui/models.json`) for non-standard setups.

### Customizing system prompts

Edit `webui/models.json` and re-run `make configure-webui`. Each model has a
`params.system` field with the system prompt. The default style is
caveman-lite: no filler, no hedging, keep articles and full sentences,
professional but tight.

### Environment variable defaults

These are set in `docker-compose.yml` for the `open-webui` service:

| Variable | Value | Effect |
|---|---|---|
| `WEBUI_NAME` | `llama.cpp` | Header branding |
| `DEFAULT_MODELS` | `Qwen3.5-27B-Q4_K_M` | Default model for new chats |
| `ENABLE_PERSISTENT_CONFIG` | `false` | Env vars always applied on restart |
| `DEFAULT_MODEL_PARAMS` | `{"function_calling":"native",...}` | Native tool calling globally |
| `DEFAULT_PROMPT_SUGGESTIONS` | `[]` | No starter suggestions |
| `ENABLE_SIGNUP` | `false` | Single-user, no registration |
| `WEBUI_ADMIN_EMAIL` | (from `.env`) | Headless admin email (optional) |
| `WEBUI_ADMIN_PASSWORD` | (from `.env`) | Headless admin password (optional) |
| `WEBUI_ADMIN_NAME` | `Admin` | Headless admin display name |

### How model capabilities flow

The proxy enriches the `/v1/models` response with metadata that Open WebUI
picks up automatically (v0.8.12+, PR #22441). Priority chain:

1. **`DEFAULT_MODEL_METADATA`** (env var) — global baseline for all models
2. **Proxy `/v1/models` meta** — per-model description + capabilities
   (vision flag derived from `MMPROJ_FILE` in preset)
3. **Workspace model overrides** (via `make configure-webui`) — system
   prompts, parameters, full capability config

Each layer overrides the previous. Workspace models have highest priority.

## Monitoring

| Command | Description |
|---|---|
| `make status` | Container status, active model, health |
| `make health` | Check proxy health endpoint |
| `make metrics` | Prometheus metrics from llama-server |
| `make gpu` | GPU utilization, power draw, VRAM usage |
| `make logs` | Follow logs for all services |
| `make logs-llama` | Follow llama-server logs only |

Health and metrics are also available via HTTP:

```bash
curl http://localhost:11434/health     # proxy health (JSON)
curl http://localhost:11434/metrics    # Prometheus metrics
```

## Environment variables

### Model preset variables (in `models/*.env`)

Each preset file defines one model's complete configuration:

| Variable | Description | Example |
|---|---|---|
| `VRAM_ESTIMATE_GB` | Weight size in GB (for VRAM budget check) | `18.0` |
| `MODEL_REPO` | HuggingFace repo for GGUF | `unsloth/Qwen3.5-27B-GGUF` |
| `MODEL_FILE` | GGUF filename | `Qwen3.5-27B-Q4_K_M.gguf` |
| `MODEL_NAME` | Display name (shown in proxy /v1/models) | `Qwen 3.5 27B Dense (local)` |
| `MMPROJ_FILE` | Multimodal projector filename (empty = text-only) | `mmproj-BF16.gguf` |
| `MMPROJ_URL` | Download URL for mmproj | HuggingFace URL |
| `TEMPLATE_FILE` | Jinja chat template filename (empty = GGUF default) | `google-gemma-4-interleaved.jinja` |
| `TEMPLATE_URL` | Download URL for template | GitHub raw URL |
| `REASONING` | Enable thinking mode (`on` or empty) | `on` |
| `CONTEXT_SIZE` | Context window size in tokens (0 = auto from model, `--fit` scales to VRAM) | `0` |
| `TEMPERATURE` | Sampling temperature | `0.6` |
| `TOP_P` | Nucleus sampling threshold | `0.95` |
| `TOP_K` | Top-k sampling | `20` |
| `MIN_P` | Min-p sampling (0 = disabled) | `0` |

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
| `HEALTH_TIMEOUT` | `900` | Seconds to wait for llama-server after model swap (15 min, accommodates first-time GGUF downloads) |
| `VRAM_LIMIT_GB` | `32` | Total GPU VRAM in GB |
| `VRAM_RESERVE_GB` | `10` | VRAM reserved for KV cache + overhead |
| `HOST_HOME` | `${HOME}` | Host HOME path (for `~` resolution in compose volumes) |
| `COMPOSE_PROJECT_NAME` | `llm-compose` | Must match host project name so proxy recreates llama-server on the existing network |

## Adding custom models

Create a new file in `models/` following the preset format:

```bash
# models/my-model.env

# Description line (shown by `make models`)
# Size info | active params
# Best for: use case
VRAM_ESTIMATE_GB=18.0
MODEL_REPO=username/My-Model-GGUF
MODEL_FILE=my-model-Q4_K_M.gguf
MODEL_NAME=My Model (local)
MMPROJ_FILE=                    # leave empty for text-only
MMPROJ_URL=
TEMPLATE_FILE=                  # leave empty to use GGUF default
TEMPLATE_URL=
REASONING=                      # "on" for thinking models, empty to disable
CONTEXT_SIZE=0
TEMPERATURE=0.7
TOP_P=0.95
TOP_K=40
MIN_P=0
```

Then:

```bash
make switch MODEL=my-model && make up
```

**Constraints:**

- `VRAM_ESTIMATE_GB` must be ≤ 22 (32 GB limit - 10 GB reserve). Both the
  Makefile and proxy enforce this. Include the mmproj size if applicable.
- `MODEL_FILE` minus `.gguf` becomes the model ID. This must match the key
  in your OpenCode config.
- The Docker image is model-agnostic — any GGUF that llama.cpp supports works.
  No rebuild required to add new models.
- Models with embedded Jinja2 chat templates (most modern GGUFs) don't need
  `TEMPLATE_FILE`. The `--jinja` flag is always passed. Only set it for
  models that need an override template (e.g., Gemma 4).

## Model details

### Qwen 3.5 27B Dense

The current best general-purpose local model. Native multimodal with the
highest benchmarks in the ≤22 GB VRAM class:

- **16.7 GB** on disk (Q4_K_M) + **~1.3 GB** mmproj = ~18 GB VRAM
- **27B active parameters** (dense — all params evaluated per token)
- **262K native context** (auto-fit to VRAM, typically ~200K+)
- Native vision + video support via early fusion (not bolted-on)
- Thinking mode by default (`<think>...</think>` blocks)
- SWE-bench Verified: 72.4%, GPQA Diamond: 85.5%, MMMU: 82.3%
- Sampling: Qwen recommended "thinking coding" params (temp 0.6, top_p 0.95)

### Qwen3.6 35B-A3B MoE

Agentic coding MoE — only 3B active parameters per token, successor to Qwen3.5-35B-A3B:

- **22.1 GB** on disk (UD-Q4_K_M), vision support via early fusion
- **256 experts**, 8 routed + 1 shared active per token
- **262K native context** (auto-fit to VRAM), extensible to 1M with YaRN
- Thinking mode by default, with optional `preserve_thinking` for agentic chains
- SWE-bench Verified: 73.4%, GPQA Diamond: 86.0%, AIME 2026: 92.7%
- Best for agentic coding, frontend workflows, and repo-level reasoning

### Gemma 4 31B Dense

Google's multimodal model with hybrid sliding-window attention:

- **~19 GB** on disk (Q4_K_M) + **1.2 GB** mmproj = ~20.2 GB VRAM
- **All 31B parameters active** per token (dense)
- **256K native context** (50 local + 10 global attention layers)
- Vision support via multimodal projector
- Thinking mode via the interleaved template (requires external jinja file)
- Fits 100% on GPU with flash attention + quantized KV cache

**Gemma 4 thinking mode notes:**

Thinking is enabled via `--reasoning on` and the interleaved template
(`google-gemma-4-interleaved.jinja` from
[PR #21418](https://github.com/ggml-org/llama.cpp/pull/21418)). This is the
only model that needs an external template file — the others use the template
embedded in the GGUF.

The 31B Dense uses **adaptive thinking** — it decides whether and how deeply
to think based on prompt complexity. Simple prompts may produce empty thinking
blocks; harder problems trigger extended reasoning.

**Known limitation:** `--reasoning-budget N` is broken for Gemma 4
([llama.cpp #21487](https://github.com/ggml-org/llama.cpp/issues/21487)).

**Important:** Do **not** set `--chat-template gemma` — that forces the
legacy Gemma 1/2 template and breaks tool calling.

### Qwen3 Coder 30B A3B

MoE coding specialist — only 3.3B active params per token for fast inference:

- **18.6 GB** on disk (Q4_K_M)
- **Non-thinking mode** — fast, direct code outputs
- **262K native context** (auto-fit to VRAM)
- Optimized for code generation, refactoring, and agentic coding
- Tool calling support for OpenCode

### Qwen3 32B

Dense general-purpose model with strong reasoning and tool calling:

- **~20 GB** on disk (Q4_K_M)
- **Thinking mode** — adaptive chain-of-thought reasoning
- **131K native context** (auto-fit to VRAM)
- Excellent at research, science, daily questions, web search, tool use

## Makefile reference

### Getting started

| Target | Description |
|---|---|
| `make setup` | First-time setup: .env, volumes, model assets, image build |
| `make deploy` | Full deploy: setup + push images to registry + start stack |
| `make up` | Start the stack in background |
| `make down` | Stop the stack |
| `make help` | Show all available targets |

### Model switching

| Target | Description |
|---|---|
| `make models` | List available model presets |
| `make switch MODEL=name` | Switch preset (updates .env + downloads assets) |
| `make run MODEL=name` | Switch + restart in one shot |
| `make download-all` | Pre-download all model GGUFs (~98 GB total) |
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
| `make dirs` | Create persistent volume directories (called by other targets) |

### Open WebUI

| Target | Description |
|---|---|
| `make configure-webui` | Import workspace models + system prompts into Open WebUI |
| `make reset-webui` | Nuke WebUI database and start fresh |

### Benchmarking

| Target | Description |
|---|---|
| `make bench` | Benchmark current model with different flag combos |
| `make bench-quick` | Quick benchmark: q8 vs q4 V cache only |
| `make bench-all` | Benchmark all model presets |

### Monitoring

| Target | Description |
|---|---|
| `make gpu` | GPU utilization, power draw, VRAM usage |
| `make metrics` | Fetch Prometheus metrics from llama-server |
| `make health` | Check llama-server health endpoint |

## Why llama.cpp instead of Ollama?

We switched from Ollama to llama-server (llama.cpp) for three reasons:

1. **Ollama 0.20.x has a bug** where flash attention causes Gemma 4 to run
   on CPU despite reporting 100% GPU
   ([ollama#15237](https://github.com/ollama/ollama/issues/15237)).
2. **llama.cpp is faster** — 57 tok/s generation (dense) / 185 tok/s (MoE) vs
   ~35 tok/s on Ollama because flash attention + quantized KV cache work correctly.
3. **No abstraction overhead** — llama.cpp is the engine Ollama wraps. Direct
   access means fewer bugs and more control.

### Benchmarks (RTX 5090, 32 GB VRAM, 65K context)

All models at Q4 quantization, single slot, q8_0 KV cache, flash attention,
4096 batch size. Run `make bench` to reproduce.

| Model | Type | Prompt | Generation | VRAM | Context |
|---|---|---|---|---|---|
| Qwen 3.5 35B MoE | MoE (3B active) | 600 tok/s | **185 tok/s** | 26.8 GB | 65K |
| Qwen3 Coder 30B MoE | MoE (3.3B active) | 354 tok/s | **203 tok/s** | 25.2 GB | 65K |
| Qwen 3.5 27B Dense | Dense (27B) | 461 tok/s | **54 tok/s** | 24.6 GB | 65K |
| Qwen3 32B | Dense (32B) | 731 tok/s | **60 tok/s** | 31.7 GB | 41K* |
| Gemma 4 31B Dense | Dense (31B) | 576 tok/s | **57 tok/s** | 30.0 GB | 65K |

*Qwen3 32B auto-fit reduced to 41K (large model, tight VRAM at 65K).

**MoE models are 3-4x faster** at generation because only ~3B parameters
are evaluated per token. Dense models process prompts faster but generate
slower because all 27-32B parameters are active.

### vs Ollama (Gemma 4 31B Dense)

| | Ollama 0.20.2 (FA off) | llama-server (this stack) |
|---|---|---|
| Generation | ~35 tok/s | **57 tok/s** |
| Prompt eval | ~40 tok/s | **576 tok/s** |
| GPU utilization | 28% | **85%** |
| Power draw | 126W | **228W** |
| KV cache | f16 | **q8_0** |

## Troubleshooting

### `dependency failed to start: container llama_server is unhealthy`

The model GGUF is still downloading or loading. Check progress:

```bash
docker logs llama_server --tail 20    # look for download progress
make status                           # check container health state
```

If models aren't pre-downloaded, the first start can take 5-10+ minutes
depending on network speed (~20 GB GGUF). The health check allows up to
12.5 minutes. To avoid this:

```bash
make download-all   # pre-cache all GGUFs, then restart
make up
```

### Container starts but health check keeps failing

```bash
docker inspect llama_server --format='{{json .State.Health}}' | python3 -m json.tool
```

Common causes:

- **Model too large for VRAM** — check `docker logs llama_server` for OOM
  errors. Ensure `VRAM_ESTIMATE_GB` ≤ 22 in the preset.
- **Missing .env** — run `make setup` or `make switch MODEL=qwen35`
- **Corrupt download** — delete `~/docker-volumes/llama-server/huggingface/`
  and restart
- **Wrong CUDA architecture** — if not on an RTX 5090, rebuild the image
  with the correct `CMAKE_CUDA_ARCHITECTURES`

### Model switching fails or times out

The proxy has a 900s (15 min) timeout for model swaps (`HEALTH_TIMEOUT`).
If the model GGUF isn't cached, it downloads first (~20 GB, 5-10 min
depending on network). Pre-download all models to avoid this:

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
docker exec llama_server curl -sf http://localhost:8080/health
```

If Open WebUI is crash-looping with `unable to open database file`, it's a
permissions issue. The container runs as UID 1000 (`user: "1000:1000"` in
compose) but Docker may have created subdirectories as root. Fix with:

```bash
make reset-webui    # nuke and recreate (loses chat history)
make up && make configure-webui
```

Or fix permissions without losing data:

```bash
docker compose stop open-webui
docker run --rm -v ~/docker-volumes/webui:/data alpine chown -R 1000:1000 /data
docker compose start open-webui
```

### Building on a different GPU

The default Dockerfile targets sm_120 (RTX 5090). To build for a different
GPU:

1. Edit `CMAKE_CUDA_ARCHITECTURES` in `llama-server.Dockerfile`
2. Optionally change the base CUDA image version
3. Run `make rebuild`

You can target multiple architectures (e.g., `"89;120"`) at the cost of
larger image size and longer build time.

## Known issues

### First start is slow

Two scenarios:

1. **First-ever start** (model not cached): GGUF downloads from HuggingFace
   (~17-21 GB), then loads into VRAM. Total: 5-10+ minutes depending on
   network. Health check window: 750s (12.5 min).
2. **Subsequent starts** (model cached): VRAM load only. ~60-90s.

Pre-download with `make download-all` to always get scenario 2.

### Model switching takes ~60-90 seconds

When the proxy detects a model mismatch, it recreates llama-server with the
new model. The delay is the GGUF load time into VRAM — there's no way around
this with 32 GB VRAM (only one ~20 GB model fits at a time). The proxy blocks
the triggering request until the new model is healthy, then forwards it.
Pre-download all models with `make download-all` to eliminate network wait
on first switch.

## Project structure

```
llm-compose/
  docker-compose.yml          # service definitions (3 containers)
  llama-server.Dockerfile     # multi-stage CUDA build for llama.cpp
  Makefile                    # all commands (setup, build, switch, etc.)
  .env                        # active model config (generated, gitignored)
  .env.example                # reference config
  .gitignore                  # excludes .env and legacy dirs
  proxy/
    Dockerfile                # Python + Docker CLI image
    proxy.py                  # reverse proxy with auto model switching
  models/
    gemma4.env                # Gemma 4 31B Dense preset
    qwen3.env                 # Qwen3 32B preset
    qwen3-coder.env           # Qwen3 Coder 30B A3B preset
    qwen35.env                # Qwen 3.5 27B Dense preset
    qwen36-moe.env            # Qwen3.6 35B-A3B MoE preset
  webui/
    models.json               # Open WebUI workspace model configs
  scripts/
    init-webui.sh             # One-shot Open WebUI model import
    bench.sh                  # Performance benchmark script
```
