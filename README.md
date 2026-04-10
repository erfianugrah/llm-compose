# llm-compose

Local LLM inference stack (llama.cpp + Open WebUI) for WSL2 with NVIDIA GPU.

Multi-model local LLM inference on an RTX 5090 with flash attention,
quantized KV cache, and 64k context — all inside Docker. Swap models without
rebuilding the image.

## Quick start

```bash
make setup   # one-time: .env, volumes, model assets, image build
make up      # start the stack
```

Open the UI at [http://localhost:3000](http://localhost:3000).

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
To pre-download all models: `make download-all`.

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

### Monitoring

- **Health:** `make health` or `curl http://localhost:11434/health`
- **Metrics:** `make metrics` or `curl http://localhost:11434/metrics` (Prometheus format)
- **GPU:** `make gpu` or `nvidia-smi --query-gpu=utilization.gpu,power.draw,memory.used --format=csv`
- **Status:** `make status` — container status + health check

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
# Update the version in llama-server.Dockerfile, then:
make rebuild
```

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
          "name": "Gemma 4 31B Dense (local)"
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

## Known issues

### First request is slow after container start

The model takes ~60-90 seconds to load into VRAM on first request. Subsequent requests
are instant. The `start_period: 120s` in the healthcheck accounts for this.

### Model switching takes ~60-90 seconds

When the proxy detects a model mismatch, it recreates llama-server with the new model.
The delay is the GGUF load time into VRAM — there's no way around this with 32 GB VRAM
(only one ~20 GB model fits at a time). The proxy blocks the triggering request until
the new model is healthy, then forwards it. Pre-download all models with
`make download-all` to eliminate network wait on first switch.
