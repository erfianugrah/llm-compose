# llm-compose

Local LLM inference stack (llama.cpp + Open WebUI) for WSL2 with NVIDIA GPU.

Runs **Gemma 4 31B Dense** on an RTX 5090 with flash attention,
quantized KV cache, and 64k context — all inside Docker.

## Quick start

```bash
make setup   # one-time: .env, volumes, mmproj download, image build (~10 min)
make up      # start the stack
```

Open the UI at [http://localhost:3000](http://localhost:3000).

Run `make help` for all available targets.

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

The recommended way is `make setup` (see Quick start above). For manual setup:

1. Create the `.env` file and volume directories:

```bash
echo "WEBUI_SECRET_KEY=$(openssl rand -hex 32)" > .env
mkdir -p ~/docker-volumes/llama-server/models ~/docker-volumes/webui
```

2. Download the multimodal projector for vision (image) support:

```bash
curl -L --progress-bar \
  -o ~/docker-volumes/llama-server/models/mmproj-gemma-4-31B-it-f16.gguf \
  "https://huggingface.co/ggml-org/gemma-4-31B-it-GGUF/resolve/main/mmproj-gemma-4-31B-it-f16.gguf"
```

3. Build the llama-server image (one-time, ~10 min):

```bash
docker compose build llama-server
```

4. Start the stack:

```bash
docker compose up -d
```

The text model (~18 GB) auto-downloads from HuggingFace on first start.

5. Open the UI at [http://localhost:3000](http://localhost:3000)

## Architecture

| Service | Address | Purpose |
|---|---|---|
| llama-server | `127.0.0.1:11434` | LLM inference (GPU), OpenAI-compatible API |
| Open WebUI | `127.0.0.1:3000` | Browser-based chat interface |

Both ports are bound to `127.0.0.1` only — not exposed to the network.

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

## Model choice: Gemma 4 31B Dense

The 31B Dense is the flagship Gemma 4 model. It fits on 32 GB VRAM at Q4_K_M
thanks to the hybrid sliding-window attention (50 local layers + 10 global layers):

- **18.7 GB** on disk (Q4_K_M quantization)
- **All 31B parameters active** per token — no routing, no sparsity
- **256K native context** (we use 64k — only the 10 global layers scale with context)
- Top benchmarks in its class (LiveCodeBench: 80%, AIME 2026: 89.2%)
- Fits **100% on GPU** with flash attention + q8_0 KV cache at 64k context (~24.6 GB total)
- Thinking mode via the interleaved template (PR #21418)

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

Gemma 4 is a multimodal model that can process both text and images. Vision requires
a separate **multimodal projector** (mmproj) GGUF file that handles image encoding
before the language model sees it.

`make setup` downloads the mmproj automatically. If you set up manually, see step 2
in the Setup section.

The mmproj is loaded at runtime via `--mmproj` — no image rebuild required.
The `libmtmd.so` library that powers multimodal inference is already compiled into
the Docker image.

### How it works

1. The text model GGUF (`gemma-4-31B-it-Q4_K_M.gguf`) handles language
2. The mmproj GGUF (`mmproj-gemma-4-31B-it-f16.gguf`) handles image encoding
3. llama-server combines both via `libmtmd` to process multimodal inputs

### Disabling vision

To run text-only (saves ~1.1 GB disk, negligible VRAM):

1. Remove the `--mmproj` and its path from `docker-compose.yml`
2. Remove the `/models` volume mount
3. Run `docker compose up -d`

## Using with OpenCode

Add the provider to `~/.config/opencode/opencode.json`:

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
        }
      }
    }
  }
}
```

Then in OpenCode, run `/models` and select the local model.

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
