# llm-compose

Local LLM inference stack (llama.cpp + Open WebUI) for WSL2 with NVIDIA GPU.

Runs **Gemma 4 26B MoE** at **167 tok/s** on an RTX 5090 with flash attention,
quantized KV cache, and 64k context — all inside Docker.

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

1. Create the `.env` file and volume directories:

```bash
echo "WEBUI_SECRET_KEY=$(openssl rand -hex 32)" > .env
mkdir -p ~/docker-volumes/{llama-server,webui}
```

2. Build the llama-server image (one-time, ~10 min):

```bash
docker compose build llama-server
```

3. Start the stack:

```bash
docker compose up -d
```

The model (~18 GB) auto-downloads from HuggingFace on first start.

4. Open the UI at [http://localhost:3000](http://localhost:3000)

## Architecture

| Service | Address | Purpose |
|---|---|---|
| llama-server | `127.0.0.1:11434` | LLM inference (GPU), OpenAI-compatible API |
| Open WebUI | `127.0.0.1:3000` | Browser-based chat interface |

Both ports are bound to `127.0.0.1` only — not exposed to the network.

### Monitoring

- **Health:** `curl http://localhost:11434/health`
- **Metrics:** `curl http://localhost:11434/metrics` (Prometheus format)
- **GPU:** `nvidia-smi --query-gpu=utilization.gpu,power.draw,memory.used --format=csv`

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

## Model choice: Gemma 4 26B MoE

The 26B MoE (Mixture of Experts) is the best fit for 32 GB VRAM:

- **18 GB** on disk (Q4_K_M quantization)
- Only **3.8B parameters active** per token (8/128 experts + 1 shared)
- **256K native context** (we use 64k for the VRAM/quality balance)
- Near-identical benchmarks to the 31B dense (LiveCodeBench: 77% vs 80%)
- Fits **100% on GPU** with flash attention + q8_0 KV cache at 64k context

## Custom Docker image

The `llama-server.Dockerfile` builds llama.cpp from source with:

- **CUDA 12.8** with native **sm_120** (Blackwell) kernels — no PTX JIT overhead
- Only the `llama-server` binary + shared libs in the runtime image (~5.7 GB)
- Multi-stage build: `nvidia/cuda:12.8.1-devel` → `nvidia/cuda:12.8.1-runtime`

To rebuild after a llama.cpp update:

```bash
# Update the version in llama-server.Dockerfile, then:
docker compose build llama-server
docker compose up -d llama-server
```

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
        "gemma-4-26b-a4b-it-Q4_K_M": {
          "name": "Gemma 4 26B MoE (local)"
        }
      }
    }
  }
}
```

Then in OpenCode, run `/models` and select the local model.

### Gemma 4 thinking mode

Gemma 4 has built-in chain-of-thought reasoning enabled by default. The thinking trace
appears in the `reasoning_content` field of the OpenAI-compatible API response. The
`@ai-sdk/openai-compatible` provider reads this field automatically.

**Controlling thinking depth:**

- Thinking is unrestricted by default (`--reasoning-budget -1`).
- Gemma 4 decides how deeply to think based on the **prompt complexity**, not an API parameter.
- `reasoning_effort` in the OpenAI API is **not supported** by llama-server for Gemma 4
  (it's a model-specific training feature, not a generic parameter).
- To disable thinking entirely, add `--reasoning-budget 0` to the docker-compose command.
- See [Google's Gemma 4 prompting guide](https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4#adaptive-thought-efficiency)
  for prompt techniques that influence thinking depth.

**Important:** Do **not** set `--chat-template gemma` — that forces the legacy Gemma 1/2
template and breaks tool calling. The Gemma 4 template is auto-detected from the GGUF metadata.

## Known issues

### Gemma 4 thinking consumes token budget before responding

With default thinking enabled, short `max_tokens` budgets may be entirely consumed by
the reasoning trace, leaving `content` empty. This is expected — the model thinks first,
then responds. OpenCode handles this automatically with adequate token budgets.

### First request is slow after container start

The model takes ~60-90 seconds to load into VRAM on first request. Subsequent requests
are instant. The `start_period: 120s` in the healthcheck accounts for this.
