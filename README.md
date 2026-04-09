# ollama-compose

Local LLM inference stack (Ollama + Open WebUI) for WSL2 with NVIDIA GPU.

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

1. Create the `.env` file:

```bash
# Generate a secret key for Open WebUI session persistence
echo "WEBUI_SECRET_KEY=$(openssl rand -hex 32)" > .env
```

2. Start the stack:

```bash
docker compose up -d
```

3. Open the UI at [http://localhost:3000](http://localhost:3000)

## Pull and run a model

```bash
docker exec -it ollama ollama run gemma4:26b
```

## Architecture

| Service | Address | Purpose |
|---|---|---|
| Ollama | `127.0.0.1:11434` | LLM inference engine (GPU) |
| Open WebUI | `127.0.0.1:3000` | Browser-based chat interface |
| Warmup | (one-shot) | Pre-loads model into VRAM on startup |

Both ports are bound to `127.0.0.1` only -- not exposed to the network.

## Model choice: gemma4:26b MoE

The 26B MoE (Mixture of Experts) variant is the best fit for 32GB VRAM GPUs like the RTX 5090:

| Model | Disk | Active params | VRAM usage (64k ctx) | GPU util | Warm response |
|---|---|---|---|---|---|
| `gemma4:26b` MoE | 18 GB | 3.8B | ~22 GB (100% GPU) | 82%, 237W | 0.36s |
| `gemma4:31b` Dense | 20 GB | 30.7B | ~32 GB (85% GPU) | 28%, 126W | 17.4s |

The 26b MoE is **48x faster** on 32GB VRAM because it fits entirely on GPU. The 31b dense
model spills layers to CPU, creating a GPU-CPU bottleneck. Benchmark scores are nearly
identical (LiveCodeBench: 77.1% vs 80.0%, AIME 2026: 88.3% vs 89.2%).

## Known issues

### Flash attention bug with Gemma 4 (Ollama 0.20.x)

`OLLAMA_FLASH_ATTENTION=1` causes Gemma 4 models to run on CPU despite `ollama ps` reporting
`100% GPU`. This is a [known Ollama bug](https://github.com/ollama/ollama/issues/15237).

**Workaround:** Flash attention is disabled in this stack's `docker-compose.yml`. Re-enable
once the fix is merged upstream (likely Ollama 0.21+).

**Side effect:** Without flash attention, `OLLAMA_KV_CACHE_TYPE=q8_0` is ignored (quantized
KV cache requires FA). The KV cache falls back to f16, using ~2x VRAM. This is still fine
for the 26b MoE at 64k context (~22 GB total).

### Gemma 4 thinking mode and OpenAI-compatible API

Gemma 4 has thinking/reasoning enabled by default. Without `reasoning_effort` set, the model
puts all output into the `reasoning` field and leaves `content` empty. The `@ai-sdk/openai-compatible`
provider (used by OpenCode) only reads `content`, so it appears to hang.

**Fix:** Set `"reasoningEffort": "high"` (or `"low"`, `"medium"`) in the model options.
This tells Ollama to populate both `reasoning` and `content` fields.

## Using with OpenCode

You can use your local Ollama instance as a provider for [OpenCode](https://opencode.ai).

1. Pull the model (Ollama recommends at least 64k context for OpenCode):

```bash
docker exec -it ollama ollama pull gemma4:26b
```

2. Add the provider to your `opencode.json` (project root or `~/.config/opencode/opencode.json`):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "ollama": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Ollama (local)",
      "options": {
        "baseURL": "http://localhost:11434/v1"
      },
      "models": {
        "gemma4:26b": {
          "name": "Gemma 4 26B MoE (local)",
          "options": {
            "reasoningEffort": "high"
          }
        }
      }
    }
  }
}
```

3. In OpenCode, run `/models` and select the local model.

## Verify GPU is actually being used

`ollama ps` can report `100% GPU` while inference runs on CPU (see flash attention bug above).
Always verify with `nvidia-smi`:

```bash
# During active inference, GPU-Util should be 50-90% and power draw 150-300W
nvidia-smi --query-gpu=utilization.gpu,power.draw,memory.used --format=csv,noheader
```

## Alternatives to Ollama

If you hit Ollama bugs or need more performance, consider:

| Tool | Best for | Gemma 4 support | Notes |
|---|---|---|---|
| **llama.cpp** (`llama-server`) | Max single-user speed | Full (GGUF) | ~44 tok/s on 26b with 12GB VRAM. FA works correctly. OpenAI-compatible API built in. |
| **vLLM** | Multi-user / production | Full (HuggingFace) | 3-5x better concurrent throughput. Requires CUDA. |
| **LM Studio** | GUI model browser | Full (GGUF) | Desktop app, good for evaluation. |

For a single-user coding setup, `llama-server` (from llama.cpp) is the fastest option and
avoids Ollama's abstraction layer. Ollama wins on convenience (Docker, model management, one-command pulls).
