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
docker exec -it ollama ollama run llama3.3:70b
```

## Architecture

| Service | Address | Purpose |
|---|---|---|
| Ollama | `127.0.0.1:11434` | LLM inference engine (GPU) |
| Open WebUI | `127.0.0.1:3000` | Browser-based chat interface |

Both ports are bound to `127.0.0.1` only -- not exposed to the network.

## Using with OpenCode

You can use your local Ollama instance as a provider for [OpenCode](https://opencode.ai).

1. Pull a model with tool-calling support (recommended `num_ctx` of 16k-32k):

```bash
docker exec -it ollama ollama pull gemma4:31b
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
        "gemma4:31b": {
          "name": "Gemma 4 31B (local)"
        }
      }
    }
  }
}
```

3. In OpenCode, run `/models` and select the local model.
