# Contributing to llm-compose

Thanks for considering a contribution. This is a personal hobbyist
project tuned for a single RTX 5090, but PRs are welcome — especially
for additional GPU/VRAM tier support, new model presets, better
benchmarking, and operational improvements.

## What's in scope

✅ Welcome:
- New model presets (`models/*.env`) — particularly for non-32GB VRAM
  tiers (12, 16, 24, 48, 80 GB)
- Bug fixes with a clear repro path
- Better default proxy behaviour (mode swapping, GGUF caching, health checks)
- Additional benchmark suites (`bench/`) — anything reproducible
- Better Open WebUI / OpenCode integration patterns
- Documentation, README clarifications, examples
- Compose tweaks that improve portability (more env-driven config,
  fewer hardcoded paths)

❓ Discuss before building:
- Major refactors (open an issue first)
- Switching backends (e.g. swapping llama.cpp for vLLM as default) —
  worth aligning on tradeoffs
- Multi-GPU support — non-trivial; the proxy currently assumes one
  GPU exclusively held at a time

❌ Out of scope:
- Hosted-service offerings, multi-tenant deployments
- Closed-source models / proprietary backends
- Apple Silicon (M-series) support — entirely different toolchain;
  worth its own project

## Hardware assumptions

This stack is built for and tested on:

- **RTX 5090** (32 GB VRAM, sm_120 / Blackwell)
- **WSL2** on Windows or native Linux
- Docker Engine (not Docker Desktop)

Build args target `CMAKE_CUDA_ARCHITECTURES=120` by default; PRs
adding support for other architectures (sm_86 Ampere, sm_89 Ada, etc.)
are welcome — see the Dockerfiles for the targeting flags.

## Development setup

```bash
git clone https://github.com/erfianugrah/llm-compose.git
cd llm-compose
make setup           # one-time: .env, volumes, image build, default model
make up              # start the stack
```

For changes to `proxy/proxy.py`:

```bash
make rebuild-proxy && docker compose up -d --no-deps model-proxy
```

For changes to a Dockerfile (heavy):

```bash
make rebuild           # llama-server (~10 min)
make rebuild-comfyui   # ComfyUI (~5 min)
make rebuild-train     # lora-train (~5 min)
```

## Validation

Smoke-test the proxy after changes:

```bash
make status                                 # current GPU mode + health
curl http://localhost:11434/v1/models       # list available models
curl -X POST http://localhost:11434/mode \
    -H 'Content-Type: application/json' \
    -d '{"mode":"llm"}'                     # explicit mode switch
```

For benchmark changes:

```bash
make bench-perf MODEL=qwen35       # latency + throughput sweep
make bench-quants                  # quantization comparison
```

## Code style

- **Python**: stdlib-first; the proxy is intentionally minimal. New
  deps need clear justification.
- **Bash**: shellcheck-clean; entrypoints assume non-root where
  possible.
- **Comments** explain *why*, not *what* — match the existing density
  for non-obvious logic (GPU coordination, mode swapping, network
  configuration).
- **No AI attribution** in commit messages, comments, or PR bodies.
- **Compose**: prefer env-var defaults (`${VAR:-default}`) over
  hardcoded values; new top-level vars get documented in `.env.example`.

## Adding a new model preset

1. Create `models/<id>.env` based on an existing preset:

   ```bash
   cp models/qwen35.env models/my-model.env
   $EDITOR models/my-model.env
   ```

2. Required fields: `MODEL_REPO`, `MODEL_FILE`, `MODEL_NAME`,
   `VRAM_ESTIMATE_GB`, `CONTEXT_SIZE`, sampler params.

3. Optional: `MMPROJ_URL` (vision models), `TEMPLATE_URL` (custom
   chat templates).

4. Validate the URLs resolve and the model fits VRAM:

   ```bash
   make switch MODEL=my-model
   make status
   ```

5. Add to the `Available models` table in README.md if you intend
   to upstream the preset.

## Commit messages

Match the existing style — run `git log --oneline -20`. Lead with
the most affected component (`Proxy:`, `ComfyUI:`, `Train:`,
`Compose:`, `Docs:`, `Bench:`). Body explains *why* over *what*.

## Pull request workflow

1. Fork, create a feature branch off `main`.
2. Validate with `docker compose config -q` before pushing.
3. Open a PR with the same shape as existing commit messages.
4. Expect review focused on: does it preserve single-GPU
   exclusive-mode discipline, does it match the codebase's existing
   patterns, does it ship with appropriate validation steps.

## Reporting bugs

Useful issue includes:
- Output of `make status`
- GPU model + driver version (`nvidia-smi`)
- Relevant log excerpts (`docker compose logs --tail 100 model-proxy`)
- The active model preset (`grep MODEL_NAME .env`)
- Reproduction steps

## License

By contributing, you agree your changes ship under the [MIT License](LICENSE)
that covers the rest of the project.
