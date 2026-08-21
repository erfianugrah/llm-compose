# Contributing to llm-compose

Thanks for considering a contribution. This is a personal hobbyist
project tuned for a single RTX 5090, but PRs are welcome — especially
for additional GPU/VRAM tier support, new model presets, better
benchmarking, and operational improvements.

## What's in scope

✅ Welcome:
- New model presets (`models/*.toml`) — particularly for non-32GB VRAM
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
# One-time: add the wrapper to PATH so `llmc` is on $PATH
echo 'export PATH="$HOME/llm-compose/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc

make deploy          # setup + build + up
```

For changes to the proxy / orchestrator / CLI (Python code under `llmc/`):

```bash
make ship-proxy      # build proxy + push to registry + restart container
                     # — or skip the push: `make build-proxy && make restart`
```

`make build-proxy` invokes `docker build` directly; Docker's layer cache
takes care of the heavy `pip install docker` step, so a code-only change
rebuilds the `COPY llmc/` layer in ~1 s.

For changes to a GPU service Dockerfile (heavy):

```bash
make build-llama        # cache-aware, fast if only entrypoint/script changed
make rebuild-llama      # --no-cache, full from-scratch rebuild (~10 min)
```

The `rebuild-X` targets are for when you actually need a clean build
(base image bump, CUDA arch change, dependency pin update). For typical
code edits, `make build-X` lets Docker cache handle incrementality.

## Tests

```bash
make test               # unit + schema (~1s, no Docker)
make test-docker        # + docker daemon integration (~30s)
make test-integration   # + end-to-end GPU swap (~90s, stack up + GPU)
```

Add tests for any new behaviour. Mock the Docker SDK for orchestrator
logic, stub the proxy HTTP for CLI commands. Real-daemon and real-GPU
tests live behind `LLMC_TEST_DOCKER=1` and `LLMC_TEST_INTEGRATION=1`
respectively.

## Validation

Smoke-test the live stack after changes:

```bash
llmc status                              # health + active mode
llmc switch qwen3-vl                     # cheap model swap (~5GB, fastest)
curl http://localhost:11434/v1/models    # list presets via the proxy
```

For benchmark changes:

```bash
llmc bench perf                          # latency + throughput
llmc bench quants                        # quantization comparison
```

## Code style

- **Python**: stdlib-first. The proxy + CLI are intentionally minimal.
  The `docker` SDK is the only third-party dep, and only in the proxy
  image — the host CLI uses subprocess + http.client.
- **Bash**: shellcheck-clean.
- **Comments**: explain *why*, not *what*. Match the existing density
  for non-obvious logic (GPU coordination, mode swapping, swap-time
  edge cases).
- **No AI attribution** in commit messages, comments, or PR bodies.
- **Schemas are strict**: TOML loaders reject unknown keys and wrong
  types. Catch config typos at load time, not at container startup.

## Adding a new model preset

1. Create `models/<id>.toml` based on an existing preset:

   ```bash
   cp models/qwen36.toml models/my-model.toml
   $EDITOR models/my-model.toml
   ```

2. Required: `name`, `vram_gb`, `[model]` with `repo` + `file`.

3. Optional: `[mmproj]` (vision), `[template]` (custom chat template),
   `[runtime]` (context_size, reasoning, sampler params).

4. Validate it parses and the URLs resolve:

   ```bash
   make test                # schema validation runs on every preset
   llmc models              # confirms it shows up
   llmc switch my-model     # actually loads it
   ```

5. Add to the `Available models` table in README.md if you intend
   to upstream the preset.

## Commit messages

Match the existing style — `git log --oneline -20`. Lead with the most
affected component (`proxy:`, `cli:`, `comfyui:`, `train:`, `docs:`,
`bench:`). Body explains *why* over *what*.

## Pull request workflow

1. Fork, create a feature branch off `main`.
2. `make test-docker` (and `make test-integration` if the change
   affects mode swapping or container lifecycle).
3. Open a PR with the same shape as existing commit messages.
4. Expect review focused on: does it preserve single-GPU
   exclusive-mode discipline, does it match the codebase's existing
   patterns, does it ship with appropriate test coverage.

## Reporting bugs

Useful issue includes:
- `llmc status` output
- GPU model + driver version (`nvidia-smi`)
- Relevant log excerpts (`docker logs model_proxy_go --tail 100`)
- The active preset (visible in `llmc status`)
- Reproduction steps

## License

By contributing, you agree your changes ship under the [MIT License](LICENSE)
that covers the rest of the project.
