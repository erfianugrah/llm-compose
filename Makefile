# llm-compose — local LLM + image / video inference + LoRA training stack
#
# Daily use goes through the llmc CLI:
#
#     python3 -m llmc <command>           # full command surface
#     python3 -m llmc --help              # list everything
#
# This Makefile only keeps the small set of targets where `make` is
# objectively more ergonomic than the CLI (running tests, building
# images, common one-liners). Everything else has been migrated to
# `llmc` — `llmc up`, `llmc switch`, `llmc train status`, etc.
#
# v1 reference: the previous 837-line Makefile is preserved as
# Makefile.v1-legacy for the duration of the v2 cutover.

LLMC := python3 -m llmc

# Image tags — keep in sync with images/*.Dockerfile and orchestrator.py
PROXY_IMAGE   := erfianugrah/llmc-proxy:v2
LLAMA_IMAGE   := erfianugrah/llama-server:cuda12.8-sm120
# Pascal / GTX 1070 (sm_61) variant — same Dockerfile, CUDA_ARCH build-arg.
# Built here on the 5090, pulled on servarr (always-on, Jellyfin-coexist).
LLAMA_PASCAL_IMAGE := erfianugrah/llama-server:cuda12.8-sm61
COMFYUI_IMAGE := erfianugrah/comfyui:cuda12.8-sm120
TRAIN_IMAGE   := erfianugrah/lora-train:latest

.PHONY: help setup up down restart status shell test test-docker test-integration \
        build build-proxy build-llama build-llama-pascal build-comfyui build-train \
        rebuild-proxy rebuild-llama rebuild-llama-pascal rebuild-comfyui rebuild-train \
        pull push push-proxy push-llama push-llama-pascal push-comfyui push-train \
        release ship ship-proxy deploy clean \
        logs-proxy logs-webui logs-llama logs-comfyui logs-train \
        gpu health metrics

# ── Stack lifecycle (pure shell — no Python startup) ──────────────────

## First-time setup: generate .env + create named volumes
## Uses llmc because it does schema validation + crypto-random secret gen.
setup:
	@$(LLMC) setup

## Start proxy + Open WebUI. Pre-flights .env and bind directories so the
## error message points at `make setup` instead of compose's cryptic
## "${WEBUI_SECRET_KEY:?...}" or a daemon-side "source path not found".
up:
	@if [ ! -f .env ]; then \
		echo "Missing .env. Run: make setup"; exit 1; fi
	@if [ ! -d $$HOME/docker-volumes/state ]; then \
		echo "Bind directories not created. Run: make setup"; exit 1; fi
	docker compose up -d
	@echo "Stack ready. Proxy at http://localhost:11434"

## Stop the stack and any running GPU service (labelled llmc.mode=...)
down:
	@gpu_ids=$$(docker ps -q --filter "label=llmc.mode"); \
	if [ -n "$$gpu_ids" ]; then \
		echo "Stopping GPU services..."; \
		echo "$$gpu_ids" | xargs docker stop >/dev/null; \
		echo "$$gpu_ids" | xargs docker rm -f >/dev/null; \
	fi
	docker compose down

## Force-recreate proxy + Open WebUI (keep any running GPU service)
restart:
	docker compose up -d --force-recreate model-proxy open-webui

## Show stack + active mode + active model.
## Uses llmc because it queries the proxy's mode endpoint and renders a table.
status:
	@$(LLMC) status

## Open a busybox shell with every named volume mounted at /vol/<name>
shell:
	@$(LLMC) volumes shell

## Stop the stack. The bind directories at $HOME/docker-volumes/* keep
## your GGUFs / LoRAs / WebUI DB on disk — `make down` doesn't touch them.
## To wipe a specific subdir (e.g. WebUI accounts): `llmc webui reset --yes`.
clean: down
	@echo "Stack stopped. Bind data at $$HOME/docker-volumes/ preserved."
	@echo "To wipe specific data: rm -rf $$HOME/docker-volumes/<name>"

# ── Logs (pure docker, no Python startup) ───────────────────────────

## Follow proxy logs
logs-proxy:
	docker logs -f --tail=100 model_proxy

## Follow Open WebUI logs
logs-webui:
	docker logs -f --tail=100 open_webui

## Follow llama-server logs (only when LLM mode is active)
logs-llama:
	docker logs -f --tail=100 llama_server

## Follow ComfyUI logs (only when comfyui mode is active)
logs-comfyui:
	docker logs -f --tail=100 comfyui

## Follow lora-train logs (only when train mode is active)
logs-train:
	docker logs -f --tail=100 lora_train

# ── Quick checks (pure shell) ───────────────────────────────────────

## GPU utilization, power, VRAM
gpu:
	nvidia-smi --query-gpu=utilization.gpu,power.draw,memory.used,memory.total --format=csv

## Proxy health via curl
health:
	@curl -sf http://localhost:11434/health | python3 -m json.tool 2>/dev/null \
		|| echo "Proxy not reachable"

## llama-server Prometheus metrics (when LLM mode active)
metrics:
	@curl -sf http://localhost:11434/metrics 2>/dev/null \
		|| echo "Proxy not reachable or no LLM running"

# ── Tests ────────────────────────────────────────────────────────────

## Run the unit + schema test suite (no Docker required)
test:
	@python3 -m unittest discover llmc.tests

## Run all tests including Docker daemon integration (~30s)
test-docker:
	@LLMC_TEST_DOCKER=1 python3 -m unittest discover llmc.tests

## Run end-to-end GPU integration tests (requires stack up + GPU + ~90s)
test-integration:
	@LLMC_TEST_INTEGRATION=1 python3 -m unittest discover llmc.tests

# ── Image builds ─────────────────────────────────────────────────────
#
# All build targets call `docker build` directly — no "skip if exists"
# check. Docker's BuildKit layer cache is what decides whether to rerun
# each step. If nothing changed in the Dockerfile or build context, the
# build resolves in <1 s (just metadata). If the entrypoint script or
# llmc/ source changed, only the affected layers rebuild — for the
# llama-server image that's typically just the COPY + chmod layers, not
# the 10-minute CUDA + llama.cpp compile.
#
# The previous `docker image inspect && skip` guard was misleading:
# it skipped rebuilds even when the source had changed.

## Build all images (Docker's layer cache makes this fast if unchanged)
build: build-proxy build-llama build-comfyui build-train

build-proxy:
	docker build -t $(PROXY_IMAGE) -f images/proxy.Dockerfile .

build-llama:
	docker build -t $(LLAMA_IMAGE) -f llama-server.Dockerfile .

## Pascal/sm_61 build for servarr's GTX 1070 (cross-compiled on the 5090).
build-llama-pascal:
	docker build --build-arg CUDA_ARCH=61 -t $(LLAMA_PASCAL_IMAGE) -f llama-server.Dockerfile .

build-comfyui:
	docker build -t $(COMFYUI_IMAGE) -f comfyui.Dockerfile .

build-train:
	docker build -t $(TRAIN_IMAGE) -f lora-train.Dockerfile .

## Force a full rebuild of a single image (busts Docker's layer cache).
## Use when you actually need to recompile (CUDA flags, llama.cpp version
## bump) — for "I changed a Python file" just `make build-proxy`.
rebuild-proxy:
	docker build --no-cache -t $(PROXY_IMAGE) -f images/proxy.Dockerfile .

rebuild-llama:
	docker build --no-cache -t $(LLAMA_IMAGE) -f llama-server.Dockerfile .

rebuild-llama-pascal:
	docker build --no-cache --build-arg CUDA_ARCH=61 -t $(LLAMA_PASCAL_IMAGE) -f llama-server.Dockerfile .

rebuild-comfyui:
	docker build --no-cache -t $(COMFYUI_IMAGE) -f comfyui.Dockerfile .

rebuild-train:
	docker build --no-cache -t $(TRAIN_IMAGE) -f lora-train.Dockerfile .

# ── Registry ─────────────────────────────────────────────────────────

## Pull all pre-built images from the registry
pull:
	docker pull $(PROXY_IMAGE)
	docker pull $(LLAMA_IMAGE)
	docker pull $(COMFYUI_IMAGE)
	docker pull $(TRAIN_IMAGE)

## Push all custom images to the registry
push: push-proxy push-llama push-comfyui push-train

push-proxy:
	docker push $(PROXY_IMAGE)

push-llama:
	docker push $(LLAMA_IMAGE)

push-llama-pascal:
	docker push $(LLAMA_PASCAL_IMAGE)

push-comfyui:
	docker push $(COMFYUI_IMAGE)

push-train:
	docker push $(TRAIN_IMAGE)

## Build all images + push + restart proxy + WebUI so the running stack
## picks up the new proxy image. Won't touch llama-server / comfyui /
## lora-train containers — those are spawned on demand by the proxy and
## the next `llmc switch` / `llmc mode X` will use the freshly-pushed
## image automatically.
release: build push restart
	@echo "All images built, pushed, and stack restarted"

## Alias for release — old muscle-memory shortcut
ship: release

## Ship just the proxy (common case: llmc/ source changed). Skips the
## ~14 GB of llama-server + comfyui + lora-train pushes that are no-ops
## on every layer when only Python changed.
ship-proxy: build-proxy push-proxy restart
	@echo "Proxy shipped and restarted"

## Full bootstrap: setup + build all + start
deploy: setup build up
	@echo "Stack deployed. Configure WebUI: llmc webui configure"

# ── Help ─────────────────────────────────────────────────────────────

help:
	@echo "llm-compose v2 — 'llmc --help' for the full CLI surface."
	@echo ""
	@echo "Stack lifecycle:"
	@echo "  make setup           First-time: generate .env + create volumes"
	@echo "  make deploy          Full bootstrap: setup + build + up"
	@echo "  make up              Start proxy + Open WebUI"
	@echo "  make down            Stop the stack (incl. any running GPU service)"
	@echo "  make restart         Recreate proxy + WebUI (keep GPU service)"
	@echo "  make status          Show stack + GPU mode + active model"
	@echo "  make shell           Busybox with every named volume at /vol/<name>"
	@echo "  make clean           Stop + remove volumes (preserves bind-mount data)"
	@echo ""
	@echo "Logs (direct docker — no Python startup):"
	@echo "  make logs-proxy      Follow proxy logs"
	@echo "  make logs-webui      Follow Open WebUI logs"
	@echo "  make logs-llama      Follow llama-server logs"
	@echo "  make logs-comfyui    Follow ComfyUI logs"
	@echo "  make logs-train      Follow lora-train logs"
	@echo ""
	@echo "Quick checks:"
	@echo "  make gpu             nvidia-smi: utilization, power, VRAM"
	@echo "  make health          curl /health"
	@echo "  make metrics         curl /metrics (llama-server Prometheus)"
	@echo ""
	@echo "Tests:"
	@echo "  make test            Unit + schema tests (~1s, no Docker)"
	@echo "  make test-docker     + Docker daemon integration (~30s)"
	@echo "  make test-integration  + end-to-end GPU tests (~90s, needs stack up)"
	@echo ""
	@echo "Images (Docker's layer cache makes incremental builds fast):"
	@echo "  make build           docker build all 4 (cache-aware, ~5s if unchanged)"
	@echo "  make build-proxy     just the proxy (use for llmc/ source changes)"
	@echo "  make build-{llama,comfyui,train}  single-image variants"
	@echo "  make rebuild-X       --no-cache (slow — for base image bumps, etc.)"
	@echo ""
	@echo "Registry:"
	@echo "  make pull            pull all 4 images from Docker Hub"
	@echo "  make push            push all 4"
	@echo "  make push-X          push just one image"
	@echo "  make release / ship  build all + push all + restart stack"
	@echo "  make ship-proxy      build proxy + push proxy + restart (daily flow)"
	@echo ""
	@echo "Operations (use the CLI):"
	@echo "  llmc switch <preset>   Hot-swap LLM model"
	@echo "  llmc mode <m>          Switch GPU mode (llm | comfyui | train)"
	@echo "  llmc models            List available presets"
	@echo "  llmc train status      Training job progress"
	@echo "  llmc dataset caption x Start a captioning job"
	@echo "  llmc eval quicktest    LoRA eval pass-through"
	@echo "  llmc bench perf        Benchmark pass-through"
	@echo "  llmc volumes refresh   Fix Docker Desktop bind-mount snapshot rot"
