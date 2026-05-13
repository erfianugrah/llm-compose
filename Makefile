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
COMFYUI_IMAGE := erfianugrah/comfyui:cuda12.8-sm120
TRAIN_IMAGE   := erfianugrah/lora-train:latest

.PHONY: help setup up down restart status shell test test-docker test-integration \
        build build-proxy build-llama build-comfyui build-train \
        rebuild-proxy rebuild-llama rebuild-comfyui rebuild-train \
        pull push release ship deploy clean \
        logs-proxy logs-webui logs-llama logs-comfyui logs-train \
        gpu health metrics

# ── Stack lifecycle ──────────────────────────────────────────────────

## First-time setup: generate .env + create named volumes
setup:
	@$(LLMC) setup

## Start the stack (proxy + Open WebUI)
up:
	@$(LLMC) up

## Stop the stack and any running GPU service
down:
	@$(LLMC) down

## Restart proxy + Open WebUI (force-recreate, keeps any running GPU service)
restart:
	docker compose up -d --force-recreate model-proxy open-webui

## Show stack + active mode + active model
status:
	@$(LLMC) status

## Open a busybox shell with every named volume mounted at /vol/<name>
shell:
	@$(LLMC) volumes shell

## Stop the stack and remove named volumes (DOES NOT delete bind-mount data)
clean:
	@$(LLMC) down
	@echo "Removing named volumes (bind-mount data at the device paths is preserved)..."
	@docker volume ls -q --filter "name=llmc-" | xargs -r docker volume rm

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

## Build any missing images (skip if already present)
build:
	@if ! docker image inspect $(PROXY_IMAGE) >/dev/null 2>&1; then \
		echo "Building $(PROXY_IMAGE)..."; \
		docker build -t $(PROXY_IMAGE) -f images/proxy.Dockerfile .; \
	else \
		echo "$(PROXY_IMAGE) already built (make rebuild-proxy to force)"; \
	fi
	@if ! docker image inspect $(LLAMA_IMAGE) >/dev/null 2>&1; then \
		echo "Building $(LLAMA_IMAGE) (~10 min)..."; \
		docker build -t $(LLAMA_IMAGE) -f llama-server.Dockerfile .; \
	else \
		echo "$(LLAMA_IMAGE) already built (make rebuild-llama to force)"; \
	fi
	@if ! docker image inspect $(COMFYUI_IMAGE) >/dev/null 2>&1; then \
		echo "Building $(COMFYUI_IMAGE) (~5 min)..."; \
		docker build -t $(COMFYUI_IMAGE) -f comfyui.Dockerfile .; \
	else \
		echo "$(COMFYUI_IMAGE) already built (make rebuild-comfyui to force)"; \
	fi
	@if ! docker image inspect $(TRAIN_IMAGE) >/dev/null 2>&1; then \
		echo "Building $(TRAIN_IMAGE) (~5 min)..."; \
		docker build -t $(TRAIN_IMAGE) -f lora-train.Dockerfile .; \
	else \
		echo "$(TRAIN_IMAGE) already built (make rebuild-train to force)"; \
	fi

build-proxy:
	docker build -t $(PROXY_IMAGE) -f images/proxy.Dockerfile .

build-llama:
	docker build -t $(LLAMA_IMAGE) -f llama-server.Dockerfile .

build-comfyui:
	docker build -t $(COMFYUI_IMAGE) -f comfyui.Dockerfile .

build-train:
	docker build -t $(TRAIN_IMAGE) -f lora-train.Dockerfile .

rebuild-proxy: build-proxy
rebuild-llama: build-llama
rebuild-comfyui: build-comfyui
rebuild-train: build-train

# ── Registry ─────────────────────────────────────────────────────────

## Pull pre-built images from the registry (skip local build)
pull:
	docker pull $(PROXY_IMAGE)
	docker pull $(LLAMA_IMAGE)
	docker pull $(COMFYUI_IMAGE)
	docker pull $(TRAIN_IMAGE)

## Push all custom images to the registry
push:
	docker push $(PROXY_IMAGE)
	docker push $(LLAMA_IMAGE)
	docker push $(COMFYUI_IMAGE)
	docker push $(TRAIN_IMAGE)

## Build all images and push to the registry
release: build push
	@echo "All images built and pushed"

## Alias for release — old muscle-memory shortcut
ship: release

## Full bootstrap: setup volumes + build images + start stack
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
	@echo "Images:"
	@echo "  make build           Build any missing images"
	@echo "  make rebuild-proxy   Force-rebuild a single image"
	@echo "  make pull / push     Sync with the registry"
	@echo "  make release / ship  Build all + push"
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
