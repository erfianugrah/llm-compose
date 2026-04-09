# llm-compose — local LLM inference stack
# See README.md for full documentation.

# ── Configuration ────────────────────────────────────────────────────
HF_REPO     := ggml-org/gemma-4-26b-a4b-it-GGUF
MMPROJ_FILE := mmproj-gemma-4-26B-A4B-it-f16.gguf
MMPROJ_URL  := https://huggingface.co/$(HF_REPO)/resolve/main/$(MMPROJ_FILE)
VOLUME_DIR  := $(HOME)/docker-volumes/llama-server
MODELS_DIR  := $(VOLUME_DIR)/models
IMAGE       := erfianugrah/llama-server:cuda12.8-sm120

# ── Primary targets ──────────────────────────────────────────────────
.PHONY: setup build up down restart logs status clean help

## First-time setup: generate .env, create volumes, download mmproj, pull or build image
setup: .env dirs mmproj build
	@echo "\n✓ Setup complete. Run 'make up' to start the stack."

## Build the llama-server Docker image (skips if already present)
build:
	@if docker image inspect $(IMAGE) >/dev/null 2>&1; then \
		echo "Image $(IMAGE) already exists (use 'make rebuild' to force)"; \
	else \
		echo "Building $(IMAGE) (~10 min)..."; \
		docker compose build llama-server; \
	fi

## Start the stack in the background
up:
	docker compose up -d

## Stop the stack
down:
	docker compose down

## Restart all services
restart:
	docker compose restart

## Follow logs for all services
logs:
	docker compose logs -f

## Follow logs for llama-server only
logs-llama:
	docker compose logs -f llama-server

## Show container status and health
status:
	@docker compose ps
	@echo ""
	@curl -sf http://localhost:11434/health 2>/dev/null \
		&& echo "llama-server: healthy" \
		|| echo "llama-server: not reachable"

## Stop stack and remove volumes (keeps downloaded models)
clean:
	docker compose down -v

# ── Image management ─────────────────────────────────────────────────
.PHONY: pull push rebuild

## Pull the llama-server image from the registry (skips local build)
pull:
	docker pull $(IMAGE)

## Push the locally built image to the registry
push:
	docker push $(IMAGE)

## Rebuild llama-server from source and restart
rebuild:
	docker compose build llama-server
	docker compose up -d llama-server

## Rebuild from source, push to registry, and restart
release: rebuild push
	@echo "✓ $(IMAGE) built, pushed, and restarted"

# ── Model management ─────────────────────────────────────────────────
.PHONY: mmproj dirs

## Download the multimodal projector (vision support)
mmproj: dirs
	@if [ -f "$(MODELS_DIR)/$(MMPROJ_FILE)" ]; then \
		echo "mmproj already downloaded: $(MODELS_DIR)/$(MMPROJ_FILE)"; \
	else \
		echo "Downloading mmproj (~1.1 GB)..."; \
		curl -L --progress-bar -o "$(MODELS_DIR)/$(MMPROJ_FILE)" "$(MMPROJ_URL)"; \
		echo "✓ Downloaded to $(MODELS_DIR)/$(MMPROJ_FILE)"; \
	fi

## Create persistent volume directories (handles root-owned Docker volumes)
dirs:
	@for d in "$(VOLUME_DIR)" "$(MODELS_DIR)" "$(HOME)/docker-volumes/webui"; do \
		if [ ! -d "$$d" ]; then \
			mkdir -p "$$d" 2>/dev/null || sudo mkdir -p "$$d"; \
		fi; \
		if [ ! -w "$$d" ]; then \
			echo "Fixing ownership on $$d (requires sudo)..."; \
			sudo chown $(shell id -u):$(shell id -g) "$$d"; \
		fi; \
	done

# ── Monitoring ───────────────────────────────────────────────────────
.PHONY: gpu metrics health

## Show GPU utilization, power draw, and VRAM usage
gpu:
	nvidia-smi --query-gpu=utilization.gpu,power.draw,memory.used,memory.total --format=csv

## Fetch Prometheus metrics from llama-server
metrics:
	@curl -sf http://localhost:11434/metrics 2>/dev/null || echo "llama-server not reachable"

## Check llama-server health endpoint
health:
	@curl -sf http://localhost:11434/health 2>/dev/null | python3 -m json.tool 2>/dev/null \
		|| echo "llama-server not reachable"

# ── Utilities ────────────────────────────────────────────────────────

## Generate .env with a random secret key
.env:
	@echo "WEBUI_SECRET_KEY=$$(openssl rand -hex 32)" > .env
	@echo "✓ Created .env"

## Show this help message
help:
	@echo "llm-compose — local LLM inference stack"
	@echo ""
	@echo "Usage: make <target>"
	@echo ""
	@echo "Getting started:"
	@echo "  make setup     First-time setup (env, volumes, mmproj, build)"
	@echo "  make up        Start the stack"
	@echo "  make down      Stop the stack"
	@echo ""
	@echo "Image management:"
	@echo "  make pull      Pull image from registry"
	@echo "  make push      Push image to registry"
	@echo "  make rebuild   Rebuild from source and restart"
	@echo "  make release   Rebuild, push, and restart"
	@echo ""
	@echo "Operations:"
	@echo "  make restart   Restart all services"
	@echo "  make logs      Follow all logs"
	@echo "  make logs-llama  Follow llama-server logs only"
	@echo "  make status    Show container status and health"
	@echo "  make clean     Stop stack and remove volumes"
	@echo ""
	@echo "Monitoring:"
	@echo "  make gpu       Show GPU stats"
	@echo "  make metrics   Fetch Prometheus metrics"
	@echo "  make health    Check llama-server health"
	@echo ""
	@echo "Model management:"
	@echo "  make mmproj    Download multimodal projector (vision)"
