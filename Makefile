# llm-compose — local LLM inference stack
# See README.md for full documentation.

# ── Configuration ────────────────────────────────────────────────────
MODEL       ?= gemma4
VOLUME_DIR  := $(HOME)/docker-volumes/llama-server
MODELS_DIR  := $(VOLUME_DIR)/models
IMAGE       := erfianugrah/llama-server:cuda12.8-sm120
PRESET      := models/$(MODEL).env

# ── Primary targets ──────────────────────────────────────────────────
.PHONY: setup build up down restart logs status clean help

## First-time setup: generate secret, create volumes, load default model, pull or build image
setup: .env dirs build
	@if ! grep -q MODEL_REPO .env 2>/dev/null; then \
		$(MAKE) --no-print-directory switch MODEL=$(MODEL); \
	else \
		echo "Model already configured in .env"; \
	fi
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
	@if grep -q MODEL_NAME .env 2>/dev/null; then \
		echo "Active model: $$(grep MODEL_NAME .env | cut -d= -f2)"; \
	fi
	@curl -sf http://localhost:11434/health 2>/dev/null \
		&& echo "llama-server: healthy" \
		|| echo "llama-server: not reachable"

## Stop stack and remove volumes (keeps downloaded models)
clean:
	docker compose down -v

# ── Model switching ──────────────────────────────────────────────────
.PHONY: switch models assets

## Switch to a model preset: make switch MODEL=gemma4|qwen3-coder|qwen3
switch: dirs
	@if [ ! -f "$(PRESET)" ]; then \
		echo "Error: preset '$(PRESET)' not found"; \
		echo "Available:"; ls -1 models/*.env | sed 's|models/||;s|\.env||;s|^|  |'; \
		exit 1; \
	fi
	@echo "Switching to $(MODEL)..."
	@# Preserve WEBUI_SECRET_KEY, replace everything else
	@SECRET=$$(grep '^WEBUI_SECRET_KEY=' .env 2>/dev/null | head -1); \
	cp "$(PRESET)" .env.tmp; \
	if [ -n "$$SECRET" ]; then \
		echo "" >> .env.tmp; \
		echo "$$SECRET" >> .env.tmp; \
	else \
		echo "" >> .env.tmp; \
		echo "WEBUI_SECRET_KEY=$$(openssl rand -hex 32)" >> .env.tmp; \
	fi; \
	mv .env.tmp .env
	@$(MAKE) --no-print-directory assets
	@echo "✓ Switched to $(MODEL). Run 'make up' to start."

## Download model-specific assets (mmproj, templates)
assets: dirs
	@# Download mmproj if specified
	@MMPROJ=$$(grep '^MMPROJ_FILE=' .env 2>/dev/null | cut -d= -f2); \
	MMPROJ_URL=$$(grep '^MMPROJ_URL=' .env 2>/dev/null | cut -d= -f2); \
	if [ -n "$$MMPROJ" ] && [ -n "$$MMPROJ_URL" ]; then \
		if [ -f "$(MODELS_DIR)/$$MMPROJ" ]; then \
			echo "mmproj: $$MMPROJ (cached)"; \
		else \
			echo "Downloading mmproj: $$MMPROJ ..."; \
			curl -L --progress-bar -o "$(MODELS_DIR)/$$MMPROJ" "$$MMPROJ_URL"; \
		fi; \
	fi
	@# Download template if specified
	@TMPL=$$(grep '^TEMPLATE_FILE=' .env 2>/dev/null | cut -d= -f2); \
	TMPL_URL=$$(grep '^TEMPLATE_URL=' .env 2>/dev/null | cut -d= -f2); \
	if [ -n "$$TMPL" ] && [ -n "$$TMPL_URL" ]; then \
		if [ -f "$(MODELS_DIR)/$$TMPL" ]; then \
			echo "template: $$TMPL (cached)"; \
		else \
			echo "Downloading template: $$TMPL ..."; \
			curl -L --progress-bar -o "$(MODELS_DIR)/$$TMPL" "$$TMPL_URL"; \
		fi; \
	fi

## List available model presets
models:
	@echo "Available models:"
	@for f in models/*.env; do \
		name=$$(basename "$$f" .env); \
		desc=$$(head -1 "$$f" | sed 's/^# //'); \
		printf "  %-16s %s\n" "$$name" "$$desc"; \
	done
	@echo ""
	@if grep -q MODEL_NAME .env 2>/dev/null; then \
		echo "Active: $$(grep MODEL_NAME .env | cut -d= -f2)"; \
	fi
	@echo ""
	@echo "Switch with: make switch MODEL=<name>"

# ── Image management ─────────────────────────────────────────────────
.PHONY: pull push rebuild release

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

## Create persistent volume directories (handles root-owned Docker volumes)
dirs:
	@for d in "$(VOLUME_DIR)" "$(MODELS_DIR)" "$(HOME)/docker-volumes/webui"; do \
		if [ ! -d "$$d" ]; then \
			mkdir -p "$$d" 2>/dev/null || sudo mkdir -p "$$d"; \
		fi; \
		if [ ! -w "$$d" ]; then \
			echo "Fixing ownership on $$d (requires sudo)..."; \
			sudo chown $$(id -u):$$(id -g) "$$d"; \
		fi; \
	done

## Generate .env with a random secret key (only if .env doesn't exist)
.env:
	@echo "WEBUI_SECRET_KEY=$$(openssl rand -hex 32)" > .env
	@echo "✓ Created .env"

## Show this help message
help:
	@echo "llm-compose — local LLM inference stack"
	@echo ""
	@echo "Usage: make <target> [MODEL=<name>]"
	@echo ""
	@echo "Getting started:"
	@echo "  make setup              First-time setup (default: gemma4)"
	@echo "  make up                 Start the stack"
	@echo "  make down               Stop the stack"
	@echo ""
	@echo "Model switching:"
	@echo "  make models             List available model presets"
	@echo "  make switch MODEL=name  Switch model and download assets"
	@echo "  make up                 Restart with the new model"
	@echo ""
	@echo "Image management:"
	@echo "  make pull               Pull image from registry"
	@echo "  make push               Push image to registry"
	@echo "  make rebuild            Rebuild from source and restart"
	@echo "  make release            Rebuild, push, and restart"
	@echo ""
	@echo "Operations:"
	@echo "  make restart            Restart all services"
	@echo "  make logs               Follow all logs"
	@echo "  make logs-llama         Follow llama-server logs only"
	@echo "  make status             Show container status and health"
	@echo "  make clean              Stop stack and remove volumes"
	@echo ""
	@echo "Monitoring:"
	@echo "  make gpu                Show GPU stats"
	@echo "  make metrics            Fetch Prometheus metrics"
	@echo "  make health             Check llama-server health"
