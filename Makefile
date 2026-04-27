# llm-compose — local LLM + image/video inference stack
# See README.md for full documentation.

# ── Configuration ────────────────────────────────────────────────────
MODEL       ?= gemma4
VOLUME_DIR  := $(HOME)/docker-volumes/llama-server
MODELS_DIR  := $(VOLUME_DIR)/models
IMAGE         := erfianugrah/llama-server:cuda12.8-sm120
COMFYUI_IMAGE := erfianugrah/comfyui:cuda12.8-sm120
PROXY_IMAGE   := erfianugrah/model-proxy:latest
PRESET        := models/$(MODEL).env
COMFYUI_DIR   := $(HOME)/docker-volumes/comfyui
TRAIN_IMAGE   := erfianugrah/lora-train:latest
TRAIN_DIR     := $(HOME)/docker-volumes/training-data

# VRAM budget — must match docker-compose.yml proxy env
VRAM_LIMIT   ?= 32
VRAM_RESERVE ?= 6

# ── Primary targets ──────────────────────────────────────────────────
.PHONY: setup build up down restart logs status clean deploy help

## First-time setup: generate secret, create volumes, load default model, build images, download ComfyUI models
setup: .env dirs build setup-comfyui
	@if ! grep -q MODEL_REPO .env 2>/dev/null; then \
		$(MAKE) --no-print-directory switch MODEL=$(MODEL); \
	else \
		echo "Model already configured in .env"; \
	fi
	@echo "\n✓ Setup complete. Run 'make up' to start the stack."
	@echo "  After first start, run 'make configure-webui' to set up Open WebUI models."

## Build all Docker images (skips if already present)
build:
	@if docker image inspect $(IMAGE) >/dev/null 2>&1; then \
		echo "Image $(IMAGE) already exists (use 'make rebuild' to force)"; \
	else \
		echo "Building $(IMAGE) (~10 min)..."; \
		docker compose --profile llm build llama-server; \
	fi
	@if docker image inspect $(COMFYUI_IMAGE) >/dev/null 2>&1; then \
		echo "Image $(COMFYUI_IMAGE) already exists (use 'make rebuild-comfyui' to force)"; \
	else \
		echo "Building $(COMFYUI_IMAGE) (~5 min)..."; \
		docker compose --profile comfyui build comfyui; \
	fi
	@if docker image inspect $(TRAIN_IMAGE) >/dev/null 2>&1; then \
		echo "Image $(TRAIN_IMAGE) already exists (use 'make rebuild-train' to force)"; \
	else \
		echo "Building $(TRAIN_IMAGE) (~5 min)..."; \
		docker compose --profile train build lora-train; \
	fi
	@docker compose build model-proxy

## Start the stack (proxy + Open WebUI). GPU service starts on first request.
up:
	docker compose up -d
	@echo "Proxy + Open WebUI started. GPU service starts on first request."
	@echo "  Force LLM mode:     make llm"
	@echo "  Force ComfyUI mode: make comfyui"
	@echo "  Force train mode:   make train"

## Stop the stack (all profiles)
down:
	docker compose --profile llm --profile comfyui --profile train down

## Restart all running services
restart:
	docker compose --profile llm --profile comfyui --profile train restart

## Follow logs for all services
logs:
	docker compose --profile llm --profile comfyui --profile train logs -f

## Follow logs for llama-server only
logs-llama:
	docker compose logs -f llama-server

## Show container status, active mode, and health
status:
	@docker compose --profile llm --profile comfyui --profile train ps
	@echo ""
	@if grep -q MODEL_NAME .env 2>/dev/null; then \
		echo "Active LLM model: $$(grep MODEL_NAME .env | cut -d= -f2)"; \
	fi
	@curl -sf http://localhost:11434/mode 2>/dev/null \
		| python3 -c "import sys,json; d=json.load(sys.stdin); print(f'GPU mode: {d[\"mode\"] or \"idle\"}')" 2>/dev/null \
		|| echo "Proxy: not reachable"
	@curl -sf http://localhost:11434/health 2>/dev/null \
		| python3 -c "import sys,json; d=json.load(sys.stdin); print(f'Health: {d[\"status\"]}')" 2>/dev/null \
		|| echo "Health: not reachable"

## Full deploy: setup, download all LLM + ComfyUI models, push images, start, configure UI
deploy: setup download-all push up configure-webui
	@echo "\n✓ Deployed. All LLM + ComfyUI models cached, images pushed, Open WebUI configured."

## Stop stack and remove volumes (keeps downloaded models)
clean:
	docker compose --profile llm --profile comfyui --profile train down -v

# ── Model switching ──────────────────────────────────────────────────
.PHONY: switch run models assets download-all

## Switch to a model preset: make switch MODEL=gemma4|qwen3-coder|qwen3
switch: dirs
	@if [ ! -f "$(PRESET)" ]; then \
		echo "Error: preset '$(PRESET)' not found"; \
		echo "Available:"; ls -1 models/*.env | sed 's|models/||;s|\.env||;s|^|  |'; \
		exit 1; \
	fi
	@# VRAM budget check — reject before writing .env
	@ESTIMATE=$$(grep '^VRAM_ESTIMATE_GB=' "$(PRESET)" 2>/dev/null | cut -d= -f2); \
	if [ -n "$$ESTIMATE" ]; then \
		MAX=$$(echo "$(VRAM_LIMIT) - $(VRAM_RESERVE)" | bc); \
		OVER=$$(echo "$$ESTIMATE > $$MAX" | bc); \
		if [ "$$OVER" = "1" ]; then \
			echo "Error: $(MODEL) needs ~$${ESTIMATE}GB VRAM for weights,"; \
			echo "       but only $${MAX}GB available after reserving $(VRAM_RESERVE)GB"; \
			echo "       for KV cache + compute buffer (total VRAM: $(VRAM_LIMIT)GB)."; \
			echo "       Use a smaller quant (e.g. Q4_K_S, UD-IQ4_XS)."; \
			exit 1; \
		fi; \
	else \
		echo "Warning: $(PRESET) missing VRAM_ESTIMATE_GB, skipping budget check"; \
	fi
	@echo "Switching to $(MODEL)..."
	@# Preserve WEBUI_SECRET_KEY, replace everything else
	@SECRET=$$(grep '^WEBUI_SECRET_KEY=' .env 2>/dev/null | head -1); \
	cp "$(PRESET)" .env.tmp; \
	echo "" >> .env.tmp; \
	echo "# Auto-derived asset filenames (based on preset name)" >> .env.tmp; \
	MMPROJ_URL=$$(grep '^MMPROJ_URL=' "$(PRESET)" | cut -d= -f2); \
	if [ -n "$$MMPROJ_URL" ]; then \
		echo "MMPROJ_FILE=$(MODEL)-mmproj.gguf" >> .env.tmp; \
	else \
		echo "MMPROJ_FILE=" >> .env.tmp; \
	fi; \
	TMPL_URL=$$(grep '^TEMPLATE_URL=' "$(PRESET)" | cut -d= -f2); \
	if [ -n "$$TMPL_URL" ]; then \
		echo "TEMPLATE_FILE=$(MODEL)-template.jinja" >> .env.tmp; \
	else \
		echo "TEMPLATE_FILE=" >> .env.tmp; \
	fi; \
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

## Switch model and restart in one shot: make run MODEL=qwen3-coder
run:
	@$(MAKE) --no-print-directory switch MODEL=$(MODEL)
	docker compose up -d
	docker compose --profile llm up -d llama-server

## Download model-specific assets (mmproj, templates) — names auto-derived from preset
assets: dirs
	@# Download mmproj if URL specified — filename is <preset>-mmproj.gguf (set by switch)
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
	@# Download template if URL specified — filename is <preset>-template.jinja (set by switch)
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

## Pre-download all model GGUFs and assets so switching is instant
download-all: dirs
	@for f in models/*.env; do \
		name=$$(basename "$$f" .env); \
		repo=$$(grep '^MODEL_REPO=' "$$f" | cut -d= -f2); \
		file=$$(grep '^MODEL_FILE=' "$$f" | cut -d= -f2); \
		mname=$$(grep '^MODEL_NAME=' "$$f" | cut -d= -f2); \
		echo ""; \
		echo "── $$mname ──"; \
		echo ""; \
		docker run --rm -t --gpus all \
			-v "$(VOLUME_DIR):/root/.cache" \
			-v "$(MODELS_DIR):/models" \
			--entrypoint /bin/sh \
			$(IMAGE) -c " \
				llama-server \
					--hf-repo $$repo \
					--hf-file $$file \
					--port 9999 --host 127.0.0.1 \
					-ngl 0 -c 512 \
					--no-warmup 2>&1 & \
				PID=\$$!; \
				for i in \$$(seq 1 240); do \
					if curl -sf http://127.0.0.1:9999/health >/dev/null 2>&1; then \
						echo 'GGUF cached ✓'; \
						kill \$$PID 2>/dev/null; \
						break; \
					fi; \
					kill -0 \$$PID 2>/dev/null || { echo 'GGUF cached ✓'; break; }; \
					sleep 5; \
				done; \
				kill \$$PID 2>/dev/null; wait \$$PID 2>/dev/null; \
				true \
			"; \
		mmproj_url=$$(grep '^MMPROJ_URL=' "$$f" | cut -d= -f2); \
		if [ -n "$$mmproj_url" ]; then \
			mmproj="$$name-mmproj.gguf"; \
			if [ -f "$(MODELS_DIR)/$$mmproj" ]; then \
				echo "mmproj: $$mmproj (cached)"; \
			else \
				echo "Downloading mmproj: $$mmproj ..."; \
				curl -L --progress-bar -o "$(MODELS_DIR)/$$mmproj" "$$mmproj_url"; \
			fi; \
		fi; \
		tmpl_url=$$(grep '^TEMPLATE_URL=' "$$f" | cut -d= -f2); \
		if [ -n "$$tmpl_url" ]; then \
			tmpl="$$name-template.jinja"; \
			if [ -f "$(MODELS_DIR)/$$tmpl" ]; then \
				echo "template: $$tmpl (cached)"; \
			else \
				echo "Downloading template: $$tmpl ..."; \
				curl -L --progress-bar -o "$(MODELS_DIR)/$$tmpl" "$$tmpl_url"; \
			fi; \
		fi; \
	done
	@echo ""
	@echo "✓ All models pre-downloaded. Switch instantly with: make switch MODEL=<name>"

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

# ── GPU mode switching ────────────────────────────────────────────────
.PHONY: llm comfyui train mode logs-comfyui logs-train

## Switch to LLM mode (stops ComfyUI, starts llama-server)
llm:
	@curl -sf -X POST http://localhost:11434/mode \
		-H 'Content-Type: application/json' \
		-d '{"mode":"llm"}' \
		| python3 -m json.tool 2>/dev/null \
		|| echo "Proxy not reachable. Start with: make up"

## Switch to ComfyUI mode (stops llama-server, starts ComfyUI)
comfyui:
	@curl -sf -X POST http://localhost:11434/mode \
		-H 'Content-Type: application/json' \
		-d '{"mode":"comfyui"}' \
		| python3 -m json.tool 2>/dev/null \
		|| echo "Proxy not reachable. Start with: make up"

## Switch to train mode (stops llama-server/ComfyUI, starts lora-train)
train:
	@curl -sf -X POST http://localhost:11434/mode \
		-H 'Content-Type: application/json' \
		-d '{"mode":"train"}' \
		| python3 -m json.tool 2>/dev/null \
		|| echo "Proxy not reachable. Start with: make up"

## Show current GPU mode (llm, comfyui, train, or idle)
mode:
	@curl -sf http://localhost:11434/mode 2>/dev/null \
		| python3 -m json.tool 2>/dev/null \
		|| echo "Proxy not reachable"

## Follow logs for ComfyUI only
logs-comfyui:
	docker compose --profile comfyui logs -f comfyui

## Follow logs for lora-train only
logs-train:
	docker compose --profile train logs -f lora-train

# ── LoRA training ─────────────────────────────────────────────────────
.PHONY: train-status train-logs deploy-lora rebuild-train

## Show current training job status (step, loss, ETA)
train-status:
	@curl -sf http://localhost:11434/train/status 2>/dev/null \
		| python3 -c "\
import sys, json; d=json.load(sys.stdin); \
s=d.get('state','idle'); \
print(f'State: {s}'); \
[print(f'Progress: {d[\"step\"]}/{d[\"total_steps\"]} ({round(d[\"step\"]/d[\"total_steps\"]*100,1)}%)') if d.get('total_steps',0)>0 else None]; \
[print(f'Epoch: {d[\"epoch\"]}/{d[\"total_epochs\"]}') if d.get('total_epochs',0)>0 else None]; \
[print(f'Loss: {d[\"loss\"]:.6f}') if d.get('loss',0)>0 else None]; \
[print(f'Elapsed: {d[\"elapsed_seconds\"]}s') if 'elapsed_seconds' in d else None]; \
[print(f'ETA: {d[\"eta_seconds\"]}s ({round(d[\"eta_seconds\"]/60,1)} min)') if d.get('eta_seconds') else None]; \
" 2>/dev/null \
		|| echo "Training service not reachable. Switch with: make train"

## Show last 50 lines of training output
train-logs:
	@curl -sf 'http://localhost:11434/train/logs?lines=50' 2>/dev/null \
		| python3 -c "import sys,json; d=json.load(sys.stdin); print('\n'.join(d.get('lines',[])))" 2>/dev/null \
		|| echo "Training service not reachable"

## Copy a trained LoRA to ComfyUI: make deploy-lora NAME=my-lora
deploy-lora:
	@if [ -z "$(NAME)" ]; then \
		echo "Usage: make deploy-lora NAME=<filename>"; \
		echo "Available LoRAs:"; \
		docker run --rm -v "$(TRAIN_DIR)/output:/out" alpine sh -c \
			'ls /out/*.safetensors 2>/dev/null | sed "s|/out/||"' || true; \
		exit 1; \
	fi
	@SRC="$(TRAIN_DIR)/output/$(NAME)"; \
	if [ ! "$${SRC%.safetensors}" != "$$SRC" ]; then SRC="$${SRC}.safetensors"; fi; \
	docker run --rm \
		-v "$(TRAIN_DIR)/output:/src:ro" \
		-v "$(COMFYUI_DIR)/models/loras:/dst" \
		alpine sh -c "cp /src/$(NAME) /dst/ 2>/dev/null || cp /src/$(NAME).safetensors /dst/ 2>/dev/null || echo 'Not found: $(NAME)'"
	@echo "✓ Deployed $(NAME) to $(COMFYUI_DIR)/models/loras/"

## Rebuild lora-train image from source
rebuild-train:
	docker compose --profile train build lora-train

# ── ComfyUI model setup ──────────────────────────────────────────────
.PHONY: setup-comfyui

## Download ComfyUI checkpoints, IP-Adapter, CLIP vision, upscaler (~17 GB)
setup-comfyui: dirs
	@./scripts/setup-comfyui.sh

# ── Open WebUI configuration ─────────────────────────────────────────
.PHONY: configure-webui reset-webui

## Import workspace models (system prompts, params) into Open WebUI
configure-webui:
	@./scripts/init-webui.sh

# ── Benchmarking ─────────────────────────────────────────────────────
.PHONY: bench bench-quick bench-all

## Benchmark current model with different flag combinations
bench:
	@./scripts/bench.sh

## Quick benchmark: old vs new flags only
bench-quick:
	@./scripts/bench.sh --quick

## Benchmark all model presets
bench-all:
	@for f in models/*.env; do \
		name=$$(basename "$$f" .env); \
		BENCH_MODEL=$$name ./scripts/bench.sh --quick; \
		echo ""; \
	done

# ── Image management ─────────────────────────────────────────────────
.PHONY: pull push rebuild rebuild-comfyui release

## Pull all custom images from the registry (skips local build)
pull:
	docker pull $(IMAGE)
	docker pull $(COMFYUI_IMAGE)
	docker pull $(TRAIN_IMAGE)
	docker pull $(PROXY_IMAGE)

## Push all custom images to the registry
push:
	docker push $(IMAGE)
	docker push $(COMFYUI_IMAGE)
	docker push $(TRAIN_IMAGE)
	docker push $(PROXY_IMAGE)

## Rebuild llama-server from source
rebuild:
	docker compose --profile llm build llama-server

## Rebuild ComfyUI from source
rebuild-comfyui:
	docker compose --profile comfyui build comfyui

## Rebuild all, push to registry
release: rebuild rebuild-comfyui rebuild-train push
	@echo "✓ All images built and pushed"

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
	@for d in \
		"$(VOLUME_DIR)" "$(MODELS_DIR)" "$(HOME)/docker-volumes/webui" \
		"$(TRAIN_DIR)" "$(TRAIN_DIR)/datasets" "$(TRAIN_DIR)/configs" \
		"$(TRAIN_DIR)/output" "$(TRAIN_DIR)/raw" \
		"$(COMFYUI_DIR)/models" "$(COMFYUI_DIR)/models/checkpoints" \
		"$(COMFYUI_DIR)/models/clip_vision" "$(COMFYUI_DIR)/models/ipadapter" \
		"$(COMFYUI_DIR)/models/loras" "$(COMFYUI_DIR)/models/vae" \
		"$(COMFYUI_DIR)/models/upscale_models" "$(COMFYUI_DIR)/models/controlnet" \
		"$(COMFYUI_DIR)/output" "$(COMFYUI_DIR)/input" \
		"$(COMFYUI_DIR)/custom_nodes" "$(COMFYUI_DIR)/user"; do \
		if [ ! -d "$$d" ]; then \
			mkdir -p "$$d" 2>/dev/null || sudo mkdir -p "$$d"; \
		fi; \
		if [ ! -w "$$d" ]; then \
			echo "Fixing ownership on $$d..."; \
			sudo chown -R $$(id -u):$$(id -g) "$$d"; \
		fi; \
	done

## Reset Open WebUI database (nuke and recreate on next start)
reset-webui:
	@echo "Stopping Open WebUI..."
	@docker compose stop open-webui 2>/dev/null || true
	@docker run --rm -v "$(HOME)/docker-volumes/webui:/data" alpine sh -c "rm -rf /data/*"
	@mkdir -p "$(HOME)/docker-volumes/webui"
	@echo "✓ WebUI data reset. Run 'make up && make configure-webui' to reconfigure."

## Generate .env with a random secret key (only if .env doesn't exist)
.env:
	@echo "WEBUI_SECRET_KEY=$$(openssl rand -hex 32)" > .env
	@echo "✓ Created .env"

## Show this help message
help:
	@echo "llm-compose — local LLM + image/video inference stack"
	@echo ""
	@echo "Usage: make <target> [MODEL=<name>]"
	@echo ""
	@echo "Getting started:"
	@echo "  make setup              First-time setup: images + LLM preset + ComfyUI models"
	@echo "  make deploy             Full deploy: setup + all LLM models + push + start"
	@echo "  make up                 Start the stack (proxy + Open WebUI)"
	@echo "  make down               Stop the stack"
	@echo ""
	@echo "GPU mode switching (proxy auto-swaps on route, or switch manually):"
	@echo "  make mode               Show current GPU mode (llm/comfyui/train/idle)"
	@echo "  make llm                Switch to LLM mode"
	@echo "  make comfyui            Switch to ComfyUI mode"
	@echo "  make train              Switch to LoRA training mode"
	@echo ""
	@echo "LLM model switching (or just select in OpenCode — proxy auto-swaps):"
	@echo "  make models             List available model presets"
	@echo "  make run MODEL=name     Switch + restart in one shot"
	@echo "  make download-all       Pre-download all LLM GGUFs for instant switching"
	@echo ""
	@echo "LoRA training (use MCP tools or API — proxy auto-swaps GPU):"
	@echo "  make train-status       Show training progress (step, loss, ETA)"
	@echo "  make train-logs         Show last 50 lines of training output"
	@echo "  make logs-train         Follow lora-train container logs"
	@echo "  make deploy-lora NAME=x Copy trained LoRA to ComfyUI"
	@echo "  make rebuild-train      Rebuild lora-train image"
	@echo ""
	@echo "ComfyUI model setup:"
	@echo "  make setup-comfyui      Download checkpoints, IP-Adapter, CLIP, upscaler (~17 GB)"
	@echo ""
	@echo "Image management:"
	@echo "  make pull               Pull all custom images from registry"
	@echo "  make push               Push all custom images to registry"
	@echo "  make rebuild            Rebuild llama-server from source"
	@echo "  make rebuild-comfyui    Rebuild ComfyUI from source"
	@echo "  make release            Rebuild all + push to registry"
	@echo ""
	@echo "Operations:"
	@echo "  make restart            Restart all running services"
	@echo "  make logs               Follow all logs"
	@echo "  make logs-llama         Follow llama-server logs only"
	@echo "  make logs-comfyui       Follow ComfyUI logs only"
	@echo "  make status             Show container status and GPU mode"
	@echo "  make clean              Stop stack and remove volumes"
	@echo ""
	@echo "Open WebUI:"
	@echo "  make configure-webui    Import workspace models + system prompts"
	@echo "  make reset-webui        Nuke WebUI database and start fresh"
	@echo ""
	@echo "Monitoring:"
	@echo "  make gpu                Show GPU stats"
	@echo "  make metrics            Fetch Prometheus metrics"
	@echo "  make health             Check proxy health"
