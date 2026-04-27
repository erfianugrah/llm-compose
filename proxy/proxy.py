#!/usr/bin/env python3
"""
Multi-backend GPU proxy for llm-compose.

Routes requests to either llama-server (LLM inference) or ComfyUI
(image/video generation) based on URL path. Only one GPU workload
runs at a time — the proxy manages container lifecycle via Docker
socket, stopping one service before starting the other.

Route table:
  /v1/*        → llama-server  (auto-starts LLM mode)
  /comfyui/*   → ComfyUI       (auto-starts ComfyUI mode, strips prefix)
  /health      → proxy self
  /v1/models   → proxy self (preset list)
  /mode        → GET current mode / POST switch mode
"""

import http.server
import http.client
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────
LLAMA_HOST = os.environ.get("LLAMA_HOST", "llama-server")
LLAMA_PORT = int(os.environ.get("LLAMA_PORT", "8080"))
COMFYUI_HOST = os.environ.get("COMFYUI_HOST", "comfyui")
COMFYUI_PORT = int(os.environ.get("COMFYUI_PORT", "8188"))
TRAIN_HOST = os.environ.get("TRAIN_HOST", "lora-train")
TRAIN_PORT = int(os.environ.get("TRAIN_PORT", "8787"))
PROXY_PORT = int(os.environ.get("PROXY_PORT", "11434"))
PRESETS_DIR = Path(os.environ.get("PRESETS_DIR", "/presets"))
PROJECT_DIR = Path(os.environ.get("PROJECT_DIR", "/project"))
ASSETS_DIR = Path(os.environ.get("ASSETS_DIR", "/assets"))
HEALTH_TIMEOUT = int(os.environ.get("HEALTH_TIMEOUT", "900"))
VRAM_LIMIT_GB = float(os.environ.get("VRAM_LIMIT_GB", "32"))
VRAM_RESERVE_GB = float(os.environ.get("VRAM_RESERVE_GB", "6"))
COMPOSE_PROJECT = os.environ.get("COMPOSE_PROJECT_NAME", "llm-compose")

# ── State ────────────────────────────────────────────────────────────
# active_mode: "llm" | "comfyui" | None (nothing running)
active_mode = None
current_model_id = None
switch_lock = threading.Lock()
switching = False


# ── Helpers ──────────────────────────────────────────────────────────
def log(msg):
    print(f"[model-proxy] {msg}", flush=True)


def _compose_env():
    """Build env dict for docker compose subprocess calls.
    Sets HOME to host HOME so ~ in volume paths resolves correctly."""
    env = os.environ.copy()
    host_home = os.environ.get("HOST_HOME")
    if host_home:
        env["HOME"] = host_home
    return env


def _compose_run(args, timeout=30):
    """Run a docker compose command. Returns (success, stderr)."""
    cmd = ["docker", "compose", "-p", COMPOSE_PROJECT, *args]
    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_DIR),
        env=_compose_env(),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        log(f"docker compose {' '.join(args)} failed: {result.stderr.strip()}")
    return result.returncode == 0, result.stderr


def _json_response(handler, status, data):
    """Send a JSON response."""
    body = json.dumps(data).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _wait_healthy(host, port, path, timeout):
    """Poll a health endpoint until 200 or timeout. Returns True on success."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            conn = http.client.HTTPConnection(host, port, timeout=3)
            conn.request("GET", path)
            resp = conn.getresponse()
            body = resp.read()
            conn.close()
            if resp.status == 200:
                # llama-server returns {"status": "ok"}, ComfyUI returns stats JSON
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


# ── Preset loading ───────────────────────────────────────────────────
def parse_env_file(path):
    """Parse a KEY=VALUE env file. Also extracts the first comment line
    as a description (used by /v1/models for Open WebUI metadata)."""
    config = {}
    description = ""
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            # First comment = description (strip "# " prefix)
            if not description:
                description = line.lstrip("# ").strip()
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            config[key.strip()] = value.strip()
    return config, description


def load_presets():
    """Build model_id -> preset mapping from models/*.env files."""
    presets = {}
    for f in sorted(PRESETS_DIR.glob("*.env")):
        config, description = parse_env_file(f)
        model_file = config.get("MODEL_FILE", "")
        # Model ID = GGUF filename without extension (matches OpenCode config key)
        model_id = model_file.rsplit(".", 1)[0] if model_file else f.stem
        presets[model_id] = {
            "preset": f.stem,
            "config": config,
            "model_id": model_id,
            "description": description,
        }
    return presets


def detect_current_model():
    """Read .env to determine which model is currently configured."""
    env_file = PROJECT_DIR / ".env"
    if not env_file.exists():
        return None
    config, _ = parse_env_file(env_file)
    model_file = config.get("MODEL_FILE", "")
    return model_file.rsplit(".", 1)[0] if model_file else None


# ── VRAM budget check ────────────────────────────────────────────────
def check_vram_budget(preset_info):
    """Return (ok, message). Rejects if model weights would leave
    insufficient VRAM for KV cache and CUDA overhead."""
    estimate = preset_info["config"].get("VRAM_ESTIMATE_GB")
    if estimate is None:
        # No estimate in preset — allow but warn
        log(f"WARNING: preset '{preset_info['preset']}' missing VRAM_ESTIMATE_GB, skipping budget check")
        return True, ""
    try:
        estimate = float(estimate)
    except ValueError:
        log(f"WARNING: invalid VRAM_ESTIMATE_GB='{estimate}' in preset '{preset_info['preset']}'")
        return True, ""

    max_weight = VRAM_LIMIT_GB - VRAM_RESERVE_GB
    if estimate > max_weight:
        msg = (
            f"Model '{preset_info['config'].get('MODEL_NAME', preset_info['model_id'])}' "
            f"needs ~{estimate}GB VRAM for weights alone, "
            f"but only {max_weight}GB available after reserving "
            f"{VRAM_RESERVE_GB}GB for KV cache + compute buffer "
            f"(total VRAM: {VRAM_LIMIT_GB}GB). "
            f"Use a smaller quant (e.g. Q4_K_S, UD-IQ4_XS)."
        )
        log(f"REJECTED: {msg}")
        return False, msg
    return True, ""


# ── Message normalization ────────────────────────────────────────────
def _needs_system_merge(messages):
    """True if there are 2+ system messages, OR any system message appears
    after a non-system message. Both cases trigger the Qwen template's
    'System message must be at the beginning' exception."""
    system_count = sum(1 for m in messages if isinstance(m, dict) and m.get("role") == "system")
    if system_count < 2:
        # Check for out-of-order single system
        for i, m in enumerate(messages):
            if not isinstance(m, dict):
                continue
            if m.get("role") == "system" and i > 0 and messages[i - 1].get("role") != "system":
                return True
        return False
    return True


def _merge_system_messages(messages):
    """Collapse all system messages into one at position 0. Content is joined
    with blank lines so each original message remains readable."""
    system_parts = []
    others = []
    for m in messages:
        if not isinstance(m, dict):
            others.append(m)
            continue
        if m.get("role") == "system":
            content = m.get("content", "")
            if isinstance(content, str) and content.strip():
                system_parts.append(content)
            elif isinstance(content, list):
                # Multimodal content array — join text parts only
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        system_parts.append(part.get("text", ""))
        else:
            others.append(m)
    if not system_parts:
        return others
    merged = {"role": "system", "content": "\n\n".join(system_parts)}
    return [merged, *others]


# ── Asset download ───────────────────────────────────────────────────
def _ensure_asset(filename, url, kind):
    """Download mmproj/template to ASSETS_DIR if missing. Called before
    recreating llama-server so the new model's assets are ready.
    Raises RuntimeError if download fails so the caller can abort the swap."""
    if not filename or not url:
        return
    dest = ASSETS_DIR / filename
    if dest.exists():
        return
    log(f"Downloading {kind}: {filename}")
    import urllib.request
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "llm-compose/proxy"})
        with urllib.request.urlopen(req, timeout=30) as response:
            with tmp.open("wb") as f:
                while True:
                    chunk = response.read(1 << 20)  # 1 MiB
                    if not chunk:
                        break
                    f.write(chunk)
        tmp.rename(dest)
        log(f"Downloaded {kind}: {filename}")
    except Exception as exc:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(f"download failed for {kind} at {url}: {exc}") from exc


# ── Mode switching (GPU exclusivity) ─────────────────────────────────
def _stop_service(service, profile):
    """Stop a profiled service. Tolerates already-stopped."""
    log(f"Stopping {service}...")
    _compose_run(["--profile", profile, "stop", service])


def _start_service(service, profile):
    """Start a profiled service."""
    log(f"Starting {service}...")
    ok, _ = _compose_run(["--profile", profile, "up", "-d", service])
    return ok


def ensure_mode(target):
    """Ensure the target mode ("llm", "comfyui", or "train") is active.
    Stops the current GPU service first. Returns True on success.
    Caller must hold switch_lock."""
    global active_mode, switching

    if active_mode == target:
        return True

    switching = True
    try:
        # Stop current GPU service
        if active_mode == "llm":
            _stop_service("llama-server", "llm")
        elif active_mode == "comfyui":
            _stop_service("comfyui", "comfyui")
        elif active_mode == "train":
            _stop_service("lora-train", "train")

        # Start target GPU service
        if target == "llm":
            ok = _start_service("llama-server", "llm")
            if not ok:
                return False
            log("Waiting for llama-server to become healthy...")
            if not _wait_healthy(LLAMA_HOST, LLAMA_PORT, "/health", HEALTH_TIMEOUT):
                log("Timeout waiting for llama-server")
                return False
        elif target == "comfyui":
            ok = _start_service("comfyui", "comfyui")
            if not ok:
                return False
            log("Waiting for ComfyUI to become healthy...")
            if not _wait_healthy(COMFYUI_HOST, COMFYUI_PORT, "/system_stats", HEALTH_TIMEOUT):
                log("Timeout waiting for ComfyUI")
                return False
        elif target == "train":
            ok = _start_service("lora-train", "train")
            if not ok:
                return False
            log("Waiting for lora-train to become healthy...")
            if not _wait_healthy(TRAIN_HOST, TRAIN_PORT, "/health", 120):
                log("Timeout waiting for lora-train")
                return False
        else:
            log(f"Unknown mode: {target}")
            return False

        active_mode = target
        log(f"Mode: {target}")
        return True
    finally:
        switching = False


# ── Model switching (within LLM mode) ────────────────────────────────
def switch_model(preset_info):
    """Update .env with new model preset and recreate llama-server.
    Assumes LLM mode is already active (or will be activated)."""
    global current_model_id, switching
    switching = True
    preset_name = preset_info["preset"]
    model_name = preset_info["config"].get("MODEL_NAME", preset_name)
    log(f"Switching to {model_name}...")

    try:
        preset_file = PRESETS_DIR / f"{preset_name}.env"
        env_file = PROJECT_DIR / ".env"

        cfg = preset_info["config"]
        mmproj_url = cfg.get("MMPROJ_URL", "").strip()
        tmpl_url = cfg.get("TEMPLATE_URL", "").strip()
        mmproj_file = f"{preset_name}-mmproj.gguf" if mmproj_url else ""
        tmpl_file = f"{preset_name}-template.jinja" if tmpl_url else ""

        # Download assets BEFORE touching .env so a failure leaves state intact
        try:
            _ensure_asset(mmproj_file, mmproj_url, "mmproj")
            _ensure_asset(tmpl_file, tmpl_url, "template")
        except RuntimeError as exc:
            log(f"ABORT swap: {exc}")
            return False

        # Preserve WEBUI_SECRET_KEY
        secret = None
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("WEBUI_SECRET_KEY="):
                    secret = line
                    break

        content = preset_file.read_text()
        content += (
            "\n# Auto-derived asset filenames (based on preset name)\n"
            f"MMPROJ_FILE={mmproj_file}\n"
            f"TEMPLATE_FILE={tmpl_file}\n"
        )
        if secret:
            content += f"\n{secret}\n"
        env_file.write_text(content)

        # Recreate llama-server with new env (docker compose reads .env).
        # If we're not in LLM mode yet, stop the other service first.
        if active_mode == "comfyui":
            _stop_service("comfyui", "comfyui")
        elif active_mode == "train":
            _stop_service("lora-train", "train")

        ok, _ = _compose_run(
            ["--profile", "llm", "up", "-d", "--force-recreate", "llama-server"]
        )
        if not ok:
            return False

        # Wait for healthy
        log(f"Waiting for {model_name} to load...")
        if not _wait_healthy(LLAMA_HOST, LLAMA_PORT, "/health", HEALTH_TIMEOUT):
            log(f"Timeout waiting for {model_name}")
            return False

        current_model_id = preset_info["model_id"]
        log(f"Loaded: {model_name}")
        return True
    finally:
        switching = False


# ── HTTP Proxy ───────────────────────────────────────────────────────
class ProxyHandler(http.server.BaseHTTPRequestHandler):
    presets = load_presets()

    # ── Route classification ─────────────────────────────────────────
    def _classify_route(self):
        """Determine which backend a request targets.
        Returns ("llm"|"comfyui"|"train", path) or (None, path)."""
        if self.path.startswith("/comfyui"):
            stripped = self.path[len("/comfyui"):]
            if not stripped:
                stripped = "/"
            return "comfyui", stripped
        if self.path.startswith("/train"):
            stripped = self.path[len("/train"):]
            if not stripped:
                stripped = "/"
            return "train", stripped
        if self.path.startswith("/v1/"):
            return "llm", self.path
        return None, self.path

    # ── Request handlers ─────────────────────────────────────────────
    def do_GET(self):
        if self.path == "/v1/models":
            self.handle_models()
            return
        if self.path == "/health":
            self.handle_health()
            return
        if self.path == "/mode":
            self.handle_mode_get()
            return

        mode, target_path = self._classify_route()
        if mode == "comfyui":
            if not self._ensure_comfyui():
                return
            self.proxy_to(COMFYUI_HOST, COMFYUI_PORT, target_path)
        elif mode == "train":
            if not self._ensure_train():
                return
            self.proxy_to(TRAIN_HOST, TRAIN_PORT, target_path)
        elif mode == "llm":
            if not self._ensure_llm():
                return
            self.proxy_to(LLAMA_HOST, LLAMA_PORT, target_path)
        else:
            self._proxy_active(target_path)

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length else b""

        if self.path == "/mode":
            self.handle_mode_post(body)
            return

        mode, target_path = self._classify_route()

        if mode == "comfyui":
            if not self._ensure_comfyui():
                return
            self.proxy_to(COMFYUI_HOST, COMFYUI_PORT, target_path, body=body)

        elif mode == "train":
            if not self._ensure_train():
                return
            self.proxy_to(TRAIN_HOST, TRAIN_PORT, target_path, body=body)

        elif mode == "llm":
            # LLM path — model switching + message normalization
            if body:
                try:
                    data = json.loads(body)
                    requested_model = data.get("model", "")
                    if not self.ensure_model(requested_model):
                        return  # Error already sent

                    # Qwen chat templates reject multiple system messages
                    messages = data.get("messages")
                    if isinstance(messages, list) and _needs_system_merge(messages):
                        data["messages"] = _merge_system_messages(messages)
                        body = json.dumps(data).encode()
                except json.JSONDecodeError:
                    pass

            if not self._ensure_llm():
                return
            self.proxy_to(LLAMA_HOST, LLAMA_PORT, target_path, body=body)

        else:
            self._proxy_active(target_path, body=body)

    def do_OPTIONS(self):
        mode, target_path = self._classify_route()
        if mode == "comfyui":
            if not self._ensure_comfyui():
                return
            self.proxy_to(COMFYUI_HOST, COMFYUI_PORT, target_path)
        elif mode == "train":
            if not self._ensure_train():
                return
            self.proxy_to(TRAIN_HOST, TRAIN_PORT, target_path)
        elif mode == "llm":
            if not self._ensure_llm():
                return
            self.proxy_to(LLAMA_HOST, LLAMA_PORT, target_path)
        else:
            self._proxy_active(target_path)

    # ── Mode management endpoints ────────────────────────────────────
    def handle_mode_get(self):
        """GET /mode — return current active mode."""
        _json_response(self, 200, {
            "mode": active_mode,
            "switching": switching,
        })

    def handle_mode_post(self, body):
        """POST /mode — explicitly switch mode. Body: {"mode": "llm"|"comfyui"}"""
        try:
            data = json.loads(body)
        except (json.JSONDecodeError, ValueError):
            _json_response(self, 400, {"error": "invalid JSON"})
            return

        target = data.get("mode")
        if target not in ("llm", "comfyui", "train"):
            _json_response(self, 400, {
                "error": f"invalid mode: {target}. Must be 'llm', 'comfyui', or 'train'."
            })
            return

        with switch_lock:
            if active_mode == target:
                _json_response(self, 200, {"mode": active_mode, "switched": False})
                return
            ok = ensure_mode(target)

        if ok:
            _json_response(self, 200, {"mode": active_mode, "switched": True})
        else:
            _json_response(self, 503, {
                "error": f"Failed to switch to {target}",
                "mode": active_mode,
            })

    # ── Ensure mode helpers ──────────────────────────────────────────
    def _ensure_llm(self):
        """Ensure LLM mode is active. Returns True or sends error."""
        global active_mode
        if active_mode == "llm":
            return True
        with switch_lock:
            if active_mode == "llm":
                return True
            ok = ensure_mode("llm")
        if not ok:
            _json_response(self, 503, {
                "error": {"message": "Failed to start llama-server", "type": "server_error", "code": 503}
            })
            return False
        return True

    def _ensure_comfyui(self):
        """Ensure ComfyUI mode is active. Returns True or sends error."""
        global active_mode
        if active_mode == "comfyui":
            return True
        with switch_lock:
            if active_mode == "comfyui":
                return True
            ok = ensure_mode("comfyui")
        if not ok:
            _json_response(self, 503, {
                "error": {"message": "Failed to start ComfyUI", "type": "server_error", "code": 503}
            })
            return False
        return True

    def _ensure_train(self):
        """Ensure train mode is active. Returns True or sends error."""
        global active_mode
        if active_mode == "train":
            return True
        with switch_lock:
            if active_mode == "train":
                return True
            ok = ensure_mode("train")
        if not ok:
            _json_response(self, 503, {
                "error": {"message": "Failed to start lora-train", "type": "server_error", "code": 503}
            })
            return False
        return True

    def _proxy_active(self, path, body=None):
        """Forward to whatever backend is currently active."""
        if active_mode == "comfyui":
            self.proxy_to(COMFYUI_HOST, COMFYUI_PORT, path, body=body)
        elif active_mode == "llm":
            self.proxy_to(LLAMA_HOST, LLAMA_PORT, path, body=body)
        elif active_mode == "train":
            self.proxy_to(TRAIN_HOST, TRAIN_PORT, path, body=body)
        else:
            _json_response(self, 503, {
                "error": {"message": "No GPU service is running. Send a request to /v1/, /comfyui/, or /train/ to start one.", "type": "server_error", "code": 503}
            })

    # ── Existing proxy endpoints ─────────────────────────────────────
    def handle_models(self):
        """Return all presets as available models.

        The meta object follows the Open WebUI upstream metadata schema
        (PR #22441) so capabilities and descriptions are picked up
        automatically without manual UI configuration.
        """
        models = []
        for model_id, info in self.presets.items():
            cfg = info["config"]
            has_vision = bool(cfg.get("MMPROJ_URL"))
            reasoning = cfg.get("REASONING", "").strip().lower() in ("on", "true", "1")
            try:
                context = int(cfg.get("CONTEXT_SIZE", "65536"))
            except ValueError:
                context = 65536
            try:
                vram_gb = float(cfg.get("VRAM_ESTIMATE_GB", "0"))
            except ValueError:
                vram_gb = 0.0
            loaded = model_id == current_model_id
            models.append({
                "id": model_id,
                "object": "model",
                "created": 0,
                "owned_by": "local",
                "meta": {
                    # Open WebUI recognized fields (v0.8.12+, PR #22441)
                    "description": info.get("description", ""),
                    "capabilities": {
                        "vision": has_vision,
                    },
                    # Custom fields for proxy consumers (OpenCode, scripts)
                    "name": cfg.get("MODEL_NAME", model_id),
                    "loaded": loaded,
                    "preset": info["preset"],
                    "context": context,
                    "reasoning": reasoning,
                    "vram_gb": vram_gb,
                    "mode": active_mode,
                },
            })
        body = json.dumps({"object": "list", "data": models}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def handle_health(self):
        """Proxy health check. Returns 200 even during switching so Docker
        doesn't kill us mid-swap. Clients can check the 'status' field."""
        if switching:
            _json_response(self, 200, {"status": "switching", "mode": active_mode})
            return
        if active_mode == "llm":
            self.proxy_to(LLAMA_HOST, LLAMA_PORT, "/health")
        elif active_mode == "comfyui":
            try:
                conn = http.client.HTTPConnection(COMFYUI_HOST, COMFYUI_PORT, timeout=3)
                conn.request("GET", "/system_stats")
                resp = conn.getresponse()
                resp.read()
                conn.close()
                _json_response(self, 200, {"status": "ok", "mode": "comfyui"})
            except Exception:
                _json_response(self, 200, {"status": "degraded", "mode": "comfyui"})
        elif active_mode == "train":
            try:
                conn = http.client.HTTPConnection(TRAIN_HOST, TRAIN_PORT, timeout=3)
                conn.request("GET", "/health")
                resp = conn.getresponse()
                resp.read()
                conn.close()
                _json_response(self, 200, {"status": "ok", "mode": "train"})
            except Exception:
                _json_response(self, 200, {"status": "degraded", "mode": "train"})
        else:
            _json_response(self, 200, {"status": "idle", "mode": None})

    def ensure_model(self, requested_model):
        """Switch LLM model if needed. Returns True if ready, False on error."""
        global current_model_id, active_mode

        if not current_model_id:
            current_model_id = detect_current_model()

        # No switch needed
        if not requested_model or requested_model == current_model_id:
            return True

        # Unknown model
        if requested_model not in self.presets:
            return True  # Let llama-server handle the unknown model

        # VRAM budget gate — reject before touching anything
        ok, vram_msg = check_vram_budget(self.presets[requested_model])
        if not ok:
            _json_response(self, 422, {
                "error": {"message": vram_msg, "type": "vram_exceeded", "code": 422}
            })
            return False

        # Switch needed
        with switch_lock:
            # Re-check after acquiring lock (another thread may have switched)
            if requested_model == current_model_id:
                return True

            success = switch_model(self.presets[requested_model])
            if success:
                active_mode = "llm"
            if not success:
                _json_response(self, 503, {
                    "error": {"message": f"Failed to load model: {requested_model}", "type": "server_error", "code": 503}
                })
                return False
            return True

    # ── Generic reverse proxy ────────────────────────────────────────
    def proxy_to(self, host, port, path, body=None):
        """Forward request to a backend, streaming the response."""
        try:
            conn = http.client.HTTPConnection(host, port, timeout=600)

            # Forward headers (drop hop-by-hop). Recompute Content-Length when
            # we forward a (possibly rewritten) body.
            headers = {}
            for key, value in self.headers.items():
                lower = key.lower()
                if lower in ("host", "transfer-encoding", "connection", "content-length"):
                    continue
                headers[key] = value
            if body is not None:
                headers["Content-Length"] = str(len(body))

            conn.request(self.command, path, body=body, headers=headers)
            resp = conn.getresponse()

            # Send status + headers
            self.send_response(resp.status)
            is_stream = False
            for key, value in resp.getheaders():
                lower = key.lower()
                if lower in ("transfer-encoding", "connection"):
                    continue
                if lower == "content-type" and "text/event-stream" in value:
                    is_stream = True
                self.send_header(key, value)
            self.end_headers()

            # Stream body
            if is_stream:
                # SSE: flush after each chunk for real-time streaming
                while True:
                    chunk = resp.read(4096)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
            else:
                # Non-streaming: read all then send
                self.wfile.write(resp.read())

            conn.close()
        except (ConnectionRefusedError, OSError) as e:
            names = {COMFYUI_HOST: "ComfyUI", TRAIN_HOST: "lora-train"}
            service = names.get(host, "llama-server")
            _json_response(self, 502, {
                "error": {"message": f"{service} unavailable: {e}", "type": "server_error", "code": 502}
            })

    def log_message(self, format, *args):
        # Quieter logging -- only log non-health/non-system_stats requests
        msg = args[0] if args else ""
        if "/health" not in msg and "/system_stats" not in msg:
            log(f"{self.address_string()} {msg}")


# ── Main ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    presets = load_presets()
    current_model_id = detect_current_model()

    log(f"Listening on :{PROXY_PORT}")
    log(f"LLM backend: {LLAMA_HOST}:{LLAMA_PORT}")
    log(f"ComfyUI backend: {COMFYUI_HOST}:{COMFYUI_PORT}")
    log(f"Train backend: {TRAIN_HOST}:{TRAIN_PORT}")
    log(f"VRAM budget: {VRAM_LIMIT_GB}GB total, {VRAM_RESERVE_GB}GB reserved → {VRAM_LIMIT_GB - VRAM_RESERVE_GB}GB max model weight")
    log(f"LLM presets: {', '.join(presets.keys())}")
    if current_model_id:
        log(f"Active LLM model: {current_model_id}")
    log(f"Mode: idle (no GPU service running — will start on first request)")

    server = http.server.ThreadingHTTPServer(("", PROXY_PORT), ProxyHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log("Shutting down")
        server.shutdown()
