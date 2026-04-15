#!/usr/bin/env python3
"""
Model-switching reverse proxy for llama-server.

Sits in front of llama-server and auto-swaps models when the requested
model differs from what's currently loaded. Select a different model in
OpenCode's /models menu and the proxy handles the rest.

The ~60-90s model load on swap is unavoidable (VRAM), but you never
need to leave OpenCode or touch the terminal.
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
PROXY_PORT = int(os.environ.get("PROXY_PORT", "11434"))
PRESETS_DIR = Path(os.environ.get("PRESETS_DIR", "/presets"))
PROJECT_DIR = Path(os.environ.get("PROJECT_DIR", "/project"))
HEALTH_TIMEOUT = int(os.environ.get("HEALTH_TIMEOUT", "180"))
VRAM_LIMIT_GB = float(os.environ.get("VRAM_LIMIT_GB", "32"))
VRAM_RESERVE_GB = float(os.environ.get("VRAM_RESERVE_GB", "10"))

# ── State ────────────────────────────────────────────────────────────
current_model_id = None
switch_lock = threading.Lock()
switching = False


# ── Preset loading ───────────────────────────────────────────────────
def parse_env_file(path):
    """Parse a KEY=VALUE env file, ignoring comments and blanks."""
    config = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            config[key.strip()] = value.strip()
    return config


def load_presets():
    """Build model_id -> preset mapping from models/*.env files."""
    presets = {}
    for f in sorted(PRESETS_DIR.glob("*.env")):
        config = parse_env_file(f)
        model_file = config.get("MODEL_FILE", "")
        # Model ID = GGUF filename without extension (matches OpenCode config key)
        model_id = model_file.rsplit(".", 1)[0] if model_file else f.stem
        presets[model_id] = {
            "preset": f.stem,
            "config": config,
            "model_id": model_id,
        }
    return presets


def detect_current_model():
    """Read .env to determine which model is currently configured."""
    env_file = PROJECT_DIR / ".env"
    if not env_file.exists():
        return None
    config = parse_env_file(env_file)
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
            f"{VRAM_RESERVE_GB}GB for KV cache + overhead "
            f"(total VRAM: {VRAM_LIMIT_GB}GB). "
            f"Use a smaller quant (Q4_K_M) or reduce context size."
        )
        log(f"REJECTED: {msg}")
        return False, msg
    return True, ""


# ── Model switching ──────────────────────────────────────────────────
def switch_model(preset_info):
    """Update .env with new model preset and recreate llama-server."""
    global current_model_id, switching
    switching = True
    preset_name = preset_info["preset"]
    model_name = preset_info["config"].get("MODEL_NAME", preset_name)
    log(f"Switching to {model_name}...")

    try:
        preset_file = PRESETS_DIR / f"{preset_name}.env"
        env_file = PROJECT_DIR / ".env"

        # Preserve WEBUI_SECRET_KEY
        secret = None
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("WEBUI_SECRET_KEY="):
                    secret = line
                    break

        content = preset_file.read_text()
        if secret:
            content += f"\n{secret}\n"
        env_file.write_text(content)

        # Recreate llama-server with new env (docker compose reads .env).
        # Set HOME to the host HOME so ~ in docker-compose.yml volume paths
        # resolves correctly (the proxy container's HOME is /root).
        compose_env = os.environ.copy()
        host_home = os.environ.get("HOST_HOME")
        if host_home:
            compose_env["HOME"] = host_home
        result = subprocess.run(
            ["docker", "compose", "up", "-d", "--force-recreate", "llama-server"],
            cwd=str(PROJECT_DIR),
            env=compose_env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            log(f"docker compose failed: {result.stderr}")
            return False

        # Wait for healthy
        log(f"Waiting for {model_name} to load...")
        deadline = time.monotonic() + HEALTH_TIMEOUT
        while time.monotonic() < deadline:
            try:
                conn = http.client.HTTPConnection(LLAMA_HOST, LLAMA_PORT, timeout=3)
                conn.request("GET", "/health")
                resp = conn.getresponse()
                body = resp.read()
                conn.close()
                if resp.status == 200:
                    data = json.loads(body)
                    if data.get("status") == "ok":
                        current_model_id = preset_info["model_id"]
                        log(f"Loaded: {model_name}")
                        return True
            except Exception:
                pass
            time.sleep(2)

        log(f"Timeout waiting for {model_name}")
        return False
    finally:
        switching = False


# ── HTTP Proxy ───────────────────────────────────────────────────────
class ProxyHandler(http.server.BaseHTTPRequestHandler):
    presets = load_presets()

    def do_GET(self):
        if self.path == "/v1/models":
            self.handle_models()
        elif self.path == "/health":
            self.handle_health()
        else:
            self.proxy_request()

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length else b""

        # Check if we need to switch models
        if self.path.startswith("/v1/") and body:
            try:
                data = json.loads(body)
                requested_model = data.get("model", "")
                if not self.ensure_model(requested_model):
                    return  # Error already sent
            except json.JSONDecodeError:
                pass

        self.proxy_request(body=body)

    def do_OPTIONS(self):
        self.proxy_request()

    def handle_models(self):
        """Return all presets as available models."""
        models = []
        for model_id, info in self.presets.items():
            cfg = info["config"]
            loaded = model_id == current_model_id
            models.append({
                "id": model_id,
                "object": "model",
                "created": 0,
                "owned_by": "local",
                # Extra metadata for visibility
                "meta": {
                    "name": cfg.get("MODEL_NAME", model_id),
                    "loaded": loaded,
                    "preset": info["preset"],
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
            body = json.dumps({"status": "switching"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.proxy_request()

    def ensure_model(self, requested_model):
        """Switch model if needed. Returns True if ready, False on error."""
        global current_model_id

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
            body = json.dumps({
                "error": {
                    "message": vram_msg,
                    "type": "vram_exceeded",
                    "code": 422,
                }
            }).encode()
            self.send_response(422)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return False

        # Switch needed
        with switch_lock:
            # Re-check after acquiring lock (another thread may have switched)
            if requested_model == current_model_id:
                return True

            success = switch_model(self.presets[requested_model])
            if not success:
                body = json.dumps({
                    "error": {
                        "message": f"Failed to load model: {requested_model}",
                        "type": "server_error",
                        "code": 503,
                    }
                }).encode()
                self.send_response(503)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return False
            return True

    def proxy_request(self, body=None):
        """Forward request to llama-server, streaming the response."""
        try:
            conn = http.client.HTTPConnection(LLAMA_HOST, LLAMA_PORT, timeout=600)

            # Forward headers (drop hop-by-hop)
            headers = {}
            for key, value in self.headers.items():
                if key.lower() not in ("host", "transfer-encoding", "connection"):
                    headers[key] = value

            conn.request(self.command, self.path, body=body, headers=headers)
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
            body = json.dumps({
                "error": {
                    "message": f"llama-server unavailable: {e}",
                    "type": "server_error",
                    "code": 502,
                }
            }).encode()
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def log_message(self, format, *args):
        # Quieter logging -- only log non-health requests
        if "/health" not in (args[0] if args else ""):
            log(f"{self.address_string()} {args[0]}")


def log(msg):
    print(f"[model-proxy] {msg}", flush=True)


# ── Main ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    presets = load_presets()
    current_model_id = detect_current_model()

    log(f"Listening on :{PROXY_PORT}")
    log(f"Backend: {LLAMA_HOST}:{LLAMA_PORT}")
    log(f"VRAM budget: {VRAM_LIMIT_GB}GB total, {VRAM_RESERVE_GB}GB reserved → {VRAM_LIMIT_GB - VRAM_RESERVE_GB}GB max model weight")
    log(f"Models: {', '.join(presets.keys())}")
    if current_model_id:
        log(f"Active: {current_model_id}")

    server = http.server.ThreadingHTTPServer(("", PROXY_PORT), ProxyHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log("Shutting down")
        server.shutdown()
