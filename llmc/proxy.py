"""HTTP reverse proxy + GPU mode orchestrator.

The v2 proxy: stdlib-only HTTP server that routes requests to llama-server,
ComfyUI, or lora-train based on URL prefix, automatically swapping GPU modes
as needed. State is persisted to /state/active.toml so a proxy restart
recovers cleanly.

Route table:
    /v1/*           → llama-server  (auto-starts LLM mode, hot-swap on `model`)
    /metrics        → llama-server  (Prometheus metrics, read-only, no swap)
    /comfyui/*      → ComfyUI       (auto-starts comfyui mode, prefix stripped)
    /train/*        → lora-train    (auto-starts train mode, prefix stripped)
    /v1/models      → proxy self    (preset list, OpenAI-API format)
    /health         → proxy self    (returns 200 even mid-swap)
    GET /mode       → proxy self    (current mode + switching flag)
    POST /mode      → proxy self    (explicit mode switch)

Configuration is read from environment variables on startup. See ProxyConfig.

Differences from v1:
    1. No .env rewriting. Preset → container env vars via Docker SDK directly.
    2. No `docker compose` shellout. Orchestrator uses the Engine API.
    3. Presets are TOML, schema-validated at load time.
    4. State (active mode + model) lives in /state/active.toml, not /project/.env.
    5. Bind mounts use direct host paths from volumes.toml — no Docker
       named-volume indirection (which rots on Docker Desktop/WSL2 restart).
"""

from __future__ import annotations

import http.client
import http.server
import json
import os
import signal
import socketserver
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from llmc import state as state_mod
from llmc.orchestrator import (
    COMFYUI_SERVICE,
    LLAMA_SERVICE,
    SERVICES,
    TRAIN_SERVICE,
    GpuService,
    Orchestrator,
    OrchestratorError,
)
from llmc.presets import Preset, load_all
from llmc.state import State
from llmc.volumes import VolumeRegistry, load as load_volumes


# ── Configuration ──────────────────────────────────────────────────────


@dataclass
class ProxyConfig:
    port: int = int(os.environ.get("LLMC_PROXY_PORT", "11434"))
    presets_dir: Path = Path(os.environ.get("LLMC_PRESETS_DIR", "/presets"))
    state_dir: Path = Path(os.environ.get("LLMC_STATE_DIR", "/state"))
    # Assets dir: where mmproj/template files are downloaded. The proxy
    # bind-mounts the llama-server/models host path here so downloads land
    # where llama-server will later read them.
    assets_dir: Path = Path(os.environ.get("LLMC_ASSETS_DIR", "/assets"))
    # Bind-mount path registry — host paths for spawning GPU services.
    # Mounted into the proxy container at /volumes.toml by compose.yaml.
    volumes_toml: Path = Path(os.environ.get("LLMC_VOLUMES_TOML", "/volumes.toml"))
    vram_limit_gb: float = float(os.environ.get("LLMC_VRAM_LIMIT_GB", "32"))
    vram_reserve_gb: float = float(os.environ.get("LLMC_VRAM_RESERVE_GB", "6"))
    network: str = os.environ.get("LLMC_NETWORK", "llmc")
    health_timeout: int = int(os.environ.get("LLMC_HEALTH_TIMEOUT", "900"))

    @property
    def state_path(self) -> Path:
        return self.state_dir / "active.toml"


# ── Proxy state singleton ──────────────────────────────────────────────


@dataclass
class ProxyContext:
    """Shared state across HTTP handler threads. Mutated under swap_lock."""

    config: ProxyConfig
    orchestrator: Orchestrator
    presets: dict[str, Preset]  # keyed by model_id (GGUF stem)
    state: State
    swap_lock: threading.Lock = field(default_factory=threading.Lock)
    switching: bool = False
    # Model lock: when set (a preset name), the proxy refuses anything that
    # would evict the locked model - model swaps, comfyui/train mode swaps,
    # unknown-model passthrough. Protects unattended multi-hour runs (loop
    # engine) from cross-client GPU eviction (whisper bot, Open WebUI).
    # In-memory only: a proxy restart clears the lock.
    lock_model: Optional[str] = None
    lock_owners: set[str] = field(default_factory=set)

    def reload_presets(self) -> None:
        """Re-scan the presets dir. Lets users add a TOML file without
        restarting the proxy."""
        self.presets = load_all(self.config.presets_dir)

    def preset_by_name(self, name: str) -> Optional[Preset]:
        """Look up by either model_id or preset name (filename stem)."""
        if name in self.presets:
            return self.presets[name]
        for preset in self.presets.values():
            if preset.name == name:
                return preset
        return None


# ── Helpers ────────────────────────────────────────────────────────────


def _log(msg: str) -> None:
    print(f"[llmc-proxy] {msg}", flush=True)


def _json_response(handler, status: int, payload: dict) -> None:
    body = json.dumps(payload).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(body)


def _check_vram_budget(preset: Preset, config: ProxyConfig) -> tuple[bool, str]:
    """Return (ok, error_message). VRAM_LIMIT - VRAM_RESERVE is the maximum
    model-weights budget; KV cache and CUDA overhead live in the reserve."""
    available = config.vram_limit_gb - config.vram_reserve_gb
    if preset.vram_gb > available:
        return False, (
            f"Model {preset.display_name!r} needs ~{preset.vram_gb}GB VRAM "
            f"for weights, but only {available}GB available after reserving "
            f"{config.vram_reserve_gb}GB for KV cache + compute buffer "
            f"(total VRAM: {config.vram_limit_gb}GB). Use a smaller quant."
        )
    return True, ""


def _merge_system_messages(messages: list) -> list:
    """Collapse multiple system messages into a single one at position 0.

    Qwen chat templates (and others) reject multiple system messages or
    out-of-order system messages with 'System message must be at the
    beginning'. This helper makes the proxy tolerate that without rejecting
    legitimate multi-system prompts from agents like OpenCode."""
    parts: list[str] = []
    others: list = []
    for m in messages:
        if not isinstance(m, dict):
            others.append(m)
            continue
        if m.get("role") == "system":
            content = m.get("content", "")
            if isinstance(content, str) and content.strip():
                parts.append(content)
            elif isinstance(content, list):
                for chunk in content:
                    if isinstance(chunk, dict) and chunk.get("type") == "text":
                        parts.append(chunk.get("text", ""))
        else:
            others.append(m)
    if not parts:
        return others
    return [{"role": "system", "content": "\n\n".join(parts)}] + others


def _needs_system_merge(messages: list) -> bool:
    """Heuristic: 2+ system messages OR a system message after a non-system."""
    if not isinstance(messages, list):
        return False
    sys_count = sum(1 for m in messages if isinstance(m, dict) and m.get("role") == "system")
    if sys_count >= 2:
        return True
    for i, m in enumerate(messages):
        if not isinstance(m, dict) or m.get("role") != "system" or i == 0:
            continue
        prev = messages[i - 1]
        if not isinstance(prev, dict) or prev.get("role") != "system":
            return True
    return False


# ── Mode + model switching ─────────────────────────────────────────────


def _ensure_mode(ctx: ProxyContext, target: str) -> tuple[bool, str]:
    """Ensure the target GPU mode is active. Caller must hold swap_lock."""
    if target not in SERVICES:
        return False, f"unknown mode {target!r}"

    if ctx.lock_model and target != "llm":
        return False, (
            f"model lock active on {ctx.lock_model!r}: refusing to leave "
            f"llm mode for {target!r} (POST /mode {{\"lock\": false}} to unlock)"
        )

    current = ctx.orchestrator.current_mode()
    if current == target:
        return True, ""

    service = SERVICES[target]
    _log(f"Switching mode: {current} → {target}")
    ctx.switching = True
    try:
        if target == "llm":
            # LLM mode requires a preset. If state has none yet (first run),
            # fail with a clear message — the user must POST /mode with
            # a model in the body, or hit /v1 with a model param.
            preset = ctx.preset_by_name(ctx.state.model) if ctx.state.model else None
            if preset is None:
                return False, (
                    "cannot enter LLM mode without an active preset; "
                    "POST /mode with {\"mode\":\"llm\", \"model\":\"<preset>\"}"
                )
            ctx.orchestrator.ensure_preset_assets(preset, ctx.config.assets_dir)
            ctx.orchestrator.spawn_llama(preset)
        elif target == "comfyui":
            ctx.orchestrator.spawn_comfyui()
        elif target == "train":
            ctx.orchestrator.spawn_train()
        else:
            return False, f"no spawn handler for mode {target!r}"

        _log(f"Waiting for {service.hostname} to become healthy...")
        if not ctx.orchestrator.wait_healthy(service, timeout=ctx.config.health_timeout):
            return False, f"timeout waiting for {service.hostname} healthcheck"

        ctx.state = state_mod.update(
            ctx.config.state_path,
            mode=target,
            model=ctx.state.model if target == "llm" else ctx.state.model,
        )
        _log(f"Mode: {target}")
        return True, ""
    except OrchestratorError as exc:
        _log(f"orchestrator error during mode switch: {exc}")
        return False, str(exc)
    finally:
        ctx.switching = False


def _ensure_model(ctx: ProxyContext, requested_model: str) -> tuple[bool, str]:
    """Hot-swap LLM preset if the requested model differs from current.

    Returns (ready, error_message). Caller MUST hold ctx.swap_lock — this
    function mutates ctx.state, calls stop_gpu_services, and spawns a new
    container. Concurrent invocations would race on the spawn (see commit
    b1f33de)."""
    if not requested_model:
        # No model specified — current model is fine
        return True, ""

    # Live-reload so a newly-added TOML is switchable without someone first
    # hitting GET /v1/models (previously the only reload trigger).
    try:
        ctx.reload_presets()
    except Exception as exc:
        _log(f"preset reload failed: {exc}")

    preset = ctx.preset_by_name(requested_model)
    if preset is None:
        # Name comparison, not preset-None rejection: if the locked preset's
        # TOML was deleted mid-lock, the locked model is still what's running
        # and a request naming it must keep working.
        if ctx.lock_model and requested_model != ctx.lock_model:
            return False, (
                f"model lock active on {ctx.lock_model!r}: rejecting unknown "
                f"model {requested_model!r} (passthrough would silently run "
                f"on the locked model)"
            )
        # Unknown model — let llama-server handle it (might be a passthrough
        # to whatever GGUF it has loaded, or it'll 404 — either way, not
        # our problem)
        _log(f"unknown model {requested_model!r}, passing through")
        return True, ""

    if ctx.lock_model and preset.name != ctx.lock_model:
        return False, (
            f"model lock active on {ctx.lock_model!r}: refusing to swap to "
            f"{preset.name!r} (POST /mode {{\"lock\": false}} to unlock)"
        )

    # VRAM budget gate — reject before stopping the current model
    ok, msg = _check_vram_budget(preset, ctx.config)
    if not ok:
        return False, msg

    # Same model already active?
    if (
        ctx.orchestrator.current_mode() == "llm"
        and ctx.state.model == preset.name
    ):
        return True, ""

    _log(f"Switching LLM model: {ctx.state.model} → {preset.name}")
    ctx.switching = True
    try:
        # Update state's model first so _ensure_mode picks it up
        ctx.state = state_mod.update(ctx.config.state_path, model=preset.name)

        # Download assets before tearing down the current container
        ctx.orchestrator.ensure_preset_assets(preset, ctx.config.assets_dir)

        # Spawn (auto-stops previous GPU service)
        ctx.orchestrator.spawn_llama(preset)

        _log(f"Waiting for llama-server with {preset.name} to become healthy...")
        if not ctx.orchestrator.wait_healthy(LLAMA_SERVICE, timeout=ctx.config.health_timeout):
            return False, f"timeout loading {preset.display_name}"

        ctx.state = state_mod.update(ctx.config.state_path, mode="llm")
        _log(f"Loaded: {preset.display_name}")
        return True, ""
    except OrchestratorError as exc:
        _log(f"orchestrator error during model swap: {exc}")
        return False, str(exc)
    finally:
        ctx.switching = False


# ── HTTP handler ───────────────────────────────────────────────────────


class ProxyHandler(http.server.BaseHTTPRequestHandler):
    # Injected by run() at server start time
    ctx: ProxyContext = None  # type: ignore

    # ── Routing ────────────────────────────────────────────────────────

    def _classify(self) -> tuple[Optional[str], str]:
        """Map URL prefix to GPU mode + the path to forward.
        Returns (mode, stripped_path) where mode is None for self-handled."""
        if self.path.startswith("/v1/"):
            return "llm", self.path
        if self.path == "/metrics":
            return "llm", "/metrics"
        if self.path.startswith("/comfyui"):
            stripped = self.path[len("/comfyui"):] or "/"
            return "comfyui", stripped
        if self.path.startswith("/train"):
            stripped = self.path[len("/train"):] or "/"
            return "train", stripped
        return None, self.path

    # ── Self-handled endpoints ─────────────────────────────────────────

    def _handle_health(self) -> None:
        if self.ctx.switching:
            _json_response(self, 200, {"status": "switching", "mode": self.ctx.state.mode})
            return
        mode = self.ctx.orchestrator.current_mode()
        _json_response(self, 200, {"status": "ok", "mode": mode})

    def _handle_models(self) -> None:
        """OpenAI-compatible model list. Live-reloads presets on each call so
        adding a TOML file doesn't require a proxy restart."""
        try:
            self.ctx.reload_presets()
        except Exception as exc:
            _log(f"preset reload failed: {exc}")

        active_model = self.ctx.state.model
        data = []
        for model_id, preset in self.ctx.presets.items():
            data.append({
                "id": model_id,
                "object": "model",
                "created": 0,
                "owned_by": "local",
                "meta": {
                    "description": preset.description.split("\n", 1)[0],
                    "capabilities": {"vision": preset.has_vision},
                    "name": preset.display_name,
                    "preset": preset.name,
                    "loaded": preset.name == active_model,
                    "context": preset.runtime.context_size,
                    "reasoning": preset.runtime.reasoning == "on",
                    "vram_gb": preset.vram_gb,
                    "mode": self.ctx.orchestrator.current_mode(),
                },
            })
        _json_response(self, 200, {"object": "list", "data": data})

    def _handle_mode_get(self) -> None:
        _json_response(self, 200, {
            "mode": self.ctx.orchestrator.current_mode(),
            "switching": self.ctx.switching,
            "model": self.ctx.state.model,
            "locked": self.ctx.lock_model,
            "lock_owners": sorted(list(self.ctx.lock_owners)),
        })

    def _handle_mode_post(self, body: bytes) -> None:
        try:
            payload = json.loads(body)
        except (json.JSONDecodeError, ValueError):
            _json_response(self, 400, {"error": "invalid JSON"})
            return

        # Model lock management: {"lock": "<preset>"} / {"lock": true} (lock
        # the currently-active model) / {"lock": false} or {"lock": null}
        # (unlock). Always allowed, including while locked.
        if "lock" in payload:
            lock_val = payload.get("lock")
            if lock_val in (None, False):
                self.ctx.lock_model = None
                _log("model lock cleared")
                _json_response(self, 200, {"locked": None})
                return
            if lock_val is True:
                lock_name = self.ctx.state.model
                if not lock_name:
                    _json_response(self, 400, {"error": "no active model to lock"})
                    return
            else:
                try:
                    self.ctx.reload_presets()
                except Exception as exc:
                    _log(f"preset reload failed: {exc}")
                preset = self.ctx.preset_by_name(str(lock_val))
                if preset is None:
                    _json_response(self, 404, {"error": f"unknown preset {lock_val!r}"})
                    return
                lock_name = preset.name
            if self.ctx.state.model != lock_name:
                _log(f"warning: locking {lock_name!r} while active model is "
                     f"{self.ctx.state.model!r} - lock pins a model that is not running")
            # Under swap_lock so a concurrent swap can't interleave between
            # another thread's lock check and this mutation.
            with self.ctx.swap_lock:
                self.ctx.lock_model = lock_name
            _log(f"model lock set: {lock_name}")
            _json_response(self, 200, {"locked": lock_name})
            return

        target = payload.get("mode")
        if target not in SERVICES:
            _json_response(self, 400, {
                "error": f"invalid mode {target!r}; must be one of {sorted(SERVICES)}"
            })
            return

        # POST /mode {mode: llm, model: X} must do a model swap even when
        # we're already in LLM mode (different model running). Route to
        # _ensure_model which handles both "enter LLM mode" and "swap model
        # within LLM mode". Pre-flight the preset existence + VRAM check
        # here so the caller gets 404/422 instead of generic 503.
        requested_model = payload.get("model")
        if target == "llm" and requested_model:
            preset = self.ctx.preset_by_name(requested_model)
            if preset is None:
                _json_response(self, 404, {"error": f"unknown preset {requested_model!r}"})
                return
            ok, vram_msg = _check_vram_budget(preset, self.ctx.config)
            if not ok:
                _json_response(self, 422, {"error": vram_msg})
                return

            with self.ctx.swap_lock:
                ok, msg = _ensure_model(self.ctx, preset.name)
            if ok:
                _json_response(self, 200, {
                    "mode": self.ctx.orchestrator.current_mode(),
                    "model": self.ctx.state.model,
                    "switched": True,
                })
            else:
                _json_response(self, 503, {
                    "error": msg,
                    "mode": self.ctx.orchestrator.current_mode(),
                })
            return

        # POST /mode {mode: X} — just mode swap, use the model already in
        # state (set by a previous call or first-run config).
        with self.ctx.swap_lock:
            ok, msg = _ensure_mode(self.ctx, target)
        if ok:
            _json_response(self, 200, {
                "mode": self.ctx.orchestrator.current_mode(),
                "model": self.ctx.state.model,
                "switched": True,
            })
        else:
            _json_response(self, 503, {
                "error": msg,
                "mode": self.ctx.orchestrator.current_mode(),
            })

    # ── Backend forwarding ─────────────────────────────────────────────

    # Methods that auto-trigger a GPU mode swap when their route's backend
    # isn't active. Read-only methods (GET, HEAD, OPTIONS) do not — they
    # 503 cleanly instead, so a status poll can't accidentally stop the
    # currently-running service.
    _SWAP_TRIGGER_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})

    def _ensure_and_forward(
        self,
        target_mode: str,
        target_path: str,
        body: Optional[bytes] = None,
    ) -> None:
        """Ensure the target mode is active, then proxy to the backend."""
        service = SERVICES[target_mode]

        if self.ctx.orchestrator.current_mode() != target_mode:
            if self.command not in self._SWAP_TRIGGER_METHODS:
                _json_response(self, 503, {
                    "error": {
                        "message": (
                            f"{target_mode} service is not active "
                            f"(current mode: {self.ctx.orchestrator.current_mode()}). "
                            f"Switch with POST /mode {{\"mode\":\"{target_mode}\"}}."
                        ),
                        "type": "service_inactive",
                        "code": 503,
                    }
                })
                return
            with self.ctx.swap_lock:
                if self.ctx.orchestrator.current_mode() != target_mode:
                    ok, msg = _ensure_mode(self.ctx, target_mode)
                    if not ok:
                        _json_response(self, 503, {
                            "error": {"message": msg, "type": "server_error", "code": 503}
                        })
                        return

        # LLM generations can legitimately exceed 600s end-to-end (long
        # thinking chains, non-streamed requests). Other backends keep the
        # tighter default.
        timeout = 3600 if target_mode == "llm" else 600
        self._forward(service.hostname, service.internal_port, target_path, body=body, timeout=timeout)

    def _forward(self, host: str, port: int, path: str, body: Optional[bytes] = None, *, timeout: int = 600) -> None:
        """Generic reverse proxy. Streams SSE responses chunk-by-chunk."""
        try:
            conn = http.client.HTTPConnection(host, port, timeout=timeout)
        except Exception as exc:
            _json_response(self, 502, {
                "error": {"message": f"connection failed: {exc}", "type": "server_error", "code": 502}
            })
            return

        # Forward headers (drop hop-by-hop; recompute Content-Length)
        headers = {}
        for key, value in self.headers.items():
            lower = key.lower()
            if lower in ("host", "transfer-encoding", "connection", "content-length"):
                continue
            headers[key] = value
        if body is not None:
            headers["Content-Length"] = str(len(body))

        try:
            conn.request(self.command, path, body=body, headers=headers)
            resp = conn.getresponse()
        except (OSError, http.client.HTTPException) as exc:
            _json_response(self, 502, {
                "error": {"message": f"upstream error: {exc}", "type": "server_error", "code": 502}
            })
            try:
                conn.close()
            except Exception:
                pass
            return

        # Header send inside the try so a client disconnect here still
        # reaches the finally's conn.close().
        try:
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
        except (BrokenPipeError, ConnectionResetError):
            conn.close()
            return

        try:
            if is_stream:
                # SSE: flush after each chunk so tokens appear in real time.
                # Distinguish upstream death from client disconnect: an
                # upstream reset must be surfaced, because a clean EOF is
                # indistinguishable from a finished completion and an
                # unattended agent would consume the truncated stream as a
                # full answer.
                while True:
                    try:
                        chunk = resp.read(4096)
                    except (OSError, http.client.HTTPException) as exc:
                        _log(f"upstream {host}:{port} died mid-stream: {exc}")
                        try:
                            self.wfile.write(
                                b'data: {"error":{"message":"upstream terminated '
                                b'mid-stream","type":"upstream_error"}}\n\n'
                            )
                            self.wfile.flush()
                        except OSError:
                            pass
                        break
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
            else:
                try:
                    payload_bytes = resp.read()
                except (OSError, http.client.HTTPException) as exc:
                    # Upstream reset before the body completed. Response
                    # headers (with upstream's Content-Length) are already
                    # sent, so the honest signal is an abrupt close: the
                    # client gets a truncated-body error instead of a clean
                    # short response.
                    _log(f"upstream {host}:{port} reset during body read: {exc}")
                    return
                self.wfile.write(payload_bytes)
        except (BrokenPipeError, ConnectionResetError):
            # Client disconnected mid-stream — not an error worth logging
            pass
        finally:
            conn.close()

    # ── Method dispatchers ─────────────────────────────────────────────

    def do_GET(self) -> None:
        if self.path == "/health":
            return self._handle_health()
        if self.path == "/v1/models":
            return self._handle_models()
        if self.path == "/mode":
            return self._handle_mode_get()

        mode, target_path = self._classify()
        if mode is None:
            _json_response(self, 404, {"error": f"unknown route {self.path!r}"})
            return
        self._ensure_and_forward(mode, target_path)

    def do_POST(self) -> None:
        if self.path == "/mode":
            body = self._read_body()
            return self._handle_mode_post(body)

        mode, target_path = self._classify()
        if mode is None:
            _json_response(self, 404, {"error": f"unknown route {self.path!r}"})
            return

        body = self._read_body()

        # LLM-mode requests: hot-swap model if `model` field is set, and
        # normalize messages for Qwen template quirks.
        #
        # _ensure_model mutates state + spawns containers, so it must run
        # under swap_lock — matches _handle_mode_post and _ensure_and_forward.
        # Without the lock two concurrent POSTs to /v1/chat/completions race:
        # both pass the same-model guard (current_mode briefly reports 'idle'
        # mid-spawn) and both call spawn_llama, the second blowing up with
        # 'container name "/llama_server" is already in use'.
        if mode == "llm" and body:
            try:
                payload = json.loads(body)
                requested_model = payload.get("model", "")
                with self.ctx.swap_lock:
                    ok, msg = _ensure_model(self.ctx, requested_model)
                if not ok:
                    _json_response(self, 422, {
                        "error": {"message": msg, "type": "model_unavailable", "code": 422}
                    })
                    return
                messages = payload.get("messages")
                if isinstance(messages, list) and _needs_system_merge(messages):
                    payload["messages"] = _merge_system_messages(messages)
                    body = json.dumps(payload).encode()
            except json.JSONDecodeError:
                # Body wasn't JSON — just forward as-is
                pass

        self._ensure_and_forward(mode, target_path, body=body)

    def do_OPTIONS(self) -> None:
        """CORS preflight — answer locally with permissive headers. Forwarding
        OPTIONS to the backend would trigger a GPU mode swap on every browser
        preflight (e.g. Open WebUI hitting /comfyui while LLM is active),
        silently stopping llama-server."""
        origin = self.headers.get("Origin", "*")
        req_method = self.headers.get("Access-Control-Request-Method", "GET, POST, OPTIONS")
        req_headers = self.headers.get("Access-Control-Request-Headers", "Content-Type, Authorization")
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", origin)
        self.send_header("Access-Control-Allow-Methods", req_method)
        self.send_header("Access-Control-Allow-Headers", req_headers)
        self.send_header("Access-Control-Max-Age", "86400")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length) if length > 0 else b""

    def log_message(self, format: str, *args) -> None:
        """Quieter logging — skip health and system_stats endpoints."""
        msg = " ".join(str(a) for a in args) if args else ""
        if "/health" not in msg and "/system_stats" not in msg:
            _log(f"{self.address_string()} {msg}")


# ── Entry point ────────────────────────────────────────────────────────


def build_context(config: Optional[ProxyConfig] = None) -> ProxyContext:
    """Construct a ProxyContext from configuration. Loads presets, recovers
    state from /state/active.toml, and reconciles with the orchestrator's
    view of the world (if the proxy crashed mid-swap, on-disk state may
    disagree with what Docker says is running)."""
    config = config or ProxyConfig()
    # Load the bind-mount path registry. Required at runtime — the orchestrator
    # needs host paths to spawn GPU services (Docker daemon binds host paths
    # directly into the new container; there are no named volumes anymore).
    volumes = load_volumes(config.volumes_toml)
    orchestrator = Orchestrator(network=config.network, volumes=volumes)
    presets = load_all(config.presets_dir)
    state = state_mod.load(config.state_path)

    # Reconcile state vs actual running containers — orchestrator wins.
    actual_mode = orchestrator.current_mode()
    if actual_mode != state.mode and not (actual_mode == "idle" and state.mode == "idle"):
        _log(
            f"State reconciliation: on-disk mode={state.mode!r}, "
            f"running={actual_mode!r}. Trusting running container."
        )
        state = state_mod.update(config.state_path, mode=actual_mode)

    return ProxyContext(
        config=config,
        orchestrator=orchestrator,
        presets=presets,
        state=state,
    )


class _ThreadingServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def run(config: Optional[ProxyConfig] = None) -> None:
    """Start the proxy. Blocks until SIGTERM/SIGINT."""
    ctx = build_context(config)
    ProxyHandler.ctx = ctx

    _log(f"Listening on :{ctx.config.port}")
    _log(f"Network: {ctx.config.network}")
    _log(f"Presets: {', '.join(sorted(ctx.presets))}")
    _log(f"Active mode: {ctx.state.mode}" + (f" ({ctx.state.model})" if ctx.state.model else ""))
    _log(
        f"VRAM budget: {ctx.config.vram_limit_gb}GB total, "
        f"{ctx.config.vram_reserve_gb}GB reserved → "
        f"{ctx.config.vram_limit_gb - ctx.config.vram_reserve_gb}GB max model weight"
    )

    server = _ThreadingServer(("", ctx.config.port), ProxyHandler)

    def _shutdown(signum, frame):
        _log(f"Shutting down (signal {signum})")
        server.shutdown()

    signal.signal(signal.SIGTERM, _shutdown)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        _log("Shutting down (KeyboardInterrupt)")
        server.shutdown()


if __name__ == "__main__":
    run()
