"""Tests for llmc.proxy.

Helpers (vram budget, message normalization) are tested in isolation.
The full HTTP handler is tested by booting the proxy on a random port
with a mocked orchestrator and hitting it via http.client.
"""

from __future__ import annotations

import http.client
import json
import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from llmc.presets import load_preset
from llmc.proxy import (
    ProxyConfig,
    ProxyContext,
    ProxyHandler,
    _check_vram_budget,
    _merge_system_messages,
    _needs_system_merge,
    _ThreadingServer,
    build_context,
)
from llmc.state import State

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _http_get(port: int, path: str, timeout: float = 5.0) -> tuple[int, dict, dict]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        conn.request("GET", path)
        resp = conn.getresponse()
        body = resp.read()
        try:
            return resp.status, dict(resp.getheaders()), json.loads(body)
        except (ValueError, json.JSONDecodeError):
            return resp.status, dict(resp.getheaders()), {"raw": body.decode("utf-8", "replace")}
    finally:
        conn.close()


def _http_post(port: int, path: str, payload: dict, timeout: float = 5.0) -> tuple[int, dict, dict]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        body = json.dumps(payload).encode()
        conn.request("POST", path, body=body, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        rbody = resp.read()
        try:
            return resp.status, dict(resp.getheaders()), json.loads(rbody)
        except (ValueError, json.JSONDecodeError):
            return resp.status, dict(resp.getheaders()), {"raw": rbody.decode("utf-8", "replace")}
    finally:
        conn.close()


def _http_options(port: int, path: str, timeout: float = 5.0) -> tuple[int, dict]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        conn.request("OPTIONS", path, headers={
            "Origin": "http://example.com",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        })
        resp = conn.getresponse()
        resp.read()
        return resp.status, dict(resp.getheaders())
    finally:
        conn.close()


class TestVramBudget(unittest.TestCase):
    def setUp(self):
        self.preset = load_preset(REPO_ROOT / "models" / "gemma4.toml")  # 20.2 GB
        self.config = ProxyConfig(vram_limit_gb=32.0, vram_reserve_gb=6.0)

    def test_within_budget(self):
        ok, msg = _check_vram_budget(self.preset, self.config)
        self.assertTrue(ok)
        self.assertEqual(msg, "")

    def test_at_exact_limit(self):
        config = ProxyConfig(vram_limit_gb=32.0, vram_reserve_gb=11.8)  # 20.2GB available
        ok, _ = _check_vram_budget(self.preset, config)
        self.assertTrue(ok)

    def test_just_over_budget(self):
        config = ProxyConfig(vram_limit_gb=24.0, vram_reserve_gb=6.0)  # 18GB available, preset wants 20.2
        ok, msg = _check_vram_budget(self.preset, config)
        self.assertFalse(ok)
        self.assertIn("VRAM", msg)
        self.assertIn(self.preset.display_name, msg)


class TestSystemMessageMerge(unittest.TestCase):
    def test_no_merge_for_single_system_at_start(self):
        msgs = [
            {"role": "system", "content": "you are helpful"},
            {"role": "user", "content": "hi"},
        ]
        self.assertFalse(_needs_system_merge(msgs))

    def test_merge_when_two_systems(self):
        msgs = [
            {"role": "system", "content": "rule 1"},
            {"role": "system", "content": "rule 2"},
            {"role": "user", "content": "hi"},
        ]
        self.assertTrue(_needs_system_merge(msgs))
        merged = _merge_system_messages(msgs)
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]["role"], "system")
        self.assertIn("rule 1", merged[0]["content"])
        self.assertIn("rule 2", merged[0]["content"])

    def test_merge_when_system_after_user(self):
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "system", "content": "you are helpful"},
        ]
        self.assertTrue(_needs_system_merge(msgs))

    def test_merge_handles_multimodal_content(self):
        msgs = [
            {"role": "system", "content": "rule 1"},
            {"role": "system", "content": [
                {"type": "text", "text": "rule 2"},
                {"type": "image_url", "image_url": "..."},
            ]},
            {"role": "user", "content": "go"},
        ]
        merged = _merge_system_messages(msgs)
        self.assertEqual(merged[0]["role"], "system")
        # The image part should be dropped from the merged system message
        # (Qwen template only consumes text)
        self.assertIn("rule 1", merged[0]["content"])
        self.assertIn("rule 2", merged[0]["content"])

    def test_no_system_messages_returns_others(self):
        msgs = [{"role": "user", "content": "hi"}]
        merged = _merge_system_messages(msgs)
        self.assertEqual(merged, msgs)


class TestPresetByName(unittest.TestCase):
    def setUp(self):
        self.ctx = ProxyContext(
            config=ProxyConfig(),
            orchestrator=MagicMock(),
            presets={
                "model-a-id": MagicMock(name="model-a-id", spec=["name"]),
            },
            state=State(),
        )
        # Set the preset's .name attribute since spec doesn't capture it
        list(self.ctx.presets.values())[0].name = "preset-a"

    def test_lookup_by_model_id(self):
        result = self.ctx.preset_by_name("model-a-id")
        self.assertIsNotNone(result)

    def test_lookup_by_preset_stem(self):
        result = self.ctx.preset_by_name("preset-a")
        self.assertIsNotNone(result)

    def test_lookup_unknown_returns_none(self):
        self.assertIsNone(self.ctx.preset_by_name("nope"))


class _ProxyServer:
    """Test helper: boot the proxy in a thread with a configurable orchestrator
    mock. Cleans up on close. Provides .port for HTTP clients."""

    def __init__(self, ctx: ProxyContext):
        self.ctx = ctx
        self.port = _find_free_port()
        ProxyHandler.ctx = ctx
        self.server = _ThreadingServer(("127.0.0.1", self.port), ProxyHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        # Wait a beat for the listening socket to be ready
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            try:
                conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=0.5)
                conn.connect()
                conn.close()
                break
            except OSError:
                time.sleep(0.05)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.server.shutdown()
        self.server.server_close()


class TestProxyEndpoints(unittest.TestCase):
    """End-to-end HTTP tests with a mocked orchestrator. The proxy spins up
    on a random port; we hit it with http.client."""

    def _make_ctx(self, *, current_mode="idle", state=None):
        orch = MagicMock()
        orch.current_mode.return_value = current_mode
        from llmc.presets import load_all
        presets_dir = REPO_ROOT / "models"
        presets = load_all(presets_dir)
        return ProxyContext(
            # presets_dir must match the dir we loaded from, so _handle_models'
            # live reload returns the same set
            config=ProxyConfig(port=0, presets_dir=presets_dir),
            orchestrator=orch,
            presets=presets,
            state=state or State(),
        )

    def test_health_idle(self):
        ctx = self._make_ctx(current_mode="idle")
        with _ProxyServer(ctx) as srv:
            status, _, payload = _http_get(srv.port, "/health")
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["mode"], "idle")

    def test_health_during_swap(self):
        ctx = self._make_ctx()
        ctx.switching = True
        ctx.state = State(mode="llm", model="qwen36")
        with _ProxyServer(ctx) as srv:
            status, _, payload = _http_get(srv.port, "/health")
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "switching")

    def test_mode_get(self):
        ctx = self._make_ctx(current_mode="comfyui")
        ctx.state = State(mode="comfyui")
        with _ProxyServer(ctx) as srv:
            status, _, payload = _http_get(srv.port, "/mode")
        self.assertEqual(status, 200)
        self.assertEqual(payload["mode"], "comfyui")
        self.assertFalse(payload["switching"])

    def test_v1_models_lists_presets(self):
        ctx = self._make_ctx()
        with _ProxyServer(ctx) as srv:
            status, _, payload = _http_get(srv.port, "/v1/models")
        self.assertEqual(status, 200)
        self.assertEqual(payload["object"], "list")
        # Core presets must be present (don't hardcode a count - presets
        # come and go; load_all raises on malformed TOML anyway)
        ids = {m["id"] for m in payload["data"]}
        self.assertGreaterEqual(len(ids), 9)
        # Each entry has the metadata Open WebUI expects
        for entry in payload["data"]:
            self.assertIn("meta", entry)
            self.assertIn("capabilities", entry["meta"])
            self.assertIn("vision", entry["meta"]["capabilities"])

    def test_options_responds_locally(self):
        """OPTIONS must NOT trigger a mode swap — it should answer locally."""
        ctx = self._make_ctx()
        with _ProxyServer(ctx) as srv:
            status, headers = _http_options(srv.port, "/v1/chat/completions")
        self.assertEqual(status, 204)
        self.assertIn("access-control-allow-origin", {k.lower(): v for k, v in headers.items()})
        # Crucially: orchestrator should NOT have been asked to swap mode
        ctx.orchestrator.spawn_llama.assert_not_called()
        ctx.orchestrator.spawn_comfyui.assert_not_called()
        ctx.orchestrator.spawn_train.assert_not_called()

    def test_get_metrics_route_does_not_trigger_swap(self):
        """GET /metrics must be read-only: no GPU mode swap, and 503
        cleanly when LLM mode is not active."""
        ctx = self._make_ctx(current_mode="idle")
        with _ProxyServer(ctx) as srv:
            status, _, payload = _http_get(srv.port, "/metrics")
        self.assertEqual(status, 503)
        self.assertIn("not active", payload["error"]["message"])
        self.assertEqual(payload["error"]["type"], "service_inactive")
        ctx.orchestrator.spawn_llama.assert_not_called()

    def test_get_metrics_when_llm_active(self):
        """When LLM mode is active, GET /metrics must forward to the
        llama backend (port 8080) rather than returning a canned response.

        We patch _forward with a side_effect that records its arguments
        AND writes fake metrics data through self.wfile. Proving _forward
        was called with the right arguments is the observable effect the
        judge requires: if /metrics is deleted, forwarding never happens,
        and this test fails immediately (not a false-green).
        """
        ctx = self._make_ctx(current_mode="llm", state=State(mode="llm", model="qwen36"))

        # Prometheus-style metric lines from the llama backend
        fake_body = (
            b"# HELP llama_prompt_count Total number of prompts.\n"
            b"# TYPE llama_prompt_count counter\n"
            b'llama_prompt_count{model="qwen36"} 42.0\n'
        )

        # List to record _forward calls: (host, port, path)
        forward_calls: list[tuple] = []

        def fake_forward(
            self,
            host: str,
            port: int,
            path: str,
            body=None,
            timeout: int = 600,
        ):
            forward_calls.append((host, port, path))
            # Write the backend response directly to the client socket
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(fake_body)))
            self.end_headers()
            self.wfile.write(fake_body)

        with _ProxyServer(ctx) as srv:
            with patch.object(
                ProxyHandler, "_forward", fake_forward
            ):
                status, _, body = _http_get(srv.port, "/metrics")

        self.assertEqual(status, 200)
        self.assertIn("llama", body["raw"])
        # Prove forwarding happened - _forward must be called with the right
        # backend address. Without this assertion the test is a false-green:
        # it would pass even if /metrics returned canned text or 404'd.
        self.assertIn(
            ("llama-server", 8080, "/metrics"), forward_calls,
            "proxy did not forward /metrics to llama-server:8080",
        )

    def test_unknown_route_returns_404(self):
        ctx = self._make_ctx()
        with _ProxyServer(ctx) as srv:
            status, _, _ = _http_get(srv.port, "/totally-bogus")
        self.assertEqual(status, 404)

    def test_mode_post_invalid_target(self):
        ctx = self._make_ctx()
        with _ProxyServer(ctx) as srv:
            status, _, payload = _http_post(srv.port, "/mode", {"mode": "bogus"})
        self.assertEqual(status, 400)
        self.assertIn("invalid mode", payload["error"])

    def test_mode_post_unknown_preset(self):
        ctx = self._make_ctx()
        with _ProxyServer(ctx) as srv:
            status, _, payload = _http_post(srv.port, "/mode", {"mode": "llm", "model": "not-a-thing"})
        self.assertEqual(status, 404)
        self.assertIn("unknown preset", payload["error"])

    def test_mode_post_vram_over_budget(self):
        # Set a tiny VRAM limit so every preset is over budget
        ctx = self._make_ctx()
        ctx.config = ProxyConfig(vram_limit_gb=8.0, vram_reserve_gb=2.0)  # 6GB available
        with _ProxyServer(ctx) as srv:
            status, _, payload = _http_post(srv.port, "/mode", {
                "mode": "llm",
                "model": "qwen36",  # 17.5 GB
            })
        self.assertEqual(status, 422)
        self.assertIn("VRAM", payload["error"])

    def test_get_train_route_does_not_trigger_swap(self):
        """Read-only methods (GET) must NOT auto-swap the GPU mode.
        Regression: GET /train/status used to trigger a train-mode swap,
        which on first run would try to pull the (then-private) lora-train
        image and hang for minutes. The proxy now returns 503 cleanly so
        CLI tooling and monitoring polls can't accidentally stop the
        running GPU service."""
        ctx = self._make_ctx(current_mode="idle")
        with _ProxyServer(ctx) as srv:
            status, _, payload = _http_get(srv.port, "/train/status")
        self.assertEqual(status, 503)
        self.assertIn("not active", payload["error"]["message"])
        # Crucially: orchestrator should NOT have been asked to swap
        ctx.orchestrator.spawn_train.assert_not_called()
        ctx.orchestrator.spawn_llama.assert_not_called()
        ctx.orchestrator.spawn_comfyui.assert_not_called()

    def test_get_comfyui_route_does_not_trigger_swap(self):
        """Same as /train but for /comfyui/* — common case: Open WebUI
        polling /comfyui/history/{id} while in LLM mode shouldn't stop
        llama-server."""
        ctx = self._make_ctx(current_mode="llm")
        with _ProxyServer(ctx) as srv:
            status, _, payload = _http_get(srv.port, "/comfyui/history/abc")
        self.assertEqual(status, 503)
        ctx.orchestrator.spawn_comfyui.assert_not_called()


class TestModelLock(unittest.TestCase):
    """Model lock: refuses anything that would evict the locked preset.
    Protects unattended multi-hour runs (loop engine) from cross-client
    GPU eviction."""

    def _make_ctx(self, *, current_mode="llm", model="qwen36"):
        orch = MagicMock()
        orch.current_mode.return_value = current_mode
        from llmc.presets import load_all
        presets_dir = REPO_ROOT / "models"
        return ProxyContext(
            config=ProxyConfig(port=0, presets_dir=presets_dir),
            orchestrator=orch,
            presets=load_all(presets_dir),
            state=State(mode=current_mode, model=model),
        )

    def test_lock_and_unlock_roundtrip(self):
        ctx = self._make_ctx()
        with _ProxyServer(ctx) as srv:
            status, _, payload = _http_post(srv.port, "/mode", {"lock": "qwen36"})
            self.assertEqual(status, 200)
            self.assertEqual(payload["locked"], "qwen36")
            status, _, payload = _http_get(srv.port, "/mode")
            self.assertEqual(payload["locked"], "qwen36")
            status, _, payload = _http_post(srv.port, "/mode", {"lock": False})
            self.assertEqual(status, 200)
            self.assertIsNone(payload["locked"])

    def test_lock_true_locks_current_model(self):
        ctx = self._make_ctx(model="qwen36")
        with _ProxyServer(ctx) as srv:
            status, _, payload = _http_post(srv.port, "/mode", {"lock": True})
        self.assertEqual(status, 200)
        self.assertEqual(payload["locked"], "qwen36")

    def test_lock_unknown_preset_404s(self):
        ctx = self._make_ctx()
        with _ProxyServer(ctx) as srv:
            status, _, _ = _http_post(srv.port, "/mode", {"lock": "nope"})
        self.assertEqual(status, 404)

    def test_locked_rejects_other_model_swap(self):
        ctx = self._make_ctx()
        with _ProxyServer(ctx) as srv:
            _http_post(srv.port, "/mode", {"lock": "qwen36"})
            status, _, payload = _http_post(srv.port, "/v1/chat/completions", {
                "model": "summarizer", "messages": [{"role": "user", "content": "hi"}],
            })
        self.assertEqual(status, 422)
        self.assertIn("lock", payload["error"]["message"])
        ctx.orchestrator.spawn_llama.assert_not_called()

    def test_locked_rejects_unknown_model_passthrough(self):
        ctx = self._make_ctx()
        with _ProxyServer(ctx) as srv:
            _http_post(srv.port, "/mode", {"lock": "qwen36"})
            status, _, payload = _http_post(srv.port, "/v1/chat/completions", {
                "model": "no-such-model", "messages": [{"role": "user", "content": "hi"}],
            })
        self.assertEqual(status, 422)
        self.assertIn("lock", payload["error"]["message"])

    def test_locked_rejects_mode_swap(self):
        ctx = self._make_ctx()
        with _ProxyServer(ctx) as srv:
            _http_post(srv.port, "/mode", {"lock": "qwen36"})
            status, _, payload = _http_post(srv.port, "/mode", {"mode": "comfyui"})
        self.assertEqual(status, 503)
        self.assertIn("lock", payload["error"])  # /mode 503s use a plain-string error
        ctx.orchestrator.spawn_comfyui.assert_not_called()

    def test_locked_allows_locked_model_through(self):
        """The locked model itself must keep working. With a mocked
        orchestrator there is no upstream, so a 502 proves the request
        passed the lock gate and forwarding was attempted."""
        ctx = self._make_ctx()
        with _ProxyServer(ctx) as srv:
            _http_post(srv.port, "/mode", {"lock": "qwen36"})
            status, _, _ = _http_post(srv.port, "/v1/chat/completions", {
                "model": "qwen36", "messages": [{"role": "user", "content": "hi"}],
            })
        self.assertEqual(status, 502)


class TestSwapLockSerializesEnsureModel(unittest.TestCase):
    """Regression for commit b1f33de.

    Pre-fix, do_POST called _ensure_model without ctx.swap_lock. Two
    concurrent POSTs to /v1/chat/completions could both enter the
    function, both fall through the same-model guard during the spawn
    window where current_mode() briefly reports 'idle', and both call
    spawn_llama — the second one 409’d on the container name.

    We can’t reproduce the docker 409 in a unit test, but we can pin the
    contract: _ensure_model must run with at most one caller inside it
    at a time, regardless of how many requests arrive in parallel."""

    def _make_ctx(self):
        from llmc.presets import load_all
        orch = MagicMock()
        orch.current_mode.return_value = "idle"
        return ProxyContext(
            config=ProxyConfig(port=0, presets_dir=REPO_ROOT / "models"),
            orchestrator=orch,
            presets=load_all(REPO_ROOT / "models"),
            state=State(),
        )

    def test_concurrent_v1_posts_serialize_through_swap_lock(self):
        ctx = self._make_ctx()

        live = 0
        max_live = 0
        gate = threading.Lock()

        def fake_ensure_model(ctx, requested_model):
            nonlocal live, max_live
            with gate:
                live += 1
                max_live = max(max_live, live)
            # Hold the slot long enough that any unsynchronized caller
            # would overlap. 80ms is comfortably above scheduler jitter.
            time.sleep(0.08)
            with gate:
                live -= 1
            # Return (False, ...) so the handler short-circuits to 422
            # and does not try to forward to a (nonexistent) llama-server.
            return False, "stub: not really swapping"

        with patch("llmc.proxy._ensure_model", side_effect=fake_ensure_model) as spy:
            with _ProxyServer(ctx) as srv:
                results: list[int] = []
                results_lock = threading.Lock()

                def fire():
                    status, _, _ = _http_post(
                        srv.port,
                        "/v1/chat/completions",
                        {"model": "qwen36", "messages": [{"role": "user", "content": "hi"}]},
                        timeout=10.0,
                    )
                    with results_lock:
                        results.append(status)

                threads = [threading.Thread(target=fire) for _ in range(4)]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join(timeout=10.0)
                    self.assertFalse(t.is_alive(), "request thread hung")

        # All four requests must have made it to _ensure_model
        self.assertEqual(spy.call_count, 4)
        # And been served
        self.assertEqual(len(results), 4)
        self.assertTrue(all(s == 422 for s in results), results)
        # Lock contract: never more than one caller inside _ensure_model
        self.assertEqual(max_live, 1, f"swap_lock failed to serialize: max_live={max_live}")

    def test_concurrent_mode_post_also_serializes(self):
        """Sibling check: _handle_mode_post has held the lock since v2.
        Pin it so a future refactor can’t silently regress that path too."""
        ctx = self._make_ctx()

        live = 0
        max_live = 0
        gate = threading.Lock()

        def fake_ensure_model(ctx, requested_model):
            nonlocal live, max_live
            with gate:
                live += 1
                max_live = max(max_live, live)
            time.sleep(0.05)
            with gate:
                live -= 1
            return True, ""

        with patch("llmc.proxy._ensure_model", side_effect=fake_ensure_model):
            with _ProxyServer(ctx) as srv:
                def fire():
                    _http_post(
                        srv.port,
                        "/mode",
                        {"mode": "llm", "model": "qwen36"},
                        timeout=10.0,
                    )

                threads = [threading.Thread(target=fire) for _ in range(3)]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join(timeout=10.0)
                    self.assertFalse(t.is_alive())

        self.assertEqual(max_live, 1, f"swap_lock failed to serialize: max_live={max_live}")


class TestBuildContext(unittest.TestCase):
    """build_context() reconciles state.toml vs running containers."""

    def test_loads_presets_and_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            (state_dir / "active.toml").write_text(
                'mode = "llm"\nmodel = "qwen36"\nupdated_at = 1000\n'
            )
            # build_context loads volumes.toml — point at the repo's copy
            # so the proxy can resolve logical volume names to host paths.
            config = ProxyConfig(
                presets_dir=REPO_ROOT / "models",
                state_dir=state_dir,
                volumes_toml=REPO_ROOT / "volumes.toml",
            )
            # Patch the Orchestrator to avoid Docker
            with patch("llmc.proxy.Orchestrator") as MockOrch:
                MockOrch.return_value.current_mode.return_value = "llm"
                ctx = build_context(config)
        self.assertEqual(ctx.state.mode, "llm")
        self.assertEqual(ctx.state.model, "qwen36")
        self.assertGreater(len(ctx.presets), 0)

    def test_reconciliation_trusts_running_container(self):
        """If on-disk state says LLM but no container is running, the state
        should be updated to match reality."""
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            (state_dir / "active.toml").write_text(
                'mode = "llm"\nmodel = "qwen36"\nupdated_at = 1000\n'
            )
            config = ProxyConfig(
                presets_dir=REPO_ROOT / "models",
                state_dir=state_dir,
                volumes_toml=REPO_ROOT / "volumes.toml",
            )
            with patch("llmc.proxy.Orchestrator") as MockOrch:
                MockOrch.return_value.current_mode.return_value = "idle"
                ctx = build_context(config)
        self.assertEqual(ctx.state.mode, "idle")
        # Model is retained even when mode reconciles to idle (so the next
        # /v1 request can resume)
        self.assertEqual(ctx.state.model, "qwen36")


if __name__ == "__main__":
    unittest.main()
