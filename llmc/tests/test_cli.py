"""Tests for llmc.cli.

We can't easily test the full argv path end-to-end without spinning up
real services, but we can:
    - exercise the argument parser for each subcommand
    - test ProxyClient against a stub HTTP server
    - test the output formatters
    - test fallback behaviors (proxy unreachable → local TOML)
"""

from __future__ import annotations

import contextlib
import http.server
import io
import json
import socket
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from llmc import cli
from llmc.cli import (
    EXIT_BACKEND_ERROR,
    EXIT_OK,
    EXIT_TRANSIENT,
    EXIT_USER_ERROR,
    ProxyClient,
    _build_parser,
    main,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _StubHandler(http.server.BaseHTTPRequestHandler):
    """Per-test stub. Override on subclasses via class attribute `responses`."""

    responses: dict[tuple[str, str], tuple[int, dict]] = {}

    def _respond(self):
        key = (self.command, self.path)
        if key in self.responses:
            status, payload = self.responses[key]
        else:
            status, payload = (404, {"error": f"no stub for {key}"})
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _record_and_respond(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        self.server.requests.append((self.command, self.path, raw))  # type: ignore[attr-defined]
        self._respond()

    do_GET = _respond
    do_POST = _respond

    def log_message(self, *args, **kwargs):
        # Quiet
        pass


@contextlib.contextmanager
def _stub_proxy(responses: dict[tuple[str, str], tuple[int, dict]]):
    """Run a stub HTTP proxy on a random port for the duration of the test.
    Yields the (host, port). Patches the CLI's defaults to point at it."""
    port = _free_port()

    class Handler(_StubHandler):
        pass

    Handler.responses = responses
    server = http.server.HTTPServer(("127.0.0.1", port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    # Patch defaults so cli.ProxyClient() picks them up
    with patch.object(cli, "DEFAULT_PROXY_HOST", "127.0.0.1"), \
            patch.object(cli, "DEFAULT_PROXY_PORT", port):
        try:
            yield ("127.0.0.1", port)
        finally:
            server.shutdown()
            server.server_close()


@contextlib.contextmanager
def _stub_proxy_recording(responses: dict[tuple[str, str], tuple[int, dict]]):
    """Like _stub_proxy, but also captures request bodies.
    Yields (host, port, requests) where requests is a list of
    (method, path, raw_body) tuples."""
    port = _free_port()

    class Handler(_StubHandler):
        do_POST = _StubHandler._record_and_respond

    Handler.responses = responses
    server = http.server.HTTPServer(("127.0.0.1", port), Handler)
    server.requests = []  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    with patch.object(cli, "DEFAULT_PROXY_HOST", "127.0.0.1"), \
            patch.object(cli, "DEFAULT_PROXY_PORT", port):
        try:
            yield ("127.0.0.1", port, server.requests)
        finally:
            server.shutdown()
            server.server_close()


@contextlib.contextmanager
def _capture():
    """Capture stdout + stderr."""
    out, err = io.StringIO(), io.StringIO()
    with patch("sys.stdout", out), patch("sys.stderr", err):
        yield out, err


class TestParser(unittest.TestCase):
    """All subcommands must parse without raising."""

    def setUp(self):
        self.parser = _build_parser()

    def test_no_args_shows_help(self):
        # Argparse doesn't error on empty argv; main() returns user error
        ns = self.parser.parse_args([])
        self.assertIsNone(ns.command)

    def test_status_subcommand(self):
        ns = self.parser.parse_args(["status"])
        self.assertEqual(ns.command, "status")

    def test_mode_without_target(self):
        ns = self.parser.parse_args(["mode"])
        self.assertEqual(ns.command, "mode")
        self.assertIsNone(ns.target)

    def test_mode_with_target_and_model(self):
        ns = self.parser.parse_args(["mode", "llm", "--model", "qwen38"])
        self.assertEqual(ns.target, "llm")
        self.assertEqual(ns.model, "qwen38")

    def test_mode_rejects_bogus_target(self):
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["mode", "totally-fake"])

    def test_switch_requires_preset(self):
        ns = self.parser.parse_args(["switch", "qwen38"])
        self.assertEqual(ns.preset, "qwen38")

    def test_volumes_subcommands(self):
        for sub in ("ls", "ensure", "shell"):
            ns = self.parser.parse_args(["volumes", sub])
            self.assertEqual(ns.command, "volumes")
            self.assertEqual(ns.volumes_command, sub)

    def test_logs_with_no_service(self):
        ns = self.parser.parse_args(["logs"])
        self.assertEqual(ns.services, [])

    def test_logs_with_multiple_services(self):
        ns = self.parser.parse_args(["logs", "model-proxy", "open-webui"])
        self.assertEqual(ns.services, ["model-proxy", "open-webui"])

    def test_json_flag(self):
        ns = self.parser.parse_args(["--json", "status"])
        self.assertTrue(ns.json)

    def test_lock_renew_flag_parses(self):
        ns = self.parser.parse_args(["lock", "--renew", "--owner", "sess-1"])
        self.assertEqual(ns.command, "lock")
        self.assertTrue(ns.renew)
        self.assertIsNone(ns.preset)
        self.assertEqual(ns.owner, "sess-1")


class TestStatusCommand(unittest.TestCase):
    def test_status_proxy_unreachable(self):
        # Use a port we know is closed (0 means "pick a free one" in bind,
        # but for an HTTPConnection it tries to connect, which fails)
        with patch.object(cli, "DEFAULT_PROXY_PORT", 1):  # priv-only, will fail
            with _capture() as (out, err):
                rc = main(["status"])
        self.assertEqual(rc, EXIT_TRANSIENT)
        self.assertIn("not reachable", err.getvalue())

    def test_status_reachable_idle(self):
        responses = {
            ("GET", "/health"): (200, {"status": "ok", "mode": "idle"}),
            ("GET", "/mode"): (200, {"mode": "idle", "model": None, "switching": False}),
            ("GET", "/v1/models"): (200, {"data": []}),
        }
        with _stub_proxy(responses):
            with _capture() as (out, err):
                rc = main(["status"])
        self.assertEqual(rc, EXIT_OK)
        self.assertIn("Mode:", out.getvalue())
        self.assertIn("idle", out.getvalue())

    def test_status_json(self):
        responses = {
            ("GET", "/health"): (200, {"status": "ok", "mode": "llm"}),
            ("GET", "/mode"): (200, {"mode": "llm", "model": "qwen38", "switching": False}),
            ("GET", "/v1/models"): (200, {"data": [{"id": "x"}]}),
        }
        with _stub_proxy(responses):
            with _capture() as (out, _):
                rc = main(["--json", "status"])
        self.assertEqual(rc, EXIT_OK)
        payload = json.loads(out.getvalue())
        self.assertTrue(payload["reachable"])
        self.assertEqual(payload["active_model"], "qwen38")
        self.assertEqual(payload["preset_count"], 1)


class TestModelsCommand(unittest.TestCase):
    def test_models_fallback_to_local_when_proxy_down(self):
        with patch.object(cli, "DEFAULT_PROXY_PORT", 1):  # unreachable
            with _capture() as (out, _):
                rc = main(["models"])
        self.assertEqual(rc, EXIT_OK)
        self.assertIn("proxy unreachable", out.getvalue())
        # Surviving presets should be listed
        for preset in ("gemma4", "qwen38", "summarizer", "loop"):
            self.assertIn(preset, out.getvalue())

    def test_models_from_proxy(self):
        responses = {
            ("GET", "/health"): (200, {"status": "ok", "mode": "llm"}),
            ("GET", "/v1/models"): (200, {"data": [
                {
                    "id": "model-a",
                    "meta": {
                        "preset": "preset-a",
                        "name": "Preset A",
                        "vram_gb": 10.0,
                        "context": 32768,
                        "capabilities": {"vision": True},
                        "loaded": True,
                    },
                },
            ]}),
        }
        with _stub_proxy(responses):
            with _capture() as (out, _):
                rc = main(["models"])
        self.assertEqual(rc, EXIT_OK)
        self.assertIn("preset-a", out.getvalue())


class TestSwitchCommand(unittest.TestCase):
    def test_switch_success(self):
        responses = {
            ("POST", "/mode"): (200, {"mode": "llm", "model": "qwen38", "switched": True}),
        }
        with _stub_proxy(responses):
            with _capture() as (out, _):
                rc = main(["switch", "qwen38"])
        self.assertEqual(rc, EXIT_OK)
        self.assertIn("Loaded: qwen38", out.getvalue())

    def test_switch_unknown_preset(self):
        responses = {
            ("POST", "/mode"): (404, {"error": "unknown preset 'nope'"}),
        }
        with _stub_proxy(responses):
            with _capture() as (_, err):
                rc = main(["switch", "nope"])
        self.assertEqual(rc, EXIT_BACKEND_ERROR)
        self.assertIn("unknown preset", err.getvalue())

    def test_switch_vram_exceeded(self):
        responses = {
            ("POST", "/mode"): (422, {"error": "needs ~80GB VRAM"}),
        }
        with _stub_proxy(responses):
            with _capture() as (_, err):
                rc = main(["switch", "giant"])
        self.assertEqual(rc, EXIT_BACKEND_ERROR)
        self.assertIn("VRAM", err.getvalue())


class TestModeCommand(unittest.TestCase):
    def test_mode_get(self):
        responses = {
            ("GET", "/mode"): (200, {"mode": "comfyui", "model": None, "switching": False}),
        }
        with _stub_proxy(responses):
            with _capture() as (out, _):
                rc = main(["mode"])
        self.assertEqual(rc, EXIT_OK)
        self.assertEqual(out.getvalue().strip(), "comfyui")

    def test_mode_set(self):
        responses = {
            ("POST", "/mode"): (200, {"mode": "comfyui", "switched": True}),
        }
        with _stub_proxy(responses):
            with _capture() as (out, _):
                rc = main(["mode", "comfyui"])
        self.assertEqual(rc, EXIT_OK)
        self.assertIn("Mode: comfyui", out.getvalue())


class TestLockCommand(unittest.TestCase):
    """lock --renew heartbeats the proxy TTL; status shows the expiry."""

    def test_lock_renew_posts_renew_true(self):
        expiry = int(time.time()) + 900
        responses = {
            ("POST", "/mode"): (200, {
                "locked": "qwen38", "lock_owners": ["sess-1"],
                "renewed": True, "lock_expires_at": expiry, "lock_ttl_seconds": 900,
            }),
        }
        with _stub_proxy_recording(responses) as (_, _, requests):
            with _capture() as (out, _):
                rc = main(["lock", "--renew", "--owner", "sess-1"])
        self.assertEqual(rc, EXIT_OK)
        self.assertIn("Lock renewed: qwen38", out.getvalue())
        self.assertEqual(len(requests), 1)
        method, path, raw = requests[0]
        self.assertEqual((method, path), ("POST", "/mode"))
        self.assertEqual(json.loads(raw), {"renew": True, "owner": "sess-1"})

    def test_lock_renew_rejects_preset_arg(self):
        with _capture() as (_, err):
            rc = main(["lock", "qwen38", "--renew"])
        self.assertEqual(rc, EXIT_USER_ERROR)
        self.assertIn("--renew takes no preset", err.getvalue())

    def test_lock_renew_non_holder_409s(self):
        responses = {
            ("POST", "/mode"): (409, {"error": 'owner "sess-9" holds no lock or queue entry to renew'}),
        }
        with _stub_proxy(responses):
            with _capture() as (_, err):
                rc = main(["lock", "--renew", "--owner", "sess-9"])
        self.assertEqual(rc, EXIT_BACKEND_ERROR)
        self.assertIn("holds no lock", err.getvalue())

    def test_status_shows_lock_expiry(self):
        expiry = int(time.time()) + 900
        responses = {
            ("GET", "/health"): (200, {"status": "ok", "mode": "llm"}),
            ("GET", "/mode"): (200, {
                "mode": "llm", "model": "qwen38", "switching": False,
                "locked": "qwen38", "lock_owners": ["sess-1"],
                "lock_expires_at": expiry, "lock_ttl_seconds": 900,
            }),
            ("GET", "/v1/models"): (200, {"data": []}),
        }
        with _stub_proxy(responses):
            with _capture() as (out, _):
                rc = main(["status"])
        self.assertEqual(rc, EXIT_OK)
        self.assertIn("qwen38 (owners: sess-1), expires in", out.getvalue())


class TestVolumesCommand(unittest.TestCase):
    def test_volumes_ls_no_json(self):
        # `volumes ls` walks the host paths from volumes.toml — no Docker
        # required after the migration to direct binds.
        with _capture() as (out, _):
            rc = main(["volumes", "ls"])
        self.assertEqual(rc, EXIT_OK)
        # Should show the volumes declared in volumes.toml
        for name in ("llmc-state", "llmc-llama-models", "llmc-comfyui-models"):
            self.assertIn(name, out.getvalue())

    def test_volumes_ensure_parses(self):
        """`volumes ensure` replaces the old `create`/`refresh` pair."""
        parser = _build_parser()
        ns = parser.parse_args(["volumes", "ensure"])
        self.assertEqual(ns.volumes_command, "ensure")


class TestTrainCommands(unittest.TestCase):
    """Train CLI commands: 503-handling + happy path."""

    def test_train_status_returns_transient_when_service_inactive(self):
        responses = {
            ("GET", "/train/status"): (503, {
                "error": {
                    "message": "train service is not active",
                    "type": "service_inactive",
                    "code": 503,
                },
            }),
        }
        with _stub_proxy(responses):
            with _capture() as (out, err):
                rc = main(["train", "status"])
        self.assertEqual(rc, EXIT_TRANSIENT)
        self.assertIn("not active", err.getvalue())
        # Critical: must NOT silently print 'State: idle' as the legacy bug did
        self.assertNotIn("State: idle", out.getvalue())

    def test_train_status_happy_path(self):
        responses = {
            ("GET", "/train/status"): (200, {
                "state": "training",
                "step": 100,
                "total_steps": 2680,
                "loss": 0.12345,
                "epoch": 1,
                "total_epochs": 4,
                "elapsed_seconds": 180,
                "eta_seconds": 4500,
            }),
        }
        with _stub_proxy(responses):
            with _capture() as (out, _):
                rc = main(["train", "status"])
        self.assertEqual(rc, EXIT_OK)
        self.assertIn("State: training", out.getvalue())
        self.assertIn("100/2680", out.getvalue())
        self.assertIn("Loss:", out.getvalue())

    def test_train_logs_when_inactive(self):
        responses = {
            ("GET", "/train/logs?lines=50"): (503, {"error": {"message": "inactive"}}),
        }
        with _stub_proxy(responses):
            with _capture() as (_, err):
                rc = main(["train", "logs"])
        self.assertEqual(rc, EXIT_TRANSIENT)
        self.assertIn("Switch to train mode", err.getvalue())

    def test_train_list_happy_path(self):
        responses = {
            ("GET", "/train/jobs"): (200, {
                "files": [
                    {"name": "lora-a.safetensors", "size_mb": 123.4},
                    {"name": "lora-b.safetensors", "size_mb": 56.7},
                ],
            }),
        }
        with _stub_proxy(responses):
            with _capture() as (out, _):
                rc = main(["train", "list"])
        self.assertEqual(rc, EXIT_OK)
        self.assertIn("lora-a.safetensors", out.getvalue())
        self.assertIn("123.4", out.getvalue())

    def test_train_list_empty(self):
        responses = {
            ("GET", "/train/jobs"): (200, {"files": []}),
        }
        with _stub_proxy(responses):
            with _capture() as (out, _):
                rc = main(["train", "list"])
        self.assertEqual(rc, EXIT_OK)
        self.assertIn("no trained LoRAs", out.getvalue())


class TestDatasetCommands(unittest.TestCase):
    """Dataset CLI commands: 503-handling + happy path."""

    def test_caption_start_when_inactive(self):
        responses = {
            ("POST", "/train/caption"): (503, {
                "error": {"message": "train service is not active"},
            }),
        }
        with _stub_proxy(responses):
            with _capture() as (_, err):
                rc = main(["dataset", "caption", "my-set", "--trigger", "subject"])
        self.assertEqual(rc, EXIT_TRANSIENT)
        self.assertIn("not active", err.getvalue())

    def test_caption_status_happy_path(self):
        responses = {
            ("GET", "/train/caption/status"): (200, {
                "state": "running",
                "engine": "blip2",
                "dataset": "my-set",
                "captions_written": 30,
                "images_total": 100,
                "elapsed_seconds": 25,
            }),
        }
        with _stub_proxy(responses):
            with _capture() as (out, _):
                rc = main(["dataset", "caption-status"])
        self.assertEqual(rc, EXIT_OK)
        self.assertIn("State: running", out.getvalue())
        self.assertIn("30/100", out.getvalue())
        self.assertIn("blip2", out.getvalue())


class TestPassthroughCommands(unittest.TestCase):
    """Eval and bench commands forward args to underlying scripts."""

    def test_eval_no_args_shows_help(self):
        with _capture() as (_, err):
            rc = main(["eval"])
        self.assertEqual(rc, EXIT_USER_ERROR)
        self.assertIn("Usage: llmc eval", err.getvalue())

    def test_bench_no_args_shows_help(self):
        with _capture() as (_, err):
            rc = main(["bench"])
        self.assertEqual(rc, EXIT_USER_ERROR)
        self.assertIn("Usage: llmc bench", err.getvalue())

    def test_bench_unknown_subcommand(self):
        with _capture() as (_, err):
            rc = main(["bench", "not-a-real-thing"])
        self.assertEqual(rc, EXIT_USER_ERROR)
        self.assertIn("Unknown bench", err.getvalue())


class TestProxyClient(unittest.TestCase):
    def test_reachable_true_on_200(self):
        responses = {("GET", "/health"): (200, {"status": "ok"})}
        with _stub_proxy(responses) as (host, port):
            client = ProxyClient(host=host, port=port)
            self.assertTrue(client.reachable())

    def test_reachable_false_on_connection_refused(self):
        client = ProxyClient(host="127.0.0.1", port=1, timeout=0.5)
        self.assertFalse(client.reachable())

    def test_set_mode_sends_payload(self):
        captured = {}

        class CaptureHandler(_StubHandler):
            responses = {("POST", "/mode"): (200, {"mode": "llm"})}

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                captured["body"] = json.loads(self.rfile.read(length))
                super().do_POST()

        port = _free_port()
        server = http.server.HTTPServer(("127.0.0.1", port), CaptureHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = ProxyClient(host="127.0.0.1", port=port)
            status, _ = client.set_mode("llm", model="qwen38")
            self.assertEqual(status, 200)
            self.assertEqual(captured["body"], {"mode": "llm", "model": "qwen38"})
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
