"""Tests for llmc.orchestrator.

Unit tests use unittest.mock to fake the Docker SDK — they exercise the
orchestrator's logic (label filtering, mutual exclusion, error paths)
without touching a real daemon.

Integration tests (LLMC_TEST_DOCKER=1) hit the real Docker daemon. They
use a tiny `alpine` image with the llmc labels to verify the
mutual-exclusion logic actually works end-to-end. They skip if docker
is not installed or not reachable.
"""

from __future__ import annotations

import os
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

try:
    import docker  # noqa: F401
    HAS_DOCKER = True
except ImportError:
    HAS_DOCKER = False

from llmc.orchestrator import (
    COMFYUI_SERVICE,
    GPU_LABEL,
    LLAMA_SERVICE,
    SERVICE_LABEL,
    SERVICES,
    TRAIN_SERVICE,
    GpuService,
    Orchestrator,
    OrchestratorError,
    _llama_command,
)
from llmc.presets import load_preset

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PRESET = load_preset(REPO_ROOT / "models" / "gemma4.toml")


class TestServiceDefinitions(unittest.TestCase):
    def test_services_have_unique_modes(self):
        modes = [svc.mode for svc in SERVICES.values()]
        self.assertEqual(len(modes), len(set(modes)))

    def test_services_have_unique_ports(self):
        ports = [svc.internal_port for svc in SERVICES.values()]
        self.assertEqual(len(ports), len(set(ports)))

    def test_all_three_modes_defined(self):
        self.assertEqual(set(SERVICES), {"llm", "comfyui", "train"})


class TestLlamaCommand(unittest.TestCase):
    def test_command_uses_model_args(self):
        cmd = _llama_command()
        self.assertIn("MODEL_FILE", cmd)
        self.assertIn("MODEL_REPO", cmd)
        self.assertIn("--hf-repo", cmd)
        self.assertIn("/models/", cmd)

    def test_command_has_required_flags(self):
        cmd = _llama_command()
        for flag in ("--port 8080", "--host 0.0.0.0", "-ngl 99", "--flash-attn on",
                     "-ctk q8_0", "-ctv q8_0", "--metrics"):
            self.assertIn(flag, cmd, f"missing {flag!r} in llama command")


@unittest.skipUnless(HAS_DOCKER, "docker SDK not installed; pip install docker")
class TestOrchestratorMockedSDK(unittest.TestCase):
    """Verify the orchestrator's logic with a mock Docker client.

    Even with a mocked client these tests need the docker package importable
    because the orchestrator imports docker.errors / docker.types lazily."""

    def _orchestrator(self, containers=None):
        client = MagicMock()
        client.containers.list.return_value = containers or []
        return Orchestrator(client=client), client

    def _make_container(self, *, status="running", name="x", mode="llm"):
        c = MagicMock()
        c.name = name
        c.status = status
        c.labels = {GPU_LABEL: mode, SERVICE_LABEL: name}
        return c

    def test_current_mode_idle_when_no_containers(self):
        orch, _ = self._orchestrator(containers=[])
        self.assertEqual(orch.current_mode(), "idle")

    def test_current_mode_returns_running_llm(self):
        c = self._make_container(mode="llm")
        orch, _ = self._orchestrator(containers=[c])
        self.assertEqual(orch.current_mode(), "llm")

    def test_current_mode_ignores_stopped(self):
        c = self._make_container(mode="llm", status="exited")
        orch, _ = self._orchestrator(containers=[c])
        self.assertEqual(orch.current_mode(), "idle")

    def test_current_mode_ignores_invalid_label(self):
        c = self._make_container(mode="not-a-mode")
        orch, _ = self._orchestrator(containers=[c])
        self.assertEqual(orch.current_mode(), "idle")

    def test_stop_gpu_services_stops_and_removes(self):
        running = self._make_container(name="r", status="running")
        stopped = self._make_container(name="s", status="exited")
        orch, _ = self._orchestrator(containers=[running, stopped])
        result = orch.stop_gpu_services()
        running.stop.assert_called_once()
        running.remove.assert_called_once()
        # exited container: no stop, but remove yes
        stopped.stop.assert_not_called()
        stopped.remove.assert_called_once()
        self.assertEqual(sorted(result), ["r", "s"])

    def test_spawn_stops_existing_then_runs(self):
        existing = self._make_container(name="old", status="running")
        orch, client = self._orchestrator(containers=[existing])
        client.containers.run.return_value = MagicMock(name="new-container")
        orch.spawn_llama(PRESET)
        existing.stop.assert_called_once()
        existing.remove.assert_called_once()
        client.containers.run.assert_called_once()

    def test_spawn_passes_correct_env_and_labels(self):
        orch, client = self._orchestrator(containers=[])
        client.containers.run.return_value = MagicMock()
        orch.spawn_llama(PRESET)
        _, kwargs = client.containers.run.call_args
        env = kwargs["environment"]
        self.assertEqual(env["MODEL_FILE"], PRESET.model.file)
        self.assertEqual(env["MMPROJ_FILE"], PRESET.mmproj_filename)
        self.assertEqual(env["CONTEXT_SIZE"], str(PRESET.runtime.context_size))
        self.assertEqual(kwargs["labels"][GPU_LABEL], "llm")
        self.assertEqual(kwargs["labels"][SERVICE_LABEL], "llama-server")
        self.assertEqual(kwargs["name"], "llama_server")
        # GPU device request present
        self.assertEqual(len(kwargs["device_requests"]), 1)

    def test_spawn_image_not_found_raises(self):
        from docker.errors import ImageNotFound

        orch, client = self._orchestrator(containers=[])
        client.containers.run.side_effect = ImageNotFound("nope")
        with self.assertRaises(OrchestratorError) as ctx:
            orch.spawn_llama(PRESET)
        self.assertIn("not found", str(ctx.exception))

    def test_spawn_uses_correct_volumes_for_llama(self):
        orch, client = self._orchestrator(containers=[])
        client.containers.run.return_value = MagicMock()
        orch.spawn_llama(PRESET)
        _, kwargs = client.containers.run.call_args
        vols = kwargs["volumes"]
        self.assertIn("llmc-llama-cache", vols)
        self.assertIn("llmc-llama-models", vols)
        self.assertEqual(vols["llmc-llama-models"]["bind"], "/models")

    def test_spawn_train_mounts_distinct_volumes_for_models_vs_loras(self):
        orch, client = self._orchestrator(containers=[])
        client.containers.run.return_value = MagicMock()
        orch.spawn_train()
        _, kwargs = client.containers.run.call_args
        vols = kwargs["volumes"]
        # /models read-only, /loras writable, different volumes
        self.assertIn("llmc-comfyui-models", vols)
        self.assertIn("llmc-comfyui-loras", vols)
        self.assertEqual(vols["llmc-comfyui-models"]["mode"], "ro")
        self.assertEqual(vols["llmc-comfyui-loras"]["mode"], "rw")


@unittest.skipUnless(
    os.environ.get("LLMC_TEST_DOCKER") == "1",
    "Set LLMC_TEST_DOCKER=1 to run Docker integration tests",
)
class TestOrchestratorIntegration(unittest.TestCase):
    """End-to-end with real Docker daemon. Uses alpine as a stand-in for
    GPU services — we verify label-based mutual exclusion, not actual
    GPU container behaviour."""

    TEST_NETWORK = "llmc-test-net"
    TEST_LABEL_PREFIX = "llmc-test-"

    def setUp(self):
        import docker as docker_pkg
        self.docker = docker_pkg.from_env()
        # Idempotent network create
        try:
            self.docker.networks.get(self.TEST_NETWORK)
        except docker_pkg.errors.NotFound:
            self.docker.networks.create(self.TEST_NETWORK, driver="bridge")
        self.orch = Orchestrator(client=self.docker, network=self.TEST_NETWORK)
        self._cleanup()

    def tearDown(self):
        self._cleanup()
        try:
            self.docker.networks.get(self.TEST_NETWORK).remove()
        except Exception:
            pass

    def _cleanup(self):
        for c in self.docker.containers.list(all=True, filters={"label": GPU_LABEL}):
            try:
                c.remove(force=True)
            except Exception:
                pass

    def _spawn_fake(self, mode: str, name: str):
        """Spawn an alpine container with our labels, to simulate a GPU service."""
        return self.docker.containers.run(
            "alpine",
            ["sleep", "60"],
            name=name,
            detach=True,
            network=self.TEST_NETWORK,
            labels={GPU_LABEL: mode, SERVICE_LABEL: name},
        )

    def test_current_mode_detects_running_container(self):
        self._spawn_fake("llm", f"{self.TEST_LABEL_PREFIX}llama")
        self.assertEqual(self.orch.current_mode(), "llm")

    def test_stop_gpu_services_removes_all_labelled(self):
        # In normal operation only one GPU service runs at a time. This test
        # spawns two to verify stop_gpu_services() cleans up both. The
        # current_mode() return value when two are running is intentionally
        # unspecified (Docker doesn't guarantee containers.list ordering)
        # — the proxy's swap protocol ensures this doesn't happen in practice.
        self._spawn_fake("llm", f"{self.TEST_LABEL_PREFIX}llama")
        self._spawn_fake("comfyui", f"{self.TEST_LABEL_PREFIX}comfy")
        self.assertIn(self.orch.current_mode(), ("llm", "comfyui"))
        stopped = self.orch.stop_gpu_services()
        self.assertEqual(len(stopped), 2)
        self.assertEqual(self.orch.current_mode(), "idle")


if __name__ == "__main__":
    unittest.main()
