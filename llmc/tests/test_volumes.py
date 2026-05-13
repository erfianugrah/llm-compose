"""Tests for llmc.volumes.

Pure-schema tests run without Docker. Integration tests that actually call
`docker volume create` are gated behind LLMC_TEST_DOCKER=1 so CI without
Docker can still validate the schema.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path

from llmc.volumes import (
    VolumeError,
    VolumeRegistry,
    VolumeSpec,
    create,
    create_all,
    inspect,
    load,
    remove,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
VOLUMES_TOML = REPO_ROOT / "volumes.toml"


class TestVolumeLoading(unittest.TestCase):
    """Schema validation. Doesn't touch Docker."""

    def test_load_repo_volumes_toml(self):
        registry = load(VOLUMES_TOML)
        self.assertGreater(len(registry.volumes), 0)
        # Sanity-check well-known volume names that the proxy depends on
        for required in (
            "llmc-state",
            "llmc-llama-models",
            "llmc-comfyui-models",
            "llmc-training-data",
            "llmc-webui-data",
        ):
            self.assertIn(required, registry.volumes,
                          f"volumes.toml missing required volume {required!r}")

    def test_paths_are_absolute_after_load(self):
        registry = load(VOLUMES_TOML)
        for spec in registry:
            self.assertTrue(spec.device.is_absolute(),
                            f"{spec.name}: device path not absolute: {spec.device}")

    def test_home_expansion(self):
        registry = load(VOLUMES_TOML)
        home = Path(os.environ["HOME"])
        state = registry.volumes["llmc-state"]
        self.assertTrue(
            str(state.device).startswith(str(home)),
            f"expected llmc-state under {home}, got {state.device}",
        )

    def _check_rejected(self, content: str, expect_substring: str) -> None:
        with tempfile.NamedTemporaryFile(suffix=".toml", mode="w", delete=False) as f:
            f.write(content)
            path = Path(f.name)
        try:
            with self.assertRaises(VolumeError) as ctx:
                load(path)
            self.assertIn(expect_substring, str(ctx.exception))
        finally:
            path.unlink()

    def test_invalid_toml(self):
        self._check_rejected("not = valid = = =", "invalid TOML")

    def test_unknown_top_level(self):
        self._check_rejected(
            'root = "/x"\nbogus = 1\n[volumes.a]\npath = "a"',
            "unknown top-level",
        )

    def test_missing_volumes(self):
        self._check_rejected(
            'root = "/x"\n[volumes]',
            "at least one volume",
        )

    def test_unknown_volume_key(self):
        self._check_rejected(
            'root = "/x"\n[volumes.a]\npath = "a"\ntypo = 1',
            "unknown key",
        )

    def test_empty_path(self):
        self._check_rejected(
            'root = "/x"\n[volumes.a]\npath = ""',
            "non-empty string",
        )

    def test_invalid_volume_name(self):
        self._check_rejected(
            'root = "/x"\n[volumes."-bad-start"]\npath = "a"',
            "invalid volume name",
        )

    def test_unresolved_env_var(self):
        # Set up a temp file referencing an env var we'll unset
        old_val = os.environ.pop("LLMC_NOT_A_REAL_VAR", None)
        try:
            self._check_rejected(
                'root = "${LLMC_NOT_A_REAL_VAR}/x"\n[volumes.a]\npath = "a"',
                "unresolved environment variable",
            )
        finally:
            if old_val is not None:
                os.environ["LLMC_NOT_A_REAL_VAR"] = old_val

    def test_absolute_path_overrides_root(self):
        with tempfile.NamedTemporaryFile(suffix=".toml", mode="w", delete=False) as f:
            f.write('root = "/default"\n[volumes.a]\npath = "/absolute/elsewhere"')
            path = Path(f.name)
        try:
            registry = load(path)
            self.assertEqual(str(registry.volumes["a"].device), "/absolute/elsewhere")
        finally:
            path.unlink()


@unittest.skipUnless(
    os.environ.get("LLMC_TEST_DOCKER") == "1",
    "Set LLMC_TEST_DOCKER=1 to run Docker integration tests",
)
class TestDockerIntegration(unittest.TestCase):
    """End-to-end with real Docker volume create/remove. Idempotent."""

    TEST_VOLUME = "llmc-test-volume-pleasedelete"

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="llmc-vol-test-"))
        self.spec = VolumeSpec(name=self.TEST_VOLUME, device=self.tmpdir)
        # Always start clean
        remove(self.TEST_VOLUME, force=True)

    def tearDown(self):
        remove(self.TEST_VOLUME, force=True)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_create_inspect_remove_roundtrip(self):
        action = create(self.spec)
        self.assertEqual(action, "created")
        info = inspect(self.TEST_VOLUME)
        self.assertIsNotNone(info)
        self.assertEqual(
            info["Options"]["device"],
            str(self.tmpdir),
        )
        # Idempotent — same spec returns "exists"
        action2 = create(self.spec)
        self.assertEqual(action2, "exists")
        self.assertTrue(remove(self.TEST_VOLUME))
        self.assertIsNone(inspect(self.TEST_VOLUME))

    def test_conflict_on_different_device(self):
        create(self.spec)
        other = tempfile.mkdtemp(prefix="llmc-vol-other-")
        try:
            conflicting = VolumeSpec(name=self.TEST_VOLUME, device=Path(other))
            with self.assertRaises(VolumeError) as ctx:
                create(conflicting)
            self.assertIn("already exists but points to", str(ctx.exception))
        finally:
            shutil.rmtree(other, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
