"""Tests for llmc.volumes.

Pure-schema tests run without Docker. Filesystem integration tests use a
tempdir — no Docker daemon required (the migration to direct binds means
this module no longer talks to Docker at all).
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
    ensure,
    ensure_all,
    load,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
VOLUMES_TOML = REPO_ROOT / "volumes.toml"


class TestVolumeLoading(unittest.TestCase):
    """Schema validation. Doesn't touch the filesystem (beyond reading TOML)."""

    def test_load_repo_volumes_toml(self):
        registry = load(VOLUMES_TOML)
        self.assertGreater(len(registry.volumes), 0)
        # Sanity-check well-known names that the proxy + compose depend on
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

    def test_device_for_lookup(self):
        """Registry.device_for() returns the host path for a known name
        and raises VolumeError on unknown names."""
        registry = load(VOLUMES_TOML)
        path = registry.device_for("llmc-state")
        self.assertEqual(path, registry.volumes["llmc-state"].device)
        with self.assertRaises(VolumeError):
            registry.device_for("does-not-exist")

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


class TestEnsure(unittest.TestCase):
    """Host directory creation. Filesystem only, no Docker."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="llmc-vol-test-"))

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_ensure_creates_missing_directory(self):
        target = self.tmpdir / "fresh"
        spec = VolumeSpec(name="test-fresh", device=target)
        self.assertFalse(target.exists())
        self.assertEqual(ensure(spec), "created")
        self.assertTrue(target.is_dir())

    def test_ensure_idempotent_on_existing(self):
        target = self.tmpdir / "exists"
        target.mkdir()
        spec = VolumeSpec(name="test-exists", device=target)
        self.assertEqual(ensure(spec), "exists")

    def test_ensure_creates_nested_path(self):
        target = self.tmpdir / "a" / "b" / "c"
        spec = VolumeSpec(name="test-nested", device=target)
        self.assertEqual(ensure(spec), "created")
        self.assertTrue(target.is_dir())

    def test_ensure_refuses_when_path_is_a_file(self):
        target = self.tmpdir / "is-a-file"
        target.write_text("hi")
        spec = VolumeSpec(name="test-conflict", device=target)
        with self.assertRaises(VolumeError) as ctx:
            ensure(spec)
        self.assertIn("not a directory", str(ctx.exception))

    def test_ensure_all_returns_per_volume_actions(self):
        a = self.tmpdir / "a"
        b = self.tmpdir / "b"
        a.mkdir()
        registry = VolumeRegistry(volumes={
            "test-a": VolumeSpec(name="test-a", device=a),
            "test-b": VolumeSpec(name="test-b", device=b),
        })
        actions = ensure_all(registry)
        self.assertEqual(actions, {"test-a": "exists", "test-b": "created"})
        self.assertTrue(b.is_dir())


if __name__ == "__main__":
    unittest.main()
