"""Tests for llmc.state."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from llmc.state import State, StateError, load, save, update


class TestStateValidation(unittest.TestCase):
    def test_default_state(self):
        s = State()
        self.assertEqual(s.mode, "idle")
        self.assertIsNone(s.model)
        self.assertEqual(s.updated_at, 0)

    def test_invalid_mode(self):
        with self.assertRaises(StateError):
            State(mode="bogus")

    def test_valid_modes(self):
        for mode in ("llm", "comfyui", "train", "idle"):
            State(mode=mode)  # should not raise

    def test_to_toml_includes_model_when_set(self):
        toml = State(mode="llm", model="qwen36", updated_at=42).to_toml()
        self.assertIn('mode = "llm"', toml)
        self.assertIn('model = "qwen36"', toml)
        self.assertIn("updated_at = 42", toml)

    def test_to_toml_omits_model_when_none(self):
        toml = State(mode="comfyui").to_toml()
        self.assertNotIn("model", toml)


class TestStatePersistence(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(
            suffix=".toml", delete=False, mode="w"
        )
        self.tmp.close()
        self.path = Path(self.tmp.name)
        # Start with no file (load should return defaults)
        self.path.unlink()

    def tearDown(self):
        if self.path.exists():
            self.path.unlink()

    def test_load_missing_returns_idle(self):
        s = load(self.path)
        self.assertEqual(s.mode, "idle")
        self.assertIsNone(s.model)

    def test_save_and_load_roundtrip(self):
        save(self.path, State(mode="llm", model="qwen36", updated_at=1000))
        loaded = load(self.path)
        self.assertEqual(loaded.mode, "llm")
        self.assertEqual(loaded.model, "qwen36")
        self.assertEqual(loaded.updated_at, 1000)

    def test_lock_roundtrip(self):
        save(self.path, State(
            mode="llm",
            model="qwen36",
            locked="loop",
            lock_owners=["a", "b"],
            updated_at=1000
        ))
        loaded = load(self.path)
        self.assertEqual(loaded.mode, "llm")
        self.assertEqual(loaded.model, "qwen36")
        self.assertEqual(loaded.locked, "loop")
        self.assertEqual(loaded.lock_owners, ["a", "b"])
        self.assertEqual(loaded.updated_at, 1000)

    def test_save_is_atomic_no_temp_file_left(self):
        save(self.path, State(mode="comfyui"))
        # No .tmp file should exist after a successful save
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        self.assertFalse(tmp_path.exists())

    def test_save_stamps_timestamp_when_zero(self):
        save(self.path, State(mode="llm", model="qwen36", updated_at=0))
        loaded = load(self.path)
        self.assertGreater(loaded.updated_at, 0)

    def test_update_is_atomic_rmw(self):
        save(self.path, State(mode="llm", model="qwen36", updated_at=1000))
        new = update(self.path, mode="comfyui", model=None)
        self.assertEqual(new.mode, "comfyui")
        self.assertIsNone(new.model)
        self.assertGreater(new.updated_at, 1000)  # bumped to now

        # Verify it's actually persisted
        loaded = load(self.path)
        self.assertEqual(loaded.mode, "comfyui")
        self.assertIsNone(loaded.model)

    def test_update_preserves_unchanged_fields(self):
        save(self.path, State(mode="llm", model="qwen36", updated_at=1000))
        # Only update mode; model should stay
        update(self.path, mode="idle")
        loaded = load(self.path)
        self.assertEqual(loaded.mode, "idle")
        # When transitioning to idle the model is intentionally retained
        # so the next /v1 request remembers the last preset to load
        self.assertEqual(loaded.model, "qwen36")


class TestStateLoadValidation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(
            suffix=".toml", delete=False, mode="w"
        )
        self.tmp.close()
        self.path = Path(self.tmp.name)

    def tearDown(self):
        if self.path.exists():
            self.path.unlink()

    def _write(self, content: str):
        self.path.write_text(content)

    def test_unknown_key(self):
        self._write('mode = "llm"\nbogus = 1')
        with self.assertRaises(StateError) as ctx:
            load(self.path)
        self.assertIn("unknown key", str(ctx.exception))

    def test_invalid_mode(self):
        self._write('mode = "bogus"')
        with self.assertRaises(StateError):
            load(self.path)

    def test_wrong_type_for_model(self):
        self._write('mode = "llm"\nmodel = 123')
        with self.assertRaises(StateError) as ctx:
            load(self.path)
        self.assertIn("model must be a string", str(ctx.exception))

    def test_invalid_toml(self):
        self._write("not toml = = =")
        with self.assertRaises(StateError) as ctx:
            load(self.path)
        self.assertIn("invalid TOML", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
