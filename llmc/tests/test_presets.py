"""Tests for llmc.presets.

Validates that:
1. All 8 TOML presets parse cleanly
2. Schema validation catches malformed inputs
3. Generated env vars match what the legacy .env files would produce
   (migration fidelity check — proves no preset config was lost)
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from llmc.presets import (
    AssetSpec,
    Preset,
    PresetError,
    load_all,
    load_preset,
    preset_to_env,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = REPO_ROOT / "models"


def _parse_dotenv(path: Path) -> dict[str, str]:
    """Minimal .env parser — same logic as the legacy proxy.py."""
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        out[key.strip()] = value.strip()
    return out


class TestPresetLoading(unittest.TestCase):
    def test_load_all_presets(self):
        presets = load_all(MODELS_DIR)
        self.assertEqual(len(presets), 8, f"expected 8 presets, got {sorted(presets)}")

    def test_model_ids_unique(self):
        presets = load_all(MODELS_DIR)
        ids = [p.model_id for p in presets.values()]
        self.assertEqual(len(ids), len(set(ids)), "duplicate model_ids")

    def test_model_id_matches_gguf_filename(self):
        for path in MODELS_DIR.glob("*.toml"):
            preset = load_preset(path)
            self.assertEqual(
                preset.model_id,
                preset.model.file.removesuffix(".gguf"),
                f"{preset.name}: model_id should be GGUF filename minus .gguf",
            )

    def test_all_presets_have_required_fields(self):
        for path in MODELS_DIR.glob("*.toml"):
            preset = load_preset(path)
            self.assertTrue(preset.display_name, f"{preset.name}: missing name")
            self.assertGreater(preset.vram_gb, 0, f"{preset.name}: invalid vram_gb")
            self.assertTrue(preset.model.repo, f"{preset.name}: missing model.repo")
            self.assertTrue(preset.model.file, f"{preset.name}: missing model.file")


class TestSchemaValidation(unittest.TestCase):
    """Catch typos and structural errors at load time."""

    def _check_rejected(self, content: str, expect_substring: str) -> None:
        with tempfile.NamedTemporaryFile(suffix=".toml", mode="w", delete=False) as f:
            f.write(content)
            path = Path(f.name)
        try:
            with self.assertRaises(PresetError) as ctx:
                load_preset(path)
            self.assertIn(expect_substring, str(ctx.exception))
        finally:
            path.unlink()

    def test_missing_required_keys(self):
        self._check_rejected('name = "x"\nvram_gb = 5', "missing required key")

    def test_unknown_top_level_key(self):
        self._check_rejected(
            'name="x"\nvram_gb=5\nbogus=1\n[model]\nrepo="r"\nfile="f.gguf"',
            "unknown key",
        )

    def test_unknown_model_key(self):
        self._check_rejected(
            'name="x"\nvram_gb=5\n[model]\nrepo="r"\nfile="f.gguf"\ntypo=1',
            "unknown key",
        )

    def test_mmproj_url_and_file_mutually_exclusive(self):
        self._check_rejected(
            'name="x"\nvram_gb=5\n[model]\nrepo="r"\nfile="f.gguf"\n[mmproj]\nurl="u"\nfile="f"',
            "mutually exclusive",
        )

    def test_template_url_and_file_mutually_exclusive(self):
        self._check_rejected(
            'name="x"\nvram_gb=5\n[model]\nrepo="r"\nfile="f.gguf"\n[template]\nurl="u"\nfile="f"',
            "mutually exclusive",
        )

    def test_invalid_reasoning(self):
        self._check_rejected(
            'name="x"\nvram_gb=5\n[model]\nrepo="r"\nfile="f.gguf"\n[runtime]\nreasoning="yes"',
            "must be 'on' or 'off'",
        )

    def test_wrong_type(self):
        self._check_rejected(
            'name="x"\nvram_gb=5\n[model]\nrepo="r"\nfile="f.gguf"\n[runtime]\ncontext_size="big"',
            "expected int",
        )

    def test_invalid_toml(self):
        self._check_rejected("not = valid = toml = = =", "invalid TOML")

    def test_empty_model_repo(self):
        self._check_rejected(
            'name="x"\nvram_gb=5\n[model]\nrepo=""\nfile="f.gguf"',
            "must be non-empty",
        )


class TestMigrationFidelity(unittest.TestCase):
    """Verify each TOML preset produces the same effective container env as
    the legacy preset.env + make switch + docker-compose.yml pipeline.

    The legacy flow:
      models/<x>.env  →  make switch  →  .env (with MMPROJ_FILE / TEMPLATE_FILE
                                          derived from preset name)
                       →  compose interpolates into container env
                          (with PARALLEL_SLOTS defaulted to 1 by `${X:-1}`)

    The new flow:
      models/<x>.toml  →  preset_to_env()  →  docker run -e ...

    For migration fidelity, both pipelines must produce the same env vars at
    the container boundary. This test computes the legacy effective env for
    each preset and compares to preset_to_env() output.
    """

    # Keys that are part of the legacy preset format but not passed to the
    # container — they're metadata for the proxy/Makefile only.
    LEGACY_METADATA = {"VRAM_ESTIMATE_GB", "MODEL_NAME", "MMPROJ_URL", "TEMPLATE_URL"}

    def _legacy_effective_env(self, env_path: Path) -> dict[str, str]:
        """Compute what the container would actually receive in the legacy
        pipeline: preset.env vars + derived MMPROJ_FILE/TEMPLATE_FILE from
        `make switch` logic + compose defaults."""
        raw = _parse_dotenv(env_path)
        effective: dict[str, str] = {}

        # Pass-through keys
        for key in ("MODEL_REPO", "MODEL_FILE", "REASONING", "CONTEXT_SIZE",
                    "TEMPERATURE", "TOP_P", "TOP_K", "MIN_P",
                    "PRESENCE_PENALTY", "REPEAT_PENALTY"):
            if key in raw:
                effective[key] = raw[key]

        # PARALLEL_SLOTS: compose defaults to "1" via ${PARALLEL_SLOTS:-1}
        effective["PARALLEL_SLOTS"] = raw.get("PARALLEL_SLOTS", "1")

        # MMPROJ_FILE / TEMPLATE_FILE: derived by make switch (lines 174-185
        # of legacy Makefile). If URL is set in preset, filename = <stem>-mmproj.gguf.
        # If explicit MMPROJ_FILE is set in preset (e.g. summarizer), use that.
        preset_name = env_path.stem
        if raw.get("MMPROJ_FILE"):
            effective["MMPROJ_FILE"] = raw["MMPROJ_FILE"]
        elif raw.get("MMPROJ_URL"):
            effective["MMPROJ_FILE"] = f"{preset_name}-mmproj.gguf"
        else:
            effective["MMPROJ_FILE"] = ""

        if raw.get("TEMPLATE_FILE"):
            effective["TEMPLATE_FILE"] = raw["TEMPLATE_FILE"]
        elif raw.get("TEMPLATE_URL"):
            effective["TEMPLATE_FILE"] = f"{preset_name}-template.jinja"
        else:
            effective["TEMPLATE_FILE"] = ""

        return effective

    def _values_equivalent(self, key: str, legacy: str, new: str) -> bool:
        """Compare two env-var values for semantic equivalence. Numeric values
        are compared as floats so '0' == '0.0', '1' == '1.0'."""
        if legacy == new:
            return True
        numeric_keys = {"TEMPERATURE", "TOP_P", "TOP_K", "MIN_P", "CONTEXT_SIZE",
                        "PRESENCE_PENALTY", "REPEAT_PENALTY", "PARALLEL_SLOTS"}
        if key in numeric_keys:
            try:
                return float(legacy) == float(new)
            except ValueError:
                pass
        return False

    def test_each_toml_matches_legacy_env(self):
        toml_presets = load_all(MODELS_DIR)
        toml_by_name = {p.name: p for p in toml_presets.values()}

        for env_path in sorted(MODELS_DIR.glob("*.env")):
            with self.subTest(preset=env_path.stem):
                legacy = self._legacy_effective_env(env_path)
                toml_preset = toml_by_name.get(env_path.stem)
                self.assertIsNotNone(toml_preset, f"{env_path.stem}: no matching .toml")
                new_env = preset_to_env(toml_preset)

                # Every legacy key must appear in new env with equivalent value
                for key, legacy_val in sorted(legacy.items()):
                    if key not in new_env:
                        # Empty legacy values are OK to drop in new env
                        if legacy_val == "":
                            continue
                        self.fail(f"{env_path.stem}: new env missing key {key!r} "
                                  f"(legacy had {legacy_val!r})")
                    self.assertTrue(
                        self._values_equivalent(key, legacy_val, new_env[key]),
                        f"{env_path.stem}.{key}: legacy={legacy_val!r}, "
                        f"new={new_env[key]!r}",
                    )

                # Any new key that's not in legacy must have an empty legacy
                # value (i.e. legacy would have passed an empty string).
                for key in set(new_env) - set(legacy):
                    if new_env[key] == "":
                        continue
                    self.fail(f"{env_path.stem}: new env adds key {key!r}={new_env[key]!r} "
                              f"that legacy never set")


class TestAssetDerivation(unittest.TestCase):
    """Asset filename derivation: url -> <preset>-<suffix>, file -> as-is."""

    def test_mmproj_url_derives_filename(self):
        spec = AssetSpec(url="https://example.com/x")
        self.assertEqual(spec.derived_filename("qwen36", "-mmproj.gguf"), "qwen36-mmproj.gguf")

    def test_mmproj_file_used_as_is(self):
        spec = AssetSpec(file="custom-name.gguf")
        self.assertEqual(spec.derived_filename("qwen36", "-mmproj.gguf"), "custom-name.gguf")

    def test_unset_asset_returns_none(self):
        spec = AssetSpec()
        self.assertIsNone(spec.derived_filename("qwen36", "-mmproj.gguf"))
        self.assertFalse(spec.is_set)


if __name__ == "__main__":
    unittest.main()
