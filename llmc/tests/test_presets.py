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
    NinferSpec,
    Preset,
    PresetError,
    load_all,
    load_preset,
    preset_to_env,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = REPO_ROOT / "models"
STAGING_DIR = REPO_ROOT / "models"


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
        # Don't hardcode a count - presets come and go. Assert the core set
        # is present and everything parsed (load_all raises on bad TOML).
        expected = {"gemma4", "summarizer", "loop", "erfi"}
        names = {p.name for p in presets.values()}
        missing = expected - names
        self.assertFalse(missing, f"missing presets: {missing}; got {sorted(names)}")

    def test_model_ids_unique(self):
        presets = load_all(MODELS_DIR)
        ids = [p.model_id for p in presets.values()]
        self.assertEqual(len(ids), len(set(ids)), "duplicate model_ids")

    def test_model_id_matches_gguf_filename(self):
        """llama.cpp presets derive their OpenAI model id from the GGUF name.
        Scoped to that engine: a NInfer artifact's served id is an alias
        chosen at serve time (--model-id), so those presets state it
        explicitly - asserted separately below rather than relaxed here."""
        checked = 0
        for path in MODELS_DIR.glob("*.toml"):
            preset = load_preset(path)
            if preset.engine != "llama":
                continue
            checked += 1
            self.assertEqual(
                preset.model_id,
                preset.model.file.removesuffix(".gguf"),
                f"{preset.name}: model_id should be GGUF filename minus .gguf",
            )
        self.assertGreater(checked, 0, "no llama presets found - did the glob break?")

    def test_ninfer_presets_state_their_model_id_explicitly(self):
        """The counterpart invariant. Without an explicit id the served alias
        would silently become the artifact filename, and every client's
        `model` parameter would stop matching.

        Globs both dirs: ninfer presets live in presets-staging/ until the Go
        proxy learns the engine field, and a test that silently iterated an
        empty set would pass vacuously."""
        paths = list(MODELS_DIR.glob("*.toml"))
        self.assertTrue(
            any(load_preset(p).engine == "ninfer" for p in paths),
            "no ninfer preset found in models/ or presets-staging/",
        )
        for path in paths:
            preset = load_preset(path)
            if preset.engine != "ninfer":
                continue
            self.assertTrue(
                preset.model.id,
                f"{preset.name}: engine=ninfer requires an explicit model.id",
            )
            self.assertNotIn(".ninfer", preset.model_id)

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
        self.assertEqual(spec.derived_filename("qwen38", "-mmproj.gguf"), "qwen38-mmproj.gguf")

    def test_mmproj_file_used_as_is(self):
        spec = AssetSpec(file="custom-name.gguf")
        self.assertEqual(spec.derived_filename("qwen38", "-mmproj.gguf"), "custom-name.gguf")

    def test_unset_asset_returns_none(self):
        spec = AssetSpec()
        self.assertIsNone(spec.derived_filename("qwen38", "-mmproj.gguf"))
        self.assertFalse(spec.is_set)


if __name__ == "__main__":
    unittest.main()


_NINFER_TOML = """
name = "Qwen3.8 27B NVFP4 on NInfer"
vram_gb = 31.7
engine = "ninfer"

[model]
repo = "neroued/Qwen3.8-27B-nvfp4-NInfer"
file = "qwen3_8_27b_nvfp4.ninfer"
id = "qwen3.8-27b-nvfp4"

[ninfer]
max_context = 262144
max_concurrency = 1
kv_dtype = "fp8"
spec = "mtp"
draft_tokens = 3
lm_head_draft = true
preserve_thinking = true
vision = true
host_state_slots = 0
host_kv_mib = 0
device_state_slots = 0
"""


class TestEngineSelection(unittest.TestCase):
    """A preset names the engine that serves it. Absent = llama.cpp, so every
    pre-existing preset keeps working untouched."""

    def _load(self, content: str) -> Preset:
        with tempfile.NamedTemporaryFile(suffix=".toml", mode="w", delete=False) as f:
            f.write(content)
            path = Path(f.name)
        try:
            return load_preset(path)
        finally:
            path.unlink()

    def test_engine_defaults_to_llama(self):
        p = self._load('name="x"\nvram_gb=5\n[model]\nrepo="r"\nfile="f.gguf"')
        self.assertEqual(p.engine, "llama")

    def test_every_shipped_preset_declares_a_known_engine(self):
        for name, preset in load_all(MODELS_DIR).items():
            self.assertIn(preset.engine, ("llama", "ninfer"), f"{name}: bad engine")

    def test_unknown_engine_rejected(self):
        with self.assertRaises(PresetError) as ctx:
            self._load('name="x"\nvram_gb=5\nengine="vllm"\n[model]\nrepo="r"\nfile="f.gguf"')
        self.assertIn("engine", str(ctx.exception))

    def test_ninfer_section_parsed(self):
        p = self._load(_NINFER_TOML)
        self.assertEqual(p.engine, "ninfer")
        self.assertIsInstance(p.ninfer, NinferSpec)
        self.assertEqual(p.ninfer.max_context, 262144)
        self.assertEqual(p.ninfer.kv_dtype, "fp8")
        self.assertEqual(p.ninfer.spec, "mtp")
        self.assertEqual(p.ninfer.draft_tokens, 3)
        self.assertTrue(p.ninfer.lm_head_draft)
        self.assertTrue(p.ninfer.vision)
        self.assertEqual(p.ninfer.device_state_slots, 0)

    def test_explicit_model_id_wins(self):
        """NInfer artifacts carry no .gguf name to derive an id from, so the
        preset states it outright."""
        p = self._load(_NINFER_TOML)
        self.assertEqual(p.model_id, "qwen3.8-27b-nvfp4")

    def test_ninfer_artifact_suffix_stripped_when_no_id(self):
        content = _NINFER_TOML.replace('id = "qwen3.8-27b-nvfp4"\n', "")
        p = self._load(content)
        self.assertEqual(p.model_id, "qwen3_8_27b_nvfp4")

    def test_ninfer_section_on_a_llama_preset_rejected(self):
        with self.assertRaises(PresetError) as ctx:
            self._load(
                'name="x"\nvram_gb=5\n[model]\nrepo="r"\nfile="f.gguf"\n'
                "[ninfer]\nmax_context=1024"
            )
        self.assertIn("ninfer", str(ctx.exception))

    def test_unknown_ninfer_key_rejected(self):
        with self.assertRaises(PresetError) as ctx:
            self._load(_NINFER_TOML + "\nbogus_flag = 1\n")
        self.assertIn("unknown key", str(ctx.exception))

    def test_llama_env_rendering_untouched_by_the_engine_field(self):
        """preset_to_env is the llama.cpp entrypoint contract; the engine
        field must not leak into it."""
        p = self._load('name="x"\nvram_gb=5\n[model]\nrepo="r"\nfile="f.gguf"')
        env = preset_to_env(p)
        self.assertNotIn("ENGINE", env)
        self.assertEqual(env["MODEL_FILE"], "f.gguf")

    def test_preset_to_env_refuses_a_ninfer_preset(self):
        """A ninfer preset has no llama.cpp entrypoint; rendering one as env
        vars would silently start the wrong engine."""
        p = self._load(_NINFER_TOML)
        with self.assertRaises(PresetError):
            preset_to_env(p)


class TestEngineAwareDisplayFields(unittest.TestCase):
    """`llmc models` read context from runtime.context_size and vision from the
    mmproj asset - both llama-only. A ninfer preset therefore showed context
    65536 (the runtime default it never uses) and vision "no" (it has no mmproj
    file; vision is a serve flag). Observed 2026-09-07."""

    def setUp(self):
        self.llama = load_preset(MODELS_DIR / "qwen38.toml")
        self.ninfer = load_preset(MODELS_DIR / "qwen38-ninfer.toml")

    def test_llama_context_unchanged(self):
        self.assertEqual(self.llama.effective_context, self.llama.runtime.context_size)

    def test_ninfer_context_comes_from_the_ninfer_section(self):
        self.assertEqual(self.ninfer.effective_context, self.ninfer.ninfer.max_context)
        self.assertEqual(self.ninfer.effective_context, 262144)
        self.assertNotEqual(self.ninfer.effective_context, self.ninfer.runtime.context_size)

    def test_llama_vision_still_derives_from_mmproj(self):
        self.assertEqual(self.llama.has_vision, self.llama.mmproj.is_set)

    def test_ninfer_vision_comes_from_the_serve_flag(self):
        self.assertTrue(self.ninfer.ninfer.vision, "fixture must have vision on")
        self.assertTrue(self.ninfer.has_vision)

    def test_ninfer_vision_off_is_reported_off(self):
        from dataclasses import replace
        off = replace(self.ninfer, ninfer=replace(self.ninfer.ninfer, vision=False))
        self.assertFalse(off.has_vision)
