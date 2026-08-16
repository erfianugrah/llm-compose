"""TOML preset loader for LLM model configurations.

Schema:
    name = "..."                  # Human-readable name (shown in UIs)
    description = "..."           # One-line summary (optional multi-line)
    vram_gb = 20.2                # VRAM estimate for weights only
                                  # (proxy enforces vram_gb <= LIMIT - RESERVE)
    [model]
    repo = "org/name"             # HuggingFace repo
    file = "name.gguf"            # GGUF filename within repo
                                  # → ID = file with .gguf stripped

    [mmproj]                      # Optional — multimodal projection weights
    url = "..."                   # If set: auto-download to <preset>-mmproj.gguf
    file = "..."                  # If set: pre-placed file in /models dir
                                  # (mutually exclusive with url)

    [template]                    # Optional — custom chat template (jinja)
    url = "..."                   # If set: auto-download to <preset>-template.jinja
    file = "..."                  # If set: pre-placed file in /models dir

    [runtime]                     # All optional, sensible defaults
    reasoning = "on" | "off"      # llama-server --reasoning flag
    context_size = 65536          # -c
    parallel_slots = 1            # -np
    temperature = 1.0             # --temp
    top_p = 0.95
    top_k = 64
    min_p = 0.05
    presence_penalty = 1.5        # --presence-penalty
    repeat_penalty = 1.0          # --repeat-penalty

Validation rules:
    - vram_gb required (proxy budget check)
    - model.repo + model.file required
    - mmproj.url and mmproj.file are mutually exclusive
    - template.url and template.file are mutually exclusive
    - Unknown keys raise PresetError (catch typos)
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


class PresetError(ValueError):
    """Raised when a preset TOML fails validation."""


@dataclass(frozen=True)
class ModelSpec:
    repo: str
    file: str

    @property
    def model_id(self) -> str:
        """OpenAI-API model ID — GGUF filename minus `.gguf`."""
        return self.file.removesuffix(".gguf")


@dataclass(frozen=True)
class AssetSpec:
    """Optional asset (mmproj or template). Either auto-download (url set)
    or pre-placed (file set), never both."""

    url: Optional[str] = None
    file: Optional[str] = None

    @property
    def is_set(self) -> bool:
        return bool(self.url or self.file)

    def derived_filename(self, preset_name: str, suffix: str) -> Optional[str]:
        """Filename to use in /models. Either the explicit `file` override
        or the auto-derived `<preset_name><suffix>`."""
        if self.file:
            return self.file
        if self.url:
            return f"{preset_name}{suffix}"
        return None


@dataclass(frozen=True)
class RuntimeSpec:
    reasoning: Optional[str] = None  # "on" | "off" | None
    context_size: int = 65536
    parallel_slots: int = 1
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 64
    min_p: float = 0.0
    presence_penalty: Optional[float] = None
    repeat_penalty: Optional[float] = None
    spec_type: Optional[str] = None  # e.g. "draft-mtp" - llama.cpp --spec-type


@dataclass(frozen=True)
class Preset:
    name: str  # filename stem (e.g. "gemma4")
    display_name: str
    description: str
    vram_gb: float
    model: ModelSpec
    mmproj: AssetSpec = field(default_factory=AssetSpec)
    template: AssetSpec = field(default_factory=AssetSpec)
    runtime: RuntimeSpec = field(default_factory=RuntimeSpec)
    bench: dict = field(default_factory=dict)  # optional [bench] section (tokenizer, tags)

    @property
    def model_id(self) -> str:
        return self.model.model_id

    @property
    def mmproj_filename(self) -> Optional[str]:
        return self.mmproj.derived_filename(self.name, "-mmproj.gguf")

    @property
    def template_filename(self) -> Optional[str]:
        return self.template.derived_filename(self.name, "-template.jinja")

    @property
    def has_vision(self) -> bool:
        return self.mmproj.is_set


# Schema definition for validation. Maps section → allowed keys with types.
_TOP_LEVEL_KEYS = {"name", "description", "vram_gb", "model", "mmproj", "template", "runtime", "bench"}
_REQUIRED_TOP = {"name", "vram_gb", "model"}
_MODEL_KEYS = {"repo": str, "file": str}
_ASSET_KEYS = {"url": str, "file": str}
_BENCH_KEYS = {"tokenizer": str, "tags": str}
_RUNTIME_KEYS = {
    "reasoning": str,
    "context_size": int,
    "parallel_slots": int,
    "temperature": (int, float),
    "top_p": (int, float),
    "top_k": int,
    "min_p": (int, float),
    "presence_penalty": (int, float),
    "repeat_penalty": (int, float),
    "spec_type": str,
}


def _check_keys(section: str, data: dict, allowed: set | dict) -> None:
    allowed_keys = set(allowed.keys()) if isinstance(allowed, dict) else allowed
    unknown = set(data) - allowed_keys
    if unknown:
        raise PresetError(f"{section}: unknown key(s) {sorted(unknown)}; allowed: {sorted(allowed_keys)}")


def _check_types(section: str, data: dict, spec: dict) -> None:
    for key, expected in spec.items():
        if key not in data:
            continue
        value = data[key]
        if not isinstance(value, expected):
            type_name = expected.__name__ if isinstance(expected, type) else "/".join(t.__name__ for t in expected)
            raise PresetError(f"{section}.{key}: expected {type_name}, got {type(value).__name__}")


def _load_asset(section: str, data: dict | None) -> AssetSpec:
    if not data:
        return AssetSpec()
    _check_keys(section, data, _ASSET_KEYS)
    _check_types(section, data, _ASSET_KEYS)
    url = data.get("url", "").strip() or None
    file = data.get("file", "").strip() or None
    if url and file:
        raise PresetError(f"{section}: 'url' and 'file' are mutually exclusive")
    return AssetSpec(url=url, file=file)


def _load_runtime(data: dict | None) -> RuntimeSpec:
    if not data:
        return RuntimeSpec()
    _check_keys("runtime", data, _RUNTIME_KEYS)
    _check_types("runtime", data, _RUNTIME_KEYS)
    if "reasoning" in data and data["reasoning"] not in ("on", "off", ""):
        raise PresetError(f"runtime.reasoning: must be 'on' or 'off', got {data['reasoning']!r}")
    reasoning = data.get("reasoning", "").strip() or None
    return RuntimeSpec(
        reasoning=reasoning,
        context_size=data.get("context_size", 65536),
        parallel_slots=data.get("parallel_slots", 1),
        temperature=float(data.get("temperature", 1.0)),
        top_p=float(data.get("top_p", 0.95)),
        top_k=data.get("top_k", 64),
        min_p=float(data.get("min_p", 0.0)),
        presence_penalty=float(data["presence_penalty"]) if "presence_penalty" in data else None,
        repeat_penalty=float(data["repeat_penalty"]) if "repeat_penalty" in data else None,
        spec_type=data.get("spec_type", "").strip() or None,
    )


def load_preset(path: Path) -> Preset:
    """Load and validate a single preset TOML file. Raises PresetError on
    any schema violation. Preset name = filename stem."""
    if not path.exists():
        raise PresetError(f"{path}: file not found")
    try:
        data = tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as exc:
        raise PresetError(f"{path}: invalid TOML: {exc}") from exc

    _check_keys(str(path), data, _TOP_LEVEL_KEYS)
    missing = _REQUIRED_TOP - set(data)
    if missing:
        raise PresetError(f"{path}: missing required key(s) {sorted(missing)}")

    if not isinstance(data["vram_gb"], (int, float)):
        raise PresetError(f"{path}: vram_gb must be a number")
    if not isinstance(data["model"], dict):
        raise PresetError(f"{path}: [model] table missing or not a table")
    _check_keys(f"{path}:model", data["model"], _MODEL_KEYS)
    _check_types(f"{path}:model", data["model"], _MODEL_KEYS)
    for key in ("repo", "file"):
        if not data["model"].get(key, "").strip():
            raise PresetError(f"{path}: model.{key} is required and must be non-empty")

    bench_data = data.get("bench") or {}
    if not isinstance(bench_data, dict):
        raise PresetError(f"{path}: [bench] table missing or not a table")
    _check_keys(f"{path}:bench", bench_data, _BENCH_KEYS)
    _check_types(f"{path}:bench", bench_data, _BENCH_KEYS)

    return Preset(
        name=path.stem,
        display_name=data["name"],
        description=data.get("description", "").strip(),
        vram_gb=float(data["vram_gb"]),
        model=ModelSpec(repo=data["model"]["repo"], file=data["model"]["file"]),
        mmproj=_load_asset(f"{path}:mmproj", data.get("mmproj")),
        template=_load_asset(f"{path}:template", data.get("template")),
        runtime=_load_runtime(data.get("runtime")),
        bench=bench_data,
    )


def load_all(presets_dir: Path) -> dict[str, Preset]:
    """Load all *.toml files from `presets_dir`. Indexed by model_id
    (GGUF filename minus .gguf) for OpenAI-API compatibility.

    Raises PresetError if `presets_dir` doesn't exist — silently returning
    an empty dict from a missing directory was a footgun in early v2."""
    if not presets_dir.is_dir():
        raise PresetError(f"presets directory not found: {presets_dir}")
    presets: dict[str, Preset] = {}
    for path in sorted(presets_dir.glob("*.toml")):
        preset = load_preset(path)
        if preset.model_id in presets:
            raise PresetError(
                f"duplicate model_id {preset.model_id!r}: "
                f"{presets[preset.model_id].name}.toml and {preset.name}.toml"
            )
        presets[preset.model_id] = preset
    return presets


def preset_to_env(preset: Preset) -> dict[str, str]:
    """Render preset as the environment variables expected by llama-server's
    entrypoint script (see llama-server.Dockerfile). Used by the proxy when
    spawning the container — passed via Docker API `environment=...`."""
    env: dict[str, str] = {
        "MODEL_REPO": preset.model.repo,
        "MODEL_FILE": preset.model.file,
        "MMPROJ_FILE": preset.mmproj_filename or "",
        "TEMPLATE_FILE": preset.template_filename or "",
        "CONTEXT_SIZE": str(preset.runtime.context_size),
        "PARALLEL_SLOTS": str(preset.runtime.parallel_slots),
        "TEMPERATURE": str(preset.runtime.temperature),
        "TOP_P": str(preset.runtime.top_p),
        "TOP_K": str(preset.runtime.top_k),
        "MIN_P": str(preset.runtime.min_p),
    }
    if preset.runtime.reasoning:
        env["REASONING"] = preset.runtime.reasoning
    if preset.runtime.presence_penalty is not None:
        env["PRESENCE_PENALTY"] = str(preset.runtime.presence_penalty)
    if preset.runtime.repeat_penalty is not None:
        env["REPEAT_PENALTY"] = str(preset.runtime.repeat_penalty)
    if preset.runtime.spec_type:
        env["SPEC_TYPE"] = preset.runtime.spec_type
    return env
