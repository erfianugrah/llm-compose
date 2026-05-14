"""Bind-mount path registry.

Reads volumes.toml at repo root, validates the schema, and provides helpers
to look up host paths and ensure their backing directories exist.

History: this module used to create Docker named volumes with `local` driver
+ `o=bind` indirection. That indirection rots on Docker Desktop / WSL2
restart (the snapshot hash under `/run/desktop/mnt/host/wsl/docker-desktop-
bind-mounts/<distro>/<hash>` goes stale). Direct bind mounts have no such
fragility. The CLI surface (`llmc volumes ls`/`ensure`/`shell`) is preserved
for muscle memory; `refresh` was a workaround for the rot bug and is gone.

Architecture: this is the host-side admin layer. The proxy reads the same
file at runtime (mounted at /volumes.toml) so it can pass host paths to
the Docker SDK when spawning GPU service containers.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path


class VolumeError(ValueError):
    """Raised when volumes.toml fails validation."""


@dataclass(frozen=True)
class VolumeSpec:
    """Named bind-mount host path.

    `name` is the logical handle used in code (and in `llmc volumes shell`
    paths). `device` is the absolute host directory the name maps to.
    """

    name: str
    device: Path  # Absolute host path (env-expanded)

    @property
    def device_str(self) -> str:
        return str(self.device)


@dataclass(frozen=True)
class VolumeRegistry:
    """Parsed volumes.toml. Maps logical name → spec."""

    volumes: dict[str, VolumeSpec]

    def names(self) -> list[str]:
        return sorted(self.volumes)

    def __iter__(self):
        for name in self.names():
            yield self.volumes[name]

    def get(self, name: str) -> VolumeSpec | None:
        return self.volumes.get(name)

    def device_for(self, name: str) -> Path:
        """Look up a host path by logical name. Raises if unknown."""
        spec = self.volumes.get(name)
        if spec is None:
            raise VolumeError(f"unknown volume {name!r}")
        return spec.device


_TOP_LEVEL_KEYS = {"root", "volumes"}
_VOLUME_KEYS = {"path"}


def _expand(value: str) -> str:
    """Expand ${VAR} and ~ in a string. Raises if a referenced env var is
    unset (better to fail loud than silently bind /undefined/whatever)."""
    expanded = os.path.expandvars(value)
    if "$" in expanded:
        raise VolumeError(f"unresolved environment variable in {value!r}: {expanded!r}")
    return os.path.expanduser(expanded)


def load(path: Path) -> VolumeRegistry:
    """Load and validate volumes.toml. Raises VolumeError on schema issues."""
    if not path.exists():
        raise VolumeError(f"{path}: file not found")
    try:
        data = tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as exc:
        raise VolumeError(f"{path}: invalid TOML: {exc}") from exc

    unknown = set(data) - _TOP_LEVEL_KEYS
    if unknown:
        raise VolumeError(f"{path}: unknown top-level key(s) {sorted(unknown)}")

    root = data.get("root", "${HOME}/docker-volumes")
    if not isinstance(root, str):
        raise VolumeError(f"{path}: root must be a string")
    root_path = Path(_expand(root))

    raw_volumes = data.get("volumes", {})
    if not isinstance(raw_volumes, dict):
        raise VolumeError(f"{path}: volumes must be a table")
    if not raw_volumes:
        raise VolumeError(f"{path}: at least one volume required")

    volumes: dict[str, VolumeSpec] = {}
    for name, spec in raw_volumes.items():
        if not isinstance(spec, dict):
            raise VolumeError(f"{path}: volumes.{name} must be a table")
        unknown_keys = set(spec) - _VOLUME_KEYS
        if unknown_keys:
            raise VolumeError(f"{path}: volumes.{name} has unknown key(s) {sorted(unknown_keys)}")
        rel_path = spec.get("path")
        if not isinstance(rel_path, str) or not rel_path:
            raise VolumeError(f"{path}: volumes.{name}.path must be a non-empty string")
        device = Path(_expand(rel_path))
        if not device.is_absolute():
            device = root_path / device
        if not _is_valid_volume_name(name):
            raise VolumeError(
                f"{path}: invalid volume name {name!r}; must match "
                f"[a-zA-Z0-9][a-zA-Z0-9_.-]*"
            )
        volumes[name] = VolumeSpec(name=name, device=device)

    return VolumeRegistry(volumes=volumes)


def _is_valid_volume_name(name: str) -> bool:
    """Valid logical names: [a-zA-Z0-9][a-zA-Z0-9_.-]*

    We keep Docker's volume-name grammar for back-compat — these names are
    still used in `llmc volumes shell` mount paths and elsewhere where
    a stable, filesystem-safe identifier is needed.
    """
    if not name:
        return False
    if not (name[0].isalnum()):
        return False
    return all(c.isalnum() or c in "_.-" for c in name)


# ── Host directory ensure ────────────────────────────────────────────


def ensure(spec: VolumeSpec) -> str:
    """Create the host directory if missing. Returns 'created' or 'exists'."""
    if spec.device.exists():
        if not spec.device.is_dir():
            raise VolumeError(
                f"{spec.name}: {spec.device} exists but is not a directory"
            )
        return "exists"
    spec.device.mkdir(parents=True, exist_ok=True)
    return "created"


def ensure_all(registry: VolumeRegistry) -> dict[str, str]:
    """Ensure every host directory in the registry exists. Returns
    {name: 'created' | 'exists'}. Raises VolumeError on first conflict."""
    actions: dict[str, str] = {}
    for spec in registry:
        actions[spec.name] = ensure(spec)
    return actions
