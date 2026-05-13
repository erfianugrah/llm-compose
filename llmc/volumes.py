"""Named Docker volume registry.

Reads volumes.toml at repo root, validates the schema, and provides helpers
to create/inspect named volumes backed by host bind mounts.

Architecture note: this is the host-side admin layer. The proxy doesn't
need to read volumes.toml at runtime — it just refers to volumes by name
when spawning containers via the Docker SDK. The registry exists to:

    1. Declaratively document which volumes the stack expects
    2. Provide a `create-all` operation for first-time setup or new machines
    3. Allow the same compose.yaml to work on any machine that's run
       `llmc volumes create` against this file
"""

from __future__ import annotations

import json
import os
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path


class VolumeError(ValueError):
    """Raised when volumes.toml fails validation or a docker command fails."""


@dataclass(frozen=True)
class VolumeSpec:
    name: str
    device: Path  # Absolute host path (env-expanded)

    @property
    def device_str(self) -> str:
        return str(self.device)


@dataclass(frozen=True)
class VolumeRegistry:
    """Parsed volumes.toml. Maps volume name → spec."""

    volumes: dict[str, VolumeSpec]

    def names(self) -> list[str]:
        return sorted(self.volumes)

    def __iter__(self):
        for name in self.names():
            yield self.volumes[name]


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
    """Docker volume names: [a-zA-Z0-9][a-zA-Z0-9_.-]*"""
    if not name:
        return False
    if not (name[0].isalnum()):
        return False
    return all(c.isalnum() or c in "_.-" for c in name)


# ── Docker CLI integration ───────────────────────────────────────────


def _docker(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    """Run `docker <args>`. Raises VolumeError on non-zero exit if check=True."""
    try:
        result = subprocess.run(
            ["docker", *args],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise VolumeError("`docker` CLI not found in PATH") from exc
    if check and result.returncode != 0:
        raise VolumeError(
            f"docker {' '.join(args)} failed:\n{result.stderr.strip()}"
        )
    return result


def inspect(name: str) -> dict | None:
    """Return docker volume metadata or None if the volume doesn't exist."""
    result = _docker(["volume", "inspect", name], check=False)
    if result.returncode != 0:
        return None
    payload = json.loads(result.stdout)
    return payload[0] if payload else None


def create(spec: VolumeSpec, *, ensure_dir: bool = True) -> str:
    """Create a named volume backed by `spec.device`. Idempotent: if a volume
    with this name already exists and points to the same device, returns
    'exists'. If it exists with a different device, raises VolumeError so
    the operator can decide whether to migrate.

    `ensure_dir=True` creates the host directory if missing.
    """
    if ensure_dir:
        spec.device.mkdir(parents=True, exist_ok=True)

    existing = inspect(spec.name)
    if existing is not None:
        existing_device = (existing.get("Options") or {}).get("device", "")
        if existing_device == spec.device_str:
            return "exists"
        raise VolumeError(
            f"volume {spec.name!r} already exists but points to "
            f"{existing_device!r}, not {spec.device_str!r}. "
            f"Run `docker volume rm {spec.name}` to recreate "
            f"(no data lost — bind-mount data lives at the device path)."
        )

    _docker([
        "volume", "create",
        "--driver", "local",
        "--opt", "type=none",
        "--opt", "o=bind",
        "--opt", f"device={spec.device_str}",
        spec.name,
    ])
    return "created"


def create_all(registry: VolumeRegistry, *, ensure_dir: bool = True) -> dict[str, str]:
    """Create every volume in the registry. Returns {name: action} where
    action is 'created' or 'exists'. Raises on first conflict."""
    actions: dict[str, str] = {}
    for spec in registry:
        actions[spec.name] = create(spec, ensure_dir=ensure_dir)
    return actions


def remove(name: str, *, force: bool = False) -> bool:
    """Remove a named volume. Returns True if it existed and was removed.
    Bind-mount data is NOT deleted — the volume is just unregistered from
    Docker; files at the device path remain."""
    if inspect(name) is None:
        return False
    args = ["volume", "rm"]
    if force:
        args.append("--force")
    args.append(name)
    _docker(args)
    return True
