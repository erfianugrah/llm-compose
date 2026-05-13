"""Proxy state — persisted to the llmc-state named volume.

The proxy keeps a small amount of mutable state on disk so that:
    1. A proxy restart recovers the active mode + model without an
       expensive "what's running?" GPU service probe
    2. The CLI can read the state directly (via `llmc status`) without
       hitting the proxy HTTP endpoint
    3. Operators can inspect / reset state with `cat` / `rm`

State lives at /state/active.toml inside the proxy container, which is
the llmc-state named volume on the host. Schema:

    mode = "llm"            # "llm" | "comfyui" | "train" | "idle"
    model = "qwen36"        # Active preset name (only set when mode = "llm")
    updated_at = 1747162456 # Unix timestamp of last update

Writes are atomic via temp-file + rename. Both fields are optional in
the on-disk file; the loader fills in defaults.
"""

from __future__ import annotations

import os
import time
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

MODES = frozenset({"llm", "comfyui", "train", "idle"})


class StateError(ValueError):
    """Raised when state file is malformed."""


@dataclass(frozen=True)
class State:
    mode: str = "idle"
    model: Optional[str] = None
    updated_at: int = 0

    def __post_init__(self):
        if self.mode not in MODES:
            raise StateError(f"invalid mode {self.mode!r}; must be one of {sorted(MODES)}")
        if self.mode == "llm" and not self.model:
            # Allowed: someone is restarting the proxy with no model yet.
            # Logged by the caller, not an error.
            pass

    def to_toml(self) -> str:
        lines = [f'mode = "{self.mode}"']
        if self.model is not None:
            lines.append(f'model = "{self.model}"')
        lines.append(f"updated_at = {self.updated_at}")
        return "\n".join(lines) + "\n"


def load(path: Path) -> State:
    """Load state from `path`. Returns idle state if the file doesn't exist
    (first-run case). Raises StateError on schema violations."""
    if not path.exists():
        return State()
    try:
        data = tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as exc:
        raise StateError(f"{path}: invalid TOML: {exc}") from exc

    allowed = {"mode", "model", "updated_at"}
    unknown = set(data) - allowed
    if unknown:
        raise StateError(f"{path}: unknown key(s) {sorted(unknown)}")

    mode = data.get("mode", "idle")
    if not isinstance(mode, str):
        raise StateError(f"{path}: mode must be a string")
    model = data.get("model")
    if model is not None and not isinstance(model, str):
        raise StateError(f"{path}: model must be a string or omitted")
    updated_at = data.get("updated_at", 0)
    if not isinstance(updated_at, int):
        raise StateError(f"{path}: updated_at must be an integer")

    return State(mode=mode, model=model, updated_at=updated_at)


def save(path: Path, state: State) -> None:
    """Atomically write state to `path`. Uses temp-file + rename so a partial
    write can't corrupt the file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    stamped = replace(state, updated_at=state.updated_at or int(time.time()))
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(stamped.to_toml())
        # fsync the file and (best-effort) the dir for crash safety.
        fd = os.open(str(tmp), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, path)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def update(path: Path, **fields) -> State:
    """Atomic read-modify-write. Returns the new state.

    Always stamps updated_at with current time unless the caller passes
    an explicit value."""
    current = load(path)
    new = replace(current, **fields)
    if "updated_at" not in fields:
        new = replace(new, updated_at=int(time.time()))
    save(path, new)
    return new
