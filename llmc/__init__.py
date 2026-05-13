"""llmc — llm-compose orchestration package.

Stdlib-only Python (plus `docker` package in proxy runtime).
Public modules:
    presets  — TOML preset loader + schema validation
    state    — atomic state file r/w (active mode, current model)
    volumes  — named volume registry (volumes.toml → docker volume create)
    orchestrator — Docker SDK container lifecycle for GPU services
    proxy    — HTTP proxy + mode switching
    cli      — `python -m llmc` entry point
"""

__version__ = "2.0.0-dev"
