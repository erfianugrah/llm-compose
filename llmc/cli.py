"""llmc CLI — operator-facing command-line tool.

Replaces the bulk of the legacy Makefile. Subcommands either talk to the
running proxy over HTTP (mode swaps, status, model switches) or operate
on the Docker daemon directly via the `docker` CLI (volume management,
stack lifecycle, logs).

Pure stdlib — no third-party deps on the host. The `docker` Python SDK
is only required inside the proxy container.

Subcommand groups:
    Core orchestration  status, health, mode, switch, models
    Volumes             volumes ls, volumes create, volumes shell
    Stack lifecycle     up, down, restart, logs, setup, build, pull, push

Each subcommand exits with:
    0 success
    1 user error (bad args, validation failure)
    2 transient error (proxy unreachable, daemon down)
    3 backend error (proxy returned 5xx, container won't start)
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

from llmc.presets import PresetError, load_all
from llmc.volumes import VolumeError, create_all, inspect, load as load_volumes


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROXY_HOST = os.environ.get("LLMC_PROXY_HOST", "127.0.0.1")
DEFAULT_PROXY_PORT = int(os.environ.get("LLMC_PROXY_PORT", "11434"))
DEFAULT_PRESETS_DIR = Path(os.environ.get("LLMC_PRESETS_DIR_HOST",
                                          str(REPO_ROOT / "models")))
DEFAULT_VOLUMES_TOML = Path(os.environ.get("LLMC_VOLUMES_TOML",
                                           str(REPO_ROOT / "volumes.toml")))


# Exit codes
EXIT_OK = 0
EXIT_USER_ERROR = 1
EXIT_TRANSIENT = 2
EXIT_BACKEND_ERROR = 3


# ── Output helpers ─────────────────────────────────────────────────────


def _err(msg: str) -> None:
    print(msg, file=sys.stderr)


def _print_json(payload) -> None:
    json.dump(payload, sys.stdout, indent=2, sort_keys=True)
    print()


def _print_kv(rows: list[tuple[str, str]], *, pad: int = 14) -> None:
    """Print 'key: value' rows aligned."""
    for k, v in rows:
        print(f"{k:<{pad}} {v}")


def _print_table(headers: list[str], rows: list[list[str]]) -> None:
    """Plain ASCII table. No external deps."""
    if not rows:
        # Still print headers so the format is consistent
        print("  ".join(headers))
        return
    cols = list(zip(*([headers] + rows)))
    widths = [max(len(str(cell)) for cell in col) for col in cols]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*["-" * w for w in widths]))
    for row in rows:
        print(fmt.format(*row))


# ── Proxy HTTP client ──────────────────────────────────────────────────


class ProxyClient:
    """Thin wrapper for the proxy's HTTP API. Returns (status, json_body).

    Default host/port are read from the module attributes at instantiation
    time, not at class definition time. This lets tests patch
    DEFAULT_PROXY_PORT/DEFAULT_PROXY_HOST and have it take effect.
    """

    def __init__(self, host: Optional[str] = None, port: Optional[int] = None,
                 timeout: float = 5.0):
        self.host = host if host is not None else DEFAULT_PROXY_HOST
        self.port = port if port is not None else DEFAULT_PROXY_PORT
        self.timeout = timeout

    def _request(self, method: str, path: str, body: Optional[dict] = None,
                 timeout: Optional[float] = None) -> tuple[int, dict]:
        conn = http.client.HTTPConnection(self.host, self.port, timeout=timeout or self.timeout)
        try:
            data = json.dumps(body).encode() if body is not None else None
            headers = {"Content-Type": "application/json"} if data else {}
            conn.request(method, path, body=data, headers=headers)
            resp = conn.getresponse()
            raw = resp.read()
            try:
                payload = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                payload = {"raw": raw.decode("utf-8", "replace")}
            return resp.status, payload
        finally:
            conn.close()

    def reachable(self) -> bool:
        try:
            status, _ = self._request("GET", "/health", timeout=1.0)
            return status == 200
        except (OSError, http.client.HTTPException):
            return False

    def status(self) -> tuple[int, dict]:
        return self._request("GET", "/mode")

    def models(self) -> tuple[int, dict]:
        return self._request("GET", "/v1/models")

    def health(self) -> tuple[int, dict]:
        return self._request("GET", "/health")

    def set_mode(self, mode: str, *, model: Optional[str] = None) -> tuple[int, dict]:
        body = {"mode": mode}
        if model:
            body["model"] = model
        # Mode swaps can take 10+ minutes for first GGUF load.
        return self._request("POST", "/mode", body=body, timeout=900)


# ── Docker CLI wrappers (host-side, no SDK dep) ────────────────────────


def _docker_available() -> bool:
    return shutil.which("docker") is not None


def _run_docker(args: list[str], *, check: bool = True, capture: bool = True) -> subprocess.CompletedProcess:
    if not _docker_available():
        _err("`docker` CLI not found in PATH")
        raise SystemExit(EXIT_USER_ERROR)
    return subprocess.run(
        ["docker", *args],
        capture_output=capture,
        text=True,
        check=False if not check else False,  # we handle errors ourselves
    )


# ── Subcommand implementations ─────────────────────────────────────────


def cmd_status(args: argparse.Namespace) -> int:
    """High-level status: proxy reachable? what's running?"""
    client = ProxyClient()
    if not client.reachable():
        if args.json:
            _print_json({"reachable": False})
        else:
            _err(f"Proxy not reachable at {client.host}:{client.port}")
            _err("Start the stack: llmc up")
        return EXIT_TRANSIENT

    status_code, mode_payload = client.status()
    _, health_payload = client.health()
    _, models_payload = client.models()

    if args.json:
        _print_json({
            "reachable": True,
            "mode": mode_payload,
            "health": health_payload,
            "active_model": mode_payload.get("model"),
            "preset_count": len(models_payload.get("data", [])),
        })
        return EXIT_OK

    active_model = mode_payload.get("model") or "(none)"
    _print_kv([
        ("Proxy:", f"{client.host}:{client.port}"),
        ("Health:", health_payload.get("status", "?")),
        ("Mode:", mode_payload.get("mode", "?")),
        ("Active model:", active_model),
        ("Switching:", str(mode_payload.get("switching", False))),
        ("Presets:", str(len(models_payload.get("data", [])))),
    ])
    return EXIT_OK


def cmd_health(args: argparse.Namespace) -> int:
    client = ProxyClient()
    try:
        status, payload = client.health()
    except (OSError, http.client.HTTPException) as exc:
        _err(f"Proxy unreachable: {exc}")
        return EXIT_TRANSIENT
    if args.json:
        _print_json(payload)
    else:
        _print_kv(list(payload.items()))
    return EXIT_OK if status == 200 else EXIT_BACKEND_ERROR


def cmd_mode(args: argparse.Namespace) -> int:
    client = ProxyClient()
    if args.target is None:
        # GET current mode
        try:
            status, payload = client.status()
        except (OSError, http.client.HTTPException) as exc:
            _err(f"Proxy unreachable: {exc}")
            return EXIT_TRANSIENT
        if args.json:
            _print_json(payload)
        else:
            print(payload.get("mode", "?"))
        return EXIT_OK

    # POST new mode
    try:
        status, payload = client.set_mode(args.target, model=args.model)
    except (OSError, http.client.HTTPException) as exc:
        _err(f"Proxy unreachable: {exc}")
        return EXIT_TRANSIENT
    if status == 200:
        if args.json:
            _print_json(payload)
        else:
            new_mode = payload.get("mode", args.target)
            model = payload.get("model")
            suffix = f" ({model})" if model and new_mode == "llm" else ""
            print(f"Mode: {new_mode}{suffix}")
        return EXIT_OK
    _err(f"Mode switch failed ({status}): {payload.get('error', payload)}")
    return EXIT_BACKEND_ERROR


def cmd_switch(args: argparse.Namespace) -> int:
    """Convenience: switch LLM preset = POST /mode {mode:llm, model:X}."""
    client = ProxyClient()
    try:
        status, payload = client.set_mode("llm", model=args.preset)
    except (OSError, http.client.HTTPException) as exc:
        _err(f"Proxy unreachable: {exc}")
        return EXIT_TRANSIENT
    if status == 200:
        if args.json:
            _print_json(payload)
        else:
            print(f"Loaded: {payload.get('model', args.preset)}")
        return EXIT_OK
    _err(f"Switch failed ({status}): {payload.get('error', payload)}")
    return EXIT_BACKEND_ERROR


def cmd_models(args: argparse.Namespace) -> int:
    """List presets. Live from proxy if reachable, else local TOML files."""
    client = ProxyClient()
    if client.reachable():
        _, payload = client.models()
        data = payload.get("data", [])
        active = next((m["meta"]["preset"] for m in data
                       if m.get("meta", {}).get("loaded")), None)
        if args.json:
            _print_json(data)
            return EXIT_OK
        rows = []
        for m in data:
            meta = m.get("meta", {})
            marker = "*" if meta.get("preset") == active else " "
            rows.append([
                marker,
                meta.get("preset", "?"),
                f'{meta.get("vram_gb", 0):.1f}',
                "yes" if meta.get("capabilities", {}).get("vision") else "no",
                str(meta.get("context", "?")),
                meta.get("name", ""),
            ])
        _print_table(
            [" ", "preset", "VRAM", "vision", "context", "name"],
            rows,
        )
        return EXIT_OK

    # Fallback: read local TOML
    try:
        presets = load_all(DEFAULT_PRESETS_DIR)
    except PresetError as exc:
        _err(f"Failed to load presets: {exc}")
        return EXIT_USER_ERROR
    if args.json:
        _print_json([{
            "name": p.name,
            "display_name": p.display_name,
            "vram_gb": p.vram_gb,
            "vision": p.has_vision,
            "context": p.runtime.context_size,
        } for p in presets.values()])
        return EXIT_OK
    rows = [
        [
            p.name,
            f"{p.vram_gb:.1f}",
            "yes" if p.has_vision else "no",
            str(p.runtime.context_size),
            p.display_name,
        ]
        for p in presets.values()
    ]
    _print_table(["preset", "VRAM", "vision", "context", "name"], rows)
    print()
    print("(proxy unreachable — showing local TOML presets only)")
    return EXIT_OK


# ── Volumes subcommands ─────────────────────────────────────────────────


def cmd_volumes_ls(args: argparse.Namespace) -> int:
    """List named volumes from volumes.toml + their actual Docker state."""
    try:
        registry = load_volumes(DEFAULT_VOLUMES_TOML)
    except VolumeError as exc:
        _err(f"Failed to load volumes.toml: {exc}")
        return EXIT_USER_ERROR

    rows = []
    for spec in registry:
        info = inspect(spec.name) if _docker_available() else None
        exists = "yes" if info is not None else "no"
        actual = (info.get("Options") or {}).get("device", "") if info else ""
        match = "✓" if actual == spec.device_str else ("✗" if actual else "-")
        rows.append([spec.name, exists, match, spec.device_str])

    if args.json:
        _print_json([{"name": r[0], "exists": r[1] == "yes",
                      "device": r[3]} for r in rows])
    else:
        _print_table(["volume", "exists", "match", "device"], rows)
    return EXIT_OK


def cmd_volumes_create(args: argparse.Namespace) -> int:
    """Create all named volumes (idempotent)."""
    if not _docker_available():
        _err("`docker` CLI not found in PATH")
        return EXIT_USER_ERROR
    try:
        registry = load_volumes(DEFAULT_VOLUMES_TOML)
        actions = create_all(registry)
    except VolumeError as exc:
        _err(f"Volume operation failed: {exc}")
        return EXIT_BACKEND_ERROR
    if args.json:
        _print_json(actions)
    else:
        for name in sorted(actions):
            print(f"  {actions[name]:<8} {name}")
    return EXIT_OK


def cmd_volumes_shell(args: argparse.Namespace) -> int:
    """Open a busybox shell with every named volume mounted at /vol/<name>.
    Useful for poking around volume contents without spinning up a service."""
    if not _docker_available():
        _err("`docker` CLI not found in PATH")
        return EXIT_USER_ERROR
    try:
        registry = load_volumes(DEFAULT_VOLUMES_TOML)
    except VolumeError as exc:
        _err(f"Failed to load volumes.toml: {exc}")
        return EXIT_USER_ERROR

    mount_args = []
    for spec in registry:
        mount_args.extend(["-v", f"{spec.name}:/vol/{spec.name}"])

    cmd = [
        "docker", "run", "--rm", "-it",
        "--name", "llmc-shell",
        *mount_args,
        "--workdir", "/vol",
        "alpine:latest",
        "/bin/sh",
    ]
    print(f"# Mounting {len(registry.volumes)} volumes under /vol/")
    # exec into docker — pass control to the user's tty
    os.execvp("docker", cmd)
    return EXIT_OK  # unreachable


# ── Stack lifecycle ────────────────────────────────────────────────────


def cmd_up(args: argparse.Namespace) -> int:
    """Start the stack: ensure .env + volumes, then `docker compose up -d`."""
    rc = _ensure_env_file()
    if rc != EXIT_OK:
        return rc
    rc = cmd_volumes_create(argparse.Namespace(json=False))
    if rc != EXIT_OK:
        return rc
    result = _run_docker(["compose", "up", "-d"], capture=False)
    if result.returncode != 0:
        return EXIT_BACKEND_ERROR

    # Wait for proxy to be healthy
    client = ProxyClient()
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if client.reachable():
            print(f"Stack ready. Proxy at http://{client.host}:{client.port}")
            return EXIT_OK
        time.sleep(1)
    _err("Proxy did not become healthy within 30s. Check: llmc logs model-proxy")
    return EXIT_TRANSIENT


def cmd_down(args: argparse.Namespace) -> int:
    """Stop the stack. Also stops any GPU-service container the proxy spawned."""
    # Stop any GPU service first (they're not in compose.yaml)
    if _docker_available():
        result = subprocess.run(
            ["docker", "ps", "-q", "--filter", "label=llmc.mode"],
            capture_output=True, text=True,
        )
        container_ids = [cid for cid in result.stdout.split() if cid]
        if container_ids:
            print(f"Stopping {len(container_ids)} GPU service(s)...")
            subprocess.run(["docker", "stop", *container_ids], capture_output=True)
            subprocess.run(["docker", "rm", *container_ids], capture_output=True)

    result = _run_docker(["compose", "down"], capture=False)
    return EXIT_OK if result.returncode == 0 else EXIT_BACKEND_ERROR


def cmd_logs(args: argparse.Namespace) -> int:
    """Follow logs from one or more services."""
    targets = args.services or ["model-proxy"]
    cmd = ["compose", "logs", "-f", "--tail=100", *targets]
    result = _run_docker(cmd, capture=False)
    return EXIT_OK if result.returncode == 0 else EXIT_BACKEND_ERROR


def cmd_setup(args: argparse.Namespace) -> int:
    """First-time setup: generate .env if missing, create volumes."""
    print("Generating .env (if missing)...")
    rc = _ensure_env_file()
    if rc != EXIT_OK:
        return rc
    print("Creating named volumes from volumes.toml...")
    rc = cmd_volumes_create(argparse.Namespace(json=False))
    if rc != EXIT_OK:
        return rc
    print()
    print("Next steps:")
    print("  llmc up               Start the stack (proxy + Open WebUI)")
    print("  llmc switch <preset>  Load a model")
    print("  llmc models           List available presets")
    return EXIT_OK


def _ensure_env_file() -> int:
    """Generate a minimal .env with WEBUI_SECRET_KEY if it doesn't exist.

    .env in v2 is ONLY for variables compose needs at YAML parse time
    (currently just WEBUI_SECRET_KEY). It is never rewritten — model
    swaps go through the proxy + state file, not .env. This is the one
    place we need a host-side env file because compose's environment-
    variable interpolation doesn't read from anywhere else."""
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        print(f"  exists   .env ({env_path})")
        return EXIT_OK
    import secrets
    secret = secrets.token_hex(32)
    env_path.write_text(
        "# Generated by `llmc setup`. Holds the one variable compose needs at\n"
        "# YAML parse time. The proxy never touches this file.\n"
        f"WEBUI_SECRET_KEY={secret}\n"
        "\n"
        "# Optional: override Open WebUI host port (default 3000).\n"
        "# WEBUI_PORT=3000\n"
    )
    print(f"  created  .env ({env_path}) with random WEBUI_SECRET_KEY")
    return EXIT_OK


# ── Argument parsing ───────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="llmc",
        description="llm-compose operator CLI",
    )
    p.add_argument("--json", action="store_true",
                   help="machine-readable JSON output where applicable")
    sub = p.add_subparsers(dest="command", metavar="<command>")

    # Core orchestration
    sp = sub.add_parser("status", help="show stack + GPU mode + active model")
    sp.set_defaults(func=cmd_status)

    sp = sub.add_parser("health", help="proxy health check")
    sp.set_defaults(func=cmd_health)

    sp = sub.add_parser("mode", help="get / set GPU mode")
    sp.add_argument("target", nargs="?", choices=["llm", "comfyui", "train"],
                    help="target mode (omit to show current)")
    sp.add_argument("--model", help="when target=llm, also switch model")
    sp.set_defaults(func=cmd_mode)

    sp = sub.add_parser("switch", help="switch LLM model (shortcut for mode llm --model X)")
    sp.add_argument("preset", help="preset name (see `llmc models`)")
    sp.set_defaults(func=cmd_switch)

    sp = sub.add_parser("models", help="list available LLM presets")
    sp.set_defaults(func=cmd_models)

    # Volumes
    vol = sub.add_parser("volumes", help="manage named Docker volumes")
    vol_sub = vol.add_subparsers(dest="volumes_command", metavar="<subcommand>")
    vp = vol_sub.add_parser("ls", help="list volumes from volumes.toml + their docker state")
    vp.set_defaults(func=cmd_volumes_ls)
    vp = vol_sub.add_parser("create", help="create all volumes (idempotent)")
    vp.set_defaults(func=cmd_volumes_create)
    vp = vol_sub.add_parser("shell", help="open busybox with all volumes mounted at /vol/*")
    vp.set_defaults(func=cmd_volumes_shell)

    # Stack lifecycle
    sp = sub.add_parser("up", help="start the stack (proxy + Open WebUI)")
    sp.set_defaults(func=cmd_up)

    sp = sub.add_parser("down", help="stop the stack (incl. any running GPU service)")
    sp.set_defaults(func=cmd_down)

    sp = sub.add_parser("logs", help="follow logs from a service")
    sp.add_argument("services", nargs="*", help="service names (default: model-proxy)")
    sp.set_defaults(func=cmd_logs)

    sp = sub.add_parser("setup", help="first-time setup: create volumes")
    sp.set_defaults(func=cmd_setup)

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return EXIT_USER_ERROR

    # `volumes` without a subcommand: show help
    if args.command == "volumes" and getattr(args, "volumes_command", None) is None:
        # find the volumes subparser to print its help
        for action in parser._subparsers._group_actions[0].choices.values():
            if action.prog.endswith("llmc volumes"):
                action.print_help()
                return EXIT_USER_ERROR
        return EXIT_USER_ERROR

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
