"""llmc CLI — operator-facing command-line tool.

Replaces the bulk of the legacy Makefile. Subcommands either talk to the
running proxy over HTTP (mode swaps, status, model switches, training
jobs) or operate on the Docker daemon directly via the `docker` CLI
(volume management, stack lifecycle, logs).

Pure stdlib — no third-party deps on the host. The `docker` Python SDK
is only required inside the proxy container.

Subcommand groups:
    Core orchestration  status, health, mode, switch, models
    Volumes             volumes ls, volumes ensure, volumes shell
    Stack lifecycle     up, down, logs, setup
    Training            train status, train logs, train cancel,
                        train list, train deploy, train cleanup
    Dataset prep        dataset audit, dataset filter, dataset focus,
                        dataset caption (+ caption-status/logs/cancel)
    Eval (pass-thru)    eval <subcommand> [args...]   → eval/run.py
    Bench (pass-thru)   bench <subcommand> [args...]  → bench/*.sh

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
from llmc.volumes import VolumeError, ensure_all, load as load_volumes


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

    def set_mode(self, mode: str, *, model: Optional[str] = None, owner: Optional[str] = None) -> tuple[int, dict]:
        body = {"mode": mode}
        if model:
            body["model"] = model
        if owner:
            body["owner"] = owner
        # Mode swaps can take 10+ minutes for first GGUF load.
        return self._request("POST", "/mode", body=body, timeout=900)

    def set_lock(self, lock, owner: Optional[str] = None, wait: bool = False) -> tuple[int, dict]:
        """lock: preset name, True (lock current model), or False/None (unlock).
        owner: name of the process/user holding the lock.
        wait: when contended, join the FIFO queue (202) instead of failing (409).
        """
        body = {"lock": lock}
        if owner:
            body["owner"] = owner
        if wait:
            body["wait"] = True
        return self._request("POST", "/mode", body=body, timeout=30)

    def renew_lock(self, owner: Optional[str] = None) -> tuple[int, dict]:
        """Heartbeat the lock TTL for an existing owner (or keep a queued
        waiter's FIFO entry alive). 409 when the owner holds nothing."""
        body = {"renew": True}
        if owner:
            body["owner"] = owner
        return self._request("POST", "/mode", body=body, timeout=30)


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
    locked = mode_payload.get("locked")
    lock_info = "no"
    if locked:
        owners = mode_payload.get("lock_owners")
        lock_info = f"{locked} (owners: {', '.join(owners)})" if owners else str(locked)
        expires_at = mode_payload.get("lock_expires_at")
        if expires_at:
            remaining = int(expires_at - time.time())
            if remaining >= 0:
                lock_info += f", expires in {remaining}s"
            else:
                lock_info += ", TTL lapsed (clears on next request)"
    _print_kv([
        ("Proxy:", f"{client.host}:{client.port}"),
        ("Health:", health_payload.get("status", "?")),
        ("Mode:", mode_payload.get("mode", "?")),
        ("Active model:", active_model),
        ("Switching:", str(mode_payload.get("switching", False))),
        ("Locked:", lock_info),
        ("Presets:", str(len(models_payload.get("data", [])))),
    ])
    queue = mode_payload.get("lock_queue") or []
    if queue:
        _print_kv([
            ("Lock queue:", "; ".join(
                f"{e['position']}. {e['owner']} -> {e['model']}" for e in queue
            )),
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


def cmd_lock(args: argparse.Namespace) -> int:
    """Lock a preset against GPU-evicting swaps (loop-engine protection).

    A contended lock (another model pinned by live owners) is never hijacked:
    without --wait this fails fast with 409; with --wait it joins the proxy's
    FIFO queue and polls until the current owners drain and the grant lands.
    --renew heartbeats the TTL for a lock you already hold.
    """
    client = ProxyClient()
    if args.renew:
        if args.preset:
            _err("--renew takes no preset (it heartbeats the lock you already hold)")
            return EXIT_USER_ERROR
        try:
            status, payload = client.renew_lock(owner=args.owner)
        except (OSError, http.client.HTTPException) as exc:
            _err(f"Proxy unreachable: {exc}")
            return EXIT_TRANSIENT
        if status == 200:
            if args.json:
                _print_json(payload)
            else:
                expires_at = payload.get("lock_expires_at")
                ttl = f" (expires in {int(expires_at - time.time())}s)" if expires_at else ""
                if payload.get("queued"):
                    print(f"Queue entry renewed: position {payload.get('position')} "
                          f"(waiting on {payload.get('locked')})")
                else:
                    print(f"Lock renewed: {payload.get('locked')}{ttl}")
            return EXIT_OK
        _err(f"Renew failed ({status}): {payload.get('error', payload)}")
        return EXIT_BACKEND_ERROR
    lock_val = args.preset if args.preset else True
    deadline = (time.monotonic() + args.wait_timeout) if args.wait_timeout else None
    last_line = None
    while True:
        try:
            status, payload = client.set_lock(lock_val, owner=args.owner, wait=args.wait)
        except (OSError, http.client.HTTPException) as exc:
            _err(f"Proxy unreachable: {exc}")
            return EXIT_TRANSIENT
        if status == 200:
            if args.json:
                _print_json(payload)
            else:
                print(f"Locked: {payload.get('locked')}")
            return EXIT_OK
        if status == 202 and args.wait:
            if deadline and time.monotonic() > deadline:
                _err(f"Timed out waiting for the lock (still held: "
                     f"{payload.get('locked')}, owners: {payload.get('lock_owners')})")
                return EXIT_TRANSIENT
            line = (f"Queued for {lock_val}: position {payload.get('position')}, "
                    f"waiting on {payload.get('locked')} "
                    f"(owners: {', '.join(payload.get('lock_owners') or [])})")
            if line != last_line:
                print(line, flush=True)
                last_line = line
            time.sleep(5)
            continue
        _err(f"Lock failed ({status}): {payload.get('error', payload)}")
        return EXIT_BACKEND_ERROR


def cmd_unlock(args: argparse.Namespace) -> int:
    client = ProxyClient()
    try:
        status, payload = client.set_lock(False, owner=args.owner)
    except (OSError, http.client.HTTPException) as exc:
        _err(f"Proxy unreachable: {exc}")
        return EXIT_TRANSIENT
    if status == 200:
        if args.json:
            _print_json(payload)
        else:
            print("Unlocked")
        return EXIT_OK
    _err(f"Unlock failed ({status}): {payload.get('error', payload)}")
    return EXIT_BACKEND_ERROR


def cmd_audit(args: argparse.Namespace) -> int:
    """Check every preset's model/mmproj file against its upstream HF repo."""
    from llmc import audit as audit_mod

    try:
        presets = load_all(DEFAULT_PRESETS_DIR)
    except PresetError as exc:
        _err(f"Failed to load presets: {exc}")
        return EXIT_USER_ERROR

    models_dir = _models_dir()
    if models_dir is None:
        _err("Cannot resolve the llama-models bind path from volumes.toml")
        return EXIT_USER_ERROR

    results = audit_mod.audit_presets(presets, models_dir, deep=args.deep)
    orphaned = audit_mod.orphans(results)

    # An orphan already on off-box storage is not an alarm. Check the backup
    # destination in read-only mode too, so `make audit` is green when the
    # only copies are safe - a report that cries wolf gets ignored.
    unbacked = list(orphaned)
    backup_state = "n/a"
    if orphaned:
        try:
            inventory = audit_mod.remote_inventory(args.dest)
            backup_state = "checked"
            unbacked = [o for o in orphaned
                        if inventory.get(o.filename) != o.local_size]
            for entry in orphaned:
                if entry not in unbacked:
                    entry.note = f"{entry.note}; backed up at {args.dest}"
        except audit_mod.AuditError as exc:
            # Cannot reach the backup host: that is an unknown, not a verdict.
            # Calling these unbacked would be the same mistake as calling a
            # rate-limited HF lookup an upstream deletion.
            backup_state = f"unknown ({exc})"
            unbacked = []

    backups: list = []
    if args.backup and orphaned:
        try:
            backups = audit_mod.backup_orphans(
                orphaned, args.dest, dry_run=args.dry_run,
                log=(lambda m: None) if args.json else (lambda m: print(f"  {m}")),
            )
        except audit_mod.AuditError as exc:
            _err(f"Backup failed: {exc}")
            return EXIT_TRANSIENT

    if args.json:
        _print_json({
            "models_dir": str(models_dir),
            "results": [r.to_dict() for r in results],
            "orphans": [r.filename for r in orphaned],
            "unbacked": [r.filename for r in unbacked],
            "backup_dest": args.dest,
            "backup_state": backup_state,
            "backups": [vars(b) for b in backups],
        })
    else:
        _print_table(
            ["status", "preset", "kind", "file", "note"],
            [[r.status, r.preset, r.kind, r.filename, r.note] for r in results],
        )
        for b in backups:
            print(f"backup {b.action}: {b.filename} {b.detail}".rstrip())

    unknown = [r for r in results if r.status == audit_mod.UNKNOWN]
    lost = [r for r in results if r.unrecoverable]
    failed = [b for b in backups if b.action == "failed"]
    copied = {b.filename for b in backups if b.action in ("copied", "skipped")}
    still_unbacked = [r for r in unbacked if r.filename not in copied]

    if lost:
        _err(f"UNRECOVERABLE: {len(lost)} file(s) absent both locally and upstream")
        return EXIT_BACKEND_ERROR
    if failed:
        _err(f"{len(failed)} backup(s) failed")
        return EXIT_BACKEND_ERROR
    if backup_state.startswith("unknown"):
        _err(f"backup state unverified for {len(orphaned)} orphan(s): {backup_state}")
        return EXIT_TRANSIENT
    if still_unbacked:
        _err(f"{len(still_unbacked)} orphaned file(s) not backed up: "
             f"{', '.join(r.filename for r in still_unbacked)} "
             f"(run: llmc audit --backup)")
        return EXIT_USER_ERROR
    if unknown:
        _err(f"{len(unknown)} file(s) could not be checked upstream (network/rate limit)")
        return EXIT_TRANSIENT
    if orphaned:
        print(f"{len(orphaned)} orphan(s), all backed up at {args.dest}")
    return EXIT_OK


def _models_dir() -> Optional[Path]:
    """Host path of the llmc-llama-models bind mount."""
    try:
        registry = load_volumes(DEFAULT_VOLUMES_TOML)
    except VolumeError:
        return None
    for spec in registry:
        if spec.name == "llmc-llama-models":
            return spec.device
    return None


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
#
# After the Docker-volume-to-direct-bind migration these commands operate
# on host directories from volumes.toml, not on Docker named volumes.
# The CLI noun ("volumes") is retained for muscle memory — each named
# entry is still the logical handle for a container <-> host bind.


def cmd_volumes_ls(args: argparse.Namespace) -> int:
    """List bind-mount paths from volumes.toml + their on-disk state."""
    try:
        registry = load_volumes(DEFAULT_VOLUMES_TOML)
    except VolumeError as exc:
        _err(f"Failed to load volumes.toml: {exc}")
        return EXIT_USER_ERROR

    rows = []
    for spec in registry:
        if spec.device.exists() and spec.device.is_dir():
            exists = "yes"
            try:
                size_bytes = sum(
                    f.stat().st_size
                    for f in spec.device.rglob("*")
                    if f.is_file()
                )
                size_str = _fmt_bytes(size_bytes)
            except (OSError, PermissionError):
                size_str = "?"
        else:
            exists = "no"
            size_str = "-"
        rows.append([spec.name, exists, size_str, spec.device_str])

    if args.json:
        _print_json([{"name": r[0], "exists": r[1] == "yes",
                      "size": r[2], "path": r[3]} for r in rows])
    else:
        _print_table(["volume", "exists", "size", "path"], rows)
    return EXIT_OK


def _fmt_bytes(n: int) -> str:
    """Human-readable byte size. K/M/G/T base-1024 like `ls -h`."""
    for unit in ("B", "K", "M", "G", "T"):
        if n < 1024:
            return f"{n:.1f}{unit}" if unit != "B" else f"{n}{unit}"
        n /= 1024
    return f"{n:.1f}P"


def cmd_volumes_ensure(args: argparse.Namespace) -> int:
    """Create any missing host directories from volumes.toml. Idempotent.

    Replaces the old `volumes create` (which created Docker named volumes
    with bind indirection) — direct binds have no metadata, just need the
    host directory to exist before a container references it."""
    try:
        registry = load_volumes(DEFAULT_VOLUMES_TOML)
        actions = ensure_all(registry)
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
    """Open a busybox shell with every bind-mount path mounted at /vol/<name>.
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
        # Direct host-path bind: docker resolves -v /host:/container as a
        # bind mount (vs. a named volume) because the source starts with /.
        mount_args.extend(["-v", f"{spec.device_str}:/vol/{spec.name}"])

    cmd = [
        "docker", "run", "--rm", "-it",
        "--name", "llmc-shell",
        *mount_args,
        "--workdir", "/vol",
        "alpine:latest",
        "/bin/sh",
    ]
    print(f"# Mounting {len(registry.volumes)} bind paths under /vol/")
    # exec into docker — pass control to the user's tty
    os.execvp("docker", cmd)
    return EXIT_OK  # unreachable


# ── Stack lifecycle ────────────────────────────────────────────────────


def cmd_up(args: argparse.Namespace) -> int:
    """Start the stack: ensure .env + bind paths exist, then `docker compose up -d`."""
    rc = _ensure_env_file()
    if rc != EXIT_OK:
        return rc
    rc = cmd_volumes_ensure(argparse.Namespace(json=False))
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


# ── Training (proxies /train/* on the proxy) ───────────────────────────


def _train_get(path: str, params: str = "") -> tuple[int, dict]:
    """GET /train<path>?<params> via the proxy."""
    full = f"/train{path}"
    if params:
        full = f"{full}?{params}"
    client = ProxyClient(timeout=30)
    return client._request("GET", full)


def _train_post(path: str, body: Optional[dict] = None) -> tuple[int, dict]:
    full = f"/train{path}"
    client = ProxyClient(timeout=60)
    return client._request("POST", full, body=body)


def _train_inactive_msg(payload: dict) -> str:
    """Extract a friendly message from a 503 'service inactive' response."""
    err = payload.get("error", {})
    if isinstance(err, dict):
        return err.get("message", "train service inactive")
    return str(err)


def _check_train_active(status: int, payload: dict, args: argparse.Namespace) -> Optional[int]:
    """Return EXIT_TRANSIENT (and print friendly message) if the train
    service is not active, else None to continue."""
    if status == 503:
        if args.json:
            _print_json(payload)
        else:
            _err(_train_inactive_msg(payload))
            _err("Switch to train mode first: llmc mode train")
        return EXIT_TRANSIENT
    return None


def cmd_train_status(args: argparse.Namespace) -> int:
    try:
        status, payload = _train_get("/status")
    except (OSError, http.client.HTTPException) as exc:
        _err(f"Train service unreachable: {exc}")
        return EXIT_TRANSIENT
    rc = _check_train_active(status, payload, args)
    if rc is not None:
        return rc
    if args.json:
        _print_json(payload)
        return EXIT_OK
    state = payload.get("state", "idle")
    print(f"State: {state}")
    if state == "idle":
        return EXIT_OK
    if payload.get("output_name"):
        print(f"Output: {payload['output_name']}")
    total = payload.get("total_steps", 0)
    step = payload.get("step", 0)
    if total > 0:
        pct = step / total * 100
        print(f"Progress: {step}/{total} ({pct:.1f}%)")
    if payload.get("total_epochs"):
        print(f"Epoch: {payload.get('epoch', 0)}/{payload['total_epochs']}")
    if payload.get("loss"):
        print(f"Loss: {payload['loss']:.6f}")
    if payload.get("elapsed_seconds"):
        print(f"Elapsed: {payload['elapsed_seconds']}s ({payload['elapsed_seconds']/60:.1f} min)")
    if payload.get("eta_seconds"):
        print(f"ETA: {payload['eta_seconds']}s ({payload['eta_seconds']/60:.1f} min)")
    if payload.get("error"):
        print(f"Error: {payload['error']}")
    return EXIT_OK


def cmd_train_logs(args: argparse.Namespace) -> int:
    try:
        status, payload = _train_get("/logs", params=f"lines={args.lines}")
    except (OSError, http.client.HTTPException) as exc:
        _err(f"Train service unreachable: {exc}")
        return EXIT_TRANSIENT
    rc = _check_train_active(status, payload, args)
    if rc is not None:
        return rc
    if args.json:
        _print_json(payload)
    else:
        for line in payload.get("lines", []):
            print(line)
    return EXIT_OK


def cmd_train_cancel(args: argparse.Namespace) -> int:
    try:
        status, payload = _train_post("/cancel")
    except (OSError, http.client.HTTPException) as exc:
        _err(f"Train service unreachable: {exc}")
        return EXIT_TRANSIENT
    rc = _check_train_active(status, payload, args)
    if rc is not None:
        return rc
    if args.json:
        _print_json(payload)
    else:
        print(payload.get("status", "cancelled"))
    return EXIT_OK


def cmd_train_list(args: argparse.Namespace) -> int:
    try:
        status, payload = _train_get("/jobs")
    except (OSError, http.client.HTTPException) as exc:
        _err(f"Train service unreachable: {exc}")
        return EXIT_TRANSIENT
    rc = _check_train_active(status, payload, args)
    if rc is not None:
        return rc
    if args.json:
        _print_json(payload)
        return EXIT_OK
    files = payload.get("files", [])
    if not files:
        print("(no trained LoRAs)")
        return EXIT_OK
    rows = [[f["name"], f'{f.get("size_mb", 0):.1f}'] for f in files]
    _print_table(["lora", "size (MB)"], rows)
    return EXIT_OK


def cmd_train_cleanup(args: argparse.Namespace) -> int:
    """Kill orphaned training/captioning subprocesses (safety net)."""
    try:
        status, payload = _train_post("/cleanup")
    except (OSError, http.client.HTTPException) as exc:
        _err(f"Train service unreachable: {exc}")
        return EXIT_TRANSIENT
    rc = _check_train_active(status, payload, args)
    if rc is not None:
        return rc
    if args.json:
        _print_json(payload)
    else:
        killed = payload.get("killed", [])
        if killed:
            for p in killed:
                print(f"killed pid {p}")
        else:
            print("no orphaned processes")
    return EXIT_OK


def cmd_train_deploy(args: argparse.Namespace) -> int:
    """Copy a trained LoRA from training-data/output → comfyui/models/loras.

    Uses direct host-path binds resolved from volumes.toml — no Docker
    named volume indirection."""
    if not _docker_available():
        _err("`docker` CLI not found in PATH")
        return EXIT_USER_ERROR
    try:
        registry = load_volumes(DEFAULT_VOLUMES_TOML)
        src_path = registry.device_for("llmc-training-data")
        dst_path = registry.device_for("llmc-comfyui-loras")
    except VolumeError as exc:
        _err(f"Failed to resolve bind paths: {exc}")
        return EXIT_USER_ERROR
    name = args.lora
    if not name.endswith(".safetensors"):
        name = f"{name}.safetensors"
    result = subprocess.run([
        "docker", "run", "--rm",
        "-v", f"{src_path}:/src:ro",
        "-v", f"{dst_path}:/dst",
        "alpine", "sh", "-c",
        f"if [ -f /src/output/{name} ]; then cp /src/output/{name} /dst/ && echo copied; "
        f"else echo 'not found: {name}'; ls /src/output/*.safetensors 2>/dev/null || true; exit 1; fi",
    ], capture_output=True, text=True)
    print(result.stdout, end="")
    if result.returncode != 0:
        if result.stderr:
            _err(result.stderr.strip())
        return EXIT_BACKEND_ERROR
    print(f"Deployed {name} to {dst_path}")
    return EXIT_OK


# ── Dataset prep (mix of HTTP + docker exec) ───────────────────────────


def cmd_dataset_caption(args: argparse.Namespace) -> int:
    """Start an async captioning job. Engines: blip2 | wd14."""
    body = {
        "dataset": args.dataset,
        "engine": args.engine,
        "trigger_word": args.trigger or "",
        "overwrite": args.overwrite,
    }
    if args.prompt:
        body["prompt"] = args.prompt
    try:
        status, payload = _train_post("/caption", body=body)
    except (OSError, http.client.HTTPException) as exc:
        _err(f"Train service unreachable: {exc}")
        return EXIT_TRANSIENT
    rc = _check_train_active(status, payload, args)
    if rc is not None:
        return rc
    if args.json:
        _print_json(payload)
    else:
        print(payload.get("status", f"caption job started for {args.dataset}"))
    return EXIT_OK


def cmd_dataset_caption_status(args: argparse.Namespace) -> int:
    try:
        status, payload = _train_get("/caption/status")
    except (OSError, http.client.HTTPException) as exc:
        _err(f"Train service unreachable: {exc}")
        return EXIT_TRANSIENT
    rc = _check_train_active(status, payload, args)
    if rc is not None:
        return rc
    if args.json:
        _print_json(payload)
        return EXIT_OK
    state = payload.get("state", "idle")
    print(f"State: {state}")
    if state == "idle":
        return EXIT_OK
    print(f"Engine: {payload.get('engine', '?')}  Dataset: {payload.get('dataset', '?')}")
    total = payload.get("images_total", 0)
    done = payload.get("captions_written", 0)
    if total:
        pct = done / total * 100
        print(f"Progress: {done}/{total} ({pct:.1f}%)")
    if payload.get("elapsed_seconds"):
        print(f"Elapsed: {payload['elapsed_seconds']}s")
    if payload.get("error"):
        print(f"Error: {payload['error']}")
    return EXIT_OK


def cmd_dataset_caption_logs(args: argparse.Namespace) -> int:
    try:
        status, payload = _train_get("/caption/logs", params=f"lines={args.lines}")
    except (OSError, http.client.HTTPException) as exc:
        _err(f"Train service unreachable: {exc}")
        return EXIT_TRANSIENT
    rc = _check_train_active(status, payload, args)
    if rc is not None:
        return rc
    if args.json:
        _print_json(payload)
    else:
        for line in payload.get("lines", []):
            print(line)
    return EXIT_OK


def cmd_dataset_caption_cancel(args: argparse.Namespace) -> int:
    try:
        status, payload = _train_post("/caption/cancel")
    except (OSError, http.client.HTTPException) as exc:
        _err(f"Train service unreachable: {exc}")
        return EXIT_TRANSIENT
    rc = _check_train_active(status, payload, args)
    if rc is not None:
        return rc
    if args.json:
        _print_json(payload)
    else:
        print(payload.get("status", "cancelled"))
    return EXIT_OK


def _dataset_exec(script: str, *args_extra: str) -> int:
    """Run a script inside the running lora_train container. The scripts
    live inside the image at /audit-dataset.py / /filter-dataset.py /
    /pick-focus-subset.py."""
    if not _docker_available():
        _err("`docker` CLI not found in PATH")
        return EXIT_USER_ERROR
    cmd = ["docker", "exec", "lora_train", "python3", script, *args_extra]
    result = subprocess.run(cmd, capture_output=False)
    return EXIT_OK if result.returncode == 0 else EXIT_BACKEND_ERROR


def cmd_dataset_audit(args: argparse.Namespace) -> int:
    """Run WD14 caption auditor inside lora_train. Requires train mode."""
    extra = []
    if args.expected:
        extra.extend(["--expected-tags", args.expected])
    extra.extend(["--reject-out", f"/data/datasets/{args.dataset}-rejects.txt"])
    return _dataset_exec(
        "/audit-dataset.py",
        f"/data/datasets/{args.dataset}",
        *extra,
    )


def cmd_dataset_filter(args: argparse.Namespace) -> int:
    return _dataset_exec(
        "/filter-dataset.py",
        f"/data/datasets/{args.src}",
        f"/data/datasets/{args.dst}",
        "--rejects", f"/data/datasets/{args.src}-rejects.txt",
    )


def cmd_dataset_focus(args: argparse.Namespace) -> int:
    return _dataset_exec(
        "/pick-focus-subset.py",
        f"/data/datasets/{args.src}",
        f"/data/datasets/{args.dst}",
        "--n", str(args.n),
        "--strategy", args.strategy,
    )


# ── Eval / Bench (pass-through to scripts) ─────────────────────────────


def _passthrough(script: list[str], rest: list[str]) -> int:
    """Run a script with positional args. stdout/stderr inherited so the
    user sees output in real time."""
    result = subprocess.run(script + rest, capture_output=False)
    return EXIT_OK if result.returncode == 0 else EXIT_BACKEND_ERROR


def cmd_eval(args: argparse.Namespace) -> int:
    """Pass through to eval/run.py. The CLI doesn't need to know the eval
    subcommand schema; it just forwards everything after `llmc eval`."""
    if not args.rest:
        _err("Usage: llmc eval <subcommand> [args...]")
        _err("Subcommands: quicktest, stages, sweep, matrix, checkpoints,")
        _err("             weights, loras, seeds, shot, i2i")
        return EXIT_USER_ERROR
    return _passthrough([sys.executable, str(REPO_ROOT / "eval" / "run.py")], args.rest)


def cmd_bench(args: argparse.Namespace) -> int:
    """Native bench framework (perf/report/watch) + legacy script pass-thru."""
    if not args.rest:
        _err("Usage: llmc bench <subcommand> [args...]")
        _err("Native: perf, report, watch. Legacy: quants, quants-quick, accuracy, image, model, model-quick")
        return EXIT_USER_ERROR
    sub = args.rest[0]
    rest = args.rest[1:]

    # Native framework (llmc/bench/) - perf/report/watch per
    # docs/plans/2026-08-15-local-model-bench-framework.md
    from llmc.bench import cli as bench_cli
    if sub in bench_cli.NATIVE:
        return bench_cli.main(args.rest)

    mapping = {
        "quants":       [str(REPO_ROOT / "bench" / "bench-quants.sh")],
        "quants-quick": [str(REPO_ROOT / "bench" / "bench-quants.sh"), "--quick"],
        "accuracy":     [str(REPO_ROOT / "bench" / "bench-quants.sh"), "--skip-perf"],
        "image":        ["docker", "build",
                         "-t", "erfianugrah/bench-eval:latest",
                         "-f", str(REPO_ROOT / "bench" / "Dockerfile.eval"),
                         str(REPO_ROOT / "bench")],
        # Single-model wrappers
        "model":        [str(REPO_ROOT / "scripts" / "bench.sh")],
        "model-quick":  [str(REPO_ROOT / "scripts" / "bench.sh"), "--quick"],
    }
    if sub not in mapping:
        _err(f"Unknown bench subcommand: {sub!r}")
        _err(f"Available: {', '.join(sorted(bench_cli.NATIVE | set(mapping)))}")
        return EXIT_USER_ERROR
    return _passthrough(mapping[sub], rest)


# ── Open WebUI helpers ────────────────────────────────────────────────


def cmd_webui_configure(args: argparse.Namespace) -> int:
    """Import workspace models from webui/models.json into Open WebUI.

    Thin wrapper around scripts/init-webui.sh. Reads WEBUI_ADMIN_EMAIL /
    WEBUI_ADMIN_PASSWORD from .env for headless auth; falls back to an
    interactive prompt if not set. Idempotent."""
    script = REPO_ROOT / "scripts" / "init-webui.sh"
    if not script.exists():
        _err(f"init-webui.sh not found at {script}")
        return EXIT_USER_ERROR
    result = subprocess.run([str(script)], capture_output=False)
    return EXIT_OK if result.returncode == 0 else EXIT_BACKEND_ERROR


def cmd_webui_reset(args: argparse.Namespace) -> int:
    """Nuke Open WebUI's data directory (accounts, chat history, settings).
    Stops the container first, wipes the host bind path, restarts.
    This is destructive — bind-mount data is permanently deleted."""
    if not args.yes:
        _err("This will permanently delete all Open WebUI accounts, chats, and settings.")
        _err("Re-run with --yes to confirm.")
        return EXIT_USER_ERROR
    try:
        registry = load_volumes(DEFAULT_VOLUMES_TOML)
        webui_path = registry.device_for("llmc-webui-data")
    except VolumeError as exc:
        _err(f"Failed to resolve webui bind path: {exc}")
        return EXIT_USER_ERROR
    print("Stopping open-webui...")
    subprocess.run(["docker", "compose", "stop", "open-webui"], capture_output=True)
    print(f"Wiping {webui_path} contents...")
    # Run inside a container so we can rm root-owned files without sudo.
    subprocess.run([
        "docker", "run", "--rm", "-v", f"{webui_path}:/data",
        "alpine", "sh", "-c", "rm -rf /data/*",
    ], capture_output=False)
    print("Done. Run `llmc up` to restart with a clean WebUI.")
    return EXIT_OK


# ── ComfyUI helpers ───────────────────────────────────────────────────


def cmd_comfyui_open(args: argparse.Namespace) -> int:
    """Print the ComfyUI URL (auto-spawning the service if needed)."""
    client = ProxyClient()
    if not client.reachable():
        _err(f"Proxy not reachable at {client.host}:{client.port}")
        return EXIT_TRANSIENT
    current = client._request("GET", "/mode")[1].get("mode")
    if current != "comfyui":
        if args.no_swap:
            _err(f"Current mode is {current!r}, not comfyui.")
            _err("Run `llmc mode comfyui` first or omit --no-swap.")
            return EXIT_TRANSIENT
        print(f"Switching mode {current} -> comfyui...")
        status, payload = client.set_mode("comfyui")
        if status != 200:
            _err(f"Mode switch failed: {payload}")
            return EXIT_BACKEND_ERROR
    print("ComfyUI:  http://127.0.0.1:8188  (direct, supports websocket previews)")
    print("Through proxy:  http://127.0.0.1:11434/comfyui/  (HTTP-only)")
    return EXIT_OK


def cmd_setup(args: argparse.Namespace) -> int:
    """First-time setup: generate .env if missing, create bind directories."""
    print("Generating .env (if missing)...")
    rc = _ensure_env_file()
    if rc != EXIT_OK:
        return rc
    print("Ensuring host directories from volumes.toml exist...")
    rc = cmd_volumes_ensure(argparse.Namespace(json=False))
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

    sp = sub.add_parser("lock", help="lock a preset in place - refuse GPU-evicting swaps")
    sp.add_argument("preset", nargs="?", default=None,
                    help="preset to lock (omit to lock the currently-active model)")
    sp.add_argument("--owner", help="identity of the owner (e.g. session ID)")
    sp.add_argument("--wait", action="store_true",
                    help="if the lock is held on another model, join the FIFO queue "
                         "and wait for the grant instead of failing (409)")
    sp.add_argument("--wait-timeout", type=float, default=0, metavar="SECONDS",
                    help="give up waiting after N seconds (default: wait forever)")
    sp.add_argument("--renew", action="store_true",
                    help="heartbeat the lock TTL for a lock you already hold "
                         "(or keep your FIFO queue entry alive)")
    sp.set_defaults(func=cmd_lock)

    sp = sub.add_parser("unlock", help="clear the model lock (also drops your queue entry)")
    sp.add_argument("--owner", help="identity of the owner to unlock")
    sp.set_defaults(func=cmd_unlock)

    sp = sub.add_parser("models", help="list available LLM presets")
    sp.set_defaults(func=cmd_models)

    sp = sub.add_parser("audit",
                        help="check preset model files against upstream (drift/orphans)")
    sp.add_argument("--deep", action="store_true",
                    help="sha256 every same-size file against the HF LFS oid (slow, exact)")
    sp.add_argument("--backup", action="store_true",
                    help="rsync orphaned files (no upstream copy) to --dest and verify")
    sp.add_argument("--dest",
                    default=os.environ.get("LLMC_MODEL_BACKUP_DEST",
                                           "servarr:/tank/backups/llm-models/orphaned"),
                    help="backup destination host:/path (env LLMC_MODEL_BACKUP_DEST)")
    sp.add_argument("--dry-run", action="store_true",
                    help="with --backup: report what would be copied, copy nothing")
    sp.set_defaults(func=cmd_audit)

    # Volumes (= bind-mount host paths from volumes.toml)
    vol = sub.add_parser("volumes", help="manage bind-mount paths")
    vol_sub = vol.add_subparsers(dest="volumes_command", metavar="<subcommand>")
    vp = vol_sub.add_parser("ls", help="list bind paths from volumes.toml + disk usage")
    vp.set_defaults(func=cmd_volumes_ls)
    vp = vol_sub.add_parser("ensure", help="create any missing host directories (idempotent)")
    vp.set_defaults(func=cmd_volumes_ensure)
    vp = vol_sub.add_parser("shell", help="open busybox with all bind paths mounted at /vol/*")
    vp.set_defaults(func=cmd_volumes_shell)

    # Stack lifecycle
    sp = sub.add_parser("up", help="start the stack (proxy + Open WebUI)")
    sp.set_defaults(func=cmd_up)

    sp = sub.add_parser("down", help="stop the stack (incl. any running GPU service)")
    sp.set_defaults(func=cmd_down)

    sp = sub.add_parser("logs", help="follow logs from a service")
    sp.add_argument("services", nargs="*", help="service names (default: model-proxy)")
    sp.set_defaults(func=cmd_logs)

    sp = sub.add_parser("setup", help="first-time setup: create .env + bind directories")
    sp.set_defaults(func=cmd_setup)

    # Open WebUI helpers
    webui = sub.add_parser("webui", help="Open WebUI: configure, reset")
    webui_sub = webui.add_subparsers(dest="webui_command", metavar="<subcommand>")
    wp = webui_sub.add_parser("configure",
                              help="import workspace models from webui/models.json")
    wp.set_defaults(func=cmd_webui_configure)
    wp = webui_sub.add_parser("reset",
                              help="DESTRUCTIVE: nuke webui data volume (accounts, chats)")
    wp.add_argument("--yes", action="store_true", help="skip confirmation")
    wp.set_defaults(func=cmd_webui_reset)

    # ComfyUI helpers
    comfy = sub.add_parser("comfyui", help="ComfyUI helpers")
    comfy_sub = comfy.add_subparsers(dest="comfyui_command", metavar="<subcommand>")
    cp = comfy_sub.add_parser("open",
                              help="print ComfyUI URL (auto-switching to comfyui mode)")
    cp.add_argument("--no-swap", action="store_true",
                    help="fail if not already in comfyui mode")
    cp.set_defaults(func=cmd_comfyui_open)

    # Training (proxies to /train/* on the proxy — needs train mode active)
    train = sub.add_parser("train", help="LoRA training: status, logs, deploy, ...")
    train_sub = train.add_subparsers(dest="train_command", metavar="<subcommand>")

    tp = train_sub.add_parser("status", help="training job progress + ETA")
    tp.set_defaults(func=cmd_train_status)

    tp = train_sub.add_parser("logs", help="tail training log output")
    tp.add_argument("--lines", type=int, default=50)
    tp.set_defaults(func=cmd_train_logs)

    tp = train_sub.add_parser("cancel", help="cancel current training job")
    tp.set_defaults(func=cmd_train_cancel)

    tp = train_sub.add_parser("list", help="list trained LoRA files")
    tp.set_defaults(func=cmd_train_list)

    tp = train_sub.add_parser("cleanup",
                              help="kill orphaned training subprocesses (safety net)")
    tp.set_defaults(func=cmd_train_cleanup)

    tp = train_sub.add_parser("deploy",
                              help="copy a trained LoRA from output → comfyui/loras")
    tp.add_argument("lora", help="LoRA filename (.safetensors suffix optional)")
    tp.set_defaults(func=cmd_train_deploy)

    # Dataset prep
    ds = sub.add_parser("dataset", help="dataset prep: audit, filter, focus, caption")
    ds_sub = ds.add_subparsers(dest="dataset_command", metavar="<subcommand>")

    dp = ds_sub.add_parser("audit", help="audit a dataset's WD14 captions for issues")
    dp.add_argument("dataset", help="dataset name under /data/datasets/")
    dp.add_argument("--expected", help="comma-separated expected tags")
    dp.set_defaults(func=cmd_dataset_audit)

    dp = ds_sub.add_parser("filter", help="copy a dataset minus rejected stems")
    dp.add_argument("src", help="source dataset name")
    dp.add_argument("dst", help="destination dataset name")
    dp.set_defaults(func=cmd_dataset_filter)

    dp = ds_sub.add_parser("focus",
                           help="pick N best images for focus training")
    dp.add_argument("src")
    dp.add_argument("dst")
    dp.add_argument("--n", type=int, default=40)
    dp.add_argument("--strategy", default="longest-caption",
                    choices=["longest-caption", "random"])
    dp.set_defaults(func=cmd_dataset_focus)

    dp = ds_sub.add_parser("caption",
                           help="start an async caption job (engine: blip2 | wd14)")
    dp.add_argument("dataset")
    dp.add_argument("--engine", default="blip2", choices=["blip2", "wd14"])
    dp.add_argument("--trigger", help="trigger word prepended to each caption")
    dp.add_argument("--prompt", help="BLIP-2 conditional prompt prefix")
    dp.add_argument("--overwrite", action="store_true",
                    help="overwrite existing .txt captions")
    dp.set_defaults(func=cmd_dataset_caption)

    dp = ds_sub.add_parser("caption-status", help="caption job progress")
    dp.set_defaults(func=cmd_dataset_caption_status)

    dp = ds_sub.add_parser("caption-logs", help="tail caption log output")
    dp.add_argument("--lines", type=int, default=50)
    dp.set_defaults(func=cmd_dataset_caption_logs)

    dp = ds_sub.add_parser("caption-cancel", help="cancel running caption job")
    dp.set_defaults(func=cmd_dataset_caption_cancel)

    # Eval — pass-through to eval/run.py
    sp = sub.add_parser(
        "eval",
        help="LoRA eval pass-through to eval/run.py",
        # argparse.REMAINDER preserves --flags as positional args
    )
    sp.add_argument("rest", nargs=argparse.REMAINDER,
                    help="forwarded verbatim to eval/run.py")
    sp.set_defaults(func=cmd_eval)

    # Bench — wrappers around bench scripts
    sp = sub.add_parser(
        "bench",
        help="benchmark pass-through: perf | quants | accuracy | report | image",
    )
    sp.add_argument("rest", nargs=argparse.REMAINDER,
                    help="<subcommand> [args...] forwarded to the right bench script")
    sp.set_defaults(func=cmd_bench)

    return p


# Sub-subcommand groups: when invoked without a sub-subcommand we should
# print that group's help, not silently crash with AttributeError.
_SUB_GROUPS = {
    "volumes": "volumes_command",
    "train": "train_command",
    "dataset": "dataset_command",
    "webui": "webui_command",
    "comfyui": "comfyui_command",
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return EXIT_USER_ERROR

    # Subcommand group without a sub-subcommand — print group help
    sub_attr = _SUB_GROUPS.get(args.command)
    if sub_attr and getattr(args, sub_attr, None) is None:
        for action in parser._subparsers._group_actions[0].choices.values():
            if action.prog.endswith(f"llmc {args.command}"):
                action.print_help()
                return EXIT_USER_ERROR
        return EXIT_USER_ERROR

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
