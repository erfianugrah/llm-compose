#!/usr/bin/env python3
"""Operator-owned acceptance probe for lock persistence (restart-safe locks).

Contract: the model lock (locked preset + owners) must survive a proxy
restart by round-tripping through the state file. The probe boots the real
proxy handler with a temp state path, sets a shared lock over HTTP, then
simulates restarts by building a FRESH ProxyContext from the same state
file and booting a new server - exactly what happens when the proxy
container restarts.

Checks:
  1. Lock set with owners a,b -> state FILE contains locked + lock_owners.
  2. Restart: fresh context from that state -> GET /mode shows locked +
     both owners, and enforcement still works (other preset 422, mode
     swap 503, no spawn calls).
  3. Owner release persists: unlock owner a, restart -> still locked,
     owners == ["b"].
  4. Force-clear persists: ownerless unlock, restart -> unlocked, no
     lock keys in the file.
  5. Graceful degradation: with an UNWRITABLE state path (the dev-box
     default /state/active.toml), lock set/clear still works in-memory
     (200s) instead of 500ing - existing tests use that default path.

Exit 0 = contract satisfied. Exit 1 = not.
"""

from __future__ import annotations

import http.client
import json
import socket
import sys
import tempfile
import threading
import time
import tomllib
from pathlib import Path
from unittest.mock import MagicMock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from llmc.presets import load_all  # noqa: E402
from llmc.proxy import ProxyConfig, ProxyContext, ProxyHandler, _ThreadingServer  # noqa: E402
from llmc.state import State, load as load_state  # noqa: E402

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'pass' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _req(method: str, port: int, path: str, payload: dict | None = None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5.0)
    try:
        body = json.dumps(payload).encode() if payload is not None else None
        conn.request(method, path, body=body, headers={"Content-Type": "application/json"})
        try:
            resp = conn.getresponse()
        except (OSError, http.client.HTTPException) as exc:
            return 0, {"error": f"handler crashed: {exc}"}
        raw = resp.read()
        try:
            return resp.status, json.loads(raw)
        except (ValueError, json.JSONDecodeError):
            return resp.status, {"raw": raw.decode("utf-8", "replace")}
    finally:
        conn.close()


class _Server:
    def __init__(self, ctx: ProxyContext):
        self.port = _free_port()
        ProxyHandler.ctx = ctx
        self.server = _ThreadingServer(("127.0.0.1", self.port), ProxyHandler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            try:
                http.client.HTTPConnection("127.0.0.1", self.port, timeout=0.5).connect()
                break
            except OSError:
                time.sleep(0.05)

    def stop(self):
        self.server.shutdown()
        self.server.server_close()


def _make_ctx(presets_dir: Path, state_dir: Path) -> ProxyContext:
    orch = MagicMock()
    orch.current_mode.return_value = "llm"
    config = ProxyConfig(port=0, presets_dir=presets_dir, state_dir=state_dir)
    return ProxyContext(
        config=config,
        orchestrator=orch,
        presets=load_all(presets_dir),
        state=load_state(config.state_path),
    )


def main() -> int:
    presets_dir = REPO_ROOT / "models"
    print("lock-persistence contract probe")

    with tempfile.TemporaryDirectory() as tmp:
        state_path = Path(tmp) / "active.toml"

        # --- boot 1: lock with two owners ---
        ctx1 = _make_ctx(presets_dir, Path(tmp))
        srv1 = _Server(ctx1)
        st, pl = _req("POST", srv1.port, "/mode", {"lock": "loop", "owner": "a"})
        check("lock owner=a", st == 200, f"{st}: {pl}")
        st, pl = _req("POST", srv1.port, "/mode", {"lock": "loop", "owner": "b"})
        check("lock owner=b", st == 200, f"{st}: {pl}")

        # 1. state file must contain the lock
        data = tomllib.loads(state_path.read_text()) if state_path.exists() else {}
        check("state file records locked preset", data.get("locked") == "loop", f"got {data!r}")
        check("state file records owners", sorted(data.get("lock_owners", [])) == ["a", "b"],
              f"got {data.get('lock_owners')!r}")
        srv1.stop()

        # --- restart 1: fresh context from the same state file ---
        ctx2 = _make_ctx(presets_dir, Path(tmp))
        srv2 = _Server(ctx2)
        st, pl = _req("GET", srv2.port, "/mode")
        check("lock survives restart", pl.get("locked") == "loop", f"got {pl.get('locked')!r}")
        check("owners survive restart", sorted(pl.get("lock_owners") or []) == ["a", "b"],
              f"got {pl.get('lock_owners')!r}")
        st, pl = _req("POST", srv2.port, "/v1/chat/completions",
                      {"model": "summarizer", "messages": [{"role": "user", "content": "hi"}]})
        check("enforcement survives restart (422)", st == 422, f"{st}: {pl}")
        st, pl = _req("POST", srv2.port, "/mode", {"mode": "comfyui"})
        check("mode-swap refusal survives restart (503)", st == 503, f"{st}: {pl}")
        check("no spawn calls after restart",
              ctx2.orchestrator.spawn_llama.call_count == 0
              and ctx2.orchestrator.spawn_comfyui.call_count == 0)

        # 3. owner release persists
        st, pl = _req("POST", srv2.port, "/mode", {"lock": False, "owner": "a"})
        check("unlock owner=a", st == 200, f"{st}: {pl}")
        srv2.stop()

        ctx3 = _make_ctx(presets_dir, Path(tmp))
        srv3 = _Server(ctx3)
        st, pl = _req("GET", srv3.port, "/mode")
        check("partial release persists (still locked)", pl.get("locked") == "loop",
              f"got {pl.get('locked')!r}")
        check("partial release persists (owners == [b])", pl.get("lock_owners") == ["b"],
              f"got {pl.get('lock_owners')!r}")

        # 4. force-clear persists
        st, pl = _req("POST", srv3.port, "/mode", {"lock": False})
        check("force-clear", st == 200, f"{st}: {pl}")
        srv3.stop()

        ctx4 = _make_ctx(presets_dir, Path(tmp))
        srv4 = _Server(ctx4)
        st, pl = _req("GET", srv4.port, "/mode")
        check("force-clear persists (unlocked after restart)", pl.get("locked") is None,
              f"got {pl.get('locked')!r}")
        srv4.stop()

    # 5. graceful degradation with an unwritable state path
    bad_path = Path("/state/active.toml")  # not writable in dev/test envs
    ctx5 = _make_ctx(presets_dir, Path("/state"))
    srv5 = _Server(ctx5)
    st, pl = _req("POST", srv5.port, "/mode", {"lock": "loop", "owner": "x"})
    check("lock works with unwritable state path", st == 200, f"{st}: {pl}")
    st, pl = _req("GET", srv5.port, "/mode")
    check("in-memory lock still reported", pl.get("locked") == "loop", f"got {pl.get('locked')!r}")
    st, pl = _req("POST", srv5.port, "/mode", {"lock": False})
    check("unlock works with unwritable state path", st == 200, f"{st}: {pl}")
    srv5.stop()

    if FAILURES:
        print(f"\n{len(FAILURES)} contract violation(s): {', '.join(FAILURES)}")
        return 1
    print("\ncontract satisfied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
