#!/usr/bin/env python3
"""Operator-owned acceptance probe for concurrent loop support (shared lock).

Boots the real proxy handler on an ephemeral port with a mocked orchestrator
(mirrors llmc/tests/test_proxy.py's _ProxyServer) and asserts the lock-owners
contract over HTTP. The loop agent cannot edit this file (outside writeScope);
it turns green only when the implementation is correct.

Contract under test:
  1. POST /mode {"lock": "<preset>", "owner": "a"} pins the preset and records
     the owner; GET /mode exposes "locked" (preset name, string|null, backward
     compatible) and "lock_owners" (list of owner strings).
  2. A second owner {"lock": "<same preset>", "owner": "b"} shares the lock.
  3. {"lock": false, "owner": "a"} releases only owner a - the preset stays
     locked while any owner remains.
  4. Releasing the last owner unlocks (locked -> null).
  5. Backward compat: {"lock": "<preset>"} (no owner) still works, and
     {"lock": false} (no owner) force-clears ALL owners.
  6. While locked: /v1 with a DIFFERENT preset -> 422; /v1 with the locked
     preset -> forwarding attempted (502 with mocked orchestrator, no
     upstream); POST /mode {"mode": "comfyui"} -> 503. No spawn calls.

Exit 0 = contract satisfied. Exit 1 = not.
"""

from __future__ import annotations

import http.client
import json
import socket
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from llmc.presets import load_all  # noqa: E402
from llmc.proxy import ProxyConfig, ProxyContext, ProxyHandler, _ThreadingServer  # noqa: E402
from llmc.state import State  # noqa: E402

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
            # Handler died mid-request - report as a failed response so one
            # crash doesn't abort the remaining checks.
            return 0, {"error": f"handler crashed: {exc}"}
        raw = resp.read()
        try:
            return resp.status, json.loads(raw)
        except (ValueError, json.JSONDecodeError):
            return resp.status, {"raw": raw.decode("utf-8", "replace")}
    finally:
        conn.close()


def main() -> int:
    orch = MagicMock()
    orch.current_mode.return_value = "llm"
    presets_dir = REPO_ROOT / "models"
    ctx = ProxyContext(
        config=ProxyConfig(port=0, presets_dir=presets_dir),
        orchestrator=orch,
        presets=load_all(presets_dir),
        state=State(mode="llm", model="loop"),
    )
    port = _free_port()
    ProxyHandler.ctx = ctx
    server = _ThreadingServer(("127.0.0.1", port), ProxyHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        try:
            http.client.HTTPConnection("127.0.0.1", port, timeout=0.5).connect()
            break
        except OSError:
            time.sleep(0.05)

    try:
        print("lock-owners contract probe")

        # 1. lock with owner a
        st, pl = _req("POST", port, "/mode", {"lock": "loop", "owner": "a"})
        check("lock owner=a accepted", st == 200, f"status {st}: {pl}")
        st, pl = _req("GET", port, "/mode")
        check("locked reports preset name", pl.get("locked") == "loop", f"got {pl.get('locked')!r}")
        owners = pl.get("lock_owners")
        check("lock_owners exposed", isinstance(owners, list), f"got {owners!r}")
        check("owner a recorded", isinstance(owners, list) and "a" in owners, f"got {owners!r}")

        # 2. second owner shares
        st, pl = _req("POST", port, "/mode", {"lock": "loop", "owner": "b"})
        check("lock owner=b accepted", st == 200, f"status {st}: {pl}")
        st, pl = _req("GET", port, "/mode")
        owners = pl.get("lock_owners") or []
        check("both owners held", set(owners) == {"a", "b"}, f"got {owners!r}")

        # 3. releasing one owner keeps the lock
        st, pl = _req("POST", port, "/mode", {"lock": False, "owner": "a"})
        check("unlock owner=a accepted", st == 200, f"status {st}: {pl}")
        st, pl = _req("GET", port, "/mode")
        check("still locked after one release", pl.get("locked") == "loop", f"got {pl.get('locked')!r}")
        check("owner b remains", (pl.get("lock_owners") or []) == ["b"], f"got {pl.get('lock_owners')!r}")

        # 6a. while locked (owner b): different preset refused
        st, pl = _req("POST", port, "/v1/chat/completions",
                      {"model": "summarizer", "messages": [{"role": "user", "content": "hi"}]})
        check("other preset refused while shared-locked", st == 422, f"status {st}: {pl}")

        # 6b. locked preset itself forwards (502: no upstream with mocked orch)
        st, pl = _req("POST", port, "/v1/chat/completions",
                      {"model": "loop", "messages": [{"role": "user", "content": "hi"}]})
        check("locked preset forwards", st == 502, f"status {st}: {pl}")

        # 6c. mode eviction refused
        st, pl = _req("POST", port, "/mode", {"mode": "comfyui"})
        check("mode swap refused while shared-locked", st == 503, f"status {st}: {pl}")
        check("no spawn calls made", orch.spawn_llama.call_count == 0
              and orch.spawn_comfyui.call_count == 0 and orch.spawn_train.call_count == 0)

        # 4. releasing the last owner unlocks
        st, pl = _req("POST", port, "/mode", {"lock": False, "owner": "b"})
        check("unlock owner=b accepted", st == 200, f"status {st}: {pl}")
        st, pl = _req("GET", port, "/mode")
        check("unlocked after last release", pl.get("locked") is None, f"got {pl.get('locked')!r}")

        # 5. backward compat: no-owner lock works, no-owner unlock force-clears
        st, pl = _req("POST", port, "/mode", {"lock": "loop"})
        check("ownerless lock accepted", st == 200, f"status {st}: {pl}")
        st, pl = _req("POST", port, "/mode", {"lock": "loop", "owner": "c"})
        check("second owner after ownerless accepted", st == 200, f"status {st}: {pl}")
        st, pl = _req("POST", port, "/mode", {"lock": False})
        check("ownerless unlock force-clears", st == 200, f"status {st}: {pl}")
        st, pl = _req("GET", port, "/mode")
        check("force-clear unlocks with owners present", pl.get("locked") is None,
              f"got {pl.get('locked')!r}, owners {pl.get('lock_owners')!r}")
    finally:
        server.shutdown()
        server.server_close()

    if FAILURES:
        print(f"\n{len(FAILURES)} contract violation(s): {', '.join(FAILURES)}")
        return 1
    print("\ncontract satisfied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
