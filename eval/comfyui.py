"""Minimal ComfyUI client — routes through the model-proxy so GPU mode
swaps automatically (no manual `make comfyui` needed).

- submit()/wait() talk to the proxy on localhost:11434/comfyui.
- The first request may 503 with {"status": "switching"} while the proxy
  stops llama-server and starts ComfyUI (~20-60s). We retry transparently.
- Poll /history/{prompt_id} for completion.
- Return list of output image paths (relative to comfyui output dir).
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

PROXY = "http://localhost:11434/comfyui"
OUTPUT_ROOT = Path("~/docker-volumes/comfyui/output").expanduser()


def _req(method: str, path: str, data: dict | None = None, timeout: int = 30) -> dict:
    url = f"{PROXY}{path}"
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(
        url, data=body, method=method,
        headers={"Content-Type": "application/json", "User-Agent": "eval/1.0"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def submit(workflow: dict, client_id: str | None = None, swap_wait: int = 120) -> str:
    """Submit a workflow. Retries through proxy swap."""
    payload = {"prompt": workflow, "client_id": client_id or str(uuid.uuid4())}
    start = time.monotonic()
    while True:
        try:
            res = _req("POST", "/prompt", payload, timeout=60)
            return res["prompt_id"]
        except urllib.error.HTTPError as e:
            if e.code == 503 and time.monotonic() - start < swap_wait:
                time.sleep(3)
                continue
            raise
        except urllib.error.URLError:
            if time.monotonic() - start < swap_wait:
                time.sleep(3)
                continue
            raise


def wait(prompt_id: str, timeout: int = 300) -> list[Path]:
    """Poll history until prompt finishes. Return output image Paths."""
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        try:
            hist = _req("GET", f"/history/{prompt_id}", timeout=10)
        except urllib.error.HTTPError:
            time.sleep(2)
            continue
        if prompt_id not in hist:
            time.sleep(2)
            continue
        outputs = hist[prompt_id].get("outputs", {})
        files: list[Path] = []
        for node_out in outputs.values():
            for img in node_out.get("images", []):
                sub = img.get("subfolder", "")
                files.append(OUTPUT_ROOT / sub / img["filename"])
        return files
    raise TimeoutError(f"prompt {prompt_id} timed out after {timeout}s")


def generate(workflow: dict, timeout: int = 300) -> list[Path]:
    """Submit + wait."""
    pid = submit(workflow)
    return wait(pid, timeout=timeout)
