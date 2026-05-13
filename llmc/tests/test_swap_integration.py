"""End-to-end swap correctness + timing tests.

Requires:
    1. The v2 stack running (proxy at localhost:11434, network 'llmc')
    2. GPU available
    3. gemma4 and qwen36 GGUFs already cached in llmc-llama-models volume
       (or fast download bandwidth — these are 17-20 GB each)

Gated behind LLMC_TEST_INTEGRATION=1 because each swap costs 30-60 s of
real GPU + model-load time. Run manually:

    LLMC_TEST_INTEGRATION=1 python3 -m unittest llmc.tests.test_swap_integration -v

These tests would have caught the bug where POST /mode {mode:llm, model:X}
returned 200 but didn't actually swap the model — _ensure_mode saw we were
already in LLM mode and short-circuited, leaving the old container running
while state and CLI claimed the swap succeeded.
"""

from __future__ import annotations

import http.client
import json
import os
import subprocess
import time
import unittest


PROXY_HOST = os.environ.get("LLMC_PROXY_HOST", "127.0.0.1")
PROXY_PORT = int(os.environ.get("LLMC_PROXY_PORT", "11434"))


def _post_mode(mode: str, *, model: str = None, timeout: float = 600.0) -> tuple[int, dict]:
    """POST /mode and return (status, json_body)."""
    conn = http.client.HTTPConnection(PROXY_HOST, PROXY_PORT, timeout=timeout)
    try:
        body = {"mode": mode}
        if model:
            body["model"] = model
        conn.request("POST", "/mode",
                     body=json.dumps(body).encode(),
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        raw = resp.read()
        return resp.status, json.loads(raw) if raw else {}
    finally:
        conn.close()


def _get_status() -> dict:
    conn = http.client.HTTPConnection(PROXY_HOST, PROXY_PORT, timeout=5)
    try:
        conn.request("GET", "/mode")
        resp = conn.getresponse()
        return json.loads(resp.read())
    finally:
        conn.close()


def _container_model_file() -> str:
    """Return the MODEL_FILE env var of the running llama_server container.
    Empty string if no container is running."""
    result = subprocess.run(
        ["docker", "inspect", "llama_server",
         "--format", '{{range .Config.Env}}{{println .}}{{end}}'],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return ""
    for line in result.stdout.splitlines():
        if line.startswith("MODEL_FILE="):
            return line.split("=", 1)[1]
    return ""


def _proxy_reachable() -> bool:
    try:
        conn = http.client.HTTPConnection(PROXY_HOST, PROXY_PORT, timeout=1)
        conn.request("GET", "/health")
        return conn.getresponse().status == 200
    except (OSError, http.client.HTTPException):
        return False


@unittest.skipUnless(
    os.environ.get("LLMC_TEST_INTEGRATION") == "1",
    "Set LLMC_TEST_INTEGRATION=1 to run end-to-end GPU swap tests",
)
class TestModelSwapCorrectness(unittest.TestCase):
    """Regression test for the POST /mode {mode:llm, model:X} bug.

    Symptom: API returned 200 with the new model name, but the running
    container was still the previous one. Root cause: _handle_mode_post
    called _ensure_mode(target) instead of _ensure_model(model) when both
    fields were present and we were already in LLM mode.

    Fix in proxy.py: route through _ensure_model when target=llm AND
    model is given, since that path correctly spawns a new container.
    """

    MODEL_A = "gemma4"
    MODEL_A_FILE = "gemma-4-31B-it-Q4_K_M.gguf"
    MODEL_B = "qwen36"
    MODEL_B_FILE = "Qwen3.6-27B-UD-Q4_K_XL.gguf"

    @classmethod
    def setUpClass(cls):
        if not _proxy_reachable():
            raise unittest.SkipTest(f"proxy not reachable at {PROXY_HOST}:{PROXY_PORT}")

    def setUp(self):
        # Start each test from a known state: MODEL_A loaded.
        status, _ = _post_mode("llm", model=self.MODEL_A)
        if status != 200:
            self.skipTest(f"could not pre-load {self.MODEL_A}: HTTP {status}")
        self.assertEqual(_container_model_file(), self.MODEL_A_FILE)

    def test_swap_to_different_model_actually_swaps_container(self):
        """The original bug: API claimed success, container didn't change."""
        before = _container_model_file()
        self.assertEqual(before, self.MODEL_A_FILE)

        status, payload = _post_mode("llm", model=self.MODEL_B)
        self.assertEqual(status, 200, f"swap failed: {payload}")
        self.assertEqual(payload.get("model"), self.MODEL_B)

        # The critical assertion — without the fix this was still MODEL_A_FILE
        after = _container_model_file()
        self.assertEqual(after, self.MODEL_B_FILE,
                         f"swap reported success but container is still running "
                         f"{after!r}, not {self.MODEL_B_FILE!r}")

    def test_swap_back_and_forth_works(self):
        # A → B → A. Each step must actually change the running container.
        _post_mode("llm", model=self.MODEL_B)
        self.assertEqual(_container_model_file(), self.MODEL_B_FILE)
        _post_mode("llm", model=self.MODEL_A)
        self.assertEqual(_container_model_file(), self.MODEL_A_FILE)

    def test_repeated_switch_to_same_model_is_noop(self):
        """If we ask for the model that's already loaded, the container
        should NOT be recreated (no unnecessary VRAM reload)."""
        # Note the container ID before
        result = subprocess.run(
            ["docker", "inspect", "llama_server", "--format", "{{.Id}}"],
            capture_output=True, text=True,
        )
        before_id = result.stdout.strip()

        status, payload = _post_mode("llm", model=self.MODEL_A)
        self.assertEqual(status, 200)

        result = subprocess.run(
            ["docker", "inspect", "llama_server", "--format", "{{.Id}}"],
            capture_output=True, text=True,
        )
        after_id = result.stdout.strip()
        self.assertEqual(before_id, after_id,
                         "container was recreated for a same-model 'swap'")

    def test_swap_state_file_matches_reality(self):
        """State file and container reality should always agree."""
        _post_mode("llm", model=self.MODEL_B)
        status = _get_status()
        self.assertEqual(status["mode"], "llm")
        self.assertEqual(status["model"], self.MODEL_B)
        self.assertEqual(_container_model_file(), self.MODEL_B_FILE)


@unittest.skipUnless(
    os.environ.get("LLMC_TEST_INTEGRATION") == "1",
    "Set LLMC_TEST_INTEGRATION=1 to run end-to-end GPU swap tests",
)
class TestModelSwapTiming(unittest.TestCase):
    """Measure swap latency. Doesn't assert specific durations (varies by
    hardware and page-cache state) but prints timing for the operator and
    catches gross regressions via an upper bound."""

    # Generous upper bound — Q4 27-30B GGUF first-load can take 60+ s on
    # cold storage. Page-cache warm should be <15 s. We allow up to 5 min
    # to catch only catastrophic regressions (e.g. accidentally redownload).
    MAX_SWAP_SECONDS = 300

    @classmethod
    def setUpClass(cls):
        if not _proxy_reachable():
            raise unittest.SkipTest(f"proxy not reachable at {PROXY_HOST}:{PROXY_PORT}")

    def _time_swap(self, model: str) -> float:
        t0 = time.monotonic()
        status, _ = _post_mode("llm", model=model, timeout=cls_timeout())
        elapsed = time.monotonic() - t0
        self.assertEqual(status, 200, f"swap to {model} failed: HTTP {status}")
        return elapsed

    def test_measure_gemma4_to_qwen36_to_gemma4(self):
        # Ensure starting state
        _post_mode("llm", model="gemma4", timeout=cls_timeout())

        t1 = self._time_swap("qwen36")
        t2 = self._time_swap("gemma4")

        print(f"\nSwap timings:")
        print(f"  gemma4 → qwen36: {t1:.1f}s")
        print(f"  qwen36 → gemma4: {t2:.1f}s")

        self.assertLess(t1, self.MAX_SWAP_SECONDS)
        self.assertLess(t2, self.MAX_SWAP_SECONDS)


def cls_timeout() -> float:
    """Per-swap timeout — matches MAX_SWAP_SECONDS plus a small buffer."""
    return TestModelSwapTiming.MAX_SWAP_SECONDS + 30


if __name__ == "__main__":
    unittest.main()
