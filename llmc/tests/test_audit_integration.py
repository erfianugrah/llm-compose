"""End-to-end audit drill: live HuggingFace API + real ssh/rsync backup.

Requires:
    1. Network access to huggingface.co
    2. ssh to the backup host in LLMC_AUDIT_TEST_DEST
       (default servarr:/tank/backups/llm-models/_audit-drill - a scratch
       directory this test creates and removes; it never touches the real
       orphan store)
    3. gemma-4-12b-it-Q4_K_M.gguf on the llama-models bind path (deep-hash
       case; 6.6 GiB read)

Gated behind LLMC_TEST_INTEGRATION=1. Run manually:

    LLMC_TEST_INTEGRATION=1 python3 -m unittest llmc.tests.test_audit_integration -v

The unit suite proves the classifier's logic against an injected tree. This
proves the parts that only fail for real: that the HF tree endpoint still
returns LFS oids where we expect them, that an oid genuinely equals the
file's sha256, and that a backup lands byte-identical on the far end.

Every case runs against a TEMPORARY presets dir, so the live proxy's preset
list (which live-reloads from models/) is never touched.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from llmc.audit import (
    GONE,
    MISSING,
    OK,
    audit_presets,
    backup_orphans,
    fetch_tree,
    orphans,
    remote_inventory,
    sha256_file,
)
from llmc.presets import load_all
from llmc.volumes import load as load_volumes

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEST = os.environ.get("LLMC_AUDIT_TEST_DEST",
                      "servarr:/tank/backups/llm-models/_audit-drill")
LIVE_REPO = "unsloth/Qwen3.8-27B-GGUF"
DEEP_REPO = "unsloth/gemma-4-12b-it-GGUF"
DEEP_FILE = "gemma-4-12b-it-Q4_K_M.gguf"


def _models_dir() -> Path:
    registry = load_volumes(REPO_ROOT / "volumes.toml")
    for spec in registry:
        if spec.name == "llmc-llama-models":
            return spec.device
    raise RuntimeError("llmc-llama-models not in volumes.toml")


def _preset_toml(*, repo: str, file: str) -> str:
    return (
        'name = "audit drill"\n'
        'description = "throwaway preset created by test_audit_integration"\n'
        "vram_gb = 1.0\n\n"
        "[model]\n"
        f'repo = "{repo}"\n'
        f'file = "{file}"\n\n'
        "[runtime]\n"
        "context_size = 4096\n"
    )


@unittest.skipUnless(
    os.environ.get("LLMC_TEST_INTEGRATION") == "1",
    "Set LLMC_TEST_INTEGRATION=1 to run the live audit drill",
)
class LiveUpstreamTest(unittest.TestCase):
    def test_tree_exposes_lfs_oids(self):
        tree = fetch_tree(LIVE_REPO)
        ggufs = [meta for path, meta in tree.items() if path.endswith(".gguf")]
        self.assertTrue(ggufs, f"{LIVE_REPO} listed no GGUFs")
        self.assertTrue(
            all(len(m["sha256"] or "") == 64 for m in ggufs),
            "expected a 64-char LFS oid on every GGUF; the API shape changed",
        )

    def test_deleted_quant_is_still_deleted(self):
        """Guards the assumption this whole feature rests on."""
        tree = fetch_tree(LIVE_REPO)
        self.assertNotIn("Qwen3.8-27B-Q4_K_M.gguf", tree)
        self.assertIn("Qwen3.8-27B-UD-Q4_K_M.gguf", tree)

    def test_lfs_oid_equals_local_sha256(self):
        """HF's oid IS the sha256 - deep mode is identity, not a heuristic."""
        path = _models_dir() / DEEP_FILE
        if not path.exists():
            self.skipTest(f"{DEEP_FILE} not on disk")
        upstream = fetch_tree(DEEP_REPO)[DEEP_FILE]
        self.assertEqual(sha256_file(path), upstream["sha256"])

    def test_deep_audit_of_a_real_preset(self):
        path = _models_dir() / DEEP_FILE
        if not path.exists():
            self.skipTest(f"{DEEP_FILE} not on disk")
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copy(REPO_ROOT / "models" / "gemma4-12b.toml", Path(tmp))
            presets = load_all(Path(tmp))
            results = audit_presets(presets, _models_dir(), deep=True)
        model = next(r for r in results if r.kind == "model")
        self.assertEqual(model.status, OK)
        self.assertEqual(model.local_sha256, model.upstream_sha256)

    def test_missing_locally_but_live_upstream_is_not_an_orphan(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "drill.toml").write_text(
                _preset_toml(repo=LIVE_REPO, file="Qwen3.8-27B-UD-Q5_K_XL.gguf")
            )
            results = audit_presets(load_all(Path(tmp)), _models_dir())
        self.assertEqual(results[0].status, MISSING)
        self.assertFalse(results[0].unrecoverable)
        self.assertEqual(orphans(results), [])


@unittest.skipUnless(
    os.environ.get("LLMC_TEST_INTEGRATION") == "1",
    "Set LLMC_TEST_INTEGRATION=1 to run the live audit drill",
)
class LiveBackupDrillTest(unittest.TestCase):
    """Plant a real orphan, back it up over ssh, prove the bytes match."""

    ORPHAN = "audit-drill-orphan.gguf"

    @classmethod
    def setUpClass(cls):
        cls.models = _models_dir()
        cls.local = cls.models / cls.ORPHAN
        cls.host, cls.remote_dir = DEST.split(":", 1)
        # 8 MiB of deterministic noise: big enough to be a real transfer,
        # small enough to run in a test.
        cls.local.write_bytes(bytes(range(256)) * (8 << 12))

    @classmethod
    def tearDownClass(cls):
        cls.local.unlink(missing_ok=True)
        subprocess.run(["ssh", cls.host, f"rm -rf {cls.remote_dir}"],
                       capture_output=True, text=True)

    def _audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "drill.toml").write_text(
                _preset_toml(repo=LIVE_REPO, file=self.ORPHAN)
            )
            return audit_presets(load_all(Path(tmp)), self.models)

    def test_drill(self):
        results = self._audit()
        self.assertEqual(results[0].status, GONE, "planted orphan not detected")
        planted = orphans(results)
        self.assertEqual(len(planted), 1)

        first = backup_orphans(planted, DEST)
        self.assertEqual([b.action for b in first], ["copied"],
                         f"backup did not copy: {[b.detail for b in first]}")
        local_hash = sha256_file(self.local)
        self.assertEqual(first[0].sha256, local_hash)

        # The far end holds the same bytes, per the far end's own sha256.
        proc = subprocess.run(
            ["ssh", self.host, f"cd {self.remote_dir} && sha256sum {self.ORPHAN}"],
            capture_output=True, text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.split()[0], local_hash)

        # Recorded in the manifest, exactly once.
        proc = subprocess.run(
            ["ssh", self.host, f"cat {self.remote_dir}/SHA256SUMS"],
            capture_output=True, text=True,
        )
        self.assertEqual(
            [ln for ln in proc.stdout.splitlines() if self.ORPHAN in ln],
            [f"{local_hash}  {self.ORPHAN}"],
        )

        # Re-run is a no-op, and the read-only path can see the backup.
        second = backup_orphans(planted, DEST)
        self.assertEqual([b.action for b in second], ["skipped"])
        self.assertEqual(remote_inventory(DEST).get(self.ORPHAN),
                         self.local.stat().st_size)

        # Local content changes -> size differs -> copied again, not skipped.
        self.local.write_bytes(bytes(range(256)) * (4 << 12))
        again = backup_orphans(self._orphan_entries(), DEST)
        self.assertEqual([b.action for b in again], ["copied"])
        self.assertEqual(again[0].sha256, sha256_file(self.local))

    def _orphan_entries(self):
        return orphans(self._audit())


if __name__ == "__main__":
    unittest.main()
