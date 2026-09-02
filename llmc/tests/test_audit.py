"""Tests for llmc.audit - upstream drift classification + orphan backup.

No network: `fetch` is injected. The regression these lock in is the
2026-08-19 unsloth deletion of Qwen3.8-27B-Q4_K_M (and the parallel
gemma-4 sweep), where a preset kept pointing at a file that no longer
existed upstream and nothing noticed.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from llmc.audit import (
    DIFF,
    GONE,
    RENAMED,
    LOCAL_ONLY,
    MISSING,
    OK,
    UNKNOWN,
    AuditError,
    audit_presets,
    orphans,
    sha256_file,
)
from llmc.presets import AssetSpec, ModelSpec, Preset, RuntimeSpec


def _preset(name: str, repo: str, file: str, mmproj: str | None = None) -> Preset:
    return Preset(
        name=name,
        display_name=name,
        description="",
        vram_gb=1.0,
        model=ModelSpec(repo=repo, file=file),
        mmproj=AssetSpec(file=mmproj) if mmproj else AssetSpec(),
        runtime=RuntimeSpec(),
    )


def _tree(**files: tuple[int, str]) -> dict[str, dict]:
    return {name: {"size": size, "sha256": sha} for name, (size, sha) in files.items()}


class AuditClassificationTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.models = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write(self, name: str, payload: bytes) -> Path:
        path = self.models / name
        path.write_bytes(payload)
        return path

    def test_ok_when_size_matches(self):
        self._write("a.gguf", b"x" * 10)
        results = audit_presets(
            {"p": _preset("p", "org/repo", "a.gguf")},
            self.models,
            fetch=lambda repo: _tree(**{"a.gguf": (10, "deadbeef")}),
        )
        self.assertEqual([r.status for r in results], [OK])

    def test_gone_when_upstream_deleted(self):
        """The 2026-08-19 unsloth case: local file fine, upstream file removed."""
        self._write("Qwen3.8-27B-Q4_K_M.gguf", b"x" * 10)
        results = audit_presets(
            {"qwen38": _preset("qwen38", "unsloth/Qwen3.8-27B-GGUF",
                               "Qwen3.8-27B-Q4_K_M.gguf")},
            self.models,
            fetch=lambda repo: _tree(**{"Qwen3.8-27B-UD-Q4_K_M.gguf": (9, "abc")}),
        )
        self.assertEqual(results[0].status, GONE)
        self.assertFalse(results[0].unrecoverable)
        self.assertEqual([r.filename for r in orphans(results)],
                         ["Qwen3.8-27B-Q4_K_M.gguf"])

    def test_unrecoverable_when_gone_and_absent_locally(self):
        results = audit_presets(
            {"p": _preset("p", "org/repo", "vanished.gguf")},
            self.models,
            fetch=lambda repo: _tree(),
        )
        self.assertTrue(results[0].unrecoverable)

    def test_diff_on_size_change(self):
        self._write("a.gguf", b"x" * 10)
        results = audit_presets(
            {"p": _preset("p", "org/repo", "a.gguf")},
            self.models,
            fetch=lambda repo: _tree(**{"a.gguf": (11, "abc")}),
        )
        self.assertEqual(results[0].status, DIFF)

    def test_deep_detects_same_size_different_hash(self):
        path = self._write("a.gguf", b"x" * 10)
        real = sha256_file(path)
        same_size = audit_presets(
            {"p": _preset("p", "org/repo", "a.gguf")},
            self.models, deep=True,
            fetch=lambda repo: _tree(**{"a.gguf": (10, "0" * 64)}),
        )
        self.assertEqual(same_size[0].status, DIFF)
        matching = audit_presets(
            {"p": _preset("p", "org/repo", "a.gguf")},
            self.models, deep=True,
            fetch=lambda repo: _tree(**{"a.gguf": (10, real)}),
        )
        self.assertEqual(matching[0].status, OK)
        self.assertEqual(matching[0].local_sha256, real)

    def test_missing_local_but_upstream_present(self):
        results = audit_presets(
            {"p": _preset("p", "org/repo", "a.gguf")},
            self.models,
            fetch=lambda repo: _tree(**{"a.gguf": (10, "abc")}),
        )
        self.assertEqual(results[0].status, MISSING)
        self.assertFalse(results[0].unrecoverable)

    def test_local_only_repo_skips_upstream(self):
        self._write("loop.gguf", b"x")
        calls: list[str] = []

        def fetch(repo):
            calls.append(repo)
            return _tree()

        results = audit_presets(
            {"loop": _preset("loop", "local/loop-engine", "loop.gguf")},
            self.models, fetch=fetch,
        )
        self.assertEqual(results[0].status, LOCAL_ONLY)
        self.assertEqual(calls, [])

    def test_network_failure_is_unknown_not_gone(self):
        """A rate limit must never be reported as an upstream deletion."""
        self._write("a.gguf", b"x")

        def fetch(repo):
            raise AuditError("429")

        results = audit_presets(
            {"p": _preset("p", "org/repo", "a.gguf")}, self.models, fetch=fetch
        )
        self.assertEqual(results[0].status, UNKNOWN)
        self.assertEqual(orphans(results), [])

    def test_mmproj_is_audited_too(self):
        self._write("a.gguf", b"x" * 10)
        self._write("a-mmproj.gguf", b"y" * 5)
        results = audit_presets(
            {"p": _preset("p", "org/repo", "a.gguf", mmproj="a-mmproj.gguf")},
            self.models,
            fetch=lambda repo: _tree(**{"a.gguf": (10, "abc")}),
        )
        kinds = {r.kind: r.status for r in results}
        self.assertEqual(kinds, {"model": OK, "mmproj": GONE})

    def test_tree_fetched_once_per_repo(self):
        self._write("a.gguf", b"x" * 10)
        self._write("b.gguf", b"x" * 10)
        calls: list[str] = []

        def fetch(repo):
            calls.append(repo)
            return _tree(**{"a.gguf": (10, "abc"), "b.gguf": (10, "abc")})

        audit_presets(
            {
                "p1": _preset("p1", "org/repo", "a.gguf"),
                "p2": _preset("p2", "org/repo", "b.gguf"),
            },
            self.models, fetch=fetch,
        )
        self.assertEqual(calls, ["org/repo"])


class RenameAndDedupTest(unittest.TestCase):
    """Local renames and hardlink fan-out must not be read as data loss."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.models = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_renamed_asset_found_by_size(self):
        """qwen38-mmproj.gguf is upstream mmproj-BF16.gguf under a local name."""
        (self.models / "a.gguf").write_bytes(b"x" * 10)
        (self.models / "p-mmproj.gguf").write_bytes(b"y" * 42)
        results = audit_presets(
            {"p": _preset("p", "org/repo", "a.gguf", mmproj="p-mmproj.gguf")},
            self.models,
            fetch=lambda repo: _tree(**{"a.gguf": (10, "abc"),
                                        "mmproj-BF16.gguf": (42, "def")}),
        )
        mmproj = next(r for r in results if r.kind == "mmproj")
        self.assertEqual(mmproj.status, RENAMED)
        self.assertIn("mmproj-BF16.gguf", mmproj.note)
        self.assertEqual(orphans(results), [])

    def test_ambiguous_size_match_is_refused(self):
        (self.models / "a.gguf").write_bytes(b"x" * 10)
        results = audit_presets(
            {"p": _preset("p", "org/repo", "a.gguf")},
            self.models,
            fetch=lambda repo: _tree(**{"twin1.gguf": (10, "abc"),
                                        "twin2.gguf": (10, "def")}),
        )
        self.assertEqual(results[0].status, GONE)

    def test_hardlinks_dedupe_to_one_orphan(self):
        real = self.models / "base.gguf"
        real.write_bytes(b"x" * 10)
        for alias in ("base-a.gguf", "base-b.gguf"):
            (self.models / alias).hardlink_to(real)
        presets = {
            "p0": _preset("p0", "org/repo", "base.gguf"),
            "p1": _preset("p1", "org/repo", "base-a.gguf"),
            "p2": _preset("p2", "org/repo", "base-b.gguf"),
        }
        results = audit_presets(presets, self.models, fetch=lambda repo: _tree())
        self.assertEqual([r.status for r in results], [GONE, GONE, GONE])
        self.assertEqual(len(orphans(results)), 1)


class BackupGuardTest(unittest.TestCase):
    def test_dest_must_be_host_path(self):
        from llmc.audit import backup_orphans

        with self.assertRaises(AuditError):
            backup_orphans([], "/tank/backups", dry_run=False)


if __name__ == "__main__":
    unittest.main()
