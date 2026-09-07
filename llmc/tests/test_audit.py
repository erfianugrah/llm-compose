"""Tests for llmc.audit - upstream drift classification + orphan backup.

No network: `fetch` is injected. The regression these lock in is the
2026-08-19 unsloth deletion of Qwen3.8-27B-Q4_K_M (and the parallel
gemma-4 sweep), where a preset kept pointing at a file that no longer
existed upstream and nothing noticed.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path

from llmc.audit import (
    DIFF,
    unreferenced,
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
from llmc import audit
from llmc.presets import AssetSpec, ModelSpec, NinferSpec, Preset, RuntimeSpec


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


class UnreferencedTest(unittest.TestCase):
    """The mirror question: which files on disk does no preset name?"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.models = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_lists_only_unnamed_files_largest_first(self):
        (self.models / "used.gguf").write_bytes(b"x" * 10)
        (self.models / "used-mmproj.gguf").write_bytes(b"x" * 5)
        (self.models / "small-stray.gguf").write_bytes(b"x" * 20)
        (self.models / "big-stray.gguf").write_bytes(b"x" * 99)
        (self.models / "notes.txt").write_bytes(b"ignored")
        presets = {"p": _preset("p", "org/repo", "used.gguf", mmproj="used-mmproj.gguf")}
        found = unreferenced(presets, self.models)
        self.assertEqual([f.filename for f in found],
                         ["big-stray.gguf", "small-stray.gguf"])

    def test_symlink_target_counts_as_referenced(self):
        """`loop` names a symlink; its target must not look unreferenced."""
        real = self.models / "base.gguf"
        real.write_bytes(b"x" * 10)
        (self.models / "loop-base.gguf").symlink_to("base.gguf")
        presets = {"loop": _preset("loop", "local/x", "loop-base.gguf")}
        self.assertEqual(unreferenced(presets, self.models), [])

    def test_hardlink_count_is_reported(self):
        real = self.models / "a.gguf"
        real.write_bytes(b"x" * 10)
        (self.models / "b.gguf").hardlink_to(real)
        found = unreferenced({}, self.models)
        self.assertEqual({f.filename: f.links for f in found},
                         {"a.gguf": 2, "b.gguf": 2})


class BackupGuardTest(unittest.TestCase):
    def test_dest_must_be_host_path(self):
        from llmc.audit import backup_orphans

        with self.assertRaises(AuditError):
            backup_orphans([], "/tank/backups", dry_run=False)


if __name__ == "__main__":
    unittest.main()


class TestUnreferencedBackupSet(unittest.TestCase):
    """`--backup` only ever covered orphans (files gone from upstream). The
    146 GB that actually fills the disk is the UNREFERENCED set - files no
    preset names, most of which still exist upstream. Re-downloading those is
    hours over the internet when servarr is on the LAN, so they get backed up
    too rather than deleted and re-fetched (2026-09-07)."""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _write(self, name, size=1024):
        p = self.dir / name
        p.write_bytes(b"x" * size)
        return p

    def test_converts_to_entries_backup_orphans_accepts(self):
        self._write("stray.gguf", 2048)
        items = [audit.UnreferencedFile(filename="stray.gguf", size=2048, links=1)]
        entries = audit.unreferenced_backup_set(items, self.dir)
        self.assertEqual(len(entries), 1)
        e = entries[0]
        # the exact attribute surface backup_orphans touches
        self.assertEqual(e.filename, "stray.gguf")
        self.assertEqual(e.local_size, 2048)
        self.assertEqual(e.local_path, self.dir / "stray.gguf")

    def test_symlinks_are_excluded(self):
        """Backing up a symlink copies a pointer, not the weights."""
        self._write("real.gguf")
        (self.dir / "link.gguf").symlink_to(self.dir / "real.gguf")
        items = [audit.UnreferencedFile(filename="link.gguf", size=1024, links=1,
                                        symlink_to="real.gguf")]
        self.assertEqual(audit.unreferenced_backup_set(items, self.dir), [])

    def test_hardlinked_names_dedupe_to_one_copy(self):
        """Same bytes under two names must not be sent twice."""
        a = self._write("a.gguf", 4096)
        b = self.dir / "b.gguf"
        os.link(a, b)
        items = [
            audit.UnreferencedFile(filename="a.gguf", size=4096, links=2),
            audit.UnreferencedFile(filename="b.gguf", size=4096, links=2),
        ]
        self.assertEqual(len(audit.unreferenced_backup_set(items, self.dir)), 1)

    def test_missing_file_is_skipped_not_fatal(self):
        items = [audit.UnreferencedFile(filename="vanished.gguf", size=10, links=1)]
        self.assertEqual(audit.unreferenced_backup_set(items, self.dir), [])


class TestDryRunIsNotABackup(unittest.TestCase):
    """`--backup --dry-run` copied nothing, so its results must never count as
    evidence of a backup. Observed 2026-09-07: a dry run against a
    non-existent destination made the audit print "3 orphan(s), all backed up
    at <dest>", because dry-run results were emitted as action="skipped" and
    the summary treated "skipped" as "already present". That is a false-safe
    in the one tool you consult before deleting 146 GB."""

    def test_dry_run_action_is_distinguishable(self):
        entries = [audit._BackupEntry("a.gguf", Path("/nonexistent/a.gguf"), 10)]
        results = audit.backup_orphans(entries, "host:/path", dry_run=True)
        self.assertEqual(len(results), 1)
        self.assertEqual(
            results[0].action, "dry-run",
            "a dry run must not report the same action as a real skip",
        )

    def test_dry_run_action_is_not_skipped(self):
        """Specifically NOT "skipped": that is what the summary trusts."""
        entries = [audit._BackupEntry("a.gguf", Path("/nonexistent/a.gguf"), 10)]
        results = audit.backup_orphans(entries, "host:/path", dry_run=True)
        self.assertNotEqual(results[0].action, "skipped")


class TestEngineAwareModelsDir(unittest.TestCase):
    """Every preset was resolved against ONE models dir - the llama volume - so
    a ninfer preset, whose artifact lives in llmc-ninfer-models, was reported
    `missing ... entrypoint would download it`. Both halves were false: the
    file was present in its own volume, and nothing downloads a .ninfer
    artifact (ensure_preset_assets only fetches mmproj/template URLs).

    A false "missing" in the tool you consult before deleting weights is the
    same class of defect as the dry-run false-safe (2026-09-07)."""

    def setUp(self):
        self.llama_dir = Path(tempfile.mkdtemp())
        self.ninfer_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        for d in (self.llama_dir, self.ninfer_dir):
            shutil.rmtree(d, ignore_errors=True)

    def _ninfer_preset(self):
        return Preset(
            name="qwen38-ninfer",
            display_name="ninfer",
            description="",
            vram_gb=22.1,
            model=ModelSpec(repo="neroued/x", file="q.ninfer", id="q-alias"),
            runtime=RuntimeSpec(),
            engine="ninfer",
            ninfer=NinferSpec(),
        )

    def test_ninfer_artifact_found_in_its_own_volume(self):
        (self.ninfer_dir / "q.ninfer").write_bytes(b"x" * 16)
        results = audit.audit_presets(
            {"q-alias": self._ninfer_preset()},
            self.llama_dir,
            fetch=lambda repo: {},
            dirs={"ninfer": self.ninfer_dir},
        )
        statuses = {r.status for r in results}
        self.assertNotIn(
            audit.MISSING, statuses,
            "artifact present in the ninfer volume must not read as missing",
        )

    def test_absent_ninfer_artifact_does_not_promise_a_download(self):
        # Present upstream, absent locally: that is the MISSING branch. With an
        # empty upstream tree it would classify as GONE first.
        upstream = {"q.ninfer": {"size": 16, "sha256": ""}}
        results = audit.audit_presets(
            {"q-alias": self._ninfer_preset()},
            self.llama_dir,
            fetch=lambda repo: upstream,
            dirs={"ninfer": self.ninfer_dir},
        )
        entry = results[0]
        self.assertEqual(entry.status, audit.MISSING)
        self.assertNotIn(
            "would download", entry.note,
            "nothing downloads a .ninfer artifact; the note must not claim otherwise",
        )
        self.assertIn("place", entry.note.lower())

    def test_llama_presets_unaffected_when_no_dirs_given(self):
        (self.llama_dir / "m.gguf").write_bytes(b"y" * 8)
        p = Preset(
            name="l", display_name="l", description="", vram_gb=1.0,
            model=ModelSpec(repo="local/x", file="m.gguf"), runtime=RuntimeSpec(),
        )
        results = audit.audit_presets({"m": p}, self.llama_dir, fetch=lambda repo: {})
        self.assertNotIn(audit.MISSING, {r.status for r in results})
