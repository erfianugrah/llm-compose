"""Upstream drift audit for preset model files.

Presets carry a (repo, file) pair that the llama-server entrypoint uses as
its download fallback when /models/<file> is absent. Nothing re-validated
that pair after the initial download, so upstream deletions went unnoticed:
on 2026-08-19 unsloth deleted every plain K-quant of Qwen3.8-27B and
re-uploaded a UD-only lineup, silently orphaning the daily driver, the loop
engine and (in the same sweep) both gemma-4 presets. The local GGUFs
survived by luck; had the volume been lost, three presets were unrecoverable.

This module compares each preset's model/mmproj file against the HuggingFace
repo tree and classifies the result:

    ok           file present upstream, same size (and same hash with --deep)
    diff         present upstream but bytes changed (re-quantised / re-upload)
    renamed      not at that path upstream, but identical bytes are (mmproj
                 files get renamed on the way into /models, so the question
                 that matters is whether the CONTENT is still obtainable)
    gone         NOT present upstream - local copy is the only copy
    missing      not on local disk (download on next spawn; fatal if also gone)
    local-only   repo is a `local/...` placeholder; no upstream to check

HuggingFace's LFS `oid` IS the file's sha256 (verified 2026-09-02 against
gemma-4-12b-it-Q4_K_M), so --deep gives byte-identity, not a size heuristic.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

HF_API = "https://huggingface.co/api/models"
USER_AGENT = "llmc-audit/1"

OK = "ok"
RENAMED = "renamed"
DIFF = "diff"
GONE = "gone"
MISSING = "missing"
LOCAL_ONLY = "local-only"
UNKNOWN = "unknown"  # upstream lookup failed (network / rate limit)


class AuditError(Exception):
    pass


@dataclass
class FileAudit:
    preset: str  # preset.name (TOML stem), not the model_id
    kind: str  # "model" | "mmproj"
    repo: str
    filename: str
    local_path: Path
    local_size: Optional[int] = None
    upstream_size: Optional[int] = None
    upstream_sha256: Optional[str] = None
    local_sha256: Optional[str] = None
    status: str = UNKNOWN
    note: str = ""

    @property
    def unrecoverable(self) -> bool:
        """No upstream copy AND no local copy - the file is simply lost."""
        return self.status == GONE and self.local_size is None

    def to_dict(self) -> dict:
        return {
            "preset": self.preset,
            "kind": self.kind,
            "repo": self.repo,
            "file": self.filename,
            "local_path": str(self.local_path),
            "local_size": self.local_size,
            "upstream_size": self.upstream_size,
            "upstream_sha256": self.upstream_sha256,
            "local_sha256": self.local_sha256,
            "status": self.status,
            "note": self.note,
        }


# ── upstream ───────────────────────────────────────────────────────────


def fetch_tree(repo: str, *, timeout: float = 20.0) -> dict[str, dict]:
    """Return {path: {"size": int, "sha256": str|None}} for a HF repo.

    Raises AuditError on any transport/parse failure so the caller can mark
    the entry UNKNOWN rather than mistaking a rate-limit for a deletion -
    the difference between "nothing to do" and "your only copy is local".
    """
    url = f"{HF_API}/{repo}/tree/main?recursive=1"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.load(resp)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        raise AuditError(f"{repo}: upstream lookup failed: {exc}") from exc
    if not isinstance(payload, list):
        raise AuditError(f"{repo}: unexpected tree payload")
    out: dict[str, dict] = {}
    for entry in payload:
        if entry.get("type") != "file":
            continue
        lfs = entry.get("lfs") or {}
        out[entry["path"]] = {
            "size": lfs.get("size", entry.get("size")),
            "sha256": lfs.get("oid"),
        }
    return out


def sha256_file(path: Path, *, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


# ── audit ──────────────────────────────────────────────────────────────


def _targets(preset) -> Iterable[tuple[str, str]]:
    """(kind, filename) pairs a preset expects to find in the models dir."""
    yield "model", preset.model.file
    if preset.mmproj_filename:
        yield "mmproj", preset.mmproj_filename


def audit_presets(
    presets: dict,
    models_dir: Path,
    *,
    deep: bool = False,
    fetch: Callable[[str], dict[str, dict]] = fetch_tree,
) -> list[FileAudit]:
    """Classify every preset file against its upstream repo.

    `fetch` is injected so tests never touch the network. Repo trees are
    fetched once per repo, not once per file.
    """
    trees: dict[str, Optional[dict[str, dict]]] = {}
    errors: dict[str, str] = {}
    results: list[FileAudit] = []

    for key in sorted(presets):
        preset = presets[key]
        repo = preset.model.repo
        for kind, filename in _targets(preset):
            path = models_dir / filename
            entry = FileAudit(
                preset=preset.name, kind=kind, repo=repo,
                filename=filename, local_path=path,
            )
            if path.exists():
                entry.local_size = path.stat().st_size
            else:
                entry.status = MISSING

            if repo.startswith("local/"):
                entry.status = LOCAL_ONLY if entry.status != MISSING else MISSING
                entry.note = "local-only preset (placeholder repo)"
                results.append(entry)
                continue

            if repo not in trees:
                try:
                    trees[repo] = fetch(repo)
                except AuditError as exc:
                    trees[repo] = None
                    errors[repo] = str(exc)
            tree = trees[repo]
            if tree is None:
                entry.status = UNKNOWN
                entry.note = errors.get(repo, "upstream lookup failed")
                results.append(entry)
                continue

            up = tree.get(filename)
            if up is None:
                # Locally renamed asset (mmproj especially): the path is gone
                # but the bytes may still be published under another name.
                twin = _find_by_content(tree, entry, deep=deep)
                if twin is not None:
                    up_name, up_meta = twin
                    entry.status = RENAMED
                    entry.upstream_size = up_meta["size"]
                    entry.upstream_sha256 = up_meta["sha256"]
                    entry.note = f"upstream name: {up_name}"
                else:
                    entry.status = GONE
                    entry.note = "not in upstream repo; local copy is the only copy"
                results.append(entry)
                continue

            entry.upstream_size = up["size"]
            entry.upstream_sha256 = up["sha256"]
            if entry.local_size is None:
                entry.status = MISSING
                entry.note = "absent locally; entrypoint would download it"
            elif entry.local_size != entry.upstream_size:
                entry.status = DIFF
                entry.note = "upstream re-uploaded; local copy is a previous build"
            elif deep and entry.upstream_sha256:
                entry.local_sha256 = sha256_file(path)
                if entry.local_sha256 == entry.upstream_sha256:
                    entry.status = OK
                else:
                    entry.status = DIFF
                    entry.note = "same size, different hash"
            else:
                entry.status = OK
            results.append(entry)

    return results


def _find_by_content(
    tree: dict[str, dict], entry: FileAudit, *, deep: bool
) -> Optional[tuple[str, dict]]:
    """Locate the same bytes elsewhere in the repo tree.

    Size alone is the match key unless --deep, where the local sha256 is
    compared against the LFS oid. A unique size match in one repo is strong;
    an ambiguous one (two files of identical size) is refused rather than
    guessed - reporting the wrong twin would hide a real orphan.
    """
    if entry.local_size is None:
        return None
    candidates = [(name, meta) for name, meta in tree.items()
                  if meta.get("size") == entry.local_size]
    if not candidates:
        return None
    if deep:
        entry.local_sha256 = entry.local_sha256 or sha256_file(entry.local_path)
        exact = [c for c in candidates if c[1].get("sha256") == entry.local_sha256]
        return exact[0] if len(exact) == 1 else None
    return candidates[0] if len(candidates) == 1 else None


def orphans(results: Iterable[FileAudit]) -> list[FileAudit]:
    """Files with no upstream copy that DO exist locally - the backup set.

    Deduplicated by inode: the qwen38 variants are hardlinks of one 15.9 GiB
    blob, so backing up each name would copy the same bytes four times.
    """
    seen: set[tuple[int, int]] = set()
    out: list[FileAudit] = []
    for r in results:
        if r.status != GONE or r.local_size is None:
            continue
        try:
            st = r.local_path.stat()
            key = (st.st_dev, st.st_ino)
        except OSError:
            key = (-1, hash(r.filename))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


# ── backup ─────────────────────────────────────────────────────────────


@dataclass
class BackupResult:
    filename: str
    action: str  # "skipped" | "copied" | "failed"
    sha256: Optional[str] = None
    detail: str = ""


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def _split_dest(dest: str) -> tuple[str, str]:
    if ":" not in dest:
        raise AuditError(f"backup dest must be host:/path, got {dest!r}")
    host, remote_dir = dest.split(":", 1)
    return host, remote_dir


def remote_inventory(dest: str, *, create: bool = False) -> dict[str, int]:
    """{filename: size} already at the backup destination.

    One ssh. Used by the read-only report too: an orphan that is already on
    off-box storage is not an alarm, and a report that cannot tell the
    difference gets ignored within a month.
    """
    host, remote_dir = _split_dest(dest)
    mkdir = f"mkdir -p {remote_dir} && " if create else ""
    proc = _run(["ssh", host, f"{mkdir}ls -l {remote_dir} 2>/dev/null || true"])
    if proc.returncode != 0:
        raise AuditError(f"ssh {host}: {proc.stderr.strip() or 'failed'}")
    sizes: dict[str, int] = {}
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 9 and parts[0].startswith("-"):
            sizes[parts[-1]] = int(parts[4])
    return sizes


def backup_orphans(
    entries: list[FileAudit],
    dest: str,
    *,
    dry_run: bool = False,
    log: Callable[[str], None] = lambda _msg: None,
) -> list[BackupResult]:
    """rsync orphaned files to `dest` (`host:/path`) and verify remotely.

    Idempotent: a remote file of the same size is left alone. Verification
    is a remote sha256sum compared against a locally computed one - a copy
    that is not hash-checked at the far end is a copy you have not made.
    """
    host, remote_dir = _split_dest(dest)
    out: list[BackupResult] = []

    if dry_run:
        return [BackupResult(e.filename, "skipped", detail="dry-run") for e in entries]

    remote_sizes = remote_inventory(dest, create=True)

    for entry in entries:
        name = entry.filename
        if remote_sizes.get(name) == entry.local_size:
            log(f"{name}: already at {dest} ({entry.local_size} bytes)")
            out.append(BackupResult(name, "skipped", detail="already present"))
            continue

        log(f"{name}: hashing {entry.local_size} bytes")
        local_hash = sha256_file(entry.local_path)
        log(f"{name}: rsync -> {dest}")
        proc = _run(["rsync", "-a", "--partial", str(entry.local_path), f"{dest}/"])
        if proc.returncode != 0:
            out.append(BackupResult(name, "failed", local_hash, proc.stderr.strip()))
            continue

        proc = _run(["ssh", host, f"sha256sum {remote_dir}/{name}"])
        remote_hash = proc.stdout.split()[0] if proc.returncode == 0 and proc.stdout else ""
        if remote_hash != local_hash:
            out.append(
                BackupResult(name, "failed", local_hash, f"remote sha256 {remote_hash or 'unavailable'}")
            )
            continue

        _run(["ssh", host,
              f"cd {remote_dir} && grep -qF '{name}' SHA256SUMS 2>/dev/null "
              f"|| echo '{local_hash}  {name}' >> SHA256SUMS"])
        log(f"{name}: verified {local_hash[:12]}")
        out.append(BackupResult(name, "copied", local_hash))

    return out
