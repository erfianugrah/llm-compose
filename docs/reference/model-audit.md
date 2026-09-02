# Model audit - upstream drift and orphan backup

`llmc audit` compares every preset's model and mmproj file against the
HuggingFace repo the preset names, and copies anything that no longer exists
upstream to off-box storage.

## Why it exists

A preset's `[model] repo` + `file` pair is the llama-server entrypoint's
download fallback: if `/models/<file>` is absent, the container fetches it
from that repo. Nothing re-checked that the pair still resolved.

On 2026-08-19 unsloth deleted every plain K-quant of `Qwen3.8-27B-GGUF` and
re-uploaded a UD-only lineup (imatrix, 08-20). The same sweep hit both
ggml-org gemma-4 repos. Four of our files stopped existing upstream while
the presets kept pointing at them:

| file | preset | discovered |
|---|---|---|
| `Qwen3.8-27B-Q4_K_M.gguf` | qwen38 (+3 hardlinked variants), loop | 2026-09-02 |
| `gemma-4-26B-A4B-it-Q4_K_M.gguf` | summarizer (whisper's cross-stack model) | 2026-09-02 |
| `gemma-4-31B-it-Q4_K_M.gguf` | gemma4 | 2026-09-02 |
| `gemma4-mmproj.gguf` | gemma4 | 2026-09-02 |

Nothing failed, because the local GGUFs survived. Had the volume been lost
first, the daily driver, the loop engine and the whisper summarisation model
were all unrecoverable. The audit turns that from luck into a check.

## Usage

```
usage: llmc audit [-h] [--deep] [--backup] [--dest DEST] [--dry-run]

  --deep       sha256 every same-size file against the HF LFS oid (slow, exact)
  --backup     rsync orphaned files (no upstream copy) to --dest and verify
  --dest DEST  backup destination host:/path (env LLMC_MODEL_BACKUP_DEST)
  --dry-run    with --backup: report what would be copied, copy nothing
```

`make audit` is the read-only report. Destination defaults to
`servarr:/tank/backups/llm-models/orphaned`.

## Statuses

| status | meaning |
|---|---|
| `ok` | present upstream at the same size (same sha256 in deep mode) |
| `renamed` | not at that path upstream, but identical bytes are - mmproj files get renamed into `/models`, so what matters is whether the content is still obtainable |
| `diff` | present upstream, bytes changed (re-quantised or re-uploaded); the local copy is a previous build |
| `gone` | not published upstream at all - the local copy is the only copy |
| `missing` | absent locally; the entrypoint would download it on next spawn |
| `local-only` | preset uses a `local/...` placeholder repo; nothing to check |
| `unknown` | the upstream lookup failed (network, rate limit) |

Exit codes: `0` clean, `1` orphans present with no backup run, `2` some file
could not be checked upstream, `3` a file is absent both locally and upstream
(unrecoverable) or a backup failed.

## Design notes

- **HF's LFS `oid` is the file's sha256.** Verified 2026-09-02 against
  `gemma-4-12b-it-Q4_K_M.gguf`: API oid `0a270ec9...` matched a local
  `sha256sum` exactly. Deep mode is therefore byte-identity, not a heuristic.
- **A failed lookup is `unknown`, never `gone`.** Reporting a rate limit as a
  deletion would trigger pointless multi-GiB copies and, worse, train the
  reader to ignore the alarm.
- **Orphans dedupe by inode.** The qwen38 variants are hardlinks of one
  15.9 GiB blob; backing up each name would copy the same bytes four times.
- **An ambiguous content match is refused.** If two upstream files share the
  local file's size, `renamed` is not claimed - a wrong twin would mask a
  real orphan.
- **Backups are verified at the far end.** `sha256sum` runs on the remote and
  must match the locally computed digest before the file is recorded in
  `SHA256SUMS`. A copy that is not hash-checked remotely is not a copy you
  have made.
- **Re-runs are cheap.** A remote file of matching size is skipped, so the
  weekly timer costs one HF API call per repo plus one ssh.

## Tests

```bash
make test         # unit: classifier logic against an injected repo tree
make test-audit   # live drill: real HF API + planted orphan over ssh (~40s)
```

The unit tests cover the logic (deletion detection, rate-limit-is-not-a-
deletion, inode dedup, ambiguous-twin refusal). The drill covers what only
fails for real: that the HF tree endpoint still returns LFS oids where we
expect them, that an oid genuinely equals the file's sha256, and that a
backed-up file lands byte-identical on the far end. It plants an 8 MiB
orthan, backs it up to a scratch remote dir, re-runs to prove the skip,
mutates the local file to prove a changed file is re-copied, then removes
both the local file and the scratch dir. Presets live in a temp dir
throughout, so the proxy's live preset list is never touched.

## Schedule

`llmc-model-audit.timer` (systemd user unit, Sundays 05:00 +/- 30 min,
`Persistent=true`) runs the audit with backup enabled at `Nice=19` /
`IOSchedulingClass=idle`. Deep hashing is off there deliberately: it would
sha256 ~90 GiB of GGUF and evict the page cache the resident model depends
on. Check it with:

```bash
systemctl --user list-timers llmc-model-audit.timer
journalctl --user -u llmc-model-audit.service -n 40
```

## What it does NOT do

It does not repoint presets or download replacements. A `diff` or `gone`
verdict is a decision for a human: switching quant changes `model_id` (the
GGUF stem), which is the id advertised on `/v1/models` and pinned in
`webui/models.json`, and any quant change wants a `llmc bench tasks` A/B
before adoption.
