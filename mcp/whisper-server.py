#!/usr/bin/env python3
"""
WhisperX MCP server for OpenCode.

Exposes YouTube download and transcription as MCP tools. Communicates over
stdio using the MCP JSON-RPC protocol. Zero external dependencies — stdlib
only.

Async job pattern: long-running tools (yt_transcribe, whisper_transcribe,
yt_transcribe_playlist) submit work to a background thread and poll for up
to `wait_max_sec` (default 50) before either returning the result or
returning a `job_id`. The LLM continues with `wait_job(job_id=…)` until the
work completes. This lets us keep individual MCP tool calls under the AI
SDK's tool-call timeout regardless of how long the underlying transcription
actually takes.

State is process-local; jobs survive only as long as the MCP server itself.
A 10-minute eviction window keeps memory bounded for completed jobs.

Tools:
  whisper_status         — Check whisper service status and GPU info
  yt_download            — Download audio from a URL
  whisper_transcribe     — Transcribe a local audio/video file
  yt_transcribe          — Download + transcribe in one step
  yt_transcribe_playlist — Process a playlist
  wait_job               — Resume polling a previously-started job
"""

import html as html_mod
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid

WHISPER_URL = os.environ.get("WHISPER_URL", "http://localhost:7860")
POLL_TIMEOUT = 1800              # max wait for the underlying HTTP call (30 min)
DEFAULT_WAIT_MAX_SEC = 50        # LLM tool-call window; keeps responses snappy
JOB_RETENTION_SEC = 600          # keep completed jobs around 10 min for late polls
JOB_POLL_INTERVAL_SEC = 2        # internal loop tick when waiting on a job


# ── HTTP helpers ─────────────────────────────────────────────────────


def _request(method, path, data=None, timeout=600):
    """Make an HTTP request to the whisper service."""
    url = f"{WHISPER_URL}{path}"
    headers = {"Content-Type": "application/json"}
    body = json.dumps(data).encode() if data else None

    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        error_body = e.read().decode() if e.fp else ""
        try:
            return json.loads(error_body)
        except Exception:
            return {"error": f"HTTP {e.code}: {error_body[:300]}"}
    except urllib.error.URLError as e:
        return {"error": f"Connection failed: {e.reason}"}
    except Exception as e:
        return {"error": str(e)}


# ── YouTube context (for hotwords) ────────────────────────────────────


def _extract_video_id(url):
    match = re.search(r"(?:v=|youtu\.be/|shorts/)([\w-]{11})", url)
    return match.group(1) if match else ""


def _fetch_video_description(video_id):
    url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0",
            "Accept-Language": "en",
        })
        with urllib.request.urlopen(req, timeout=10) as resp:
            page = resp.read().decode("utf-8", errors="replace")
        match = re.search(r"var ytInitialPlayerResponse\s*=\s*(\{.+?\});", page)
        if match:
            try:
                data = json.loads(match.group(1))
                desc = data.get("videoDetails", {}).get("shortDescription", "")
                if desc:
                    return desc
            except (json.JSONDecodeError, KeyError):
                pass
        match = re.search(r'<meta\s+name="description"\s+content="([^"]*)"', page)
        if match:
            return html_mod.unescape(match.group(1))
    except Exception as e:
        log(f"Description fetch failed: {e}")
    return ""


def _extract_hotwords(text):
    if not text:
        return ""
    terms = set()
    terms.update(re.findall(r"\b[A-Z][a-zA-Z''\u2019-]{2,}(?:\s[A-Z][a-zA-Z''\u2019-]{2,})*\b", text))
    terms.update(re.findall(r'"([^"]{2,30})"', text))
    terms.update(re.findall(r"\b[A-Za-z]+[''\u2019-][A-Za-z]+\b", text))
    hotwords = ", ".join(sorted(t for t in terms if len(t) >= 3)[:150])
    if hotwords:
        log(f"Extracted hotwords: {hotwords[:120]}")
    return hotwords


# ── Job manager (async pattern) ──────────────────────────────────────


_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def _new_job_id() -> str:
    return uuid.uuid4().hex[:12]


def _submit_job(label: str, work_fn, *args, **kwargs) -> str:
    """Spawn a thread to run work_fn(*args, **kwargs). Returns job_id.

    `label` is a human-readable tag for log lines (e.g. "yt_transcribe abc123").
    """
    job_id = _new_job_id()
    with _jobs_lock:
        _jobs[job_id] = {
            "label": label,
            "status": "running",
            "started": time.time(),
            "finished": None,
            "result": None,
            "error": None,
        }

    def _runner():
        log(f"[job {job_id}] start: {label}")
        try:
            result = work_fn(*args, **kwargs)
            with _jobs_lock:
                _jobs[job_id]["status"] = "done"
                _jobs[job_id]["result"] = result
                _jobs[job_id]["finished"] = time.time()
            log(f"[job {job_id}] done in {time.time() - _jobs[job_id]['started']:.0f}s")
        except Exception as e:
            with _jobs_lock:
                _jobs[job_id]["status"] = "error"
                _jobs[job_id]["error"] = f"{type(e).__name__}: {e}"
                _jobs[job_id]["finished"] = time.time()
            log(f"[job {job_id}] error: {e}")

    threading.Thread(target=_runner, daemon=True).start()
    return job_id


def _poll_job(job_id: str, max_wait_sec: float) -> dict:
    """Block up to max_wait_sec waiting for job completion.

    Returns one of:
      {"status": "done", "result": ...}
      {"status": "error", "error": ...}
      {"status": "running", "elapsed_sec": int, "job_id": str}
      {"status": "unknown", "job_id": str}   (never seen / evicted)
    """
    with _jobs_lock:
        if job_id not in _jobs:
            return {"status": "unknown", "job_id": job_id}

    deadline = time.time() + max_wait_sec
    while time.time() < deadline:
        with _jobs_lock:
            current = _jobs[job_id]["status"]
        if current != "running":
            break
        time.sleep(JOB_POLL_INTERVAL_SEC)

    with _jobs_lock:
        job = dict(_jobs[job_id])  # snapshot

    if job["status"] == "done":
        return {"status": "done", "result": job["result"]}
    if job["status"] == "error":
        return {"status": "error", "error": job["error"]}
    return {
        "status": "running",
        "job_id": job_id,
        "elapsed_sec": int(time.time() - job["started"]),
    }


def _evict_old_jobs():
    """Background sweep: drop completed jobs older than JOB_RETENTION_SEC."""
    cutoff = time.time() - JOB_RETENTION_SEC
    with _jobs_lock:
        stale = [jid for jid, j in _jobs.items()
                 if j["status"] != "running"
                 and (j.get("finished") or 0) < cutoff]
        for jid in stale:
            del _jobs[jid]
    if stale:
        log(f"Evicted {len(stale)} completed jobs > {JOB_RETENTION_SEC}s old")


def _start_eviction_loop():
    def _loop():
        while True:
            time.sleep(60)
            try:
                _evict_old_jobs()
            except Exception as e:
                log(f"eviction loop error: {e}")
    threading.Thread(target=_loop, daemon=True).start()


# ── Async tool wrapper ────────────────────────────────────────────────


def _running_response(job_id: str, elapsed: int, label: str) -> str:
    return (
        f"{label} is still running (elapsed {elapsed}s). The work continues in "
        f"the background — call `wait_job` with the job_id below to keep "
        f"waiting:\n\n  job_id: {job_id}\n\n"
        f"You can also call wait_job multiple times if needed; each call waits "
        f"up to ~50 seconds before returning."
    )


def _async_run(label: str, work_fn, args, *, default_wait_max: int = DEFAULT_WAIT_MAX_SEC):
    """Submit work_fn() in a background thread, poll up to `wait_max_sec`,
    return either the result string or a job_id reminder.

    Tool args may include `wait_max_sec` to override the polling window.
    """
    wait_max = int(args.get("wait_max_sec", default_wait_max))
    job_id = _submit_job(label, work_fn)
    poll = _poll_job(job_id, wait_max)
    if poll["status"] == "done":
        return poll["result"]
    if poll["status"] == "error":
        return f"{label} failed: {poll['error']}"
    return _running_response(job_id, poll["elapsed_sec"], label)


# ── Worker bodies (run on background threads) ────────────────────────


def _check_service():
    result = _request("GET", "/api/status", timeout=5)
    if "error" in result:
        return (
            f"Whisper service is not reachable at {WHISPER_URL}.\n"
            f"Error: {result['error']}\n\n"
            f"Make sure the whisper-transcribe container is running:\n"
            f"  cd ~/whisper-transcribe && docker compose up -d"
        )
    return None


def _format_duration(seconds: int) -> str:
    return f"{seconds // 60:.0f}m {seconds % 60:.0f}s" if seconds >= 60 else f"{seconds}s"


def _submit_and_poll_whisper_job(params: dict, label: str = "whisper-job") -> dict:
    """Submit transcription via the server-side queue (/api/jobs) and poll
    until terminal. Returns the result dict on success or {"error": ...} on
    failure — preserves the contract of the previous _request("POST",
    "/api/transcribe", ...) call so the surrounding tool functions need
    minimal changes.

    Falls back to the legacy /api/transcribe with wait=true when the queue
    backend is down (whisper running without valkey), so the MCP tools keep
    working even when the queue layer is unavailable.
    """
    params = {**params, "consumer": params.get("consumer", "mcp")}
    sub = _request("POST", "/api/jobs", params, timeout=30)
    if "error" in sub:
        err_str = str(sub.get("error", ""))
        if "queue backend unavailable" in err_str or "503" in err_str:
            log(f"[{label}] queue unavailable; falling back to /api/transcribe wait=true")
            return _request(
                "POST", "/api/transcribe",
                {**params, "wait": True}, timeout=POLL_TIMEOUT,
            )
        return sub

    job_id = sub.get("job_id")
    if not job_id:
        return {"error": f"submit returned no job_id: {sub}"}
    log(f"[{label}] queued as {job_id} (position={sub.get('position')})")

    deadline = time.time() + POLL_TIMEOUT
    while time.time() < deadline:
        time.sleep(2)  # 2s tick — cheap (single HGETALL on valkey)
        poll = _request("GET", f"/api/jobs/{job_id}", timeout=10)
        if "error" in poll:
            # Transient network blip — keep polling. The job continues on
            # the server regardless of whether we're listening.
            continue
        status = poll.get("status")
        if status == "done":
            return poll.get("result") or {}
        if status == "failed":
            err = poll.get("error", "unknown")
            return {"error": err, "permanent": poll.get("permanent", False)}
        if status == "cancelled":
            return {"error": "cancelled"}
        # queued or running — keep polling
    return {"error": f"timed out polling job {job_id} after {POLL_TIMEOUT}s"}


def _do_yt_transcribe(args):
    """Background-thread body for yt_transcribe."""
    url = args["url"]
    err = _check_service()
    if err:
        return err

    log(f"yt_transcribe: downloading {url}")
    dl_result = _request("POST", "/api/yt-download", {"url": url}, timeout=600)
    if "error" in dl_result:
        return f"Download failed: {dl_result['error']}"

    filename = dl_result.get("filename", "")
    title = dl_result.get("title", "unknown")
    duration = dl_result.get("duration", 0)
    if not filename:
        return "Download succeeded but no filename returned"

    log(f"yt_transcribe: downloaded '{title}' ({duration}s), transcribing...")

    user_hotwords = args.get("hotwords", "")
    if not user_hotwords:
        video_id = _extract_video_id(url)
        description = _fetch_video_description(video_id) if video_id else ""
        hotwords = _extract_hotwords(f"{title}\n{description}")
    else:
        hotwords = user_hotwords

    params = {
        "file_path": filename,
        "model": args.get("model", "turbo"),
        "language": args.get("language", "Auto-detect"),
        "format": "txt",
        "diarize": args.get("diarize", False),
        "min_speakers": args.get("min_speakers", 0),
        "max_speakers": args.get("max_speakers", 0),
        "hotwords": hotwords,
        "initial_prompt": args.get("initial_prompt", ""),
        # Server-side translate policy. "auto" (default) translates
        # non-English sources to English — best for LLM consumption per
        # CS-FLEURS (arXiv:2509.14161). Pass translate=False to preserve
        # the source language.
        "translate": args.get("translate", "auto"),
        "cleanup": True,
    }
    if args.get("batch_size"):
        params["batch_size"] = args["batch_size"]

    result = _submit_and_poll_whisper_job(params, label=f"yt_transcribe {title[:40]}")
    if "error" in result:
        return f"Download succeeded but transcription failed: {result['error']}"

    transcript = result.get("transcript", "")
    status = result.get("status", "")
    cached_tag = " (cache hit)" if result.get("cached") else ""
    header = (
        f"Video: {title}\n"
        f"Duration: {_format_duration(duration)}\n"
        f"Transcription: {status}{cached_tag}\n"
        f"\n--- TRANSCRIPT ---\n\n"
    )
    return header + transcript


def _do_transcribe(args):
    """Background-thread body for whisper_transcribe."""
    file_path = args["file_path"]
    err = _check_service()
    if err:
        return err

    params = {
        "file_path": file_path,
        "model": args.get("model", "turbo"),
        "language": args.get("language", "Auto-detect"),
        "format": args.get("format", "txt"),
        "diarize": args.get("diarize", False),
        "min_speakers": args.get("min_speakers", 0),
        "max_speakers": args.get("max_speakers", 0),
        "hotwords": args.get("hotwords", ""),
        "initial_prompt": args.get("initial_prompt", ""),
        "suppress_numerals": args.get("suppress_numerals", False),
        "translate": args.get("translate", "auto"),
    }
    if args.get("batch_size"):
        params["batch_size"] = args["batch_size"]

    log(f"Transcribing: {file_path}")
    result = _submit_and_poll_whisper_job(params, label=f"transcribe {os.path.basename(file_path)}")
    if "error" in result:
        return f"Transcription failed: {result['error']}"

    status = result.get("status", "")
    transcript = result.get("transcript", "")
    subtitle_file = result.get("subtitle_file", "")
    lines = transcript.count("\n") + 1 if transcript else 0
    chars = len(transcript)

    cached_tag = " (cache hit)" if result.get("cached") else ""
    header = f"Transcription complete: {status}{cached_tag}\n({lines} lines, {chars} chars)\n"
    if subtitle_file:
        header += f"Subtitle file: {subtitle_file}\n"
    header += "\n--- TRANSCRIPT ---\n\n"
    return header + transcript


def _do_yt_transcribe_playlist(args):
    """Background-thread body for yt_transcribe_playlist."""
    url = args["url"]
    err = _check_service()
    if err:
        return err

    log(f"yt_transcribe_playlist: downloading {url}")
    dl_result = _request("POST", "/api/yt-download",
                         {"url": url, "playlist": True}, timeout=3600)
    if "error" in dl_result:
        return f"Download failed: {dl_result['error']}"

    items = dl_result.get("items", [dl_result] if "filename" in dl_result else [])
    if not items:
        return "No videos found in playlist"

    log(f"yt_transcribe_playlist: {len(items)} items to transcribe")
    all_transcripts = []
    for i, item in enumerate(items):
        filename = item.get("filename", "")
        title = item.get("title", "unknown")
        duration = item.get("duration", 0)
        duration_str = _format_duration(duration)

        if not filename:
            all_transcripts.append(
                f"\n--- [{i+1}/{len(items)}] {title} ---\nError: no file\n"
            )
            continue

        log(f"  [{i+1}/{len(items)}] Transcribing: {title}")
        params = {
            "file_path": filename,
            "model": args.get("model", "turbo"),
            "language": args.get("language", "Auto-detect"),
            "format": "txt",
            "diarize": args.get("diarize", False),
            "hotwords": _extract_hotwords(title),
            "translate": args.get("translate", "auto"),
            "cleanup": True,
        }
        result = _submit_and_poll_whisper_job(
            params, label=f"playlist[{i+1}/{len(items)}] {title[:40]}"
        )
        if "error" in result:
            all_transcripts.append(
                f"\n--- [{i+1}/{len(items)}] {title} ({duration_str}) ---\n"
                f"Transcription failed: {result['error']}\n"
            )
        else:
            transcript = result.get("transcript", "")
            cached_tag = " (cache hit)" if result.get("cached") else ""
            all_transcripts.append(
                f"\n--- [{i+1}/{len(items)}] {title} ({duration_str}){cached_tag} ---\n"
                f"{transcript}\n"
            )

    header = f"Playlist: {len(items)} videos transcribed\n"
    return header + "\n".join(all_transcripts)


# ── Tool implementations (synchronous + async wrappers) ──────────────


def tool_status(args):
    result = _request("GET", "/api/status", timeout=5)
    if "error" in result:
        return (
            f"Whisper service unavailable: {result['error']}\n\n"
            f"Start it with: cd ~/whisper-transcribe && docker compose up -d"
        )
    vision = result.get("vision", {})
    return (
        f"Status: {result.get('status', 'unknown')}\n"
        f"GPU: {result.get('gpu', 'unknown')}\n"
        f"Device: {result.get('device', 'unknown')}\n"
        f"Diarization: {'available' if result.get('diarization_available') else 'unavailable'}\n"
        f"Default batch size: {result.get('default_batch_size', '?')}\n"
        f"Vision: {vision.get('model', 'unconfigured')} "
        f"(fps_interval={vision.get('fps_interval', '?')}, "
        f"max_frames={vision.get('max_frames', '?')})"
    )


def tool_yt_download(args):
    """Download is fast (~30-90s for typical videos). Synchronous is fine."""
    url = args.get("url", "").strip()
    if not url:
        return "Error: 'url' parameter is required"
    err = _check_service()
    if err:
        return err

    log(f"Downloading: {url}")
    result = _request("POST", "/api/yt-download", {"url": url}, timeout=600)
    if "error" in result:
        return f"Download failed: {result['error']}"

    filename = result.get("filename", "unknown")
    title = result.get("title", "unknown")
    duration = result.get("duration", 0)
    return (
        f"Downloaded successfully:\n"
        f"  Title: {title}\n"
        f"  Duration: {_format_duration(duration)}\n"
        f"  File: {filename}\n\n"
        f"Use whisper_transcribe with file_path=\"{filename}\" to transcribe."
    )


def tool_yt_transcribe(args):
    if not args.get("url", "").strip():
        return "Error: 'url' parameter is required"
    return _async_run(
        f"yt_transcribe({args['url'][:60]})",
        lambda: _do_yt_transcribe(args),
        args,
    )


def tool_transcribe(args):
    if not args.get("file_path", "").strip():
        return "Error: 'file_path' parameter is required"
    return _async_run(
        f"transcribe({args['file_path']})",
        lambda: _do_transcribe(args),
        args,
    )


def tool_yt_transcribe_playlist(args):
    if not args.get("url", "").strip():
        return "Error: 'url' parameter is required"
    return _async_run(
        f"yt_transcribe_playlist({args['url'][:60]})",
        lambda: _do_yt_transcribe_playlist(args),
        args,
        # Playlists can run for hours; default a longer initial wait so
        # short ones complete in one tool call.
        default_wait_max=120,
    )


def tool_wait_job(args):
    """Resume polling a previously-submitted job."""
    job_id = args.get("job_id", "").strip()
    if not job_id:
        return "Error: 'job_id' parameter is required"
    wait_max = int(args.get("max_wait_sec", DEFAULT_WAIT_MAX_SEC))

    poll = _poll_job(job_id, wait_max)
    if poll["status"] == "done":
        return poll["result"]
    if poll["status"] == "error":
        return f"Job failed: {poll['error']}"
    if poll["status"] == "unknown":
        return (
            f"Unknown job_id: {job_id}\n"
            f"Either it never existed, the MCP server restarted, or it "
            f"completed more than {JOB_RETENTION_SEC}s ago and was evicted."
        )
    return _running_response(poll["job_id"], poll["elapsed_sec"], "Job")


# ── MCP protocol ─────────────────────────────────────────────────────


_ASYNC_NOTE = (
    "If the underlying work runs longer than ~50 seconds the call returns "
    "a `job_id` and you should call `wait_job(job_id=...)` repeatedly to "
    "continue waiting. Each `wait_job` call also waits up to ~50 seconds. "
    "The work runs in the background regardless of how many tool calls "
    "you make."
)


TOOLS = [
    {
        "name": "whisper_status",
        "description": "Check the whisper transcription service status, GPU info, and availability.",
        "inputSchema": {"type": "object", "properties": {}}
    },
    {
        "name": "yt_download",
        "description": "Download audio from a YouTube URL. Saves to /tmp on the whisper server. Use whisper_transcribe afterward to transcribe it. Synchronous (fast).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "YouTube video URL"}
            },
            "required": ["url"]
        }
    },
    {
        "name": "whisper_transcribe",
        "description": (
            "Transcribe a local audio/video file on the whisper server. "
            "Returns the full transcript. The file must exist on the whisper "
            "server's filesystem (e.g. from yt_download or /media mount).\n\n"
            f"{_ASYNC_NOTE}"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to the audio/video file on the whisper server"},
                "model": {"type": "string", "description": "Whisper model. Default: turbo",
                          "enum": ["tiny", "base", "small", "medium", "large", "turbo"]},
                "language": {"type": "string", "description": "Language code (e.g. 'en', 'fr') or 'Auto-detect'. Default: Auto-detect"},
                "translate": {
                    "description": (
                        "Translation policy. 'auto' (default) — server runs a "
                        "30s LID pre-pass and translates non-English sources to "
                        "English (best for downstream summarisation). true — "
                        "force task=translate. false — preserve source language."
                    ),
                    "oneOf": [
                        {"type": "string", "enum": ["auto"]},
                        {"type": "boolean"},
                    ],
                },
                "diarize": {"type": "boolean", "description": "Enable speaker diarization. Default: false"},
                "hotwords": {"type": "string", "description": "Comma-separated proper-noun bias terms"},
                "initial_prompt": {"type": "string", "description": "Context hint for the first transcription window"},
                "wait_max_sec": {"type": "integer", "description": f"Max seconds to wait inline before returning a job_id. Default {DEFAULT_WAIT_MAX_SEC}."}
            },
            "required": ["file_path"]
        }
    },
    {
        "name": "yt_transcribe",
        "description": (
            "Download a YouTube video and transcribe it in one step. Returns "
            "the full transcript ready for summarisation.\n\n"
            f"{_ASYNC_NOTE}"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "YouTube video URL"},
                "model": {"type": "string", "description": "Whisper model. Default: turbo",
                          "enum": ["tiny", "base", "small", "medium", "large", "turbo"]},
                "language": {"type": "string", "description": "Language code or 'Auto-detect'. Default: Auto-detect"},
                "translate": {
                    "description": (
                        "Translation policy. 'auto' (default) translates "
                        "non-English sources to English; true forces translate; "
                        "false preserves source language."
                    ),
                    "oneOf": [
                        {"type": "string", "enum": ["auto"]},
                        {"type": "boolean"},
                    ],
                },
                "diarize": {"type": "boolean", "description": "Enable speaker diarization. Default: false"},
                "hotwords": {"type": "string", "description": "Words the model might mishear"},
                "initial_prompt": {"type": "string", "description": "Context hint for transcription"},
                "wait_max_sec": {"type": "integer", "description": f"Max seconds to wait inline before returning a job_id. Default {DEFAULT_WAIT_MAX_SEC}."}
            },
            "required": ["url"]
        }
    },
    {
        "name": "yt_transcribe_playlist",
        "description": (
            "Download and transcribe all videos in a YouTube playlist. "
            "Returns combined transcripts with headers per video. "
            "Playlists can run for many minutes — default inline wait is "
            "120 seconds (override via wait_max_sec).\n\n"
            f"{_ASYNC_NOTE}"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "YouTube playlist URL"},
                "model": {"type": "string", "description": "Whisper model. Default: turbo",
                          "enum": ["tiny", "base", "small", "medium", "large", "turbo"]},
                "language": {"type": "string", "description": "Language code or 'Auto-detect'. Default: Auto-detect"},
                "translate": {
                    "description": (
                        "Translation policy applied to every video in the "
                        "playlist. 'auto' (default), true (force), false (native)."
                    ),
                    "oneOf": [
                        {"type": "string", "enum": ["auto"]},
                        {"type": "boolean"},
                    ],
                },
                "diarize": {"type": "boolean", "description": "Enable speaker diarization. Default: false"},
                "wait_max_sec": {"type": "integer", "description": "Max seconds to wait inline before returning a job_id. Default 120."}
            },
            "required": ["url"]
        }
    },
    {
        "name": "wait_job",
        "description": (
            "Resume polling a previously-started job. Returns the result if "
            "the job has completed, the error if it failed, or another "
            "`job_id` reminder if still running. Each call waits up to "
            "`max_wait_sec` (default 50) before returning.\n\n"
            "When yt_transcribe / whisper_transcribe / yt_transcribe_playlist "
            "returns 'still running' with a job_id, call this tool with that "
            "job_id. Repeat until you get the transcript."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "job_id": {"type": "string", "description": "Job ID returned by a previous tool call"},
                "max_wait_sec": {"type": "integer", "description": f"Max seconds to wait inline. Default {DEFAULT_WAIT_MAX_SEC}."}
            },
            "required": ["job_id"]
        }
    }
]

TOOL_HANDLERS = {
    "whisper_status": tool_status,
    "yt_download": tool_yt_download,
    "whisper_transcribe": tool_transcribe,
    "yt_transcribe": tool_yt_transcribe,
    "yt_transcribe_playlist": tool_yt_transcribe_playlist,
    "wait_job": tool_wait_job,
}

SERVER_INFO = {"name": "whisper", "version": "2.0.0"}
CAPABILITIES = {"tools": {}}


def handle_request(req):
    method = req.get("method", "")
    req_id = req.get("id")
    params = req.get("params", {})

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "serverInfo": SERVER_INFO,
                "capabilities": CAPABILITIES,
            }
        }

    if method == "notifications/initialized":
        return None

    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOLS}}

    if method == "tools/call":
        tool_name = params.get("name", "")
        tool_args = params.get("arguments", {})
        handler = TOOL_HANDLERS.get(tool_name)
        if not handler:
            return {
                "jsonrpc": "2.0", "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": f"Unknown tool: {tool_name}"}],
                    "isError": True
                }
            }
        try:
            result_text = handler(tool_args)
        except Exception as e:
            result_text = f"Error: {e}"
            log(f"Tool {tool_name} failed: {e}")
        return {
            "jsonrpc": "2.0", "id": req_id,
            "result": {
                "content": [{"type": "text", "text": result_text}],
                "isError": False
            }
        }

    if method == "ping":
        return {"jsonrpc": "2.0", "id": req_id, "result": {}}

    return {
        "jsonrpc": "2.0", "id": req_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"}
    }


def main():
    log("Whisper MCP server v2 starting (async job pattern)")
    log(f"Whisper URL: {WHISPER_URL}")
    _start_eviction_loop()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            log(f"Invalid JSON: {e}")
            continue
        response = handle_request(req)
        if response is not None:
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()


def log(msg):
    print(f"[whisper-mcp] {msg}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
