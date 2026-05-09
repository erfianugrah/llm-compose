#!/usr/bin/env python3
"""
WhisperX MCP server for OpenCode.

Exposes YouTube download and transcription as MCP tools so the LLM can
fetch videos and get transcripts from within OpenCode. Communicates over
stdio using the MCP JSON-RPC protocol. Zero external dependencies — stdlib only.

Tools:
  whisper_status      — Check whisper service status and GPU info
  yt_download         — Download audio from a YouTube URL
  whisper_transcribe  — Transcribe a local audio/video file
  yt_transcribe       — Download + transcribe in one step (for summaries)

The server talks to the whisper-transcribe service at WHISPER_URL
(default: http://localhost:7860). The service runs on the GPU and
handles yt-dlp downloads internally.

Typical flow for summarization:
  1. User: "transcribe and summarize this video: <url>"
  2. OpenCode calls yt_transcribe(url=<url>)
  3. This server → whisper-transcribe API → downloads audio → transcribes
  4. Returns full transcript text to the LLM
  5. LLM reads transcript and writes summary
"""

import json
import sys
import urllib.request
import urllib.error
import time
import os

WHISPER_URL = os.environ.get("WHISPER_URL", "http://localhost:7860")
LLM_URL = os.environ.get("LLM_URL", "http://localhost:11434/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "Qwen3.5-4B-Q8_0")
POLL_INTERVAL = 5  # seconds between status polls for long transcriptions
POLL_TIMEOUT = 1800  # max 30 min for very long videos


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


# ── Smart hotwords ───────────────────────────────────────────────────

import re
import html as html_mod


def _fetch_video_description(video_id):
    """Fetch YouTube video description for proper noun context."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            page = resp.read().decode("utf-8", errors="replace")
        match = re.search(r'<meta\s+name="description"\s+content="([^"]*)"', page)
        if match:
            return html_mod.unescape(match.group(1))
        match = re.search(r'<meta\s+property="og:description"\s+content="([^"]*)"', page)
        if match:
            return html_mod.unescape(match.group(1))
    except Exception as e:
        log(f"Description fetch failed: {e}")
    return ""


def _generate_hotwords(title, description=""):
    """Ask LLM to generate hotwords from video title + description."""
    if not title:
        return ""

    prompt = (
        "Given this video title and description, list proper nouns, technical terms, "
        "jargon, and names that a speech-to-text model might mishear. Include character "
        "names, place names, game/product terminology, brand names, and specialized vocabulary. "
        "Output ONLY a comma-separated list, nothing else. No explanations.\n\n"
        f"Title: {title}\n"
        f"Description: {description or '(none)'}"
    )

    payload = json.dumps({
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 256,
    }).encode()

    try:
        req = urllib.request.Request(
            f"{LLM_URL}/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
        hotwords = data["choices"][0]["message"]["content"].strip()
        log(f"Generated hotwords: {hotwords[:120]}")
        return hotwords
    except Exception as e:
        log(f"Hotword generation failed (non-fatal): {e}")
        return ""


def _extract_video_id(url):
    """Extract video ID from various YouTube URL formats."""
    patterns = [
        r"(?:v=|youtu\.be/|shorts/)([\w-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return ""



# ── Tool implementations ─────────────────────────────────────────────

def _check_service():
    """Quick connectivity check. Returns error string or None."""
    result = _request("GET", "/api/status", timeout=5)
    if "error" in result:
        return (
            f"Whisper service is not reachable at {WHISPER_URL}.\n"
            f"Error: {result['error']}\n\n"
            f"Make sure the whisper-transcribe container is running:\n"
            f"  cd ~/whisper-transcribe && docker compose up -d"
        )
    return None


def tool_status(args):
    """Check whisper service status."""
    result = _request("GET", "/api/status", timeout=5)
    if "error" in result:
        return f"Whisper service unavailable: {result['error']}\n\nStart it with: cd ~/whisper-transcribe && docker compose up -d"
    return (
        f"Status: {result.get('status', 'unknown')}\n"
        f"GPU: {result.get('gpu', 'unknown')}\n"
        f"Device: {result.get('device', 'unknown')}\n"
        f"Diarization: {'available' if result.get('diarization_available') else 'unavailable'}\n"
        f"Default batch size: {result.get('default_batch_size', '?')}"
    )


def tool_yt_download(args):
    """Download audio from a YouTube URL."""
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
    duration_str = f"{duration // 60:.0f}m {duration % 60:.0f}s" if duration >= 60 else f"{duration}s"

    return (
        f"Downloaded successfully:\n"
        f"  Title: {title}\n"
        f"  Duration: {duration_str}\n"
        f"  File: {filename}\n\n"
        f"Use whisper_transcribe with file_path=\"{filename}\" to transcribe."
    )


def tool_transcribe(args):
    """Transcribe a local audio/video file."""
    file_path = args.get("file_path", "").strip()
    if not file_path:
        return "Error: 'file_path' parameter is required"

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
    }
    if args.get("batch_size"):
        params["batch_size"] = args["batch_size"]

    log(f"Transcribing: {file_path}")
    # Transcription can take a long time for large files
    result = _request("POST", "/api/transcribe", params, timeout=POLL_TIMEOUT)

    if "error" in result:
        return f"Transcription failed: {result['error']}"

    status = result.get("status", "")
    transcript = result.get("transcript", "")
    subtitle_file = result.get("subtitle_file", "")

    lines = transcript.count("\n") + 1 if transcript else 0
    chars = len(transcript)

    header = f"Transcription complete: {status}\n({lines} lines, {chars} chars)\n"
    if subtitle_file:
        header += f"Subtitle file: {subtitle_file}\n"
    header += "\n--- TRANSCRIPT ---\n\n"

    return header + transcript


def tool_yt_transcribe(args):
    """Download a YouTube video and transcribe it in one step."""
    url = args.get("url", "").strip()
    if not url:
        return "Error: 'url' parameter is required"

    err = _check_service()
    if err:
        return err

    # Step 1: Download
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

    # Step 2: Smart hotwords — generate from title + description if user didn't provide
    user_hotwords = args.get("hotwords", "")
    if not user_hotwords:
        video_id = _extract_video_id(url)
        description = _fetch_video_description(video_id) if video_id else ""
        hotwords = _generate_hotwords(title, description)
    else:
        hotwords = user_hotwords

    # Step 3: Transcribe (cleanup=true removes the temp download after)
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
        "cleanup": True,
    }
    if args.get("batch_size"):
        params["batch_size"] = args["batch_size"]

    result = _request("POST", "/api/transcribe", params, timeout=POLL_TIMEOUT)

    if "error" in result:
        return f"Download succeeded but transcription failed: {result['error']}"

    transcript = result.get("transcript", "")
    status = result.get("status", "")
    duration_str = f"{duration // 60:.0f}m {duration % 60:.0f}s" if duration >= 60 else f"{duration}s"

    header = (
        f"Video: {title}\n"
        f"Duration: {duration_str}\n"
        f"Transcription: {status}\n"
        f"\n--- TRANSCRIPT ---\n\n"
    )

    return header + transcript


def tool_yt_transcribe_playlist(args):
    """Download and transcribe all videos in a YouTube playlist."""
    url = args.get("url", "").strip()
    if not url:
        return "Error: 'url' parameter is required"

    err = _check_service()
    if err:
        return err

    # Step 1: Download all items
    log(f"yt_transcribe_playlist: downloading {url}")
    dl_result = _request("POST", "/api/yt-download", {"url": url, "playlist": True}, timeout=3600)

    if "error" in dl_result:
        return f"Download failed: {dl_result['error']}"

    # Handle single video (not actually a playlist)
    if "items" not in dl_result:
        items = [dl_result]
    else:
        items = dl_result.get("items", [])

    if not items:
        return "No videos found in playlist"

    log(f"yt_transcribe_playlist: {len(items)} items to transcribe")

    # Step 2: Transcribe each item
    all_transcripts = []
    for i, item in enumerate(items):
        filename = item.get("filename", "")
        title = item.get("title", "unknown")
        duration = item.get("duration", 0)

        if not filename:
            all_transcripts.append(f"\n--- [{i+1}/{len(items)}] {title} ---\nError: no file\n")
            continue

        log(f"  [{i+1}/{len(items)}] Transcribing: {title}")
        # Generate hotwords per video from title
        item_hotwords = _generate_hotwords(title, "")
        params = {
            "file_path": filename,
            "model": args.get("model", "turbo"),
            "language": args.get("language", "Auto-detect"),
            "format": "txt",
            "diarize": args.get("diarize", False),
            "hotwords": item_hotwords,
            "cleanup": True,
        }

        result = _request("POST", "/api/transcribe", params, timeout=POLL_TIMEOUT)
        duration_str = f"{duration // 60:.0f}m {duration % 60:.0f}s" if duration >= 60 else f"{duration}s"

        if "error" in result:
            all_transcripts.append(f"\n--- [{i+1}/{len(items)}] {title} ({duration_str}) ---\nTranscription failed: {result['error']}\n")
        else:
            transcript = result.get("transcript", "")
            all_transcripts.append(f"\n--- [{i+1}/{len(items)}] {title} ({duration_str}) ---\n{transcript}\n")

    header = f"Playlist: {len(items)} videos transcribed\n"
    return header + "\n".join(all_transcripts)


# ── MCP protocol ─────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "whisper_status",
        "description": "Check the whisper transcription service status, GPU info, and availability.",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "yt_download",
        "description": "Download audio from a YouTube URL. Saves to /media/yt-dlp/ on the whisper server. Use whisper_transcribe afterward to transcribe it.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "YouTube video URL"
                }
            },
            "required": ["url"]
        }
    },
    {
        "name": "whisper_transcribe",
        "description": "Transcribe a local audio/video file on the whisper server. Returns the full transcript text. The file must exist on the whisper server's filesystem (e.g. /media/ mount or a file from yt_download).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the audio/video file on the whisper server"
                },
                "model": {
                    "type": "string",
                    "description": "Whisper model (tiny, base, small, medium, large, turbo). Default: turbo",
                    "enum": ["tiny", "base", "small", "medium", "large", "turbo"]
                },
                "language": {
                    "type": "string",
                    "description": "Language code (e.g. 'en', 'fr') or 'Auto-detect'. Default: Auto-detect"
                },
                "diarize": {
                    "type": "boolean",
                    "description": "Enable speaker diarization. Default: false"
                },
                "hotwords": {
                    "type": "string",
                    "description": "Comma-separated words the model might mishear (proper nouns, jargon)"
                },
                "initial_prompt": {
                    "type": "string",
                    "description": "Context hint for the first transcription window"
                }
            },
            "required": ["file_path"]
        }
    },
    {
        "name": "yt_transcribe",
        "description": "Download a YouTube video and transcribe it in one step. Returns the full transcript text ready for summarization. Best for: 'transcribe this YouTube video and summarize it'.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "YouTube video URL"
                },
                "model": {
                    "type": "string",
                    "description": "Whisper model. Default: turbo",
                    "enum": ["tiny", "base", "small", "medium", "large", "turbo"]
                },
                "language": {
                    "type": "string",
                    "description": "Language code or 'Auto-detect'. Default: Auto-detect"
                },
                "diarize": {
                    "type": "boolean",
                    "description": "Enable speaker diarization. Default: false"
                },
                "hotwords": {
                    "type": "string",
                    "description": "Words the model might mishear"
                },
                "initial_prompt": {
                    "type": "string",
                    "description": "Context hint for transcription"
                }
            },
            "required": ["url"]
        }
    },
    {
        "name": "yt_transcribe_playlist",
        "description": "Download and transcribe all videos in a YouTube playlist. Returns combined transcripts with headers for each video. Use for multi-video summarization.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "YouTube playlist URL"
                },
                "model": {
                    "type": "string",
                    "description": "Whisper model. Default: turbo",
                    "enum": ["tiny", "base", "small", "medium", "large", "turbo"]
                },
                "language": {
                    "type": "string",
                    "description": "Language code or 'Auto-detect'. Default: Auto-detect"
                },
                "diarize": {
                    "type": "boolean",
                    "description": "Enable speaker diarization. Default: false"
                }
            },
            "required": ["url"]
        }
    }
]

TOOL_HANDLERS = {
    "whisper_status": tool_status,
    "yt_download": tool_yt_download,
    "whisper_transcribe": tool_transcribe,
    "yt_transcribe": tool_yt_transcribe,
    "yt_transcribe_playlist": tool_yt_transcribe_playlist,
}

SERVER_INFO = {
    "name": "whisper",
    "version": "1.0.0",
}

CAPABILITIES = {
    "tools": {}
}


def handle_request(req):
    """Process a JSON-RPC request and return a response."""
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
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": TOOLS}
        }

    if method == "tools/call":
        tool_name = params.get("name", "")
        tool_args = params.get("arguments", {})
        handler = TOOL_HANDLERS.get(tool_name)

        if not handler:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
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
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "content": [{"type": "text", "text": result_text}],
                "isError": False
            }
        }

    if method == "ping":
        return {"jsonrpc": "2.0", "id": req_id, "result": {}}

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {
            "code": -32601,
            "message": f"Method not found: {method}"
        }
    }


def main():
    """Run the MCP server on stdio."""
    log("Whisper MCP server starting")
    log(f"Whisper URL: {WHISPER_URL}")

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
    """Log to stderr (stdout is reserved for JSON-RPC)."""
    print(f"[whisper-mcp] {msg}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
