#!/usr/bin/env python3
"""
ComfyUI MCP server for OpenCode.

Exposes ComfyUI's workflow API as MCP tools so the LLM can generate
images/videos from within OpenCode. Communicates over stdio using the
MCP JSON-RPC protocol. Zero external dependencies — stdlib only.

Tools:
  comfyui_generate  — Submit a txt2img workflow and wait for output
  comfyui_status    — Check current GPU mode and ComfyUI queue
  comfyui_history   — Get results of a previous generation

The server talks to the llm-compose proxy at COMFYUI_PROXY_URL
(default: http://localhost:11434). The proxy handles GPU mode
switching automatically — when this server hits /comfyui/*, the
proxy stops llama-server and starts ComfyUI.

IMPORTANT: This means the LLM that invoked the tool will lose its
backend during generation. OpenCode handles this gracefully because
tool calls are synchronous — the model waits for the result, and
by the time it needs to respond, the proxy will swap back to LLM
mode on the next /v1/ request.
"""

import json
import sys
import urllib.request
import urllib.error
import urllib.parse
import time
import os
import copy
from pathlib import Path

PROXY_URL = os.environ.get("COMFYUI_PROXY_URL", "http://localhost:11434")
OUTPUT_DIR = os.environ.get("COMFYUI_OUTPUT_DIR",
                            os.path.expanduser("~/docker-volumes/comfyui/output"))
POLL_INTERVAL = 2  # seconds between status polls
POLL_TIMEOUT = 300  # max seconds to wait for generation


# ── Local config overlay ─────────────────────────────────────────────
# comfyui.local.env overrides defaults (checkpoint, prompts, sampler
# settings). Gitignored — keeps model-specific config out of the repo.
def _load_local_config():
    """Load KEY=VALUE pairs from comfyui.local.env if it exists."""
    config = {}
    for search in [
        Path(__file__).resolve().parent.parent / "comfyui.local.env",
        Path.cwd() / "comfyui.local.env",
    ]:
        if search.exists():
            for line in search.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, _, value = line.partition("=")
                    config[key.strip()] = value.strip()
            break
    return config

_LOCAL = _load_local_config()

# Defaults — overridden by comfyui.local.env if present
DEFAULT_CHECKPOINT = _LOCAL.get("CHECKPOINT", "sd_xl_base_1.0.safetensors")
DEFAULT_WIDTH = int(_LOCAL.get("WIDTH", "1024"))
DEFAULT_HEIGHT = int(_LOCAL.get("HEIGHT", "1024"))
DEFAULT_STEPS = int(_LOCAL.get("STEPS", "20"))
DEFAULT_CFG = float(_LOCAL.get("CFG", "7.0"))
DEFAULT_SAMPLER = _LOCAL.get("SAMPLER", "euler")
DEFAULT_SCHEDULER = _LOCAL.get("SCHEDULER", "normal")
DEFAULT_POSITIVE_PREFIX = _LOCAL.get("POSITIVE_PREFIX", "masterpiece, best quality")
DEFAULT_NEGATIVE = _LOCAL.get("NEGATIVE",
    "lowres, bad anatomy, bad hands, missing fingers, extra digits, fewer digits, "
    "cropped, worst quality, low quality, jpeg artifacts, signature, watermark, blurry")


# ── Default workflow ─────────────────────────────────────────────────
# Minimal txt2img workflow in ComfyUI API format.
# Node IDs are arbitrary strings; this uses simple numbers.
# The model will override prompt text, seed, dimensions, and steps.
DEFAULT_WORKFLOW = {
    "4": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {
            "ckpt_name": DEFAULT_CHECKPOINT
        }
    },
    "5": {
        "class_type": "EmptyLatentImage",
        "inputs": {
            "width": DEFAULT_WIDTH,
            "height": DEFAULT_HEIGHT,
            "batch_size": 1
        }
    },
    "6": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": DEFAULT_POSITIVE_PREFIX,
            "clip": ["4", 1]
        }
    },
    "7": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": DEFAULT_NEGATIVE,
            "clip": ["4", 1]
        }
    },
    "3": {
        "class_type": "KSampler",
        "inputs": {
            "seed": 42,
            "steps": DEFAULT_STEPS,
            "cfg": DEFAULT_CFG,
            "sampler_name": DEFAULT_SAMPLER,
            "scheduler": DEFAULT_SCHEDULER,
            "denoise": 1.0,
            "model": ["4", 0],
            "positive": ["6", 0],
            "negative": ["7", 0],
            "latent_image": ["5", 0]
        }
    },
    "8": {
        "class_type": "VAEDecode",
        "inputs": {
            "samples": ["3", 0],
            "vae": ["4", 2]
        }
    },
    "9": {
        "class_type": "SaveImage",
        "inputs": {
            "filename_prefix": "opencode",
            "images": ["8", 0]
        }
    }
}


# ── HTTP helpers ─────────────────────────────────────────────────────
def _request(method, path, data=None, timeout=30):
    """Make an HTTP request to the proxy. Returns parsed JSON or raises."""
    url = f"{PROXY_URL}{path}"
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(
        url, data=body, method=method,
        headers={"Content-Type": "application/json", "User-Agent": "comfyui-mcp/1.0"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Connection failed: {e.reason}") from e


def _poll_completion(prompt_id):
    """Poll /comfyui/history/{prompt_id} until generation completes."""
    deadline = time.monotonic() + POLL_TIMEOUT
    while time.monotonic() < deadline:
        try:
            history = _request("GET", f"/comfyui/history/{prompt_id}")
            if prompt_id in history:
                return history[prompt_id]
        except RuntimeError:
            pass
        time.sleep(POLL_INTERVAL)
    raise RuntimeError(f"Generation timed out after {POLL_TIMEOUT}s")


# ── Tool implementations ─────────────────────────────────────────────
def tool_generate(params):
    """Submit a txt2img workflow to ComfyUI and wait for output."""
    prompt_text = params.get("prompt", DEFAULT_POSITIVE_PREFIX)
    negative = params.get("negative_prompt", DEFAULT_NEGATIVE)
    width = params.get("width", DEFAULT_WIDTH)
    height = params.get("height", DEFAULT_HEIGHT)
    steps = params.get("steps", DEFAULT_STEPS)
    cfg = params.get("cfg", DEFAULT_CFG)
    seed = params.get("seed")
    checkpoint = params.get("checkpoint")
    workflow_json = params.get("workflow")

    if workflow_json:
        # User provided a full workflow JSON — use as-is
        if isinstance(workflow_json, str):
            workflow = json.loads(workflow_json)
        else:
            workflow = workflow_json
    else:
        # Build from default template
        workflow = copy.deepcopy(DEFAULT_WORKFLOW)
        workflow["6"]["inputs"]["text"] = prompt_text
        workflow["7"]["inputs"]["text"] = negative
        workflow["5"]["inputs"]["width"] = width
        workflow["5"]["inputs"]["height"] = height
        workflow["3"]["inputs"]["steps"] = steps
        workflow["3"]["inputs"]["cfg"] = cfg
        if seed is not None:
            workflow["3"]["inputs"]["seed"] = seed
        else:
            workflow["3"]["inputs"]["seed"] = int(time.time()) % (2**32)
        if checkpoint:
            workflow["4"]["inputs"]["ckpt_name"] = checkpoint

    # Submit to ComfyUI via proxy (triggers mode swap if needed)
    result = _request("POST", "/comfyui/prompt", {"prompt": workflow})
    prompt_id = result.get("prompt_id")
    if not prompt_id:
        return f"Failed to queue prompt: {json.dumps(result)}"

    # Poll for completion
    try:
        history = _poll_completion(prompt_id)
    except RuntimeError as e:
        return f"Generation failed: {e}"

    # Extract output files and download them to CWD
    outputs = history.get("outputs", {})
    files = []
    for node_id, node_output in outputs.items():
        for img in node_output.get("images", []):
            filename = img.get("filename", "")
            subfolder = img.get("subfolder", "")
            img_type = img.get("type", "output")
            if not filename:
                continue

            # Download via ComfyUI's /view endpoint to the caller's working dir
            local_path = os.path.join(os.getcwd(), filename)
            try:
                view_params = urllib.parse.urlencode({
                    "filename": filename,
                    "subfolder": subfolder,
                    "type": img_type,
                })
                view_url = f"{PROXY_URL}/comfyui/view?{view_params}"
                req = urllib.request.Request(view_url, headers={"User-Agent": "comfyui-mcp/1.0"})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    with open(local_path, "wb") as f:
                        while True:
                            chunk = resp.read(1 << 20)
                            if not chunk:
                                break
                            f.write(chunk)
                files.append(local_path)
            except Exception as exc:
                # Fallback: report the volume path
                vol_path = os.path.join(OUTPUT_DIR, subfolder, filename) if subfolder else os.path.join(OUTPUT_DIR, filename)
                files.append(f"{vol_path} (download failed: {exc})")

    if files:
        file_list = "\n".join(f"  - {f}" for f in files)
        return f"Generated {len(files)} image(s):\n{file_list}"
    return f"Generation completed but no output files found. Prompt ID: {prompt_id}"


def tool_status(params):
    """Check current GPU mode and ComfyUI queue status."""
    parts = []

    # Get proxy mode
    try:
        mode = _request("GET", "/mode")
        parts.append(f"GPU mode: {mode.get('mode') or 'idle'}")
        if mode.get("switching"):
            parts.append("Status: switching modes...")
    except RuntimeError as e:
        parts.append(f"Proxy: {e}")

    # If ComfyUI is active, get queue info
    try:
        queue = _request("GET", "/comfyui/prompt")
        running = queue.get("exec_info", {}).get("queue_remaining", 0)
        parts.append(f"ComfyUI queue: {running} remaining")
    except RuntimeError:
        parts.append("ComfyUI: not currently active")

    return "\n".join(parts)


def tool_history(params):
    """Get results from a previous generation by prompt ID."""
    prompt_id = params.get("prompt_id", "")
    if not prompt_id:
        # Return recent history
        try:
            history = _request("GET", "/comfyui/history")
            if not history:
                return "No generation history available."
            recent = list(history.keys())[-5:]
            lines = [f"Recent generations (last {len(recent)}):"]
            for pid in recent:
                entry = history[pid]
                status = entry.get("status", {})
                completed = status.get("completed", False)
                lines.append(f"  {pid}: {'completed' if completed else 'pending'}")
            return "\n".join(lines)
        except RuntimeError as e:
            return f"Failed to fetch history: {e}"

    try:
        history = _request("GET", f"/comfyui/history/{prompt_id}")
        if prompt_id not in history:
            return f"No results found for prompt ID: {prompt_id}"
        entry = history[prompt_id]
        outputs = entry.get("outputs", {})
        files = []
        for node_output in outputs.values():
            for img in node_output.get("images", []):
                filename = img.get("filename", "")
                subfolder = img.get("subfolder", "")
                path = os.path.join(OUTPUT_DIR, subfolder, filename) if subfolder else os.path.join(OUTPUT_DIR, filename)
                files.append(path)
        if files:
            return f"Output files:\n" + "\n".join(f"  - {f}" for f in files)
        return f"Generation completed but no output files. Raw: {json.dumps(outputs)[:500]}"
    except RuntimeError as e:
        return f"Failed to fetch history: {e}"


# ── MCP Protocol (JSON-RPC over stdio) ───────────────────────────────
TOOLS = [
    {
        "name": "comfyui_generate",
        "description": (
            "Generate an image using ComfyUI. Submits a Stable Diffusion workflow "
            "and waits for completion. Returns file paths of generated images. "
            "Default: portrait ratio (832x1216), 30 steps, Euler sampler, CFG 5. "
            "Prefix prompts with quality tags like 'masterpiece, best quality'. "
            "IMPORTANT: Do NOT call the Read tool on the returned PNG paths. The user "
            "sees the generated images directly in their viewer; reading them inlines "
            "base64 image data into the conversation and quickly exceeds the model's "
            "input limit ('Input is too long' errors). Only Read an output image if "
            "the user explicitly asks you to visually inspect or analyze it. "
            "WARNING: Triggers GPU mode swap — llama-server stops during generation "
            "(~20-60s), restarts automatically on next chat request."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Text description of the image to generate"
                },
                "negative_prompt": {
                    "type": "string",
                    "description": "Things to avoid in the image. Defaults from comfyui.local.env or standard quality negatives."
                },
                "width": {
                    "type": "integer",
                    "description": "Image width in pixels. Good SDXL ratios: 832x1216, 1024x1024, 1216x832, 768x1344"
                },
                "height": {
                    "type": "integer",
                    "description": "Image height in pixels."
                },
                "steps": {
                    "type": "integer",
                    "description": "Number of sampling steps. More = higher quality but slower."
                },
                "cfg": {
                    "type": "number",
                    "description": "Classifier-free guidance scale."
                },
                "seed": {
                    "type": "integer",
                    "description": "Random seed for reproducibility (omit for random)"
                },
                "checkpoint": {
                    "type": "string",
                    "description": "Checkpoint model filename. Must exist in ~/docker-volumes/comfyui/models/checkpoints/. Default from comfyui.local.env or sd_xl_base_1.0.safetensors."
                },
                "workflow": {
                    "type": ["object", "string"],
                    "description": "Full ComfyUI API-format workflow JSON. Overrides all other parameters if provided."
                }
            },
            "required": ["prompt"]
        }
    },
    {
        "name": "comfyui_status",
        "description": "Check the current GPU mode (llm/comfyui/idle) and ComfyUI queue status.",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "comfyui_history",
        "description": "Get results from a previous ComfyUI generation. Call without prompt_id to list recent generations.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt_id": {
                    "type": "string",
                    "description": "The prompt ID returned by a previous generation. Omit to list recent history."
                }
            }
        }
    }
]

TOOL_HANDLERS = {
    "comfyui_generate": tool_generate,
    "comfyui_status": tool_status,
    "comfyui_history": tool_history,
}

SERVER_INFO = {
    "name": "comfyui",
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
        # Client acknowledgment — no response needed
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

    # Unknown method
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
    log("ComfyUI MCP server starting")
    log(f"Proxy URL: {PROXY_URL}")
    log(f"Output dir: {OUTPUT_DIR}")

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
    print(f"[comfyui-mcp] {msg}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
