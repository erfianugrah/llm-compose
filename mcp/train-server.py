#!/usr/bin/env python3
"""
LoRA training MCP server for OpenCode.

Exposes the lora-train HTTP API as MCP tools so the LLM can start,
monitor, and cancel LoRA training jobs from within OpenCode.
Communicates over stdio using the MCP JSON-RPC protocol.
Zero external dependencies — stdlib only.

Tools:
  train_start    — Start a LoRA training job
  train_status   — Check current training progress
  train_logs     — Get recent training log lines
  train_cancel   — Cancel current training job
  train_list     — List trained LoRA files
  train_datasets — List available datasets
  train_deploy   — Copy a trained LoRA to ComfyUI's loras dir

The server talks to the llm-compose proxy at TRAIN_PROXY_URL
(default: http://localhost:11434). The proxy handles GPU mode
switching automatically — when this server hits /train/*, the
proxy stops llama-server/ComfyUI and starts the lora-train service.

IMPORTANT: This means the LLM that invoked the tool will lose its
backend during training. Training is long-running (10-60+ min), so
the LLM will only be available again after training completes and
the proxy swaps back to LLM mode on the next /v1/ request.
"""

import json
import sys
import urllib.request
import urllib.error
import time
import os
import shutil

PROXY_URL = os.environ.get("TRAIN_PROXY_URL", "http://localhost:11434")
LORAS_DIR = os.environ.get("LORAS_DIR",
                           os.path.expanduser("~/docker-volumes/comfyui/models/loras"))
OUTPUT_DIR = os.environ.get("TRAIN_OUTPUT_DIR",
                            os.path.expanduser("~/docker-volumes/training-data/output"))
POLL_INTERVAL = 10  # seconds between status polls during training
POLL_TIMEOUT = 7200  # max 2 hours for training


def _request(method, path, data=None, timeout=30):
    url = f"{PROXY_URL}{path}"
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(
        url, data=body, method=method,
        headers={"Content-Type": "application/json", "User-Agent": "train-mcp/1.0"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Connection failed: {e.reason}") from e


# ── Tool implementations ─────────────────────────────────────────────
def tool_start(params):
    """Start a LoRA training job."""
    if not params.get("dataset_config"):
        return "Error: 'dataset_config' is required. Use train_datasets to list available datasets, then provide the TOML config path (e.g. /data/configs/my-dataset.toml)"
    config = {
        "dataset_config": params["dataset_config"],
        "base_model": params.get("base_model", "JuggernautXL_v9.safetensors"),
        "output_name": params.get("output_name", "lora-output"),
        "epochs": params.get("epochs", 4),
        "network_dim": params.get("network_dim", 32),
        "network_alpha": params.get("network_alpha", 16),
        "learning_rate": params.get("learning_rate", "1e-4"),
        "unet_lr": params.get("unet_lr", params.get("learning_rate", "1e-4")),
        "text_encoder_lr": params.get("text_encoder_lr", "5e-5"),
        "save_every_n_epochs": params.get("save_every_n_epochs", 1),
        "gradient_checkpointing": params.get("gradient_checkpointing", True),
    }

    result = _request("POST", "/train/train", config)
    status = result.get("status", "unknown")

    if status == "started":
        return (
            f"Training started: {config['output_name']}\n"
            f"Base model: {config['base_model']}\n"
            f"Dataset config: {config['dataset_config']}\n"
            f"Epochs: {config['epochs']}, dim={config['network_dim']}, alpha={config['network_alpha']}\n"
            f"WARNING: GPU is now in training mode. LLM backend is stopped.\n"
            f"Use train_status to monitor progress. Training will take 10-60+ minutes.\n"
            f"The LLM will restart automatically when you send your next chat message after training."
        )
    return f"Failed to start training: {json.dumps(result)}"


def tool_status(params):
    """Check current training progress."""
    parts = []

    # Check proxy mode first
    try:
        mode = _request("GET", "/mode")
        current = mode.get("mode") or "idle"
        parts.append(f"GPU mode: {current}")
        if mode.get("switching"):
            parts.append("Status: switching modes...")
            return "\n".join(parts)
    except RuntimeError as e:
        return f"Proxy: {e}"

    if current != "train":
        parts.append("Training service is not active.")
        return "\n".join(parts)

    # Get training status
    try:
        status = _request("GET", "/train/status")
        state = status.get("state", "unknown")
        parts.append(f"State: {state}")

        if state in ("training", "starting"):
            step = status.get("step", 0)
            total = status.get("total_steps", 0)
            epoch = status.get("epoch", 0)
            total_epochs = status.get("total_epochs", 0)
            loss = status.get("loss", 0)
            elapsed = status.get("elapsed_seconds", 0)
            eta = status.get("eta_seconds")

            if total > 0:
                pct = round(step / total * 100, 1)
                parts.append(f"Progress: {step}/{total} steps ({pct}%)")
            if total_epochs > 0:
                parts.append(f"Epoch: {epoch}/{total_epochs}")
            if loss > 0:
                parts.append(f"Loss: {loss:.6f}")
            parts.append(f"Elapsed: {elapsed}s")
            if eta:
                parts.append(f"ETA: {eta}s ({round(eta/60, 1)} min)")
        elif state == "completed":
            parts.append(f"Output: {status.get('output_name', 'unknown')}")
            parts.append(f"Elapsed: {status.get('elapsed_seconds', 0)}s")
        elif state == "failed":
            parts.append(f"Error: {status.get('error', 'unknown')}")

    except RuntimeError as e:
        parts.append(f"Training API: {e}")

    return "\n".join(parts)


def tool_logs(params):
    """Get recent training log lines."""
    lines = params.get("lines", 50)
    try:
        result = _request("GET", f"/train/logs?lines={lines}")
        state = result.get("state", "idle")
        log_lines = result.get("lines", [])
        if not log_lines:
            return f"State: {state}\nNo log output yet."
        header = f"State: {state}\nLast {len(log_lines)} lines:\n"
        return header + "\n".join(log_lines)
    except RuntimeError as e:
        return f"Failed to get logs: {e}"


def tool_cancel(params):
    """Cancel current training job."""
    try:
        result = _request("POST", "/train/cancel")
        return f"Cancel: {result.get('status', 'unknown')}"
    except RuntimeError as e:
        return f"Failed to cancel: {e}"


def tool_list(params):
    """List trained LoRA files."""
    try:
        result = _request("GET", "/train/jobs")
        files = result.get("files", [])
        if not files:
            return "No trained LoRA files found."
        lines = ["Trained LoRA files:"]
        for f in files:
            lines.append(f"  {f['name']} ({f['size_mb']} MB)")
        return "\n".join(lines)
    except RuntimeError as e:
        return f"Failed to list: {e}"


def tool_datasets(params):
    """List available training datasets."""
    try:
        result = _request("GET", "/train/datasets")
        datasets = result.get("datasets", [])
        if not datasets:
            return "No datasets found in /data/datasets/"
        lines = ["Available datasets:"]
        for d in datasets:
            lines.append(f"  {d['name']}: {d['images']} images, {d['captions']} captions")
        return "\n".join(lines)
    except RuntimeError as e:
        return f"Failed to list datasets: {e}"


def tool_deploy(params):
    """Copy a trained LoRA to ComfyUI's loras directory."""
    name = params.get("name", "")
    if not name:
        return "Error: 'name' parameter required (filename of the LoRA in output/)"

    if not name.endswith(".safetensors"):
        name += ".safetensors"

    src = os.path.join(OUTPUT_DIR, name)
    if not os.path.exists(src):
        available = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".safetensors")]
        return f"Not found: {src}\nAvailable: {', '.join(available) or 'none'}"

    dst = os.path.join(LORAS_DIR, name)
    try:
        shutil.copy2(src, dst)
        size_mb = round(os.path.getsize(dst) / 1024 / 1024, 1)
        return f"Deployed {name} ({size_mb} MB) to {LORAS_DIR}"
    except Exception as e:
        return f"Failed to deploy: {e}"


# ── MCP Protocol ─────────────────────────────────────────────────────
TOOLS = [
    {
        "name": "train_start",
        "description": (
            "Start a LoRA fine-tuning job. Triggers GPU mode swap — stops LLM/ComfyUI, "
            "starts training service. Training runs 10-60+ minutes. The LLM backend will "
            "be unavailable during training and restarts automatically on next chat request. "
            "Default: JuggernautXL base, batch_size=16 (max VRAM), "
            "dim=32, 4 epochs. Override any parameter via arguments."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "output_name": {
                    "type": "string",
                    "description": "Name for the output LoRA file (without .safetensors extension)"
                },
                "dataset_config": {
                    "type": "string",
                    "description": "Path to dataset TOML config inside the container. e.g. /data/configs/my-dataset.toml"
                },
                "base_model": {
                    "type": "string",
                    "description": "Base checkpoint filename. Must exist in checkpoints dir. Default: JuggernautXL_v9.safetensors"
                },
                "epochs": {
                    "type": "integer",
                    "description": "Number of training epochs. Default: 4"
                },
                "network_dim": {
                    "type": "integer",
                    "description": "LoRA rank/dimension. Higher = more capacity but larger file. Default: 32"
                },
                "network_alpha": {
                    "type": "integer",
                    "description": "LoRA alpha scaling factor. Default: 16"
                },
                "learning_rate": {
                    "type": "string",
                    "description": "UNet learning rate. Default: 1e-4"
                },
                "text_encoder_lr": {
                    "type": "string",
                    "description": "Text encoder learning rate. Default: 5e-5"
                },
                "save_every_n_epochs": {
                    "type": "integer",
                    "description": "Save checkpoint every N epochs. Default: 1"
                },
                "gradient_checkpointing": {
                    "type": "boolean",
                    "description": "Enable gradient checkpointing (slower but uses less VRAM). Default: true"
                }
            }
        }
    },
    {
        "name": "train_status",
        "description": (
            "Check current LoRA training progress. Returns state (idle/training/completed/failed), "
            "step count, loss, epoch, elapsed time, and ETA."
        ),
        "inputSchema": {"type": "object", "properties": {}}
    },
    {
        "name": "train_logs",
        "description": "Get recent training log output lines.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "lines": {
                    "type": "integer",
                    "description": "Number of log lines to return. Default: 50"
                }
            }
        }
    },
    {
        "name": "train_cancel",
        "description": "Cancel the current training job. GPU remains in train mode until next request triggers swap.",
        "inputSchema": {"type": "object", "properties": {}}
    },
    {
        "name": "train_list",
        "description": "List all trained LoRA files in the output directory with sizes.",
        "inputSchema": {"type": "object", "properties": {}}
    },
    {
        "name": "train_datasets",
        "description": "List available training datasets (image + caption sets).",
        "inputSchema": {"type": "object", "properties": {}}
    },
    {
        "name": "train_deploy",
        "description": "Copy a trained LoRA from the output directory to ComfyUI's loras directory so it can be used in generation workflows.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "LoRA filename (with or without .safetensors extension)"
                }
            },
            "required": ["name"]
        }
    }
]

TOOL_HANDLERS = {
    "train_start": tool_start,
    "train_status": tool_status,
    "train_logs": tool_logs,
    "train_cancel": tool_cancel,
    "train_list": tool_list,
    "train_datasets": tool_datasets,
    "train_deploy": tool_deploy,
}

SERVER_INFO = {"name": "lora-train", "version": "1.0.0"}
CAPABILITIES = {"tools": {}}


def handle_request(req):
    method = req.get("method", "")
    req_id = req.get("id")
    params = req.get("params", {})

    if method == "initialize":
        return {
            "jsonrpc": "2.0", "id": req_id,
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
    log("LoRA training MCP server starting")
    log(f"Proxy URL: {PROXY_URL}")
    log(f"Output dir: {OUTPUT_DIR}")
    log(f"LoRAs dir: {LORAS_DIR}")

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
    print(f"[train-mcp] {msg}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
