#!/usr/bin/env python3
"""
LoRA training HTTP API server for llm-compose.

Runs inside the lora-train container, wraps kohya sd-scripts with a
simple REST API so the proxy (and MCP tools) can start/monitor/cancel
training jobs without manual docker commands.

Endpoints:
  POST /train     — Start a training job (JSON config)
  GET  /status    — Current job state + progress + loss
  GET  /logs      — Last N lines of training output
  POST /cancel    — Kill current training job
  GET  /jobs      — List completed LoRA files
  GET  /datasets  — List available datasets
  GET  /configs   — List available training configs
  GET  /health    — Health check

All paths are served relative to the container's /data volume:
  /data/datasets/    — curated image+caption sets
  /data/configs/     — TOML training configs
  /data/output/      — trained LoRA checkpoints
  /data/raw/         — raw scraped data (not used by server directly)

Zero external dependencies — stdlib only.
"""

import http.server
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

PORT = int(os.environ.get("TRAIN_PORT", "8787"))
DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
DATASETS_DIR = DATA_DIR / "datasets"
CONFIGS_DIR = DATA_DIR / "configs"
OUTPUT_DIR = DATA_DIR / "output"
CHECKPOINTS_DIR = Path(os.environ.get("CHECKPOINTS_DIR", "/checkpoints"))

# ── Job state ────────────────────────────────────────────────────────
job_lock = threading.Lock()
current_job = None      # active TrainingJob or None
caption_job_lock = threading.Lock()
current_caption = None  # active CaptionJob or None


class CaptionJob:
    """Manages a single captioning run (WD14 / BLIP-2 / Florence-2).

    Matches the TrainingJob pattern so captioning is non-blocking:
    POST /caption returns 202 immediately, client polls /caption/status
    and /caption/logs. This avoids hanging the HTTP thread (which in turn
    breaks healthchecks and triggers proxy restart cascades).
    """

    def __init__(self, cmd, engine, dataset, trigger_word):
        self.cmd = cmd
        self.engine = engine
        self.dataset = dataset
        self.trigger_word = trigger_word
        self.process = None
        self.state = "starting"  # starting, running, completed, failed, cancelled
        self.start_time = time.monotonic()
        self.end_time = None
        self.log_lines = []
        self.log_lock = threading.Lock()
        self.error = None
        self.captions_written = 0
        self.images_total = 0

    def to_dict(self):
        elapsed = (self.end_time or time.monotonic()) - self.start_time
        return {
            "state": self.state,
            "engine": self.engine,
            "dataset": self.dataset,
            "trigger_word": self.trigger_word,
            "captions_written": self.captions_written,
            "images_total": self.images_total,
            "elapsed_seconds": round(elapsed),
            "error": self.error,
        }

    def append_log(self, line):
        with self.log_lock:
            self.log_lines.append(line)
            if len(self.log_lines) > 2000:
                self.log_lines = self.log_lines[-1000:]
        # Parse progress: "Found N images in ..." and "[N/M] filename: ..."
        m = re.search(r"Found (\d+) images", line)
        if m:
            self.images_total = int(m.group(1))
        m = re.search(r"\[(\d+)/(\d+)\]", line)
        if m:
            self.captions_written = int(m.group(1))
            if self.state == "starting":
                self.state = "running"

    def get_logs(self, n=100):
        with self.log_lock:
            return self.log_lines[-n:]

    def cancel(self):
        if self.process and self.process.poll() is None:
            _killpg(self.process)
            self.state = "cancelled"
            self.end_time = time.monotonic()


def _killpg(proc, grace: int = 10):
    """Kill the entire process group (proc + all its descendants).

    Without this, accelerate/torch-inductor fork helper workers that
    survive a simple proc.terminate() on the parent and linger as
    orphans holding PID slots + compile cache + file handles until
    the container restarts.

    Requires the subprocess to have been started with
    `preexec_fn=os.setsid` (or `start_new_session=True`) so it's the
    leader of its own process group.
    """
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=grace)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def _run_caption(job):
    """Run the captioning subprocess in a background thread."""
    global current_caption

    try:
        job.append_log(f"[CMD] {' '.join(job.cmd)}")
        job.process = subprocess.Popen(
            job.cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            # New session so cancel can killpg the whole subprocess tree
            start_new_session=True,
        )
        job.state = "running"

        for line in job.process.stdout:
            job.append_log(line.rstrip("\n"))

        rc = job.process.wait()
        job.end_time = time.monotonic()

        if job.state == "cancelled":
            pass
        elif rc == 0:
            job.state = "completed"
            job.append_log("[DONE] Captioning completed successfully")
        else:
            job.state = "failed"
            job.error = f"Process exited with code {rc}"
            job.append_log(f"[ERROR] Captioning failed with exit code {rc}")

    except Exception as e:
        job.state = "failed"
        job.error = str(e)
        job.end_time = time.monotonic()
        job.append_log(f"[ERROR] {e}")


class TrainingJob:
    """Manages a single training run."""

    def __init__(self, config):
        self.config = config
        self.process = None
        self.state = "starting"  # starting, training, completed, failed, cancelled
        self.step = 0
        self.total_steps = 0
        self.epoch = 0
        self.total_epochs = 0
        self.loss = 0.0
        self.start_time = time.monotonic()
        self.end_time = None
        self.output_name = config.get("output_name", "lora-output")
        self.log_lines = []
        self.log_lock = threading.Lock()
        self.error = None

    def to_dict(self):
        elapsed = (self.end_time or time.monotonic()) - self.start_time
        eta = None
        if self.step > 0 and self.total_steps > 0 and self.state == "training":
            rate = elapsed / self.step
            remaining = (self.total_steps - self.step) * rate
            eta = round(remaining)

        return {
            "state": self.state,
            "step": self.step,
            "total_steps": self.total_steps,
            "epoch": self.epoch,
            "total_epochs": self.total_epochs,
            "loss": round(self.loss, 6),
            "elapsed_seconds": round(elapsed),
            "eta_seconds": eta,
            "output_name": self.output_name,
            "config": self.config,
            "error": self.error,
        }

    def append_log(self, line):
        with self.log_lock:
            self.log_lines.append(line)
            # Keep last 2000 lines
            if len(self.log_lines) > 2000:
                self.log_lines = self.log_lines[-1000:]

    def get_logs(self, n=100):
        with self.log_lock:
            return self.log_lines[-n:]

    def parse_progress(self, line):
        """Parse sd-scripts output for step/loss/epoch progress."""
        # Progress bar: "steps:  42%|████      | 2000/4760 [05:30<07:35, 6.05it/s]"
        m = re.search(r'steps:\s+\d+%\|[^|]*\|\s*(\d+)/(\d+)', line)
        if m:
            self.step = int(m.group(1))
            self.total_steps = int(m.group(2))
            if self.state == "starting":
                self.state = "training"

        # Loss: "avr_loss=0.123456" or "loss=0.123456"
        m = re.search(r'(?:avr_)?loss[=:]\s*([\d.]+)', line)
        if m:
            self.loss = float(m.group(1))

        # Epoch: "epoch 2/4"
        m = re.search(r'epoch\s+(\d+)/(\d+)', line)
        if m:
            self.epoch = int(m.group(1))
            self.total_epochs = int(m.group(2))

        # Total steps from config: "total optimization steps / ...: 4760"
        m = re.search(r'total optimization steps.*?:\s*(\d+)', line)
        if m:
            self.total_steps = int(m.group(1))

        # Saving: "saving checkpoint"
        if "saving" in line.lower() and "checkpoint" in line.lower():
            self.append_log(f"[SAVE] {line.strip()}")

    def cancel(self):
        if self.process and self.process.poll() is None:
            _killpg(self.process)
            self.state = "cancelled"
            self.end_time = time.monotonic()


def _detect_model_type(config):
    """Auto-detect model_type from base_model filename if not set explicitly.

    Flux patterns: 'flux', 'flux1-dev', 'flux.1'
    SDXL patterns: illustrious, noobai, juggernaut, sd_xl, animagine, pony
    """
    explicit = config.get("model_type")
    if explicit:
        return explicit
    base = (config.get("base_model") or "").lower()
    if "flux" in base:
        return "flux"
    return "sdxl"


def _build_train_command(config):
    """Build the training command. Dispatches to SDXL or Flux based on model_type."""
    model_type = _detect_model_type(config)
    config["model_type"] = model_type  # persist for logging
    if model_type == "flux":
        return _build_flux_command(config)
    return _build_sdxl_command(config)


def _common_config(config):
    """Extract common config fields shared by SDXL and Flux."""
    dataset_config = config.get("dataset_config")
    if not dataset_config:
        raise ValueError("dataset_config is required")
    return {
        "dataset_config": dataset_config,
        "output_name": config.get("output_name", "lora-output"),
        "epochs": config.get("epochs", 4),
        "dim": config.get("network_dim", 32),
        "alpha": config.get("network_alpha", config.get("network_dim", 32)),
        "lr": config.get("learning_rate", "1e-4"),
        "optimizer": config.get("optimizer", "AdamW"),
        "scheduler": config.get("lr_scheduler", "cosine"),
        "warmup": config.get("warmup_steps", 100),
        "save_every": config.get("save_every_n_epochs", 1),
        "grad_ckpt": config.get("gradient_checkpointing", True),
    }


def _build_sdxl_command(config):
    """Build command for SDXL LoRA training (Illustrious / NoobAI / JuggernautXL).

    Key settings for face-likeness LoRAs:
    - UNet-only training — critical for SDXL face fidelity
    - Cached text encoder outputs — faster, less RAM
    - No noise offset — preserves face detail, cross-base compatible
    - alpha = dim — disables alpha scaling for maximum detail
    """
    c = _common_config(config)
    base_model = config.get("base_model", "Illustrious-XL-v0.1.safetensors")
    unet_lr = config.get("unet_lr", c["lr"])
    noise_offset = config.get("noise_offset", "0")
    min_snr_gamma = config.get("min_snr_gamma", 0)

    cmd = [
        "accelerate", "launch", "--mixed_precision=bf16",
        "/sd-scripts/sdxl_train_network.py",
        f"--pretrained_model_name_or_path=/models/checkpoints/{base_model}",
        f"--dataset_config={c['dataset_config']}",
        "--output_dir=/data/output",
        f"--output_name={c['output_name']}",
        "--network_module=networks.lora",
        f"--network_dim={c['dim']}",
        f"--network_alpha={c['alpha']}",
        f"--learning_rate={c['lr']}",
        f"--unet_lr={unet_lr}",
        f"--optimizer_type={c['optimizer']}",
        f"--lr_scheduler={c['scheduler']}",
        f"--lr_warmup_steps={c['warmup']}",
        f"--max_train_epochs={c['epochs']}",
        f"--save_every_n_epochs={c['save_every']}",
        "--mixed_precision=bf16",
        "--network_train_unet_only",
        "--cache_latents", "--cache_latents_to_disk",
        "--cache_text_encoder_outputs", "--cache_text_encoder_outputs_to_disk",
        "--max_data_loader_n_workers=0",
        "--sdpa",
        "--save_precision=bf16",
        "--logging_dir=/data/output/logs",
        f"--log_prefix={c['output_name']}",
    ]

    if c["grad_ckpt"]:
        cmd.append("--gradient_checkpointing")
    if float(noise_offset) > 0:
        cmd.append(f"--noise_offset={noise_offset}")
    if min_snr_gamma > 0:
        cmd.append(f"--min_snr_gamma={min_snr_gamma}")

    v_pred = config.get("v_parameterization", False)
    if v_pred:
        cmd.extend(["--v_parameterization", "--zero_terminal_snr",
                     "--scale_v_pred_loss_like_noise_pred"])

    return cmd


def _build_flux_command(config):
    """Build command for Flux LoRA training.

    Key settings:
    - Uses flux_train_network.py + networks.lora_flux
    - guidance_scale=1.0 (mandatory — Flux dev distilled at guidance=1)
    - timestep_sampling=sigmoid + discrete_flow_shift=3.1582
    - model_prediction_type=raw
    - Separate --ae, --clip_l, --t5xxl paths for Flux components
    - fp8_base optional for VRAM savings
    - AdamW8bit default — kohya's recommended optimizer for Flux
      (saves VRAM, similar quality to AdamW).
    """
    c = _common_config(config)
    # Flux prefers AdamW8bit over AdamW — override only if caller
    # didn't pick an optimizer explicitly.
    if "optimizer" not in config:
        c["optimizer"] = "AdamW8bit"
    base_model = config.get("base_model", "flux1-dev.safetensors")
    fp8_base = config.get("fp8_base", True)  # default True: 32GB VRAM needs fp8 for Flux 12B
    guidance_scale = config.get("guidance_scale", "1.0")

    cmd = [
        "accelerate", "launch", "--mixed_precision=bf16",
        "/sd-scripts/flux_train_network.py",
        f"--pretrained_model_name_or_path=/models/diffusion_models/{base_model}",
        "--clip_l=/models/text_encoders/clip_l.safetensors",
        "--t5xxl=/models/text_encoders/t5xxl_fp8_e4m3fn.safetensors",
        "--ae=/models/vae/ae.safetensors",
        f"--dataset_config={c['dataset_config']}",
        "--output_dir=/data/output",
        f"--output_name={c['output_name']}",
        "--network_module=networks.lora_flux",
        f"--network_dim={c['dim']}",
        f"--network_alpha={c['alpha']}",
        f"--learning_rate={c['lr']}",
        f"--optimizer_type={c['optimizer']}",
        f"--lr_scheduler={c['scheduler']}",
        f"--lr_warmup_steps={c['warmup']}",
        f"--max_train_epochs={c['epochs']}",
        f"--save_every_n_epochs={c['save_every']}",
        "--mixed_precision=bf16",
        "--network_train_unet_only",
        "--cache_latents", "--cache_latents_to_disk",
        "--cache_text_encoder_outputs", "--cache_text_encoder_outputs_to_disk",
        "--max_data_loader_n_workers=0",
        "--sdpa",
        "--save_precision=bf16",
        "--logging_dir=/data/output/logs",
        f"--log_prefix={c['output_name']}",
        f"--timestep_sampling=sigmoid",
        f"--discrete_flow_shift=3.1582",
        f"--model_prediction_type=raw",
        f"--guidance_scale={guidance_scale}",
    ]

    if c["grad_ckpt"]:
        cmd.append("--gradient_checkpointing")
    if fp8_base:
        cmd.append("--fp8_base")

    return cmd


def _run_training(job):
    """Run training in a thread, updating job state.

    Progress tracking uses two channels:
    1. stdout pipe — captures log lines (epoch transitions, config, errors)
    2. progress file — written by tqdm hook injected via .pth file at Python
       startup. The hook patches tqdm.update/set_postfix to write step/loss
       to a JSON file every 2s. This works because accelerate's subprocess
       inherits the .pth and TRAIN_PROGRESS_FILE env var.
    """
    global current_job

    try:
        cmd = _build_train_command(job.config)
        job.append_log(f"[CMD] {' '.join(cmd)}")

        progress_file = "/tmp/train_progress.json"
        try:
            os.unlink(progress_file)
        except OSError:
            pass

        # Add --console_log_simple to disable rich formatting in logs
        cmd.append("--console_log_simple")

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["TRAIN_PROGRESS_FILE"] = progress_file
        # Limit torch._inductor's compile worker pool so it doesn't fork
        # 16 helper processes that turn the dev machine into a slideshow.
        env.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "4")

        job.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
            # New session so cancel can killpg the whole accelerate tree
            # (accelerate launcher -> python script -> inductor workers)
            start_new_session=True,
        )

        # Reader thread for stdout (log lines — NOT tqdm progress)
        def read_output():
            for line in job.process.stdout:
                line = line.rstrip('\n')
                if line:
                    job.append_log(line)
                    job.parse_progress(line)

        reader = threading.Thread(target=read_output, daemon=True)
        reader.start()

        # Poll progress file written by tqdm hook
        while job.process.poll() is None:
            time.sleep(2)
            try:
                with open(progress_file, "r") as f:
                    data = json.loads(f.read())
                    step = data.get("step", 0)
                    if step > job.step or data.get("loss", 0) > 0:
                        job.step = step
                        job.total_steps = data.get("total", job.total_steps) or job.total_steps
                        job.loss = data.get("loss", job.loss) or job.loss
                        if job.state == "starting":
                            job.state = "training"
            except (OSError, ValueError, KeyError):
                pass

        reader.join(timeout=5)

        rc = job.process.wait()
        job.end_time = time.monotonic()

        if job.state == "cancelled":
            pass  # already set
        elif rc == 0:
            job.state = "completed"
            job.append_log("[DONE] Training completed successfully")
        else:
            job.state = "failed"
            job.error = f"Process exited with code {rc}"
            job.append_log(f"[ERROR] Training failed with exit code {rc}")

    except Exception as e:
        job.state = "failed"
        job.error = str(e)
        job.end_time = time.monotonic()
        job.append_log(f"[ERROR] {e}")


def log(msg):
    print(f"[train-server] {msg}", flush=True)


# ── HTTP handler ─────────────────────────────────────────────────────
def _json_response(handler, status, data):
    body = json.dumps(data, indent=2).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _read_body(handler):
    length = int(handler.headers.get("Content-Length", 0))
    if length:
        return json.loads(handler.rfile.read(length))
    return {}


class TrainHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        # Suppress default access log spam
        pass

    def do_GET(self):
        path = self.path.rstrip("/") or "/"

        if path == "/health":
            _json_response(self, 200, {"status": "ok", "mode": "train"})

        elif path == "/status":
            if current_job:
                _json_response(self, 200, current_job.to_dict())
            else:
                _json_response(self, 200, {"state": "idle"})

        elif path.startswith("/logs"):
            n = 100
            # Parse ?lines=N
            if "?" in path:
                for param in path.split("?")[1].split("&"):
                    if param.startswith("lines="):
                        n = int(param.split("=")[1])
            if current_job:
                _json_response(self, 200, {
                    "state": current_job.state,
                    "lines": current_job.get_logs(n),
                })
            else:
                _json_response(self, 200, {"state": "idle", "lines": []})

        elif path == "/jobs":
            # List completed LoRA files
            files = []
            for f in sorted(OUTPUT_DIR.glob("*.safetensors")):
                stat = f.stat()
                files.append({
                    "name": f.name,
                    "size_mb": round(stat.st_size / 1024 / 1024, 1),
                    "modified": stat.st_mtime,
                })
            _json_response(self, 200, {"files": files})

        elif path == "/datasets":
            datasets = []
            for d in sorted(DATASETS_DIR.iterdir()):
                if d.is_dir():
                    images = list(d.glob("*.png")) + list(d.glob("*.jpg"))
                    captions = list(d.glob("*.txt"))
                    datasets.append({
                        "name": d.name,
                        "images": len(images),
                        "captions": len(captions),
                    })
            _json_response(self, 200, {"datasets": datasets})

        elif path == "/caption/status":
            if current_caption:
                _json_response(self, 200, current_caption.to_dict())
            else:
                _json_response(self, 200, {"state": "idle"})

        elif path.startswith("/caption/logs"):
            n = 100
            if "?" in path:
                for param in path.split("?")[1].split("&"):
                    if param.startswith("lines="):
                        n = int(param.split("=")[1])
            if current_caption:
                _json_response(self, 200, {
                    "state": current_caption.state,
                    "engine": current_caption.engine,
                    "lines": current_caption.get_logs(n),
                })
            else:
                _json_response(self, 200, {"state": "idle", "lines": []})

        elif path == "/configs":
            configs = []
            for f in sorted(CONFIGS_DIR.glob("*.toml")):
                configs.append({
                    "name": f.name,
                    "size": f.stat().st_size,
                })
            _json_response(self, 200, {"configs": configs})

        else:
            _json_response(self, 404, {"error": f"Not found: {path}"})

    def do_POST(self):
        global current_job
        path = self.path.rstrip("/") or "/"

        if path == "/train":
            body = _read_body(self)

            with job_lock:
                if current_job and current_job.state in ("starting", "training"):
                    _json_response(self, 409, {
                        "error": "Training already in progress",
                        "state": current_job.state,
                        "step": current_job.step,
                        "total_steps": current_job.total_steps,
                    })
                    return

                job = TrainingJob(body)
                current_job = job

            thread = threading.Thread(target=_run_training, args=(job,), daemon=True)
            thread.start()
            log(f"Started training job: {body.get('output_name', 'lora-output')}")
            _json_response(self, 202, {"status": "started", "config": body})

        elif path == "/cancel":
            with job_lock:
                if current_job and current_job.state in ("starting", "training"):
                    current_job.cancel()
                    log("Training cancelled")
                    _json_response(self, 200, {"status": "cancelled"})
                else:
                    _json_response(self, 200, {"status": "no active job"})

        elif path == "/caption":
            # Async caption job. Returns 202 immediately; poll /caption/status.
            # Supported engines: blip2 (default, natural language for Flux T5),
            # florence (Florence-2, broken on transformers>=4.54 — prefer blip2),
            # wd14 (Danbooru tags, for SDXL/anime models).
            global current_caption
            body = _read_body(self)
            dataset = body.get("dataset")
            engine = body.get("engine", "blip2")
            trigger_word = body.get("trigger_word", "")
            overwrite = body.get("overwrite", False)

            if not dataset:
                _json_response(self, 400, {"error": "dataset required"})
                return

            dataset_dir = DATASETS_DIR / dataset
            if not dataset_dir.is_dir():
                _json_response(self, 404, {"error": f"Dataset not found: {dataset}"})
                return

            with caption_job_lock:
                if current_caption and current_caption.state in ("starting", "running"):
                    _json_response(self, 409, {
                        "error": "Captioning already in progress",
                        "engine": current_caption.engine,
                        "dataset": current_caption.dataset,
                    })
                    return

                if engine == "blip2":
                    cmd = ["python", "/train-hooks/caption_blip2.py", str(dataset_dir)]
                    if body.get("prompt"):
                        cmd.extend(["--prompt", body["prompt"]])
                elif engine == "florence":
                    cmd = ["python", "/train-hooks/caption_florence.py",
                           str(dataset_dir),
                           "--task", body.get("task", "MORE_DETAILED_CAPTION")]
                elif engine == "wd14":
                    cmd = ["python",
                           "/sd-scripts/finetune/tag_images_by_wd14_tagger.py",
                           str(dataset_dir),
                           "--repo_id=SmilingWolf/wd-swinv2-tagger-v3",
                           f"--thresh={body.get('threshold', 0.35)}",
                           "--onnx", "--remove_underscore"]
                    if trigger_word:
                        cmd.append(f"--always_first_tags={trigger_word}")
                else:
                    _json_response(self, 400, {"error": f"Unknown engine: {engine}"})
                    return

                if engine != "wd14":
                    if trigger_word:
                        cmd.extend(["--trigger-word", trigger_word])
                    if overwrite:
                        cmd.append("--overwrite")

                job = CaptionJob(cmd, engine, dataset, trigger_word)
                current_caption = job

            thread = threading.Thread(target=_run_caption, args=(job,), daemon=True)
            thread.start()
            log(f"Started caption job: engine={engine} dataset={dataset}")
            _json_response(self, 202, {
                "status": "started",
                "engine": engine,
                "dataset": dataset,
                "trigger_word": trigger_word,
            })

        elif path == "/cleanup":
            # Safety-net: kill any orphaned training/captioning subprocesses
            # that escaped normal cancel (e.g. after container restart mid-run,
            # or a previous server instance that crashed). Only touches known
            # process names — never touches /train-server.py itself.
            import glob
            killed = []
            for proc_dir in glob.glob("/proc/[0-9]*"):
                try:
                    pid = int(os.path.basename(proc_dir))
                    if pid == os.getpid():
                        continue
                    with open(f"{proc_dir}/cmdline", "rb") as f:
                        cmdline = f.read().replace(b"\x00", b" ").decode(errors="replace")
                    # Target known stragglers
                    if any(marker in cmdline for marker in (
                        "flux_train_network", "sdxl_train_network",
                        "caption_blip2", "caption_florence",
                        "compile_worker", "tag_images_by_wd14",
                        "accelerate",
                    )):
                        try:
                            os.kill(pid, signal.SIGTERM)
                            killed.append({"pid": pid, "cmdline": cmdline[:120]})
                        except ProcessLookupError:
                            pass
                except (OSError, ValueError):
                    continue
            # Give them a moment to exit cleanly, then SIGKILL any survivors
            time.sleep(1)
            for k in killed:
                try:
                    os.kill(k["pid"], signal.SIGKILL)
                except ProcessLookupError:
                    pass
            _json_response(self, 200, {"killed": len(killed), "details": killed})

        elif path == "/caption/cancel":
            with caption_job_lock:
                if current_caption and current_caption.state in ("starting", "running"):
                    current_caption.cancel()
                    log("Caption cancelled")
                    _json_response(self, 200, {"status": "cancelled"})
                else:
                    _json_response(self, 200, {"status": "no active caption job"})

        else:
            _json_response(self, 404, {"error": f"Not found: {path}"})


def main():
    server = http.server.HTTPServer(("0.0.0.0", PORT), TrainHandler)
    log(f"Training API server listening on :{PORT}")
    log(f"Data dir: {DATA_DIR}")
    log(f"Checkpoints dir: {CHECKPOINTS_DIR}")

    # Graceful shutdown
    def shutdown(signum, frame):
        log("Shutting down...")
        if current_job and current_job.state in ("starting", "training"):
            current_job.cancel()
        server.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    server.serve_forever()


if __name__ == "__main__":
    main()
