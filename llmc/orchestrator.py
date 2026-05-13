"""Docker SDK wrapper for the v2 proxy.

Replaces the legacy `docker compose --profile X up/stop` shellout pattern
with direct calls to the Docker Engine API via the `docker` Python SDK.
This eliminates:

    - .env file rewriting (env vars are passed directly to containers.run)
    - compose profile dance (each GPU service is just a container we run/stop)
    - the docker-compose CLI dependency in the proxy image
    - shell-parsing of compose error output

The orchestrator manages three mutually-exclusive GPU services
(llama-server, comfyui, lora-train). Only one runs at a time — the
proxy invokes `stop_gpu_services()` before spawning a new one.

Containers are tagged with two labels so the orchestrator can find them
again after a proxy restart:

    llmc.service = "llama-server" | "comfyui" | "lora-train"
    llmc.mode    = "llm" | "comfyui" | "train"

`current_mode()` reads those labels off running containers and returns
the active mode (or "idle" if none are running).
"""

from __future__ import annotations

import http.client
import io
import os
import tarfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TYPE_CHECKING

# The `docker` package is the proxy's only runtime dep, but the rest of
# the llmc package (CLI, presets, volumes, state) is stdlib-only. Import
# docker lazily so test modules that don't construct an Orchestrator can
# still be imported on a host without the SDK installed.
if TYPE_CHECKING:
    import docker
    from docker.models.containers import Container

from llmc.presets import Preset, preset_to_env


# ── Service definitions ────────────────────────────────────────────────


@dataclass(frozen=True)
class GpuService:
    name: str           # Container name (matches `docker ps`)
    hostname: str       # Hostname inside the user-defined network
    mode: str           # "llm" | "comfyui" | "train"
    image: str
    internal_port: int  # Port the service listens on inside the container
    health_path: str    # Path on the service to poll for readiness


LLAMA_SERVICE = GpuService(
    name="llama_server",
    hostname="llama-server",
    mode="llm",
    image=os.environ.get("LLMC_LLAMA_IMAGE", "erfianugrah/llama-server:cuda12.8-sm120"),
    internal_port=8080,
    health_path="/health",
)

COMFYUI_SERVICE = GpuService(
    name="comfyui",
    hostname="comfyui",
    mode="comfyui",
    image=os.environ.get("LLMC_COMFYUI_IMAGE", "erfianugrah/comfyui:cuda12.8-sm120"),
    internal_port=8188,
    health_path="/system_stats",
)

TRAIN_SERVICE = GpuService(
    name="lora_train",
    hostname="lora-train",
    mode="train",
    image=os.environ.get("LLMC_TRAIN_IMAGE", "erfianugrah/lora-train:latest"),
    internal_port=8787,
    health_path="/health",
)

SERVICES = {svc.mode: svc for svc in (LLAMA_SERVICE, COMFYUI_SERVICE, TRAIN_SERVICE)}

GPU_LABEL = "llmc.mode"
SERVICE_LABEL = "llmc.service"
DEFAULT_NETWORK = os.environ.get("LLMC_NETWORK", "llmc")
DEFAULT_HEALTH_TIMEOUT = int(os.environ.get("LLMC_HEALTH_TIMEOUT", "900"))


class OrchestratorError(RuntimeError):
    """Raised when a Docker operation fails in a way the proxy should surface
    to the client (e.g. image missing, healthcheck timeout)."""


# ── llama-server command builder ───────────────────────────────────────


def _llama_command() -> str:
    """The shell command that the llama-server container runs.

    This is identical to the long inline command in the legacy compose.yaml.
    Kept here so the orchestrator can pass it without modifying the image.
    Once Phase 7 bakes this into the image's ENTRYPOINT, this can be deleted
    and `command=None` will use the image default.
    """
    return r"""
if [ -f "/models/${MODEL_FILE}" ]; then
  MODEL_ARGS="-m /models/${MODEL_FILE}"
else
  MODEL_ARGS="--hf-repo ${MODEL_REPO} --hf-file ${MODEL_FILE}"
fi
exec llama-server \
  $MODEL_ARGS \
  ${MMPROJ_FILE:+--mmproj /models/${MMPROJ_FILE}} \
  --jinja \
  ${TEMPLATE_FILE:+--chat-template-file /models/${TEMPLATE_FILE}} \
  ${REASONING:+--reasoning ${REASONING}} \
  --port 8080 \
  --host 0.0.0.0 \
  -ngl 99 \
  --flash-attn on \
  -ctk q8_0 \
  -ctv q8_0 \
  -c ${CONTEXT_SIZE:-65536} \
  --fit on \
  --fit-ctx ${CONTEXT_SIZE:-65536} \
  --temp ${TEMPERATURE:-1.0} \
  --top-p ${TOP_P:-0.95} \
  --top-k ${TOP_K:-64} \
  ${MIN_P:+--min-p ${MIN_P}} \
  ${PRESENCE_PENALTY:+--presence-penalty ${PRESENCE_PENALTY}} \
  ${REPEAT_PENALTY:+--repeat-penalty ${REPEAT_PENALTY}} \
  -np ${PARALLEL_SLOTS:-1} \
  -b 2048 \
  -ub 2048 \
  --threads 8 \
  --threads-batch 8 \
  -v \
  --metrics
""".strip()


# ── Orchestrator ───────────────────────────────────────────────────────


class Orchestrator:
    """High-level Docker operations for GPU service lifecycle."""

    def __init__(
        self,
        client=None,
        network: str = DEFAULT_NETWORK,
    ):
        if client is None:
            import docker
            client = docker.from_env()
        self.client = client
        self.network = network

    # ── Mode introspection ─────────────────────────────────────────────

    def current_mode(self) -> str:
        """Return the active GPU mode based on running containers labelled
        with llmc.mode. Returns 'idle' if no labelled GPU service is up."""
        for container in self._labelled_containers():
            if container.status != "running":
                continue
            mode = container.labels.get(GPU_LABEL)
            if mode in SERVICES:
                return mode
        return "idle"

    def _labelled_containers(self) -> list:
        return self.client.containers.list(
            all=True,
            filters={"label": GPU_LABEL},
        )

    # ── Lifecycle ──────────────────────────────────────────────────────

    def stop_gpu_services(self, *, timeout: int = 10) -> list[str]:
        """Stop and remove every container with the llmc.mode label.
        Returns the names of containers that were stopped."""
        from docker.errors import NotFound
        stopped = []
        for container in self._labelled_containers():
            try:
                if container.status == "running":
                    container.stop(timeout=timeout)
                container.remove(force=True)
                stopped.append(container.name)
            except NotFound:
                continue
        return stopped

    def spawn(
        self,
        service: GpuService,
        *,
        environment: Optional[dict] = None,
        volumes: Optional[dict] = None,
        ports: Optional[dict] = None,
        command=None,
        entrypoint: Optional[list[str]] = None,
        shm_size: str = "2g",
    ):
        """Generic container spawn for a GPU service. Stops any other GPU
        service first (mutual exclusion). Returns the created container.
        Caller should call `wait_healthy()` before forwarding requests.

        IMPORTANT: pass `command` as a list[str], not a bare string. The
        Docker SDK shlex-splits string commands at whitespace, which mangles
        any non-trivial shell script (sees `if [` as three tokens, etc.).
        Use `["the script"]` to keep it as one Cmd entry."""
        from docker.errors import APIError, ImageNotFound
        from docker.types import DeviceRequest

        self.stop_gpu_services()

        run_kwargs = dict(
            image=service.image,
            name=service.name,
            hostname=service.hostname,
            detach=True,
            network=self.network,
            environment=environment or {},
            volumes=volumes or {},
            device_requests=[DeviceRequest(count=-1, capabilities=[["gpu"]])],
            shm_size=shm_size,
            restart_policy={"Name": "unless-stopped"},
            labels={SERVICE_LABEL: service.hostname, GPU_LABEL: service.mode},
        )
        if ports is not None:
            run_kwargs["ports"] = ports
        if entrypoint is not None:
            run_kwargs["entrypoint"] = entrypoint
        if command is not None:
            run_kwargs["command"] = command

        try:
            return self.client.containers.run(**run_kwargs)
        except ImageNotFound as exc:
            raise OrchestratorError(
                f"image {service.image!r} not found. "
                f"Run `make build` or `make pull` to fetch it."
            ) from exc
        except APIError as exc:
            raise OrchestratorError(
                f"docker API error starting {service.name}: {exc.explanation or exc}"
            ) from exc

    def spawn_llama(self, preset: Preset):
        """Start llama-server with the given preset. Existing GPU services
        are stopped first.

        Note: command is passed as a single-element list, not a bare string.
        Docker SDK shlex-splits string commands at whitespace, which mangles
        the multi-line bash script. Wrapping in [] yields Cmd=[<script>]
        on the engine, which is what `entrypoint=["/bin/sh", "-c"]` consumes.
        """
        return self.spawn(
            LLAMA_SERVICE,
            environment=preset_to_env(preset),
            entrypoint=["/bin/sh", "-c"],
            command=[_llama_command()],
            volumes={
                "llmc-llama-cache": {"bind": "/root/.cache", "mode": "rw"},
                "llmc-llama-models": {"bind": "/models", "mode": "rw"},
            },
        )

    def spawn_comfyui(self):
        return self.spawn(
            COMFYUI_SERVICE,
            shm_size="4g",
            ports={
                # Bind 8188 to host loopback so the ComfyUI web UI is directly
                # reachable while comfyui mode is active. Proxy at /comfyui/*
                # also works, but the websocket-based live preview only works
                # via direct connection.
                "8188/tcp": ("127.0.0.1", 8188),
            },
            volumes={
                "llmc-comfyui-models": {"bind": "/app/ComfyUI/models", "mode": "rw"},
                "llmc-comfyui-output": {"bind": "/app/ComfyUI/output", "mode": "rw"},
                "llmc-comfyui-input": {"bind": "/app/ComfyUI/input", "mode": "rw"},
                "llmc-comfyui-custom-nodes": {"bind": "/app/ComfyUI/custom_nodes", "mode": "rw"},
                "llmc-comfyui-user": {"bind": "/app/ComfyUI/user", "mode": "rw"},
            },
        )

    def spawn_train(self):
        return self.spawn(
            TRAIN_SERVICE,
            shm_size="4g",
            environment={
                "TRAIN_PORT": "8787",
                "DATA_DIR": "/data",
                "CHECKPOINTS_DIR": "/checkpoints",
            },
            volumes={
                "llmc-training-data": {"bind": "/data", "mode": "rw"},
                "llmc-comfyui-models": {"bind": "/models", "mode": "ro"},
                "llmc-comfyui-loras": {"bind": "/loras", "mode": "rw"},
            },
        )

    # ── Health check ───────────────────────────────────────────────────

    def wait_healthy(
        self,
        service: GpuService,
        *,
        timeout: Optional[int] = None,
    ) -> bool:
        """Poll the service's health endpoint until 200 or timeout.
        Uses the service hostname over the user-defined network, so this
        only works from within the proxy container (or another container
        on the same network)."""
        deadline = time.monotonic() + (timeout or DEFAULT_HEALTH_TIMEOUT)
        while time.monotonic() < deadline:
            try:
                conn = http.client.HTTPConnection(
                    service.hostname, service.internal_port, timeout=3
                )
                conn.request("GET", service.health_path)
                resp = conn.getresponse()
                resp.read()
                conn.close()
                if resp.status == 200:
                    return True
            except (OSError, http.client.HTTPException):
                pass
            time.sleep(2)
        return False

    # ── Asset downloads (mmproj, template, GGUF prefetch) ──────────────

    def ensure_asset(
        self,
        assets_dir: Path,
        filename: str,
        url: str,
        *,
        kind: str = "asset",
        timeout: int = 300,
    ) -> Path:
        """Download `url` to `assets_dir/filename` if missing. Returns the
        destination path. Idempotent. The assets_dir is the host-side bind
        path of the llmc-llama-models volume (writable from the proxy
        container which mounts it)."""
        assets_dir.mkdir(parents=True, exist_ok=True)
        dest = assets_dir / filename
        if dest.exists():
            return dest

        tmp = dest.with_suffix(dest.suffix + ".tmp")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "llmc/2.0"})
            with urllib.request.urlopen(req, timeout=timeout) as response:
                with tmp.open("wb") as f:
                    while True:
                        chunk = response.read(1 << 20)  # 1 MiB
                        if not chunk:
                            break
                        f.write(chunk)
            tmp.rename(dest)
            return dest
        except (OSError, urllib.error.URLError) as exc:
            if tmp.exists():
                tmp.unlink()
            raise OrchestratorError(
                f"failed to download {kind} {filename} from {url}: {exc}"
            ) from exc

    def ensure_preset_assets(self, preset: Preset, assets_dir: Path) -> None:
        """Download mmproj and template files for a preset if missing.
        Skipped if the preset uses an explicit `file` reference (manually-
        placed) rather than a `url`."""
        if preset.mmproj.url and preset.mmproj_filename:
            self.ensure_asset(
                assets_dir, preset.mmproj_filename, preset.mmproj.url, kind="mmproj"
            )
        if preset.template.url and preset.template_filename:
            self.ensure_asset(
                assets_dir,
                preset.template_filename,
                preset.template.url,
                kind="template",
            )
