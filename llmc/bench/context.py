"""llmc bench context - occupancy sweep driver."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable, Optional

from llmc.bench import store
from llmc.presets import load_all, Preset
from llmc.cli import ProxyClient

# ── Pure helpers (unit-tested) ─────────────────────────────────────────

def run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S")

# ── The runner ─────────────────────────────────────────────────────────

def run_context_sweep(
    preset_name: str,
    ctx_sizes: list[int],
    slots: int,
    occupancies: list[int],
    gen_tokens: int,
    dry_run: bool = False,
    proxy: str = "http://127.0.0.1:11434",
    log: Callable[[str], None] = print,
) -> int:
    stores_dir = store.REPO_ROOT / "models"
    presets = load_all(stores_dir)
    base_preset = next((p for p in presets.values() if p.name == preset_name), None)
    
    if not base_preset:
        log(f"error: base preset {preset_name} not found")
        return 1
    
    log(f"Starting sweep for {preset_name} (ctx={ctx_sizes[0]}, occ={occupancies[0]})")
    if dry_run:
        log("[dry-run] mode enabled")
    
    client = ProxyClient()
    rid = run_id()
    
    if not dry_run:
        log("Performing real sweep (modifying environment)...")
        client.set_lock(preset_name, owner="bench-context")
    else:
        log("[dry-run] would lock base preset")

    try:
        for ctx in ctx_sizes:
            if dry_run:
                log(f"[dry-run] sweep ctx={ctx}")
                continue

            sweep_id = f"ctx-sweep-{ctx}"
            gguf_path = stores_dir / f"{sweep_id}.gguf"
            toml_path = stores_dir / f"{sweep_id}.toml"
            
            base_gguf = stores_dir / base_preset.model.file
            if gguf_path.exists(): gguf_path.unlink()
            os.symlink(base_gguf, gguf_path)

            toml_str = f'name = "{sweep_id}"\n[model]\nrepo = "{base_preset.model.repo}"\nfile = "{sweep_id}.gguf"\n[runtime]\ncontext_size = {ctx}\nparallel_slots = {int(slots)}'
            toml_path.write_text(toml_str)

            client.set_lock(sweep_id, owner="bench-context")
            client.set_mode("llm", model=sweep_id)
            
            # Trigger
            body = json.dumps({"model": sweep_id, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}).encode()
            urllib.request.urlopen(f"{proxy}/v1/chat/completions", data=body, timeout=60)

            for occ in occupancies:
                target_tokens = int(ctx * occ) - gen_tokens - 64
                if target_tokens <= 0: continue
                
                prompt = "a" * target_tokens + "\n\n?"
                body = json.dumps({"model": sweep_id, "messages": [{"role": "user", "content": prompt}], "max_tokens": gen_tokens, "temperature": 0.0}).encode()
                
                try:
                    req = urllib.request.Request(f"{proxy}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
                    with urllib.request.urlopen(req, timeout=60) as r:
                        r_data = json.loads(r.read())
                    t = r_data.get("timings", {})
                    gen_per_s = t.get("predicted_per_second", 0.0)
                    
                    vram = 0
                    try:
                        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
                        if out.returncode == 0 and out.stdout.strip():
                            vram = int(out.stdout.splitlines()[0].strip())
                    except: pass

                    metrics = {"gen_tok_s": gen_per_s, "vram_mib": vram, "ctx": ctx, "occupancy": occ}
                    rec = store.make_record("context-occupancy", sweep_id, toml_path, metrics, rid, extra={"gen_tokens": gen_tokens})
                    store.append(rec)
                    log(f"  {sweep_id} | occ={occ} | gen={gen_per_s} tok/s | vram={vram} MiB")
                except Exception as e:
                    log(f"  error: {e}")

            client.set_lock(False, owner="bench-context")
            if gguf_path.exists(): gguf_path.unlink()
            if toml_path.exists(): toml_path.unlink()
        return 0
    except Exception as e:
        log(f"error: {e}")
        return 1
    finally:
        if not dry_run: client.set_lock(False, owner="bench-context")

def main() -> int:
    p = argparse.ArgumentParser(prog="llmc bench context")
    p.add_argument("--preset", required=True)
    p.add_argument("--ctx", type=int, required=True)
    p.add_argument("--occupancy", type=float, required=True)
    p.add_argument("--gen-tokens", type=int, default=200)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--slots", type=int, default=1)
    args = p.parse_args()
    return run_context_sweep(args.preset, [args.ctx], args.slots, [args.occupancy], args.gen_tokens, args.dry_run)

if __name__ == "__main__":
    sys.exit(main())
