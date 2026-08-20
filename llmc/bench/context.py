"""llmc bench context - occupancy sweep driver."""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable, Optional

from llmc.bench import store
from llmc.presets import load_all, Preset
from llmc.cli import ProxyClient

try:
    from llmc.volumes import load as load_volumes
except ImportError:
    load_volumes = None

# ── Pure helpers (unit-tested) ─────────────────────────────────────────

def run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S")

def fill_to_tokens(
    target: int,
    source_text: str,
    tokenize_fn: Callable[[str], list[str]],
) -> str:
    """Greedily/bin-search builds a string with exactly `target` tokens."""
    if target <= 0:
        return ""
    
    # Ensure source is large enough
    current_tokens = tokenize_fn(source_text)
    if len(current_tokens) < target:
        # Repeat source until we have enough
        repeats = (target // len(current_tokens)) + 1
        source_text = source_text * repeats

    # Greedy approach with chunks
    chunk_size = 500
    chunks = [source_text[i:i+chunk_size] for i in range(0, len(source_text), chunk_size)]
    
    current_text = ""
    for chunk in chunks:
        test_text = current_text + chunk
        test_tokens = tokenize_fn(test_text)
        if len(test_tokens) <= target:
            current_text = test_text
        else:
            # Binary search within this chunk
            low = 0
            high = len(chunk)
            best_chunk = ""
            while low <= high:
                mid = (low + high) // 2
                mid_text = current_text + chunk[:mid]
                mid_tokens = tokenize_fn(mid_text)
                if len(mid_tokens) <= target:
                    best_chunk = chunk[:mid]
                    low = mid + 1
                else:
                    high = mid - 1
            current_text += best_chunk
            break
    
    # Final check: if we are still below target, it's because we ran out of chunks.
    # But we ensured source_text was large enough.
    return current_text

def get_prose_chunk(stores_dir: Path) -> str:
    """Get a random prose chunk from docs/*.md."""
    docs = list(stores_dir.glob("../docs/**/*.md"))
    if not docs:
        return "placeholder text"
    
    doc_path = random.choice(docs)
    try:
        content = doc_path.read_text()
        # Split into paragraphs
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        if not paragraphs:
            return "placeholder text"
        return random.choice(paragraphs)
    except:
        return "placeholder text"

# _______________________________________________________________________

# ── The runner ─────────────────────────────────────────────────────────

def run_context_sweep(
    preset_name: str,
    ctx_sizes: list[int],
    slots: int,
    occupancies: list[float],
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
    
    # Track original preset to rollback
    original_preset = preset_name
    
    # Track created files for cleanup
    created_files: list[Path] = []
    
    if not dry_run:
        log("Performing real sweep (modulating environment)...")
        client.set_lock(preset_name, owner="bench-context")
    else:
        log("[dry-run] would lock base preset")

    try:
        # Load volumes to resolve GGUF
        volumes = None
        try:
            volumes = load_volumes(store.REPO_ROOT / "../volumes.toml")
        except:
            log("warning: could not load volumes.toml, using fallback")

        for ctx in ctx_sizes:
            if dry_run:
                log(f"[dry-run] sweep ctx={ctx}")
                continue

            sweep_id = f"ctx-sweep-{ctx}"
            gguf_path = stores_dir / f"{sweep_id}.gguf"
            toml_path = stores_dir / f"{sweep_id}.toml"
            
            # 1. Resolve real GGUF target
            if volumes and "llmc-llama-models" in volumes.names():
                model_dir = volumes.device_for("llmc-llama-models")
                base_gguf_target = model_dir / base_preset.model.file
            else:
                # Fallback to current broken logic
                base_gguf_target = stores_dir / base_preset.model.file
            
            if gguf_path.exists(): gguf_path.unlink()
            os.symlink(base_gguf_target, gguf_path)
            created_files.append(gguf_path)

            # 2. Create TOML
            toml_str = f'name = "{sweep_id}"\n[model]\nrepo = "{base_preset.model.repo}"\nfile = "{sweep_id}.gguf"\n[runtime]\ncontext_size = {ctx}\nparallel_slots = {int(slots)}'
            toml_path.write_text(toml_str)
            created_files.append(toml_path)

            # 3. Switch mode
            client.set_lock(sweep_id, owner="bench-context")
            client.set_mode("llm", model=sweep_id)
            
            # Trigger
            body = json.dumps({"model": sweep_id, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}).encode()
            urllib.request.urlopen(f"{proxy}/v1/chat/completions", data=body, timeout=60)

            for occ in occupancies:
                target_tokens = int(ctx * occ) - gen_tokens - 64
                if target_tokens <= 0: continue
                
                # Use prose filler
                prompt_text = get_prose_chunk(stores_dir)
                prompt = fill_to_tokens(target_tokens, prompt_text, lambda t: tokenize_fn_placeholder(t, proxy)) + "\n\n?"
                
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
            # We don't unlink the files immediately to allow inspection, but the 'finally' block handles it if error.
            
        return 0
    except Exception as e:
        log(f"error: {e}")
        return 1
    finally:
        # Cleanup
        for f in created_files:
            if f.exists(): f.unlink()
        if not dry_run:
            client.set_lock(False, owner="bench-context")
            # Rollback switch
            try:
                client.set_mode("llm", model=original_preset)
            except: pass

def tokenize_fn_placeholder(text: str, proxy: str) -> list[str]:
    """In reality, this calls the proxy. For testing, we can mock it."""
    try:
        body = json.dumps({"content": text}).encode()
        req = urllib.request.Request(f"{proxy}/tokenize", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())["tokens"]
    except:
        # Fall fallback for test/offline
        return list(text)

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
