#!/usr/bin/env python3
"""run-evals.py — drive HumanEval / HellaSwag / BFCL against an OpenAI endpoint.

Outputs a single JSON dict to stdout (also written to --out):
    {
      "label": "Q4_K_M",
      "humaneval": {"pass@1": 0.506, "n": 164},
      "hellaswag": {"acc": 0.86, "acc_norm": 0.872, "n": 10042},
      "bfcl":      {"overall": 0.63, "ast": 0.71, "exec": 0.55}
    }

Each eval is opt-in via flags so a single run can do one or many.
Designed for a local llama-server / proxy at OPENAI_BASE_URL.

Usage inside the container:
    python run-evals.py --label Q4_K_M --base-url http://host.docker.internal:11434/v1 \\
        --model bench --humaneval --hellaswag-subset 1000 --bfcl-subset 100
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time, tempfile, shutil
from pathlib import Path

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)

def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    log("$ " + " ".join(cmd))
    return subprocess.run(cmd, check=False, **kw)

# ── HumanEval via EvalPlus ──────────────────────────────────────────────
def eval_humaneval(base_url: str, model: str, n_samples: int, workdir: Path) -> dict:
    out = workdir / "humaneval"
    out.mkdir(exist_ok=True)
    env = {**os.environ, "OPENAI_API_KEY": "sk-noop", "OPENAI_BASE_URL": base_url}
    # evalplus.evaluate hits OpenAI-compatible endpoint when --backend openai
    cmd = [
        "evalplus.evaluate",
        "--dataset", "humaneval",
        "--model", model,
        "--backend", "openai",
        "--base-url", base_url,
        "--root", str(out),
        "--greedy",
    ]
    if n_samples > 0:
        cmd += ["--n-samples", str(n_samples)]
    r = run(cmd, env=env, capture_output=True, text=True)
    log(r.stdout[-500:] if r.stdout else "")
    if r.returncode != 0:
        log(f"evalplus failed: {r.stderr[-500:]}")
        return {"pass@1": None, "n": 0, "error": r.stderr[-200:]}
    # Parse the eval_results.json that evalplus writes
    for p in out.rglob("eval_results.json"):
        d = json.loads(p.read_text())
        base = d.get("pass_at_k", {}).get("base", {}).get("pass@1")
        plus = d.get("pass_at_k", {}).get("plus", {}).get("pass@1")
        return {"pass@1": base, "pass@1_plus": plus, "n": d.get("hint", {}).get("ntotal", 0)}
    return {"pass@1": None, "n": 0, "error": "no eval_results.json"}

# ── HellaSwag via lm-eval ───────────────────────────────────────────────
def eval_hellaswag(base_url: str, model: str, limit: int, workdir: Path, tokenizer: str) -> dict:
    """HellaSwag is a loglikelihood task — lm-eval needs a HF tokenizer that
    matches the GGUF under test. Default tokenizer matches the Qwen3 family;
    override via --hellaswag-tokenizer for other model families.
    """
    out = workdir / "hellaswag"
    out.mkdir(exist_ok=True)
    model_args = (
        f"base_url={base_url}/completions,model={model},"
        f"tokenizer={tokenizer},tokenizer_backend=huggingface,"
        f"num_concurrent=4,max_retries=2,tokenized_requests=False"
    )
    cmd = [
        "lm_eval",
        "--model", "local-completions",
        "--model_args", model_args,
        "--tasks", "hellaswag",
        "--output_path", str(out),
        "--apply_chat_template", "False",
        "--batch_size", "1",
    ]
    if limit > 0:
        cmd += ["--limit", str(limit)]
    r = run(cmd, capture_output=True, text=True)
    log(r.stdout[-500:] if r.stdout else "")
    if r.returncode != 0:
        log(f"lm-eval failed: {r.stderr[-500:]}")
        return {"acc": None, "error": r.stderr[-200:]}
    for p in out.rglob("results_*.json"):
        d = json.loads(p.read_text())
        res = d.get("results", {}).get("hellaswag", {})
        return {
            "acc": res.get("acc,none"),
            "acc_norm": res.get("acc_norm,none"),
            "n": d.get("n-samples", {}).get("hellaswag", {}).get("effective", limit or 0),
        }
    return {"acc": None, "error": "no results json"}

# ── BFCL ────────────────────────────────────────────────────────────────
def eval_bfcl(base_url: str, model: str, limit: int, workdir: Path) -> dict:
    """BFCL evaluator — uses its OpenAI-style handler against our endpoint.

    BFCL CLI is `bfcl generate` then `bfcl evaluate`. We point it at the
    local endpoint via env vars its OpenAI handler reads.
    """
    out = workdir / "bfcl"
    out.mkdir(exist_ok=True)
    env = {
        **os.environ,
        "OPENAI_API_KEY": "sk-noop",
        "OPENAI_BASE_URL": base_url,
        # BFCL reads model name from CLI — pass our placeholder
    }
    # Generate model responses. NOTE: BFCL has no working subset mechanism -
    # the old --bfcl-subset mapped to a no-op --num-gpus and the FULL category
    # always ran (the 'limit silently dropped' bug). We run the full ast
    # category and ignore `limit` entirely.
    if limit > 0:
        print(f"[run-evals] WARNING: bfcl subset {limit} requested but unsupported; running full ast category", file=sys.stderr)
    gen = ["bfcl", "generate",
           "--model", model,
           "--test-category", "ast",  # AST + executable subset
           "--num-threads", "4"]
    r = run(gen, env=env, cwd=out, capture_output=True, text=True)
    log(r.stdout[-300:] if r.stdout else "")
    if r.returncode != 0:
        log(f"bfcl generate failed: {r.stderr[-500:]}")
        return {"overall": None, "error": "generate failed"}
    # Evaluate
    ev = ["bfcl", "evaluate", "--model", model, "--test-category", "ast"]
    r = run(ev, env=env, cwd=out, capture_output=True, text=True)
    log(r.stdout[-500:] if r.stdout else "")
    # BFCL prints overall accuracy to stdout — parse it
    overall = None
    for line in (r.stdout or "").splitlines():
        if "Overall Accuracy" in line:
            try:
                overall = float(line.split(":")[-1].strip().rstrip("%")) / 100
            except ValueError:
                pass
    return {"overall": overall, "raw_tail": (r.stdout or "")[-500:]}

# ── Driver ──────────────────────────────────────────────────────────────
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True, help="quant label for this run (e.g. Q4_K_M)")
    p.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1"))
    p.add_argument("--model", default="bench", help="model id to send in OpenAI requests")
    p.add_argument("--out", default="-", help="path for JSON result, or - for stdout")
    p.add_argument("--humaneval", action="store_true")
    p.add_argument("--humaneval-samples", type=int, default=0, help="0 = all 164")
    p.add_argument("--hellaswag", action="store_true")
    p.add_argument("--hellaswag-subset", type=int, default=1000, help="0 = all 10042")
    p.add_argument("--hellaswag-tokenizer", default="Qwen/Qwen3-0.6B",
                   help="HF tokenizer for loglikelihood scoring (must match GGUF family)")
    p.add_argument("--bfcl", action="store_true")
    p.add_argument("--bfcl-subset", type=int, default=100)
    p.add_argument("--workdir", default="/tmp/evals")
    args = p.parse_args()

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    result: dict = {"label": args.label, "base_url": args.base_url, "model": args.model}

    if args.humaneval:
        log(f"=== HumanEval ({args.humaneval_samples or 164} problems) ===")
        result["humaneval"] = eval_humaneval(args.base_url, args.model, args.humaneval_samples, workdir)
    if args.hellaswag:
        log(f"=== HellaSwag (limit={args.hellaswag_subset or 'all'}, tokenizer={args.hellaswag_tokenizer}) ===")
        result["hellaswag"] = eval_hellaswag(args.base_url, args.model, args.hellaswag_subset, workdir, args.hellaswag_tokenizer)
    if args.bfcl:
        log(f"=== BFCL (limit={args.bfcl_subset}) ===")
        result["bfcl"] = eval_bfcl(args.base_url, args.model, args.bfcl_subset, workdir)

    js = json.dumps(result, indent=2, default=str)
    if args.out == "-":
        print(js)
    else:
        Path(args.out).write_text(js)
        log(f"wrote {args.out}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
