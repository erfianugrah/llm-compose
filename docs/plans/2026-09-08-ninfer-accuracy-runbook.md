# Resuming the NInfer accuracy runs (GPU required)

Everything below needs the GPU and the stack up. Current state: GPU free,
stack proxy + webui running, no model loaded. Steps in dependency order.

## 0. Bring the engine up

```bash
cd ~/infra/ai/llm-compose && export PATH="$PWD/bin:$PATH"
llmc switch qwen38-ninfer     # loads the 22 GiB artifact, ~minutes
llmc status                    # expect: Active model qwen38-ninfer, Locked: no
```

## 1. Standardized accuracy (the headline missing number)

```bash
llmc bench eval --presets qwen38-ninfer --humaneval --bfcl
# HellaSwag needs a tokenizer per preset - [bench] tokenizer is already set
# (unsloth/Qwen3.8-27B-GGUF), so add it if you want the language-modeling number:
llmc bench eval --presets qwen38-ninfer --hellaswag 1000
```
Results land in bench/results/ and feed `llmc bench report`. Compare against
the published Qwen3.8-27B numbers and the llama.cpp Q4_K_M rows already in
runs.jsonl.

## 2. Long-context QUALITY (needle) - tokenizer fixed; NInfer needs a no-swap probe mode

The tokenizer blocker is FIXED: `make_tokenizer` now falls back to a local
HF tokenizer when the engine has no `/tokenize` (commit 776fa17), and a
`*-GGUF` repo name (which ships no HF tokenizer config) maps to the Qwen3
tokenizer-only model `Qwen/Qwen3-0.6B` (same BPE family). So the probe can
tokenize against NInfer.

But there is a SECOND, structural blocker for NInfer. needle sweeps context
sizes by registering an ephemeral preset per ctx (`needle-<ctx>`) and
locking+switching the proxy to it. That swap path assumes a hot-swappable
llama.cpp engine whose GGUF the ephemeral preset points at. NInfer keeps
ONE resident model, cannot hot-swap, and the ephemeral preset's model_id
(`needle-4096`) is not a loadable NInfer artifact - the swap timed out and
tore the engine down (proxy log 2026-09-08 03:12). It also collided with a
concurrently-running eval on the same engine. Do NOT run
`llmc bench needle --preset qwen38-ninfer` as-is.

To measure NInfer long-context quality, drive it WITHOUT the ephemeral
swap: probe the already-resident `qwen38-ninfer` at its configured
max_context (262144) across depths, no per-ctx re-registration. That means
a new probe mode (a proposed `--no-swap` flag, NOT yet implemented): probe
the loaded model at depth fractions of its CURRENT context, skipping the
ephemeral/lock/switch block, reusing the 776fa17 tokenizer fallback. Good
loop task - and run it SERIALLY, never alongside an eval on the same
engine.

## 3. Churn stability (the spike's open p6)

Repeated agentic traffic; watch for the 6.2% sub-100 tok/s tail growing with
session age. Cheapest proxy: `llmc bench tasks --presets qwen38-ninfer --runs 3`
back-to-back and compare per-task wall time across runs - a decay across runs
is the churn signature. The unexplained tail is the strongest known caveat.

## 4. Frontier anchor (no GPU, but needs the stack's eval image)

```bash
# OpenRouter, verified live. Pick the frontier model id from your OpenRouter account.
python3 bench/run-evals.py --label gpt-5-high --base-url https://openrouter.ai/api/v1 \
    --model <frontier-model-id> --humaneval --bfcl-subset 100
# plus the decisive suite against the same model:
llmc bench tasks --external https://openrouter.ai/api/v1 --model-id <frontier-model-id> --runs 1
```

## 5. Cleanup note

The needle probe records to bench/results/needle-runs.jsonl (separate store,
not runs.jsonl). runs.jsonl is tracked; commit the new task/eval rows after a
run so the history grows.
