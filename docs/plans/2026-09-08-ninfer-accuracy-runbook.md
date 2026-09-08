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

## 2. Long-context QUALITY (blocked - needs a code change first)

`llmc bench needle` 404s on NInfer: it sizes filler via llama.cpp's
`/tokenize`, which NInfer doesn't serve. Fix BEFORE running:

- Add a local-tokenizer fallback in `llmc/bench/context.py::make_tokenizer`:
  when the preset's engine is ninfer (or `/tokenize` 404s), tokenize locally
  with the HF tokenizer named by the preset's `[bench] tokenizer` field
  (`unsloth/Qwen3.8-27B-GGUF`) via `transformers.AutoTokenizer`. The field is
  already declared on every preset; only eval.py's HellaSwag reads it today.
- Then: `llmc bench needle --preset qwen38-ninfer --depths 0.05,0.25,0.5,0.75,0.95 --ctxs 4096,16384,32768,65536 --runs 2`

This is a good loop task (the needle loop pattern already proved itself).

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
