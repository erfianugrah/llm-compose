# Eval harness reference (HumanEval / HellaSwag / BFCL)

How `llmc bench eval` works and what it took to make bfcl-eval 2026.3.23 run
against a local OpenAI-compatible endpoint. Read this before touching
bench/Dockerfile.eval or bench/run-evals.py.

## Architecture

- `llmc bench eval --presets a,b --humaneval --bfcl` switches the proxy to each
  preset (locked, owner `bench` - fails fast with 409 if another preset is
  pinned; no queue, rerun when the GPU is free), then runs one throwaway
  container per preset from `erfianugrah/bench-eval:latest`.
- The container entrypoint is bench/run-evals.sh -> run-evals.py, which shells
  out to `evalplus.evaluate` / `lm_eval` / `bfcl` and writes one JSON per
  preset to bench/results/eval-<preset>-<ts>.json. llmc parses that into
  metrics and appends a record to bench/results/runs.jsonl (trend store).
- The image carries TWO python environments:
  - global site-packages: evalplus + lm-eval + core deps
  - `/opt/venv-bfcl`: bfcl-eval + its pinned deps, `bfcl` symlinked into
    /usr/local/bin

## bfcl-eval 2026.3.23 compatibility landmines (all hit 2026-08-17)

1. **faiss-cpu pin is uninstallable.** bfcl-eval pins faiss-cpu==1.11.0, which
   PyPI no longer carries (1.12+ only). Install bfcl with `--no-deps`, then
   install its deps explicitly with `faiss-cpu>=1.12`.

2. **tree-sitter ABI split - two harnesses cannot share site-packages.**
   evalplus requires tree-sitter>=0.22.0 and calls `Language(capsule)` (1-arg).
   bfcl's java_parser calls `Language(capsule, "java")` (2-arg), which only
   works on tree-sitter 0.21.x. No single core version accepts both call
   shapes with matching grammar ABIs (verified against 0.21.3 / 0.22.x /
   0.26.0). Hence the dual-venv layout.

3. **qwen_agent hard-imports soundfile without declaring it.**
   `qwen_agent/utils/utils.py` does `import soundfile` at module top; with
   bfcl installed `--no-deps` this is a ModuleNotFoundError. Install
   `soundfile` explicitly into the bfcl venv.

4. **The `ast` test category no longer exists.** bfcl-eval 2026.3.23 renamed
   the category tree. `non_live` is the old ast set: simple_python,
   simple_java, simple_javascript, multiple, parallel, parallel_multiple,
   irrelevance. Avoid `live`/`web_search`/`agentic`/`memory` - they need
   external API keys (google-search-results etc).

5. **Unknown model names hard-KeyError.** Both `bfcl generate` and
   `bfcl evaluate` resolve `--model` via `MODEL_CONFIG_MAPPING[name]`. GGUF
   stems are not in the 175-entry map. bench/bfcl-sitecustomize.py is
   installed into the venv's site-packages and registers names from the
   `BFCL_LOCAL_MODELS` env var (set by run-evals.py) to the
   OpenAICompletionsHandler, which reads OPENAI_BASE_URL/OPENAI_API_KEY.
   **Register both the raw name and the underscore->slash variant**: bfcl
   evaluate does `model_name.replace("_", "/")` for config lookups (their
   convention: file paths use `_` for `/`), which mangles GGUF stems like
   `Q4_K_M` into `Q4/K/M`.

6. **OPEN: leaderboard CSV formatter crash.** `bfcl evaluate` scores fine but
   crashes in `generate_leaderboard_csv`
   (`TypeError: unsupported operand type(s) for *: 'NoneType' and 'int'`)
   because a locally-registered model has no latency/cost data and the
   formatter blindly scales Nones. This is the remaining blocker for a full
   BFCL number. Two candidate fixes if revisited: register the model with
   `input_price=0.0, output_price=0.0` (may still leave latency stats None),
   or stop parsing data_overall.csv and read the per-category `*_score.json`
   files that scoring writes to --score-dir before the leaderboard step runs.

## evalplus gotchas

- pass@k is only PRINTED to stdout (`pass@1:\t0.451` via cprint); the saved
  `<model>_openai_temp_0.0_eval_results.json` contains per-task statuses, not
  pass_at_k. run-evals.py parses stdout (base line first, plus second).
- HumanEval absolute scores through this harness run far below published
  model-card numbers (greedy decoding through the proxy's preset sampling
  params). Use for relative ranking only.

## runs.jsonl hygiene

Records are append-only and carry a preset-hash provenance. Bad records from
broken-harness runs (empty metrics) have been surgically dropped/patched
twice already (2026-08-17: 3 phase-1 junk records dropped, humaneval numbers
injected from run logs; 1 empty followup record dropped). When a run is
killed mid-flight, also check for a stale `bench` lock via `llmc status` and
clear with `llmc unlock --owner bench`.

## Re-running

```bash
llmc bench eval --presets loop,gemma4,qwen38 --humaneval          # ~1.5h
llmc bench eval --presets loop,gemma4,qwen38 --bfcl               # ~2.5h, see landmine 6
llmc bench report --markdown
llmc bench report --compare qwen38 qwen38-mtp
```
