# Plan: MTP speculative-decoding speed track (qwen38 + small models)

Date: 2026-08-17
Status: planned; code lands now, execution queued behind the P3 matrix
Parent: 2026-08-15-local-model-bench-framework.md

## 1. Goal

User directive (2026-08-17): "I want speed and accuracy and context window."
Speed via MTP speculative decoding on the existing llama.cpp stack; accuracy
= no regression vs the non-speculative baseline (verified, not assumed);
context = keep 196608x2 (loop role) and probe toward 262144 native.

## 2. Verified ground (this session)

- llama.cpp pin b10362 supports `--spec-type draft-mtp` (common/speculative.cpp
  has COMMON_SPECULATIVE_TYPE_DRAFT_MTP + the nextn embeddings API).
- The CURRENT unsloth Qwen3.8-27B Q4_K_M GGUF carries the MTP head metadata
  (`qwen35.nextn_predict_layers` present in the file header). No separate
  drafter file needed - enabling is ONE FLAG.
- Community NVFP4-MTP GGUF family (esatapedico) exists if we later want the
  Blackwell-native quant: MEDIUM 16.38 GB / HIGH 17.57 GB, MTP baked in,
  mmproj byte-identical to unsloth's. NVFP4 GGML-tensor support at b10362 is
  UNVERIFIED - that is S3's first question.
- The gittensor vLLM checkpoint (full 262K, 80 tok/s claims) is safetensors -
  out of scope here; it is the parked "second stack" option.

## 3. Spike ladder

- **S1 - MTP on the current Q4_K_M** (the 30-min win): new preset
  `qwen38-mtp` (clone of qwen38 + `spec_type = "draft-mtp"`), measure
  `llmc bench perf` gen tok/s vs the committed qwen38 baseline (74.0 tok/s,
  run 20260816-091630). Success = >1.3x decode. Also record llama.cpp's
  draft acceptance rate from the server log.
- **S2 - ctx ceiling with MTP**: the draft head adds VRAM (blk.64 tensors +
  its own KV path). Re-run the P0-style ctx matrix for qwen38-mtp:
  {131072, 196608} x {1, 2} slots, plus one probe at 262144 x 1. The loop-role
  requirement (2x98K) must still fit or MTP is chat-only.
- **S3 - NVFP4-MTP tier** (optional, if S1 shows MTP works but Q4_K_M speed
  still trails): download esatapedico HIGH (17.57 GB), preset `qwen38-nvfp4`,
  same perf + ctx measurements. First verify the pinned llama.cpp loads NVFP4
  GGML tensors at all (load test; if unsupported, decide the pin bump).
- **S4 - accuracy guard**: speculative decoding is distribution-preserving in
  theory; verify in practice: re-run `llmc bench tasks --tasks t1,t2 --runs 2`
  on qwen38-mtp and require the same pass profile as the qwen38 baseline
  (t1/t2 both 3/3 one-shot in the matrix). Plus one gumshoe repeat block
  (JSON protocol is fragile - a bad draft interaction shows up there first).

## 4. Implementation (code lands now, presets later)

- presets.py: `_RUNTIME_KEYS` + RuntimeSpec gain `spec_type` (str, optional).
- preset_to_env: `SPEC_TYPE` env when set.
- llama-server-entrypoint.sh: `SPEC_TYPE` -> `--spec-type "$SPEC_TYPE"`.
- NO preset TOML gains spec_type until BOTH images are rebuilt and the proxy
  restarted (the running proxy validates top-level/runtime keys from its
  baked code; an unknown key in /presets would crash preset reload mid-matrix).
- Entrypoint COPY is late in the Dockerfile - `make build` re-bakes cheaply.

## 5. Execution (autonomous, queued)

bench/p4-mtp.sh, launched now, gated on the P3 orchestrator exiting:

1. wait for pi-bg-p3-orchestrator to finish
2. write models/qwen38-mtp.toml (qwen38 + spec_type = "draft-mtp")
3. make build (proxy + llama images with new code) && make restart
4. `llmc bench perf --presets qwen38-mtp` (S1 numbers into the store)
5. ctx probe at 131072x2 + 196608x2 via raw docker runs (S2 core)
6. `llmc bench tasks --presets qwen38-mtp --tasks t1-go-add-truncate,t2-go-fix-palindrome --runs 2` (S4 guard)
7. commit preset + results, print S1/S2/S4 summary

S3 (NVFP4) stays manual - it wants a human call on the quant family switch.

## 6. Decision rules

- MTP ON permanently for qwen38 if: S1 >= 1.3x decode AND S2 keeps 2x98K AND
  S4 passes. Revert = delete the preset, nothing else changes.
- If S1 < 1.3x or acceptance rate is poor (<~50%): MTP off, record why;
  dense-model decode on Blackwell via llama.cpp is then the ceiling, and the
  vLLM question reopens if speed still matters.
- NVFP4 (S3) only if it beats Q4_K_M-MTP on BOTH speed and ctx headroom;
  accuracy parity must be shown via the same task suite before any switch.
