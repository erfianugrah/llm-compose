# Model proxy v2 (router redesign) Spec

**Goal:** turn the model proxy from a swapper-with-a-mutex into a small scheduler: requests are admitted, queued, and served in place when the resident model can do the job, and model swaps are graceful, durable, and never kill in-flight work.
**Workflow:** design-first (technically constrained: single GPU timeshare, porting an existing Python design to Go).
**Non-goals:** the two-resident-model VRAM split (deferred to the 3080 Ti decision); priority/preemption policies beyond FIFO+drain; multi-GPU scheduling; any change to the llmc bench CLI surface.

## Context and motivation

Current proxy (`llmc/proxy.py`, Python, threaded): routes by exact model name, swaps presets unconditionally (stopping the old container with a 10s timeout, killing in-flight streams), and coordinates tenants via an in-memory lock + FIFO queue. Observed failure modes this week:

- Stale locks survive proxy restarts in practice (two manual `llmc unlock` interventions: `pi-compact-test`, `doc-audit`).
- A whisper-bot LLM call (text or vision, both default to gemma-4-26B) during a qwen38 coding session forces a swap that kills the in-flight coding request, or 409s when a loop holds the lock - even though qwen38 is vision-capable (mmproj loaded) and could have served the bot in place.
- The degradation hunt was slowed by swaps that killed in-flight requests mid-measurement.

The proxy is never the bottleneck (the GPU is); the design target is correctness and tenant fairness, not throughput.

## Requirements

### R1: Durable lock and queue state
**Story:** As the operator, I want lock ownership and the FIFO queue to survive proxy restarts, so that a crashed or redeployed proxy never leaves the stack thinking a model is free when it is locked (or vice versa).
**Acceptance criteria:**
- THE SYSTEM SHALL persist lock owner set and FIFO queue contents to the state file on every mutation.
- WHEN the proxy starts, THE SYSTEM SHALL reload the persisted lock state.
- IF a persisted lock owner has not polled or held an active request for a configurable TTL (default 900s), THEN THE SYSTEM SHALL drop that owner on load and log the drop.
- WHEN the last owner of a lock drains, THE SYSTEM SHALL release the lock and grant it to the queue head for its queued model.

### R2: Graceful drain before swap
**Story:** As a tenant with an in-flight request, I want the proxy to finish my stream before swapping models, so that a model switch never errors my request mid-generation.
**Acceptance criteria:**
- THE SYSTEM SHALL track in-flight request counts per active model, incremented on admission and decremented on response completion or client disconnect.
- WHEN a swap is requested, THE SYSTEM SHALL wait for the active model's in-flight count to reach zero before stopping the container.
- IF in-flight does not drain within a configurable grace period (default 60s), THEN THE SYSTEM SHALL proceed with the swap and log the forced stop.
- WHILE a swap is draining, THE SYSTEM SHALL hold new requests for the target model in a pending state and respond to status polls with the drain state.

### R3: Capability routing
**Story:** As a client that needs a capability (vision) rather than a specific model, I want the proxy to serve me on the resident model when it satisfies the capability, so that my request never forces a swap.
**Acceptance criteria:**
- THE SYSTEM SHALL read a `capabilities` list from each preset TOML (e.g. `["vision", "code"]`).
- WHEN a request carries a capability key (header `X-LLM-Capability` or model field form `cap:vision`), THE SYSTEM SHALL serve it on the resident model if that preset lists the capability.
- IF the resident model does not satisfy the capability and another preset does, THEN THE SYSTEM SHALL swap (subject to R2) and serve there.
- WHEN a request names an exact model, THE SYSTEM SHALL keep current exact-name behavior, including passthrough for unknown models when unlocked.
- IF a request carries a capability no preset satisfies, THEN THE SYSTEM SHALL return 400 listing the known capabilities.

### R4: Go rewrite with REST parity
**Story:** As the maintainer, I want the proxy in Go with the existing REST surface unchanged, so that llmc (Python CLI/bench), the whisper bot, and Open WebUI need no changes.
**Acceptance criteria:**
- THE SYSTEM SHALL serve the same endpoints as the Python proxy: `GET/POST /mode`, `POST /lock`, `POST /unlock`, `GET /status`, `GET /v1/models`, and transparent `/v1/*` forwarding including SSE streaming.
- THE SYSTEM SHALL reproduce the current status codes for contention: 409 for refused lock/swap, 202 + position for queued lock requests.
- THE SYSTEM SHALL pass a Go port of the proxy test cases from `llmc/tests/test_proxy.py`.
- THE SYSTEM SHALL run as a drop-in compose service on the same port with the same volume/env contract.

### R5: Client compatibility (pi, Claude Code, Open WebUI) + MCP transparency
**Story:** As a user, I want every client in the stack to work through the proxy without client-side workarounds, so that the router is the single front door.
**Acceptance criteria:**
- THE SYSTEM SHALL serve OpenAI-compatible `POST /v1/chat/completions` with streaming (SSE) for pi (provider `openai-completions`) and Open WebUI, including `tools`/`tool_calls` payloads passed through byte-identical.
- THE SYSTEM SHALL serve `GET /v1/models` listing every configured preset (Open WebUI's model picker and pi's /model flow depend on it).
- THE SYSTEM SHALL serve an Anthropic-compatible `POST /v1/messages` (Messages API subset: `system`, `messages`, `tools`, streaming SSE with `content_block_*` events) translated to the resident/scheduled model, so Claude Code can target the proxy via `ANTHROPIC_BASE_URL`.
- IF a `/v1/messages` request uses an untranslatable feature (prompt caching blocks, citations), THEN THE SYSTEM SHALL degrade to the closest translatable form and log the dropped feature, never silently corrupt.
- THE SYSTEM SHALL pass through request/response fields it does not understand without stripping (MCP-adjacent payloads, structured-output `response_format`, vendor extensions); the proxy's job is scheduling, not schema enforcement.
- WHERE llama-server exposes MCP-related endpoints (e.g. its UI MCP proxy), THE SYSTEM SHALL forward them transparently under the same path prefix.

### R6: Soak and cutover
**Story:** As the operator, I want both proxies runnable side by side, so that cutover is reversible.
**Acceptance criteria:**
- THE SYSTEM SHALL support running the Go proxy on an alternate port against the same presets and state dir.
- WHEN the soak passes (bench perf + one full task suite through the Go proxy with zero behavior diffs vs the Python proxy's recorded results), THE SYSTEM SHALL switch the compose service to the Go proxy.
- IF the soak fails, THEN THE SYSTEM SHALL keep the Python proxy live and record the diff in the repo.

## Design

**Language/architecture:** Go, stdlib-first (house convention; cf. drawbridge). `net/http` + a hand-rolled `httputil.ReverseProxy` wrapper that increments/decrements the in-flight counter per model around each proxied request, with SSE streams held open by the handler goroutine. `BurntSushi/toml` for preset parsing (schema mirrors `llmc/presets.py`). State file format unchanged (JSON) so both proxies read the same store; the Go proxy writes lock/queue state into it (R1).

**Scheduler architecture: steal llama-swap's.** Their `internal/router` split (a single-goroutine run loop consuming events off channels; scheduler state mutated only on that goroutine so no mutex is needed; swaps as fire-and-forget goroutines reporting back `SwapDone`; a pure-function Swapper for eviction decisions; in-flight grants counted only after the caller provably took the handler) is the right shape and well documented in their design.md. Notably, llama-swap already implements drain-before-evict (in-flight WaitGroup; eviction deferred until the count hits zero) - R2 is validated prior art, and their exact race (request between lock-release and in-flight increment) is the one to write the test for first. What llama-swap does NOT do: capability serve-in-place routing (it always swaps to the named group member), docker-container lifecycle (it manages raw processes), GPU-mode arbitration against comfyui/train/whisper, or the lock API - so we adopt the architecture, not the tool.

**Rejected alternatives:** LiteLLM (provider load-balancer; no local model lifecycle, heavyweight Python dep, Redis for state - solves a different problem), llama.cpp `--models-preset` router mode (no queueing/locks/drain; config-file only), llama-swap as-is (see above - missing four integration surfaces we already depend on).

**Anthropic shim (R5):** a `POST /v1/messages` handler translating Anthropic Messages -> OpenAI chat completions (system -> system message, `tool_use`/`tool_result` blocks -> OpenAI tool calls, streaming `content_block_delta` -> OpenAI deltas) and back. Scope deliberately to what Claude Code sends; unknown blocks pass through when possible, otherwise logged-and-dropped per the degradation criterion.

**Module layout:** `proxy-go/` in this repo: `cmd/proxy/main.go`, `internal/proxy/{server,scheduler,presets,state,orchestrator}.go`. The orchestrator part shells to docker via the socket (same operations as `llmc/orchestrator.py`: stop_gpu_services, spawn container from preset, wait healthy).

**Scheduler decision order (per request):**
1. Exact model name matches resident model -> serve.
2. Capability key -> resident preset satisfies -> serve in place.
3. Capability key -> other preset satisfies -> drain (R2) + swap + serve.
4. Exact name, different preset -> drain + swap (unlocked) or 409 (locked); lock acquisition with `wait` -> 202 + FIFO position (unchanged semantics).

**Config additions:** preset TOML gains `capabilities = [...]`; proxy env gains `LOCK_TTL_S` (default 900), `DRAIN_GRACE_S` (default 60).

**Whisper bot change (follow-on, separate compose repo):** set its LLM envs to capability form once R3 ships, so bot traffic stops naming gemma-4 explicitly.

**Testing strategy:** Go table tests for the scheduler decision order and lock/queue state machine (ported from `llmc/tests/test_proxy.py`); a hurl suite against the running service for REST parity incl. SSE; soak per R6 using `llmc bench perf` + one task suite (the degradation-informed watch: confirm no wedge, decode rate sane).

## Open questions

- **Preemption/priority** (interactive request preempting a loop): deferred; default assumption is FIFO + drain only.
- **Two-resident split (qwen38 @ ~98k ctx + gemma4-12b permanent):** parked until the 3080 Ti decision; the capability table in R3 is designed so this becomes a topology change, not a protocol change.
- **Capability granularity:** v1 is a flat string list; defaults: qwen38 gets `["vision","code"]`, loop gets `["vision","code"]`, gemma4-12b gets `["vision","small"]`.
