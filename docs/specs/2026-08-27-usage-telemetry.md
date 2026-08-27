# Usage telemetry Spec

**Goal:** capture per-request token usage (prompt / completion) locally, attributed to model and mode, so the operator can see which presets actually get used and how much - the one thing OpenRouter / Zen / Aperture all give you and this stack currently does not.
**Workflow:** requirements-first
**Non-goals:** dashboards or hosted aggregation; billing / cost math; per-user attribution (single user, no auth layer); retention pruning / rotation; capturing prompt *content* (only counts, never text); telemetry for comfyui/train modes (no token concept there).

## Context and motivation

`forwardTo` today streams upstream bytes to the client verbatim and logs only `req start` / `req done` with model + body size + duration. There is no record of token counts, so questions like "is `summarizer` earning its VRAM" or "what does my `auto:cheap` traffic actually cost in tokens" are unanswerable. The proxy is the natural chokepoint: every request already passes through it, and llama-server already returns usage - it is just being discarded.

Capture points that already exist and cost nothing to reuse:

- The Anthropic shim (`anthropic.go`) already parses `usage` from every response - `translateResponse` extracts `prompt_tokens`/`completion_tokens` for non-stream, and `streamState.handleChunk` accumulates `inTokens`/`outTokens` for streams. It throws them into the translated body and forgets them.
- The native `/v1/*` path (`forwardTo`) does no parsing at all.

The gap is therefore not "how to get usage" but "tee it to a local append-only store without touching the streaming hot path."

## Requirements

### R1: Usage available for streaming requests
**Story:** As the operator, I want streaming requests to carry usage too, so that the majority of real traffic (SSE) is measurable, not just non-stream calls.
**Acceptance criteria:**
- WHEN the proxy forwards a streaming request to llama-server (`stream: true`), THE SYSTEM SHALL inject `stream_options: {"include_usage": true}` into the upstream request body when it is not already present.
- WHEN `include_usage` is already set by the client, THE SYSTEM SHALL leave it untouched.
- THE SYSTEM SHALL apply this injection on both the native `/v1/*` path and the Anthropic `/v1/messages` path. (The shim already sets it in `translateRequest`; the native path does not - parity is the point.)
- THE SYSTEM SHALL NOT inject `stream_options` into non-streaming requests.

### R2: Capture usage at the upstream boundary
**Story:** As the operator, I want one usage record per request regardless of which client protocol (OpenAI or Anthropic) was used, so that the two paths are measured identically.
**Acceptance criteria:**
- WHEN a non-streaming `/v1/*` request completes, THE SYSTEM SHALL extract `usage` from the upstream JSON response body.
- WHEN a streaming `/v1/*` request completes, THE SYSTEM SHALL extract `usage` from the final SSE chunk that carries it (the chunk with empty `choices` and a `usage` object).
- WHEN a non-streaming `/v1/messages` request completes, THE SYSTEM SHALL capture the usage `translateResponse` already extracts (`input_tokens`/`output_tokens`).
- WHEN a streaming `/v1/messages` request completes, THE SYSTEM SHALL capture the `streamState.inTokens`/`outTokens` already accumulated at `finish`.
- WHERE no usage is seen (upstream died mid-stream, non-200, or malformed body), THE SYSTEM SHALL record zero token counts plus the failure status - a request that happened is still a data point.

### R3: One append-only JSONL record per request
**Story:** As the operator, I want the data in a file I can query with jq/duckdb/mlr, so that I do not need a server or dashboard to use it.
**Acceptance criteria:**
- WHEN a request completes (success or failure), THE SYSTEM SHALL append exactly one JSON object line to the telemetry file.
- THE SYSTEM SHALL write each record atomically as a single line (one `write` syscall per record, `O_APPEND`), so a crash between requests loses at most the in-flight line, never a partial line.
- THE SYSTEM SHALL store the file at `LLMC_TELEMETRY_FILE` (default `/state/usage.jsonl`, beside `active.toml` in the already bind-mounted state dir).

### R4: Record shape
**Story:** As the operator, I want enough fields to answer per-model and per-mode questions without re-deriving anything.
**Acceptance criteria:**
- THE SYSTEM SHALL record, per line: `ts` (unix seconds), `mode` (`llm`), `model` (preset name as granted, or `-` when unknown), `model_id` (GGUF stem, or `-`), `prompt_tokens`, `completion_tokens`, `total_tokens`, `duration_ms`, `status` (HTTP status the client saw).
- THE SYSTEM SHALL compute `total_tokens` as `prompt_tokens + completion_tokens`.
- WHEN the alias feature (2026-08-24 task-routing spec) ships, THE SYSTEM SHALL additionally record `alias` (the `auto:` name, or empty) - reserved now so the schema does not churn later.

### R5: Non-blocking, failure-isolated
**Story:** As the operator, I want telemetry to never slow down or break a generation, so that a slow disk or a full state dir cannot degrade inference.
**Acceptance criteria:**
- THE SYSTEM SHALL decouple recording from the request path via a dedicated telemetry goroutine fed by a buffered channel; the forwarding goroutine only does a non-blocking send.
- IF the telemetry channel buffer is full, THEN THE SYSTEM SHALL drop the record (and count the drop) rather than block the stream.
- IF a telemetry write fails, THEN THE SYSTEM SHALL log once and continue - a telemetry failure SHALL NOT affect the response, the status code, or the scheduler.
- THE SYSTEM SHALL drain the telemetry channel on shutdown (`Close`) so completed requests are not lost on a clean stop.

### R6: Query surface
**Story:** As the operator, I want a quick aggregate view without leaving the box, so that "which model did I actually use this week" is one curl.
**Acceptance criteria:**
- THE SYSTEM SHALL expose `GET /usage` returning per-model aggregates (request count, `prompt_tokens`, `completion_tokens`, `total_tokens` summed) over a configurable trailing window.
- THE SYSTEM SHALL default the window to the full file; `GET /usage?days=7` SHALL restrict to records with `ts` within the last 7 days.
- THE SYSTEM SHALL serve `/usage` from the telemetry component directly (read-only, no scheduler acquire, no swap) so it works in any mode.
- THE SYSTEM SHALL keep `/usage` an additive endpoint: existing `/v1/models`, `/status`, `/mode` behavior is unchanged.

### R7: Configuration
**Story:** As the operator, I want to opt out or relocate the store, so that telemetry fits existing volume layouts and disk constraints.
**Acceptance criteria:**
- THE SYSTEM SHALL enable telemetry by default.
- THE SYSTEM SHALL disable telemetry when `LLMC_TELEMETRY_ENABLED` is `false` (or `0`), in which case no injection (R1), no capture, and no file writes occur.
- WHEN `LLMC_TELEMETRY_FILE` is set to an empty string, THE SYSTEM SHALL treat telemetry as disabled.

## Design

### Component

A `Telemetry` struct owning a `chan Record` and a goroutine that drains it, marshals each record to JSON, and appends to the configured file. One instance lives on the `Server` (wired in `main.go`), passed to both the native forwarding path and the `AnthropicTranslator` so both routes emit through the same sink. This mirrors how `logf` is threaded today.

```go
type Telemetry struct {
    file  *os.File
    ch    chan Record
    drops atomic.Int64   // dropped-when-full counter, surfaced in logs
}
func NewTelemetry(path string) (*Telemetry, error)
func (t *Telemetry) Record(r Record)      // non-blocking send
func (t *Telemetry) Close()               // drain + close file
```

### Capture points

1. **Native non-stream** (`forwardTo`): buffer the upstream body (chat-completions JSON is small; non-stream only), extract `usage`, forward it, then `Record`.
2. **Native streaming** (`forwardTo`): tee the SSE stream - as each `data:` line is written to the client, scan it for a JSON object with a `usage` key; on the final usage chunk, `Record`. The tee reuses the `buf` loop that already runs, not a second read.
3. **Anthropic non-stream** (`Serve` -> `translateResponse`): capture the `usage` map already computed; `Record` before returning.
4. **Anthropic streaming** (`streamState.finish`): `st.inTokens`/`st.outTokens` are already the final counts; `Record` there.

All four call the same `Telemetry.Record` with the granted model (`res.Key` / preset name) already in scope.

### Injection (R1)

In `forwardTo`, after body read and before the upstream request is built: if the body's `stream` is `true` and it has no `stream_options`, set `stream_options: {"include_usage": true}`. The shim already does this in `translateRequest`; keep its own (a no-op when present). The extra final usage chunk is harmless to OpenAI-compatible clients (empty `choices` is ignored by SDKs and Open WebUI).

### Failure modes considered

- **Upstream dies mid-stream**: no final usage chunk -> record zeros + the `upstream_died_midstream` status, per R2.
- **State dir full / read-only**: `NewTelemetry` open fails -> log, run with telemetry disabled for the process lifetime (never crash the proxy over a telemetry file).
- **Channel full under burst**: drop + count, per R5; the drop counter is surfaced in the periodic log and `GET /usage` response as `dropped`.
- **Schema drift**: R4 reserves `alias` now; adding fields later is backward-compatible because consumers query JSONL by key, not by fixed-width column.

### Testing strategy

- Unit: `Record` JSONL line is single-line valid JSON; usage extraction from a synthetic SSE chunk and a synthetic chat.completions body; injection adds `stream_options` only when `stream:true` and absent.
- Integration (`server_test.go`): a streaming request through a fake upstream with a usage chunk writes one JSONL line with non-zero `prompt_tokens`; a non-stream request likewise; an Anthropic streaming request likewise.
- Smoke (`make smoke-proxy-go`): `GET /usage` returns per-model sums after a real request; `GET /usage?days=7` filters.
- Manual: `tail -n 20 ~/docker-volumes/state/usage.jsonl | jq .` shows recent requests; `duckdb -c "SELECT model, sum(total_tokens) FROM '~/docker-volumes/state/usage.jsonl' GROUP BY 1"` gives the per-model answer.

## Open questions

- **Rotation**: default is no pruning (file grows slowly at one line per request). If it ever matters, add a `LLMC_TELEMETRY_MAX_BYTES` that rotates to `usage.jsonl.1`. Default assumption: skip until the file proves large.
- **`GET /usage` output shape**: assume `{models: [...], dropped: N, window_days: N}`. Exact field names are an implementation detail; the requirement is the per-model token sums.
