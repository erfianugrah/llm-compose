### Ninfer context-materialization wedge - mitigation

## Root cause (confirmed 2026-09-10, prior session)

Upstream bug: Neroued/ninfer#184 (open). With `--max-concurrency 1`, a client
that disconnects while the engine is materializing a long-context request
leaves the sole slot stuck. The SSE heartbeat cannot run during synchronous
materialization, so the disconnect is never detected, and the engine's own
cancellation path explicitly skips cancellation while a context transaction
is open. Symptom: prefill crawls at ~30 tok/s (baseline ~2.7k tok/s), the
HTTP listener goes unresponsive, and a bare `docker restart ninfer_server`
did NOT clear it in the 2026-09-10 reproduction - only a full stack restart
(model_proxy_go + engine) did. Practical ceiling for `qwen38-ninfer` is
~120k ctx; past that fall back to llama.cpp `qwen38`. Related upstream:
#210 (KV race, full GPU lockup), #181 (prefix cache eviction per interleaved
request).

## Done this session

1. **Wired three previously-unused `ninfer-serve` flags into the preset
   schema** (`proxy-go/internal/proxy/presets.go`, `orchestrator.go`):
   `prefill_chunk`, `max_pending_requests`, `pending_timeout_ms`. Confirmed
   present via `ninfer-serve --help` against the running image
   (`erfianugrah/ninfer:cuda13.1-sm120a-487f897`). None are set on the live
   `qwen38-ninfer.toml` preset yet - wiring only, no behavioural change
   until a preset sets them. Tests: `TestNinferOptionalQueueAndPrefillFlags`,
   `TestNinferQueueAndPrefillFlagsOmittedWhenUnset`.

2. **Built `WedgeWatchdog`** (`proxy-go/internal/proxy/wedge_watchdog.go`):
   on a `client_gone` note against the ninfer engine (server.go's
   `forwardTo`), wait a grace period (20s default), then probe
   `GET /health` (10s timeout default). If the probe fails, call
   `Scheduler.NoteUpstreamDead` - the same path a connection-level upstream
   death already uses, which flips scheduler state to idle so the *next*
   acquire does a full stop+remove+create respawn
   (`DockerOrchestrator.spawnCmd` -> `stopGPU` + `CreateAndStart`), not a
   bare restart. **Disabled by default** - `LLMC_NINFER_WEDGE_WATCHDOG=1` to
   enable. 6 unit tests (healthy/unresponsive/error-status probes, disabled
   no-op, nil-receiver safety, burst coalescing), race-clean.

## What is still unverified

- **Whether the respawn-only recovery path actually clears a real wedge.**
  The 2026-09-10 session's only confirmed recovery was a full stack restart
  (proxy + engine); it is not known whether restarting the proxy itself was
  necessary or just what the operator did at the time. `WedgeWatchdog`
  deliberately reuses the less invasive, already-tested respawn path
  (`NoteUpstreamDead`) rather than a new self-restart mechanism, on the
  reasoning that `stopGPU`'s stop+remove+create is a full container
  teardown (unlike the bare `docker restart` that was observed NOT to work)
  - but this reasoning has not been tested against a real wedge.
- **Whether `--pending-timeout-ms` reaches a request already inside context
  materialization** (the wedge state) or only bounds requests still queued
  before processing starts. `ninfer-serve --help` documents neither the
  queue-vs-processing boundary nor what happens to a request past the
  timeout. No local ninfer source to check; nothing found in upstream docs
  as of this session.
- **The watchdog's grace/probe thresholds (20s / 10s) are unvalidated
  guesses**, not measurements. They need tuning against a real wedge (false
  negative if too short relative to legitimate slow prefill = never fires;
  false positive if too aggressive = kills healthy-but-slow requests).

## Deliberately not done this session

Live reproduction of the wedge (build a long-context prompt against
`qwen38-ninfer`, abort the client mid-materialization via a short
`curl -m N`, and observe recovery with/without
`LLMC_NINFER_WEDGE_WATCHDOG=1` and `pending_timeout_ms` set) was scoped but
not run: it requires taking the shared GPU stack's `qwen38-ninfer` preset
out of service for other consumers (loops, other pi sessions) for the
duration, with a real risk of leaving it wedged if the mitigations don't
work and needing a manual `make clean` + `make deploy` to recover. Needs an
explicit go-ahead and a window when nothing else needs the GPU.

## Next steps

1. Get a go-ahead + quiet GPU window, then reproduce the wedge:
   - Baseline: reproduce without either mitigation, confirm the wedge and
     that respawn-only recovery (watchdog's path) does or does not clear it.
   - With `pending_timeout_ms` set: does the flag prevent the wedge from
     forming at all, or only bound something else?
   - With `LLMC_NINFER_WEDGE_WATCHDOG=1`: does detection fire, and does the
     resulting respawn actually restore normal throughput?
2. Set `prefill_chunk` on `qwen38-ninfer.toml` if reproduction shows it
   changes materialization duration (shorter chunks -> more SSE-heartbeat
   opportunities -> smaller wedge window) - currently just a hypothesis.
3. Tune grace/probe thresholds from what reproduction actually measures.
4. Separately noticed, out of scope for wedge mitigation: `anthropic.go`'s
   `/v1/messages` handler hardcodes `Services["llm"]` (`LlamaService`)
   instead of `Server.activeLLMService()` - unlike `server.go`'s
   OpenAI-compatible route, it cannot currently route to ninfer at all
   regardless of the active preset. Not touched here; flag before relying
   on Anthropic-format requests reaching `qwen38-ninfer`.
