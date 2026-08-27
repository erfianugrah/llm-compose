# Idle auto-unload Spec

**Goal:** release the GPU after `LLMC_IDLE_UNLOAD_S` seconds of no activity in LLM mode, so a model no longer sits resident holding VRAM and power indefinitely after the last request.
**Workflow:** requirements-first (technically constrained: single-goroutine scheduler loop owns all mutable state; an idle timer is a new event in that loop, not a new goroutine of its own).
**Non-goals:** idle-unload for comfyui/train (those are explicit operator sessions with no "idle request" notion); partial unload / KV-cache eviction inside llama-server; suspend/hibernate; any change to the lock TTL semantics (a lock still pins the model; idle-unload is a form of eviction and stays subordinate to the lock).

## Context and motivation

Today a model stays resident until something explicitly evicts it: another swap, a mode switch, or a lock release followed by a swap. The lock TTL (`LLMC_LOCK_TTL_S`, 900s) only protects against *crashed* consumers - it expires a lock nobody is renewing; it does nothing about a healthy, unlocked model that simply has no traffic. After a session ends, the 22.5 GiB model sits there drawing power for hours. The liveness-recovery path (`handleUpstreamDead`) already established the exact pattern this feature needs: flip `st.Mode` to `idle` while *keeping* `st.Model`, so the next acquire respawns the same model instead of 502-looping. Idle auto-unload is the same move, triggered by a timer instead of an upstream death.

The scheduler loop is the right home: it already sees every acquire and every release, already tracks `inflight`, already knows lock state and pending swaps. An idle timer is one more event type, armed on the last release and disarmed on the next acquire.

## Requirements

### R1: Opt-in configuration
**Story:** As the operator, I want this off by default so that enabling it is a deliberate choice and existing deployments see zero behavior change.
**Acceptance criteria:**
- THE SYSTEM SHALL read `LLMC_IDLE_UNLOAD_S` as an integer number of seconds, default `0`.
- WHEN `LLMC_IDLE_UNLOAD_S` is `0` (or negative), THE SYSTEM SHALL NOT arm any idle timer - behavior is byte-identical to today.
- WHEN `LLMC_IDLE_UNLOAD_S` is positive, THE SYSTEM SHALL arm an idle timer per R2.

### R2: Arm the timer when the last request drains
**Story:** As the operator, I want the clock to start only once the GPU is actually quiet, so that an active burst is never cut off mid-conversation.
**Acceptance criteria:**
- WHEN a release event drops the last in-flight request to zero, AND mode is `llm`, AND no lock is active, AND no swap is pending, AND `LLMC_IDLE_UNLOAD_S > 0`, THEN THE SYSTEM SHALL arm an idle timer for `LLMC_IDLE_UNLOAD_S` seconds.
- THE SYSTEM SHALL compute "last in-flight" from the loop's own `inflight` map after the release decrement (the same place `maybeStartSwap` runs), so the arming decision is race-free.

### R3: Disarm on activity
**Story:** As the operator, I want any new traffic to cancel the countdown, so that a request arriving 1 second before the deadline does not evict the model out from under itself.
**Acceptance criteria:**
- WHEN an acquire event is granted for the resident model (or an in-place grant, or a passthrough grant), THE SYSTEM SHALL stop/cancel any armed idle timer and clear the reference.
- WHEN a lock is acquired, THE SYSTEM SHALL stop/cancel any armed idle timer (a lock now owns the residency decision).

### R4: Fire -> evict -> idle (keep model name)
**Story:** As the operator, I want the model container stopped and the state marked idle but the model name remembered, so the next request respawns the same preset instead of erroring.
**Acceptance criteria:**
- WHEN the idle timer fires, THE SYSTEM SHALL re-check on the loop that the scheduler is still idle (no in-flight, no lock, no pending swap); IF not, THEN THE SYSTEM SHALL take no action.
- WHEN the re-check passes, THE SYSTEM SHALL stop the llama-server container and set `st.Mode = "idle"` while preserving `st.Model`, then persist - identical to the `handleUpstreamDead` outcome.
- THE SYSTEM SHALL log the eviction with the model name and the configured idle duration.
- WHEN the next request arrives in idle mode, THE SYSTEM SHALL respawn the preserved model via the existing idle->swap path (`acquireLLM` already handles `st.Mode != "llm"` by `swapOrJoin` to the stored preset) - no new respawn code.

### R5: Lock and swap are higher priority
**Story:** As the operator, I want the idle feature to never fight the lock or a swap, so that the two existing residency guarantees stay intact.
**Acceptance criteria:**
- WHILE a lock is active, THE SYSTEM SHALL NOT arm the idle timer and SHALL NOT fire an unload.
- WHILE a swap is pending (draining or running), THE SYSTEM SHALL NOT arm the idle timer; an armed timer SHALL be cancelled when a swap starts.
- IF a lock is acquired between arming and firing, THEN THE SYSTEM SHALL cancel the timer before it fires (the R4 re-check is the last line of defense, not the only one).

### R6: Observable state
**Story:** As the operator, I want to see that the unload happened and when, so that a surprise "model gone" is never unexplained.
**Acceptance criteria:**
- THE SYSTEM SHALL reflect the resulting `idle` mode through the existing `/status` and `GET /mode` responses (no new fields required).
- THE SYSTEM SHALL include the configured `idle_unload_seconds` in `/status` so the operator can confirm the feature is armed with the value they set.
- THE SYSTEM SHALL log both arming and firing (and any cancellation) at the existing `logf` level, matching the current swap/lock log verbosity.

## Design

### Scheduler additions

`SchedulerConfig` gains `IdleUnload time.Duration` (wired from `LLMC_IDLE_UNLOAD_S` in `main.go`). The scheduler loop gains:

- a field `idleTimer *time.Timer` (nil when not armed), touched only on the loop goroutine - the same discipline as `pending`;
- an event `evIdleTimeout struct{}`;
- a `gen` guard so a fired timer whose request arrived first is ignored (the same stale-generation pattern `evDrainTimeout` already uses).

Arm/disarm:

- **Arm** in `handle(ev)` after an `evRelease` reaches zero inflight, mirroring `maybeStartSwap`'s placement. Guard: `s.cfg.IdleUnload > 0 && s.st.Mode == "llm" && s.st.Locked == "" && s.pending == nil && allInFlightZero()`.
- **Disarm** in `handleAcquire` (before granting) and in `handleLock` (on successful lock) - `s.stopIdleTimer()`.

Fire path:

```go
case *evIdleTimeout:
    if s.st.Mode != "llm" || s.st.Locked != "" || s.pending != nil || !allInFlightZero() {
        return // lost the race to a request / lock / swap
    }
    s.logf("idle timeout: evicting %q after %s of no activity", s.st.Model, s.cfg.IdleUnload)
    if err := s.orch.StopLlama(); err != nil {
        s.logf("idle unload failed: %v (staying resident)", err)
        return
    }
    s.st.Mode = "idle" // keep st.Model; next acquire respawns via the existing idle path
    s.persist()
```

### Orchestrator change

`stopGPU()` is already implemented but unexported and only invoked inside `spawn`. Expose it as a `StopLlama() error` (or `Stop(mode string) error`) on the `Orchestrator` interface so the scheduler can evict without a spawn. The fake orchestrator in `scheduler_test.go` gains the method. No new Docker logic - the stop-and-remove container flow already exists.

### Interaction with the swap path

`spawn` already calls `stopGPU()` before starting the replacement, so idle-unload reuses the same primitive and introduces no second teardown path. The only new thing is a *stop without a spawn*, which is the `handleUpstreamDead` move generalized.

### Failure modes considered

- **Race: request arrives between timer fire and stop**: the loop is single-goroutine, so the fire re-check and the stop happen atomically relative to any acquire - no acquire can interleave. The re-check exists to catch the case where the timer fired *before* an already-queued event was processed; in that case the acquire (or lock, or swap) event is simply processed first and the fire no-ops.
- **Stop fails (Docker error)**: log and stay resident; the model remains servable. Next release re-arms (the feature is self-healing per-burst).
- **Lock acquired after arm**: `handleLock` disarms; if it somehow raced, the fire re-check sees `s.st.Locked != ""` and no-ops.
- **ComfyUI/train resident, no llm traffic**: `s.st.Mode != "llm"`, so no timer is ever armed - the feature is LLM-only by the guard, not by special-casing.

### Testing strategy

- `scheduler_test.go` (TDD, `go test -race`): timer fires -> mode `idle` + model preserved + `StopLlama` called; activity (new acquire) before fire -> no unload; lock acquired before fire -> no unload; pending swap -> timer never armed; release-to-zero arms, release-not-to-zero does not; fire after a grant (lost race) -> no-op; `IdleUnload == 0` -> timer never armed.
- Orchestrator fake: `StopLlama` records invocation; assertion covers "stop called exactly once per unload".
- Integration: after unload, a follow-up request respawns the model and serves (existing idle->swap path), asserting no 502.
- Smoke (`make smoke-proxy-go`): set `LLMC_IDLE_UNLOAD_S=1`, make a request, wait, assert `GET /status` shows `mode: idle` with the model name retained; then a second request respawns and succeeds.

## Open questions

- **Default value**: ship `0` (disabled) so nothing changes for anyone who does not opt in. If the feature proves itself, promote a default (e.g. 600s) in a later change - a behavior change, so it is a deliberate bump, not part of this spec.
- **Should comfyui/train ever idle-unload?** Out of scope here (no "idle request" concept - a training job or a generation session is an explicit, long-lived activity). Revisit only if VRAM contention during mixed workloads becomes a real complaint.
- **`stopGPU` scope**: the exposed `StopLlama` stops only the llm-labelled container, not comfyui/train. If a future feature needs cross-mode teardown, generalize to `Stop(mode)` then; today only llm unloads.
