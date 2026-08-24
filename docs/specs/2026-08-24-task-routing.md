# Task routing (auto aliases + resident-preference scheduling) Spec

**Goal:** let clients ask for a task class (`auto:code`, `auto:cheap`, `auto`) instead of a specific model, and make the scheduler prefer the resident model over a swap whenever the resident model is an acceptable answer.
**Workflow:** design-first (technically constrained: single GPU, one resident model, swap cost 5-60s; extends the proxy-go scheduler built under docs/specs/2026-08-19-model-proxy-v2.md).
**Non-goals:** prompt-content classification (no trained router, no embedding kNN, no prompt-length heuristics - research 2026-08-24 found these solve a dollar-cost problem this stack does not have and go near-random out-of-distribution); multi-model residency; queue priority/preemption beyond the greedy-serve rule in R4; changes to the llmc bench CLI; routing for comfyui/train modes.

## Context and motivation

Research basis (survey of tooling, academic literature, and practitioner reports, 2026-08-24):

- The dominant production mechanism (vLLM Semantic Router, NVIDIA blueprint, RouteLLM) is classifier-in-proxy priced in dollars-per-token, assuming all candidate models are simultaneously resident. None model the single-GPU eviction cost; that cost term is a gap in the literature, not something to copy.
- Practitioner consensus on single-GPU boxes: caller-declared intent beats inferred difficulty (the router never knows the task as well as the caller), and swap-aware scheduling ("serve the request that matches the loaded model first; one swap per burst, not N") is the policy that pays - jano's greedy reorder and ollama-agent-router's resident bonus are the two concrete implementations.
- The most-reported multi-model failure in this stack's own ecosystem is Open WebUI's task subsystem forcing reload churn between chat messages; the community fix people ask for is "just use the loaded model".

proxy-go already has the primitives: capability serve-in-place (R3 of the v2 spec), drain-before-swap, a single-goroutine scheduler loop, and per-request model resolution in `prepLLM`. This spec adds two thin layers on top: a declarative alias table (caller-declared intent) and a resident-preference rule at the one place requests currently stall (the swap-pending deferral).

Design physics: routing to a non-resident preset = container swap = 5-60s latency plus eviction of whatever is running (or a 422/409 when locked). A route therefore only pays when it picks the resident model; the alias table is ordered so that *when a swap is unavoidable*, the target is the operator's declared best answer for the task, not whatever the caller happened to name.

## Requirements

### R1: Route table
**Story:** As the operator, I want task aliases declared in one TOML file mapping an alias to an ordered chain of presets, so that routing policy is configuration, reviewed and reloaded like presets, with no code change per route.
**Acceptance criteria:**
- THE SYSTEM SHALL load routes from a routes file whose path is set by `LLMC_ROUTES_FILE` (default `/routes.toml`), bind-mounted beside the presets dir (the file MUST NOT live in the presets dir: `PresetStore.Reload` globs `*.toml` and a routes file there fails preset validation).
- THE SYSTEM SHALL treat each table entry as one alias: `chain = ["<preset>", ...]` ordered by preference, plus an optional `description` string.
- THE SYSTEM SHALL validate the routes file at load: unknown keys rejected, every chain entry SHALL resolve to a known preset (by name or model ID) at request time.
- WHEN the routes file is absent, THE SYSTEM SHALL run with zero aliases and no behavior change (existing `cap:` and exact-name routing untouched).
- WHEN the routes file is edited, THE SYSTEM SHALL pick up the change without a proxy restart (reload on request, same discipline as preset live-reload).
- IF an alias chain is empty (`chain = []`), THEN THE SYSTEM SHALL treat the alias as "whatever is resident": serve the current resident model regardless of identity.

### R2: Alias resolution at acquire
**Story:** As a client, I want to send `model: "auto"` or `model: "auto:<name>"` and have the proxy pick the cheapest correct thing - resident model if it is in the chain, the chain head otherwise - so that I never need to know what is loaded.
**Acceptance criteria:**
- WHEN a request's model field is `auto`, THE SYSTEM SHALL resolve the alias named `default`; WHEN it is `auto:<name>`, THE SYSTEM SHALL resolve the alias `<name>`.
- WHEN the resolved chain contains the resident preset, THE SYSTEM SHALL grant the request in place with no swap and rewrite the body model field to the resident model's ID (same mechanism as capability serve-in-place `ServeAs`).
- WHEN the resolved chain does not contain the resident preset, THE SYSTEM SHALL swap (subject to drain-before-swap) to the first chain entry and serve there.
- WHEN the chain is empty (R1 "whatever is resident") and LLM mode is active, THE SYSTEM SHALL grant on the resident model with the body model rewritten to its ID.
- WHEN the chain is empty and LLM mode is not active, THE SYSTEM SHALL return 503 with the existing switch-hint message (never swap on an empty chain - there is no declared target).
- IF the request names an unknown alias (`auto:<name>` with no table entry), THEN THE SYSTEM SHALL return 404 with `unknown route alias` in the error.
- WHEN a request names an exact preset or uses `cap:<name>`, THE SYSTEM SHALL keep current behavior unchanged (aliases are an opt-in namespace; `auto`-prefixed model names are reserved).
- THE SYSTEM SHALL apply the same alias resolution in the Anthropic `/v1/messages` shim as in the OpenAI `/v1/*` path.

### R3: Lock interplay
**Story:** As a loop owner holding a lock, I want alias requests to respect the pin exactly like exact-name requests, so that an `auto:` caller can never evict the locked model.
**Acceptance criteria:**
- WHILE a lock is active, WHEN an alias request's chain contains the locked preset, THE SYSTEM SHALL serve the request in place on the locked model (chain membership is the lock-safe form of "good enough").
- WHILE a lock is active, WHEN an alias request's chain does not contain the locked preset, THE SYSTEM SHALL return 422 `model_unavailable` naming the lock - identical to the exact-name refusal, never a swap.
- WHILE a lock is active, WHEN an empty-chain (`whatever is resident`) alias request arrives and the locked model is resident, THE SYSTEM SHALL serve it in place.
- THE SYSTEM SHALL refresh the lock TTL on alias grants in place exactly as it does for exact-name grants.

### R4: Greedy serve during a pending swap
**Story:** As a tenant whose model is being drained, I want short requests that the resident model can satisfy to still be served during the drain window, so that a burst of small calls around one big-model request costs one swap instead of a full stall for everyone.
**Acceptance criteria:**
- WHILE a swap is pending (draining or running), WHEN an incoming request is alias-routed or capability-routed AND the still-resident model satisfies it (chain membership / advertised capability), THE SYSTEM SHALL grant it in place instead of deferring it behind the swap.
- WHILE a swap is pending, WHEN an incoming request names a different exact model than the swap target, THE SYSTEM SHALL defer it exactly as today.
- THE SYSTEM SHALL bound greedy grants by the existing drain-grace deadline: WHEN the grace elapses, THE SYSTEM SHALL start the swap regardless of greedy in-flight counts (no new starvation class; `evDrainTimeout` already forces).
- WHEN a greedy-granted request is cut off by a forced swap, THE SYSTEM SHALL surface the same mid-stream error event as any other interrupted stream (no silent truncation).

### R5: Discovery and observability
**Story:** As a user in Open WebUI (or debugging from llmc), I want aliases to be visible and routing decisions to be in the access log, so that `auto:code` is selectable from the model picker and I can reconstruct why a request did or did not swap.
**Acceptance criteria:**
- THE SYSTEM SHALL list each alias in `GET /v1/models` as a pseudo-model entry with `meta.alias = true`, the resolved chain, and `meta.loaded = true` when the resident preset is in the chain.
- THE SYSTEM SHALL log one access-log line per alias resolution naming the alias, the decision (`in-place` / `swap to <preset>` / `rejected`), and the reason (resident-in-chain / chain-head / lock).
- THE SYSTEM SHALL include the alias name (not the rewritten model ID) in the request start/done log lines, alongside the resolved model.

## Design

### Route file

`routes.toml` at the repo root, bind-mounted to `/routes.toml`:

```toml
[default]                    # model: "auto"
description = "general chat: big model if loaded, small otherwise"
chain = ["qwen38", "qwen35-9b"]

[code]                       # model: "auto:code"
chain = ["qwen38", "qwen3-coder"]

[cheap]                      # model: "auto:cheap"
chain = ["qwen35-9b", "summarizer"]

[resident]                   # model: "auto:resident" - never swap
chain = []
```

Chain entries are preset names (filename stems); resolution to presets happens at request time against the live-reloaded store so a renamed preset fails loudly at the next request, not silently at load. Validation at load: unknown top-level keys, non-string chain entries, and alias names containing `:` are rejected; chain-vs-preset resolution is deferred to request time (presets reload independently).

### Resolution flow

`prepLLM` (and the shim's model resolution) gains one branch before preset lookup: model field starts with `auto` (`auto`, `auto:<name>`) -> look up the route, set `req.Route` (resolved chain + name) instead of `req.Preset`; unknown alias -> 404 at the handler, never reaching the scheduler. The `auto` prefix is reserved; presets cannot collide (alias lookup runs first, so a preset literally named `auto` is shadowed; the route-table validator warns, the preset still serves exact-name via its model ID).

Scheduler `acquireLLM` gains a `req.Route != nil` path that runs before the existing branches:

```
locked:
    locked preset in chain (or empty chain and locked is resident)
        -> grant in place, ServeAs = locked model ID
    else -> 422 model_unavailable (same message shape as exact-name)
unlocked:
    empty chain -> resident llm active ? grant + ServeAs : 503 switch-hint
    resident preset in chain -> grant in place, ServeAs = resident model ID
    else -> swapOrJoin(chain[0])
```

The grant path reuses `grant(ev, key, serveAs)` unchanged; the body model rewrite in `forward` already keys off `res.ServeAs`.

### Greedy serve (R4) in the scheduler

`handleAcquire`'s `s.pending != nil` branch currently defers anything that doesn't match the swap target. New order:

1. swap-target match -> join waiters (unchanged)
2. alias or capability request satisfiable by the still-resident preset -> grant in place (new)
3. everything else -> defer (unchanged)

Step 2 reuses `tryServeInPlace` (capability) and a chain-membership check (alias). Safety: greedy grants increment `inflight[drainKey]`, which delays `maybeStartSwap` - but `evDrainTimeout` fires at the grace deadline regardless and forces the swap, so a greedy burst can never starve the swap past `DrainGrace`. Deferred requests are still reprocessed post-swap and re-evaluate against the new resident, which is where a wrong-side request lands correctly.

### Failure modes considered

- **Chain head VRAM-infeasible**: the existing VRAM gate (`checkVRAMBody` in `forward`) runs on `req.Preset` at the handler, before the scheduler sees the request. Aliases bypass it because `req.Preset` is nil (resolution sets `req.Route` instead). The handler CANNOT set `req.Preset` to chain head to fix this - it doesn't know whether a swap is coming (that depends on what's resident, a scheduler decision), and an unconditional set would falsely 422 a request whose chain head is VRAM-infeasible but whose resident model (chain[1]) could have served in place. Fix: `SchedulerConfig` gains `VRAMLimitGB` / `VRAMReserveGB`, and `acquireLLM` runs `CheckVRAMBudget` on the chain head at swap-decision time (the point where the target is known). The 422 error shape is identical to the handler's. The existing handler-level gate stays for exact-name requests.
- **Stale chain (preset deleted)**: request-time resolution returns 404 `route alias X: chain entry Y not found`; no swap. Logged.
- **Alias during comfyui/train mode**: empty chain -> 503; non-empty chain -> swap to chain head (identical to an exact-name request for that preset).
- **Lock pinned on a model not in any chain**: alias traffic 422s; exact-name and capability traffic behave as today.

### Testing strategy

- `scheduler_test.go` (TDD, go test -race): alias resident-in-chain grants with ServeAs and no swap; resident-not-in-chain swaps to chain head; locked-in-chain grants; locked-not-in-chain 422; empty-chain resident grant; empty-chain non-llm-mode 503; greedy grant during pending drain; greedy grant does not survive past drain grace (forced swap still fires); deferred exact-name request unchanged.
- `server_test.go` / new `routes_test.go`: routes TOML load, unknown-key rejection, missing file = zero aliases, live reload, unknown alias 404, alias listed in /v1/models with `meta.alias`.
- Anthropic shim test: `auto:code` through /v1/messages resolves and rewrites.
- Smoke hurl (`make smoke-proxy-go`): `auto:resident` served in place while qwen38 loaded (assert no swap in log), unknown alias 404, alias under lock 422.
- Manual: Open WebUI model picker shows `auto:code`; selecting it serves chat without a swap when qwen38 is resident.

## Open questions

- **Default alias contents**: the `default` chain above is a placeholder; the real membership per alias is a tuning decision at rollout, not a design blocker. Default assumption: ship with `default`/`code`/`cheap`/`resident` and iterate.
- **Should exact-name requests also get greedy-serve during a drain when they name the resident model?** They already join via `pendingSwap.matches` only when naming the *target*; naming the draining model today defers. Default assumption: leave exact-name deferral unchanged (a caller who names a model precisely can wait); revisit if the smoke tests show real stalls.
- **llmc surface**: no CLI changes in scope; if `llmc models` showing aliases proves useful, add later as a display-only change.
