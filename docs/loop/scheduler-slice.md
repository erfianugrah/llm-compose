# Slice 1: scheduler.go task-routing logic

`routes.go`, the `AcquireRequest.Route` field, and the `SchedulerConfig`
VRAMLimitGB/VRAMReserveGB fields are ALREADY DONE and committed. Do NOT touch
routes.go. Do NOT re-add the Route field or the VRAM fields - they are already
present.

Remaining scheduler.go work ONLY. Make these exact edits:

## 1. SchedulerConfig VRAM fields: ALREADY PRESENT - skip

VRAMLimitGB and VRAMReserveGB are already on SchedulerConfig. Do not re-add them.

## 2. containsPreset helper

Add this function anywhere near the other acquire helpers:

```go
// containsPreset reports whether the resolved chain contains presetName
// (by stem or model ID).
func containsPreset(chain []*Preset, presetName string) bool {
	for _, p := range chain {
		if p.Name == presetName || p.ModelID() == presetName {
			return true
		}
	}
	return false
}
```

## 3. Wire the route path into acquireLLM

In `func (s *Scheduler) acquireLLM(...)`, at the TOP (before `p :=
ev.req.Preset`), insert:

```go
	if ev.req.Route != nil {
		s.acquireRoute(ev, now)
		return
	}
```

## 4. acquireRoute

Add this function. It handles the alias request (req.Route != nil):

```go
// acquireRoute handles an alias request (req.Route != nil). Resolution order:
// locked -> serve only if the locked preset is in the chain (or empty chain
// and the locked model is resident); unlocked -> empty chain serves whatever
// is resident, resident-in-chain serves in place, otherwise swap to chain head.
func (s *Scheduler) acquireRoute(ev *evAcquire, now time.Time) {
	route := ev.req.Route
	locked := s.st.Locked != ""

	chain := make([]*Preset, 0, len(route.Chain))
	for _, entry := range route.Chain {
		p := s.presets.ByName(entry)
		if p == nil {
			s.reject(ev, 404, "route_chain",
				fmt.Sprintf("route alias %q: chain entry %q not found", route.Name, entry))
			return
		}
		chain = append(chain, p)
	}

	if locked {
		if len(chain) == 0 {
			if s.st.Mode == "llm" && s.st.Model == s.st.Locked {
				s.refreshLockExpiry(now)
				if rp := s.residentPreset(); rp != nil {
					s.grant(ev, s.residentKey(), rp.ModelID())
				} else {
					s.grant(ev, s.residentKey(), "")
				}
				return
			}
			s.reject(ev, 422, "model_unavailable",
				fmt.Sprintf("model lock active on %q: alias %q has no resident target",
					s.st.Locked, route.Name))
			return
		}
		if containsPreset(chain, s.st.Locked) {
			s.refreshLockExpiry(now)
			if s.st.Mode == "llm" && s.st.Model == s.st.Locked {
				if rp := s.residentPreset(); rp != nil {
					s.grant(ev, s.st.Locked, rp.ModelID())
				} else {
					s.grant(ev, s.st.Locked, "")
				}
			} else {
				s.swapOrJoin(ev, "llm", s.presets.ByName(s.st.Locked))
			}
			return
		}
		s.reject(ev, 422, "model_unavailable",
			fmt.Sprintf("model lock active on %q: alias %q chain does not include it (POST /mode {\"lock\": false} to unlock)",
				s.st.Locked, route.Name))
		return
	}

	if len(chain) == 0 {
		if s.st.Mode == "llm" {
			if rp := s.residentPreset(); rp != nil {
				s.logf("route %q: empty chain - serving resident %q", route.Name, rp.Name)
				s.grant(ev, rp.Name, rp.ModelID())
			} else {
				s.reject(ev, 503, "server_error",
					fmt.Sprintf("alias %q: no resident model to serve", route.Name))
			}
		} else {
			s.reject(ev, 503, "service_inactive",
				fmt.Sprintf("alias %q has no declared target; switch to llm mode first (POST /mode {\"mode\":\"llm\", ...})", route.Name))
		}
		return
	}

	resident := s.residentPreset()
	if resident != nil && containsPreset(chain, resident.Name) {
		s.logf("route %q: resident %q in chain - serve in place", route.Name, resident.Name)
		s.grant(ev, resident.Name, resident.ModelID())
		return
	}

	head := chain[0]
	if s.cfg.VRAMLimitGB > 0 {
		if ok, msg := CheckVRAMBudget(head, s.cfg.VRAMLimitGB, s.cfg.VRAMReserveGB); !ok {
			s.reject(ev, 422, "model_unavailable", msg)
			return
		}
	}
	s.logf("route %q: swapping to chain head %q", route.Name, head.Name)
	s.swapOrJoin(ev, "llm", head)
}
```

## 5. greedyServe

```go
// greedyServe grants an alias/capability request in place during a pending
// swap when the still-resident (draining) model can satisfy it. Only fires
// while the swap is still draining (started == false). Returns true if granted.
func (s *Scheduler) greedyServe(ev *evAcquire) bool {
	if s.pending == nil || s.pending.started {
		return false
	}
	resident := s.residentPreset()
	if resident == nil {
		return false
	}
	if ev.req.Route != nil {
		for _, entry := range ev.req.Route.Chain {
			if entry == resident.Name || entry == resident.ModelID() {
				s.logf("greedy serve: route %q served by draining resident %q", ev.req.Route.Name, resident.Name)
				s.grant(ev, resident.Name, resident.ModelID())
				return true
			}
		}
		return false
	}
	if ev.req.Capability != "" && resident.HasCapability(ev.req.Capability) {
		s.logf("greedy serve: capability %q served by draining resident %q", ev.req.Capability, resident.Name)
		s.grant(ev, resident.Name, resident.ModelID())
		return true
	}
	return false
}
```

## 6. handleAcquire: greedy-serve branch

In `handleAcquire`, find the `if s.pending != nil { ... }` block. Replace it with
(this adds the greedyServe branch in the middle):

```go
	if s.pending != nil {
		// A swap is draining or running. Join if this request targets the
		// same swap; serve in place if the still-resident model can satisfy
		// an alias/capability request (R4); defer otherwise.
		if s.pending.matches(ev.req) {
			s.pending.waiters = append(s.pending.waiters, ev)
		} else if s.greedyServe(ev) {
			// granted in place
		} else {
			s.deferred = append(s.deferred, ev)
		}
		return
	}
```

## 7. handleSwapDone: alias waiters get ServeAs

In `handleSwapDone`, find the `for _, w := range p.waiters` loop. Replace its
grant so alias waiters get the head model ID in ServeAs. Also DELETE the
standalone `serveAs := ""` declaration that precedes the loop:

```go
	for _, w := range p.waiters {
		s.grantedWaiter[w] = key
		s.inflight[key]++
		serveAs := ""
		if w.req.Route != nil && p.preset != nil {
			serveAs = p.preset.ModelID()
		}
		w.reply <- AcqResult{OK: true, Granted: true, Key: key, ServeAs: serveAs}
	}
```

## Verify

After every edit run: `cd proxy-go && go build ./...`

The go-test contract is NOT green yet (server.go/anthropic.go/main.go wiring is
the next slice). Your job is ONLY: scheduler.go compiles + go build passes +
the acquireRoute/greedyServe/containsPreset functions exist.
