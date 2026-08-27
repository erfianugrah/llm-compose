# Task Routing (auto aliases + resident-preference) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or implement this plan task-by-task in-session. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let clients ask for a task class (`auto`, `auto:code`, `auto:cheap`) instead of a specific model, and make the scheduler prefer the resident model over a swap whenever the resident model is in the alias chain.

**Architecture:** A `routes.toml` alias table (declarative, live-reloaded) is resolved in the HTTP layer to a `*Route` (ordered preset chain), attached to `AcquireRequest`, and handled by a new `acquireRoute` path in the single-goroutine scheduler. Resident-in-chain grants serve in place (reuse `grant` + `ServeAs`); otherwise the scheduler swaps to the chain head (with a VRAM check moved into the scheduler, since the handler cannot set `req.Preset` without breaking the resident-in-place case). Greedy-serve during a pending drain is one new branch in `handleAcquire`.

**Tech Stack:** Go (stdlib + BurntSushi/toml), the proxy-go scheduler, hurl smoke tests.

**Spec:** `docs/specs/2026-08-24-task-routing.md` (R1-R5). Every task names the requirement IDs it satisfies.

**Verification command (whole suite):** `make test-proxy-go` = `cd proxy-go && go test ./... -race -count=1`.

**Loop note:** `.pi/harness.json` currently targets the `context` bench task, not this feature. This plan keeps per-task commit steps (human/inline execution). If this plan will run under the self-correcting loop instead, drop the commit steps and write a new harness.json first.

---

### Task 1: routes.go - Route type + RouteStore loader

**Satisfies:** R1 (load, validation, live-reload, absent-file=zero-aliases), R2 (alias-name parsing)

**Files:**
- Create: `proxy-go/internal/proxy/routes.go`
- Test: `proxy-go/internal/proxy/routes_test.go`

- [ ] **Step 1: Write the failing test**

```go
package proxy

import (
	"os"
	"path/filepath"
	"testing"
)

func writeRoutes(t *testing.T, content string) *RouteStore {
	t.Helper()
	path := filepath.Join(t.TempDir(), "routes.toml")
	if content != "" {
		if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
			t.Fatalf("writing routes: %v", err)
		}
	}
	rs, err := NewRouteStore(path)
	if err != nil {
		t.Fatalf("NewRouteStore: %v", err)
	}
	return rs
}

func TestRouteStoreLoad(t *testing.T) {
	rs := writeRoutes(t, `
[default]
description = "general"
chain = ["a", "b"]

[code]
chain = ["c"]
`)
	if r := rs.ByName("default"); r == nil || len(r.Chain) != 2 || r.Chain[0] != "a" || r.Chain[1] != "b" || r.Description != "general" {
		t.Fatalf("default route wrong: %+v", r)
	}
	if r := rs.ByName("code"); r == nil || r.Chain[0] != "c" {
		t.Fatalf("code route wrong: %+v", r)
	}
	if rs.ByName("nope") != nil {
		t.Fatal("unknown alias must be nil")
	}
	if len(rs.All()) != 2 {
		t.Fatalf("expected 2 routes, got %d", len(rs.All()))
	}
}

func TestRouteStoreMissingFile(t *testing.T) {
	rs := writeRoutes(t, "")
	if rs == nil {
		t.Fatal("missing routes file must still produce a store")
	}
	if len(rs.All()) != 0 {
		t.Fatalf("missing file must be zero aliases, got %d", len(rs.All()))
	}
}

func TestRouteStoreRejectsUnknownKey(t *testing.T) {
	path := filepath.Join(t.TempDir(), "routes.toml")
	os.WriteFile(path, []byte("[x]\nchain = [\"a\"]\nbogus = 1\n"), 0o644)
	if _, err := NewRouteStore(path); err == nil {
		t.Fatal("unknown key must be rejected")
	}
}

func TestRouteStoreRejectsColonInAlias(t *testing.T) {
	path := filepath.Join(t.TempDir(), "routes.toml")
	os.WriteFile(path, []byte("[\"a:b\"]\nchain = [\"a\"]\n"), 0o644)
	if _, err := NewRouteStore(path); err == nil {
		t.Fatal("alias containing ':' must be rejected")
	}
}

func TestRouteStoreRejectsNonStringChain(t *testing.T) {
	path := filepath.Join(t.TempDir(), "routes.toml")
	os.WriteFile(path, []byte("[x]\nchain = [1]\n"), 0o644)
	if _, err := NewRouteStore(path); err == nil {
		t.Fatal("non-string chain entry must be rejected")
	}
}

func TestRouteStoreLiveReload(t *testing.T) {
	path := filepath.Join(t.TempDir(), "routes.toml")
	os.WriteFile(path, []byte("[a]\nchain = [\"x\"]\n"), 0o644)
	rs, err := NewRouteStore(path)
	if err != nil {
		t.Fatalf("NewRouteStore: %v", err)
	}
	os.WriteFile(path, []byte("[b]\nchain = [\"y\"]\n"), 0o644)
	if err := rs.Reload(); err != nil {
		t.Fatalf("Reload: %v", err)
	}
	if rs.ByName("a") != nil || rs.ByName("b") == nil {
		t.Fatalf("reload must replace the table: %+v", rs.All())
	}
}

func TestRouteAliasName(t *testing.T) {
	cases := map[string]string{
		"auto":         "default",
		"auto:code":    "code",
		"auto:cheap":   "cheap",
		"qwen38":       "",
		"autocomplete": "",
		"cap:vision":   "",
	}
	for in, want := range cases {
		if got := routeAliasName(in); got != want {
			t.Fatalf("routeAliasName(%q) = %q, want %q", in, got, want)
		}
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd proxy-go && go test ./internal/proxy -run 'TestRoute' -count=1`
Expected: FAIL - `undefined: NewRouteStore`, `undefined: routeAliasName`.

- [ ] **Step 3: Write the implementation**

```go
// Task-routing alias table: maps an alias name ("default", "code", ...) to
// an ordered chain of presets. Mirrors the preset loader's strictness
// (unknown keys rejected) but chain-vs-preset resolution is deferred to
// request time (presets reload independently). See
// docs/specs/2026-08-24-task-routing.md.
package proxy

import (
	"fmt"
	"os"
	"strings"

	"github.com/BurntSushi/toml"
)

// RouteError is raised on any routes-file schema violation.
type RouteError struct{ Msg string }

func (e *RouteError) Error() string { return e.Msg }

func routeErr(format string, args ...any) *RouteError {
	return &RouteError{Msg: fmt.Sprintf(format, args...)}
}

// Route is one alias: an ordered chain of preset names/model IDs.
type Route struct {
	Name        string   // alias name (table key)
	Description string   // optional one-line summary
	Chain       []string // ordered by preference; empty = "whatever is resident"
}

// parseRoutes decodes a routes.toml document into a map of alias -> Route.
func parseRoutes(data []byte) (map[string]*Route, error) {
	var raw map[string]map[string]any
	if _, err := toml.Decode(string(data), &raw); err != nil {
		return nil, routeErr("routes: invalid TOML: %v", err)
	}
	out := map[string]*Route{}
	for name, table := range raw {
		if strings.Contains(name, ":") {
			return nil, routeErr("routes: alias %q must not contain ':'", name)
		}
		r := &Route{Name: name}
		for k, v := range table {
			switch k {
			case "description":
				d, ok := v.(string)
				if !ok {
					return nil, routeErr("routes: alias %q: description must be a string", name)
				}
				r.Description = d
			case "chain":
				arr, ok := v.([]any)
				if !ok {
					return nil, routeErr("routes: alias %q: chain must be an array of strings", name)
				}
				for _, e := range arr {
					s, ok := e.(string)
					if !ok {
						return nil, routeErr("routes: alias %q: chain entries must be strings", name)
					}
					r.Chain = append(r.Chain, s)
				}
			default:
				return nil, routeErr("routes: alias %q: unknown key %q", name, k)
			}
		}
		out[name] = r
	}
	return out, nil
}

// RouteStore is the routes registry. Reload() rescans the file so an edit
// is picked up without a proxy restart (parity with preset live-reload).
type RouteStore struct {
	Path   string
	routes map[string]*Route
}

func NewRouteStore(path string) (*RouteStore, error) {
	s := &RouteStore{Path: path, routes: map[string]*Route{}}
	if err := s.Reload(); err != nil {
		return nil, err
	}
	return s, nil
}

// Reload reads and reparses the routes file. A missing file yields zero
// aliases (no error); a malformed file returns an error and keeps the last
// good snapshot (or empty on first load).
func (s *RouteStore) Reload() error {
	data, err := os.ReadFile(s.Path)
	if os.IsNotExist(err) {
		s.routes = map[string]*Route{}
		return nil
	}
	if err != nil {
		return routeErr("routes: %v", err)
	}
	routes, err := parseRoutes(data)
	if err != nil {
		return err
	}
	s.routes = routes
	return nil
}

func (s *RouteStore) ByName(name string) *Route { return s.routes[name] }

func (s *RouteStore) All() map[string]*Route {
	out := make(map[string]*Route, len(s.routes))
	for k, v := range s.routes {
		out[k] = v
	}
	return out
}

// routeAliasName maps a model field to an alias name: "auto" -> "default",
// "auto:<name>" -> "<name>", anything else -> "" (not an alias).
func routeAliasName(model string) string {
	if model == "auto" {
		return "default"
	}
	if strings.HasPrefix(model, "auto:") {
		return strings.TrimPrefix(model, "auto:")
	}
	return ""
}

// aliasError marks an unknown alias (handler maps it to 404).
type aliasError struct{ name string }

func (e *aliasError) Error() string {
	return fmt.Sprintf("unknown route alias %q", e.name)
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd proxy-go && go test ./internal/proxy -run 'TestRoute' -count=1`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add proxy-go/internal/proxy/routes.go proxy-go/internal/proxy/routes_test.go
git commit -m "feat(proxy): route table loader (auto aliases, strict TOML)"
```

---

### Task 2: scheduler struct fields

**Satisfies:** R2 (Route on AcquireRequest), R2 failure-mode (VRAM in SchedulerConfig)

**Files:**
- Modify: `proxy-go/internal/proxy/scheduler.go` (SchedulerConfig + AcquireRequest)

- [ ] **Step 1: Add the fields**

In `scheduler.go`, extend the two structs (exact replacement):

```go
type SchedulerConfig struct {
	StatePath     string
	AssetsDir     string
	DrainGrace    time.Duration
	LockTTL       time.Duration
	HealthTimeout time.Duration
	VRAMLimitGB   float64 // chain-head VRAM gate at swap-decision time (R2 failure mode)
	VRAMReserveGB float64
}

// AcquireRequest is one request for GPU capacity.
type AcquireRequest struct {
	Mode       string  // llm | comfyui | train
	Preset     *Preset // resolved llm target; nil = passthrough/unknown/current
	RawModel   string  // model field as requested (for passthrough + errors)
	Capability string  // X-LLM-Capability / cap:<name> hint
	Route      *Route  // resolved alias (auto / auto:<name>); nil = not an alias
}
```

- [ ] **Step 2: Build to verify it compiles**

Run: `cd proxy-go && go build ./...`
Expected: success (fields are additive; nothing references them yet).

- [ ] **Step 3: Commit**

```bash
git add proxy-go/internal/proxy/scheduler.go
git commit -m "feat(proxy): AcquireRequest.Route + SchedulerConfig VRAM fields"
```

---

### Task 3: acquireRoute + greedyServe in the scheduler

**Satisfies:** R2 (scheduler half), R3 (lock interplay), R4 (greedy serve)

**Files:**
- Modify: `proxy-go/internal/proxy/scheduler.go`

- [ ] **Step 1: Wire the route path into acquireLLM**

At the top of `func (s *Scheduler) acquireLLM(ev *evAcquire, now time.Time)`, before the `p := ev.req.Preset` line, insert:

```go
	if ev.req.Route != nil {
		s.acquireRoute(ev, now)
		return
	}
```

- [ ] **Step 2: Add the route branch to handleAcquire (greedy serve)**

Replace the `if s.pending != nil { ... }` block in `handleAcquire` with:

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

- [ ] **Step 3: Add acquireRoute, greedyServe, and containsPreset**

Append after `resolveCapability` (near the other acquire helpers):

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

// acquireRoute handles an alias request (req.Route != nil). Resolution order:
// locked -> serve only if the locked preset is in the chain (or empty chain
// and the locked model is resident); unlocked -> empty chain serves whatever
// is resident, resident-in-chain serves in place, otherwise swap to chain head.
func (s *Scheduler) acquireRoute(ev *evAcquire, now time.Time) {
	route := ev.req.Route
	locked := s.st.Locked != ""

	// Resolve the chain to presets once; a stale entry is a 404.
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
			// "whatever is resident": valid only when the locked model is resident.
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

	// Unlocked.
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

// greedyServe grants an alias/capability request in place during a pending
// swap when the still-resident (draining) model can satisfy it. Only fires
// while the swap is still draining (started == false): once the swap starts
// the resident container is being torn down. Returns true if granted.
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

- [ ] **Step 4: Rewrite the swap grant so alias waiters get ServeAs**

In `handleSwapDone`, inside the `for _, w := range p.waiters` loop, replace the grant call so alias waiters get the head model ID rewritten into their body:

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

(The existing `serveAs := ""` variable declared earlier in the function is now unused - delete that declaration line that precedes the loop.)

- [ ] **Step 5: Build + run the whole suite to confirm no regression**

Run: `cd proxy-go && go build ./... && go test ./... -race -count=1`
Expected: PASS (existing tests unchanged; new paths not exercised yet).

- [ ] **Step 6: Commit**

```bash
git add proxy-go/internal/proxy/scheduler.go
git commit -m "feat(proxy): alias routing + greedy serve in scheduler (R2-R4)"
```

---

### Task 4: scheduler tests for R2/R3 (alias resolution + lock)

**Satisfies:** R2 (resident-in-chain, swap-to-head, empty-chain, stale-chain 404), R3 (locked-in-chain, locked-not-in-chain)

**Files:**
- Modify: `proxy-go/internal/proxy/scheduler_test.go`

- [ ] **Step 1: Write the tests**

Append a new section (the fixtures `tomlA`/`tomlB` already exist; `a` = `alpha.gguf`, `b` = `beta.gguf`):

```go
// ── 17. task routing (alias) ─────────────────────────────────────────

func TestAliasResidentInChainServesInPlace(t *testing.T) {
	e := startSched(t, newFakeOrch("llm"), newTestStore(t, map[string]string{"a": tomlA, "b": tomlB}),
		&State{Mode: "llm", Model: "a"}, SchedulerConfig{})
	route := &Route{Name: "code", Chain: []string{"b", "a"}}

	res := e.s.Acquire(ctx(), AcquireRequest{Mode: "llm", Route: route, RawModel: "auto:code"})
	if !res.OK || !res.Granted || res.Key != "a" || res.ServeAs != "alpha" {
		t.Fatalf("alias resident-in-chain: %+v", res)
	}
	defer e.s.Release(res.Key)
	if got := e.orch.llamaCount(); got != 0 {
		t.Fatalf("resident-in-chain must not spawn, got %d", got)
	}
}

func TestAliasSwapToChainHead(t *testing.T) {
	e := startSched(t, newFakeOrch("llm"), newTestStore(t, map[string]string{"a": tomlA, "b": tomlB}),
		&State{Mode: "llm", Model: "a"}, SchedulerConfig{})
	route := &Route{Name: "code", Chain: []string{"b"}} // resident "a" not in chain

	res := e.s.Acquire(ctx(), AcquireRequest{Mode: "llm", Route: route, RawModel: "auto:code"})
	if !res.OK || !res.Granted || res.Key != "b" || res.ServeAs != "beta" {
		t.Fatalf("alias swap-to-head: %+v", res)
	}
	defer e.s.Release(res.Key)
	if got := e.orch.llamaNames(); len(got) != 1 || got[0] != "b" {
		t.Fatalf("expected SpawnLlama(b), got %v", got)
	}
}

func TestAliasEmptyChainServesResident(t *testing.T) {
	e := startSched(t, newFakeOrch("llm"), newTestStore(t, map[string]string{"a": tomlA, "b": tomlB}),
		&State{Mode: "llm", Model: "a"}, SchedulerConfig{})
	route := &Route{Name: "resident", Chain: []string{}}

	res := e.s.Acquire(ctx(), AcquireRequest{Mode: "llm", Route: route, RawModel: "auto:resident"})
	if !res.OK || !res.Granted || res.Key != "a" || res.ServeAs != "alpha" {
		t.Fatalf("empty-chain resident: %+v", res)
	}
	e.s.Release(res.Key)
	if got := e.orch.llamaCount(); got != 0 {
		t.Fatalf("empty-chain must not spawn, got %d", got)
	}
}

func TestAliasEmptyChainNonLLM503(t *testing.T) {
	e := startSched(t, newFakeOrch("idle"), newTestStore(t, map[string]string{"a": tomlA, "b": tomlB}),
		nil, SchedulerConfig{})
	route := &Route{Name: "resident", Chain: []string{}}

	res := e.s.Acquire(ctx(), AcquireRequest{Mode: "llm", Route: route, RawModel: "auto:resident"})
	if res.OK || res.Status != 503 {
		t.Fatalf("empty-chain non-llm: want 503, got %+v", res)
	}
}

func TestAliasStaleChainEntry404(t *testing.T) {
	e := startSched(t, newFakeOrch("llm"), newTestStore(t, map[string]string{"a": tomlA, "b": tomlB}),
		&State{Mode: "llm", Model: "a"}, SchedulerConfig{})
	route := &Route{Name: "code", Chain: []string{"deleted"}}

	res := e.s.Acquire(ctx(), AcquireRequest{Mode: "llm", Route: route, RawModel: "auto:code"})
	if res.OK || res.Status != 404 || !strings.Contains(res.ErrMsg, "not found") {
		t.Fatalf("stale chain entry: want 404, got %+v", res)
	}
}

func TestAliasLockedInChainServesInPlace(t *testing.T) {
	e := startSched(t, newFakeOrch("llm"), newTestStore(t, map[string]string{"a": tomlA, "b": tomlB}),
		&State{Mode: "llm", Model: "a"}, SchedulerConfig{})
	if r := e.s.Lock("a", false, "A", false); r.Status != 200 {
		t.Fatalf("lock: %d %#v", r.Status, r.Body)
	}
	t.Cleanup(func() { e.s.Unlock("A") })
	route := &Route{Name: "code", Chain: []string{"b", "a"}}

	res := e.s.Acquire(ctx(), AcquireRequest{Mode: "llm", Route: route, RawModel: "auto:code"})
	if !res.OK || !res.Granted || res.Key != "a" || res.ServeAs != "alpha" {
		t.Fatalf("locked-in-chain: %+v", res)
	}
	e.s.Release(res.Key)
	if got := e.orch.llamaCount(); got != 0 {
		t.Fatalf("locked-in-chain must not spawn, got %d", got)
	}
}

func TestAliasLockedNotInChain422(t *testing.T) {
	e := startSched(t, newFakeOrch("llm"), newTestStore(t, map[string]string{"a": tomlA, "b": tomlB}),
		&State{Mode: "llm", Model: "a"}, SchedulerConfig{})
	if r := e.s.Lock("a", false, "A", false); r.Status != 200 {
		t.Fatalf("lock: %d %#v", r.Status, r.Body)
	}
	t.Cleanup(func() { e.s.Unlock("A") })
	route := &Route{Name: "code", Chain: []string{"b"}}

	res := e.s.Acquire(ctx(), AcquireRequest{Mode: "llm", Route: route, RawModel: "auto:code"})
	if res.OK || res.Status != 422 || res.ErrType != "model_unavailable" {
		t.Fatalf("locked-not-in-chain: want 422, got %+v", res)
	}
}
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `cd proxy-go && go test ./internal/proxy -run 'TestAlias' -race -count=1`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add proxy-go/internal/proxy/scheduler_test.go
git commit -m "test(proxy): alias routing + lock interplay (R2/R3)"
```

---

### Task 5: scheduler tests for R4 (greedy serve)

**Satisfies:** R4

**Files:**
- Modify: `proxy-go/internal/proxy/scheduler_test.go`

- [ ] **Step 1: Write the tests**

```go
// ── 18. greedy serve during drain (R4) ───────────────────────────────

func TestGreedyServeDuringDrain(t *testing.T) {
	e := startSched(t, newFakeOrch("llm"), newTestStore(t, map[string]string{"a": tomlA, "b": tomlB}),
		&State{Mode: "llm", Model: "a"}, SchedulerConfig{})
	pA, pB := e.s.presets.ByName("a"), e.s.presets.ByName("b")

	// Hold a in-flight; a swap to b starts draining.
	resA := e.s.Acquire(ctx(), AcquireRequest{Mode: "llm", Preset: pA})
	if !resA.OK {
		t.Fatalf("acquire a: %+v", resA)
	}
	resCh := make(chan AcqResult, 1)
	go func() { resCh <- e.s.Acquire(ctx(), AcquireRequest{Mode: "llm", Preset: pB}) }()
	waitFor(t, 2*time.Second, "pending swap to register", func() bool {
		return e.s.Status().Switching
	})

	// An alias whose chain contains the draining resident is served in place.
	route := &Route{Name: "code", Chain: []string{"b", "a"}}
	gRes := e.s.Acquire(ctx(), AcquireRequest{Mode: "llm", Route: route, RawModel: "auto:code"})
	if !gRes.OK || !gRes.Granted || gRes.Key != "a" || gRes.ServeAs != "alpha" {
		t.Fatalf("greedy serve: %+v", gRes)
	}
	e.s.Release(gRes.Key)
	if got := e.orch.llamaCount(); got != 0 {
		t.Fatalf("greedy serve must not spawn while draining, got %d", got)
	}

	// Release A; the swap proceeds.
	e.s.Release(resA.Key)
	select {
	case res := <-resCh:
		if !res.OK || res.Key != "b" {
			t.Fatalf("swap acquire: %+v", res)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("swap to b did not grant")
	}
}

func TestGreedyServeDoesNotStarveSwap(t *testing.T) {
	cfg := SchedulerConfig{DrainGrace: 300 * time.Millisecond}
	e := startSched(t, newFakeOrch("llm"), newTestStore(t, map[string]string{"a": tomlA, "b": tomlB}),
		&State{Mode: "llm", Model: "a"}, cfg)
	pA, pB := e.s.presets.ByName("a"), e.s.presets.ByName("b")

	resA := e.s.Acquire(ctx(), AcquireRequest{Mode: "llm", Preset: pA})
	if !resA.OK {
		t.Fatalf("acquire a: %+v", resA)
	}
	resCh := make(chan AcqResult, 1)
	go func() { resCh <- e.s.Acquire(ctx(), AcquireRequest{Mode: "llm", Preset: pB}) }()
	waitFor(t, 2*time.Second, "pending swap to register", func() bool {
		return e.s.Status().Switching
	})

	// A greedy grant that never releases cannot hold the swap past the grace
	// deadline: evDrainTimeout forces the swap.
	route := &Route{Name: "code", Chain: []string{"b", "a"}}
	gRes := e.s.Acquire(ctx(), AcquireRequest{Mode: "llm", Route: route, RawModel: "auto:code"})
	if !gRes.OK {
		t.Fatalf("greedy grant: %+v", gRes)
	}
	// Never release gRes; release A so only the greedy grant holds the drain.
	e.s.Release(resA.Key)
	select {
	case res := <-resCh:
		if !res.OK || res.Key != "b" {
			t.Fatalf("swap after grace: %+v", res)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("grace timeout did not force the swap past the greedy grant")
	}
	e.s.Release(gRes.Key)
}
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `cd proxy-go && go test ./internal/proxy -run 'TestGreedy' -race -count=1`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add proxy-go/internal/proxy/scheduler_test.go
git commit -m "test(proxy): greedy serve during drain (R4)"
```

---

### Task 6: server.go - resolveModel, prepLLM, forward 404, /v1/models listing

**Satisfies:** R2 (handler resolution + 404), R5 (discovery + listing)

**Files:**
- Modify: `proxy-go/internal/proxy/server.go`

- [ ] **Step 1: Add the routes field to Server and NewServer**

Locate `type Server struct { ... }` and `func NewServer(...)`. Add a `routes *RouteStore` field and a `routes *RouteStore` parameter. If the struct fields are not visible in one read, first `grep -n "type Server struct\|func NewServer" proxy-go/internal/proxy/server.go`, then edit.

```go
type Server struct {
	sched   *Scheduler
	presets *PresetStore
	routes  *RouteStore
	cfg     ServerConfig
	logf    func(string, ...any)
}

func NewServer(sched *Scheduler, presets *PresetStore, routes *RouteStore, cfg ServerConfig, logf func(string, ...any)) *Server {
	if logf == nil {
		logf = func(string, ...any) {}
	}
	return &Server{sched: sched, presets: presets, routes: routes, cfg: cfg, logf: logf}
}
```

(Confirm the exact existing field names with `grep` first - the struct may differ; keep every existing field and only add `routes`.)

- [ ] **Step 2: Add resolveModel**

Append a helper near `prepLLM`:

```go
// resolveModel turns the raw model field into (preset, route, err).
// err is non-nil (an *aliasError) only for an unknown alias.
func (s *Server) resolveModel(model string) (*Preset, *Route, error) {
	if alias := routeAliasName(model); alias != "" {
		if err := s.routes.Reload(); err != nil {
			s.logf("routes reload failed: %v", err)
		}
		r := s.routes.ByName(alias)
		if r == nil {
			return nil, nil, &aliasError{name: alias}
		}
		return nil, r, nil
	}
	if err := s.presets.Reload(); err != nil {
		s.logf("preset reload failed: %v", err)
	}
	return s.presets.ByName(model), nil, nil
}
```

- [ ] **Step 3: Rewrite prepLLM to use resolveModel**

Replace the body of `func (s *Server) prepLLM(...)` with:

```go
func (s *Server) prepLLM(r *http.Request, req *AcquireRequest, payload map[string]any) error {
	model, _ := payload["model"].(string)
	cap := r.Header.Get("X-LLM-Capability")
	if strings.HasPrefix(model, "cap:") {
		cap = strings.TrimPrefix(model, "cap:")
		model = ""
	}
	p, route, err := s.resolveModel(model)
	if err != nil {
		return err
	}
	req.Preset = p
	req.Route = route
	req.RawModel = model
	req.Capability = cap
	return nil
}
```

- [ ] **Step 4: Handle the error in forward**

In `forward`, the call site currently reads roughly `s.prepLLM(r, &req, payload)` with no return. Replace with:

```go
		if err := s.prepLLM(r, &req, payload); err != nil {
			errPlain(w, 404, err.Error(), 404, "unknown_route")
			return
		}
```

(Confirm the exact surrounding block first - it is inside `if mode == "llm" && body != nil` after the `json.Unmarshal` succeeds.)

- [ ] **Step 5: List aliases in handleModels**

In `handleModels`, after reloading presets, reload routes, and after the preset loop append an alias loop. The exact shape of the existing handler must be confirmed by reading it first. Add:

```go
	if s.routes != nil {
		if err := s.routes.Reload(); err != nil {
			s.logf("routes reload failed: %v", err)
		}
	}
```

and, after the preset `data = append(data, ...)` loop:

```go
	snap := s.sched.Status()
	for name, route := range s.routes.All() {
		loaded := false
		if len(route.Chain) == 0 {
			loaded = snap.Mode == "llm" // empty chain = "whatever is resident"
		} else if snap.Mode == "llm" {
			if rp := s.presets.ByName(snap.Model); rp != nil {
				for _, c := range route.Chain {
					if c == rp.Name || c == rp.ModelID() {
						loaded = true
						break
					}
				}
			}
		}
		id := "auto:" + name
		if name == "default" {
			id = "auto"
		}
		data = append(data, map[string]any{
			"id": id, "object": "model", "created": 0, "owned_by": "local",
			"meta": map[string]any{
				"alias": true, "name": route.Name,
				"chain": route.Chain, "loaded": loaded,
				"description": route.Description,
			},
		})
	}
```

- [ ] **Step 6: Build + run the suite**

Run: `cd proxy-go && go build ./... && go test ./... -race -count=1`
Expected: PASS. (server_test.go's `startServer` now fails to compile because `NewServer` gained a parameter - see Task 7.)

- [ ] **Step 7: Commit (after Task 7 makes the suite green)**

Do not commit yet; commit together with Task 7.

---

### Task 7: server tests - 404 + /v1/models listing + startServer wiring

**Satisfies:** R2 (handler 404), R5 (listing)

**Files:**
- Modify: `proxy-go/internal/proxy/server_test.go`
- Create: `proxy-go/internal/proxy/routes_test.go` (server-level route tests)

- [ ] **Step 1: Refactor startServer to wire a routes store**

In `server_test.go`, change `startServer` to delegate to a new `startServerWithRoutes`, so existing call sites are untouched:

```go
func startServer(t *testing.T, orch *fakeOrch, store *PresetStore, st *State, cfg ServerConfig) *httptest.Server {
	return startServerWithRoutes(t, orch, store, st, cfg, "")
}

func startServerWithRoutes(t *testing.T, orch *fakeOrch, store *PresetStore, st *State, cfg ServerConfig, routesContent string) *httptest.Server {
	t.Helper()
	if cfg.VRAMLimitGB == 0 && cfg.VRAMReserveGB == 0 {
		cfg = ServerConfig{VRAMLimitGB: 24, VRAMReserveGB: 4}
	}
	statePath := t.TempDir() + "/state.toml"
	if st != nil {
		if err := SaveState(statePath, st); err != nil {
			t.Fatalf("SaveState: %v", err)
		}
	}
	routesPath := t.TempDir() + "/routes.toml"
	if routesContent != "" {
		if err := os.WriteFile(routesPath, []byte(routesContent), 0o644); err != nil {
			t.Fatalf("writing routes: %v", err)
		}
	}
	routes, err := NewRouteStore(routesPath)
	if err != nil {
		t.Fatalf("NewRouteStore: %v", err)
	}
	scfg := SchedulerConfig{StatePath: statePath, DrainGrace: 3 * time.Second,
		LockTTL: 900 * time.Second, HealthTimeout: time.Second}
	s, err := NewScheduler(scfg, orch, store, func(string, ...any) {})
	if err != nil {
		t.Fatalf("NewScheduler: %v", err)
	}
	go s.Run()
	t.Cleanup(s.Close)
	ts := httptest.NewServer(NewServer(s, store, routes, cfg, func(string, ...any) {}))
	t.Cleanup(ts.Close)
	return ts
}
```

(Add `"os"` to the imports if not already present.)

- [ ] **Step 2: Write the server-level route tests**

Append to `routes_test.go`:

```go
package proxy

import (
	"testing"
)

const routesForServer = `
[default]
chain = ["a", "b"]

[resident]
chain = []
`

func TestUnknownAlias404(t *testing.T) {
	ts := startServerWithRoutes(t, newFakeOrch("llm"),
		newTestStore(t, map[string]string{"a": tomlA, "b": tomlB}),
		&State{Mode: "llm", Model: "a"}, ServerConfig{}, routesForServer)
	code, body := doPost(t, ts.URL+"/v1/chat/completions", map[string]any{
		"model": "auto:nope", "messages": []any{map[string]any{"role": "user", "content": "hi"}},
	})
	if code != 404 {
		t.Fatalf("unknown alias: want 404, got %d %#v", code, body)
	}
	if !containsString(errMsg(body), "unknown route alias") {
		t.Fatalf("404 body must name the alias, got %#v", body)
	}
}

func TestAliasListedInModels(t *testing.T) {
	ts := startServerWithRoutes(t, newFakeOrch("llm"),
		newTestStore(t, map[string]string{"a": tomlA, "b": tomlB}),
		&State{Mode: "llm", Model: "a"}, ServerConfig{}, routesForServer)
	_, body := doGet(t, ts.URL+"/v1/models")
	data, ok := body["data"].([]any)
	if !ok {
		t.Fatalf("models data missing: %#v", body)
	}
	found := map[string]bool{}
	for _, m := range data {
		mm, _ := m.(map[string]any)
		meta, _ := mm["meta"].(map[string]any)
		if alias, _ := meta["alias"].(bool); alias {
			found[mm["id"].(string)] = true
			if meta["loaded"] != true {
				t.Fatalf("alias %v should be loaded (resident a in chain), meta: %#v", mm["id"], meta)
			}
		}
	}
	if !found["auto"] || !found["auto:resident"] {
		t.Fatalf("expected auto + auto:resident aliases, got %v", found)
	}
}

func containsString(haystack, needle string) bool {
	for i := 0; i+len(needle) <= len(haystack); i++ {
		if haystack[i:i+len(needle)] == needle {
			return true
		}
	}
	return false
}
```

- [ ] **Step 3: Run the full suite**

Run: `cd proxy-go && go test ./... -race -count=1`
Expected: PASS (all existing + new).

- [ ] **Step 4: Commit Task 6 + Task 7 together**

```bash
git add proxy-go/internal/proxy/server.go proxy-go/internal/proxy/server_test.go proxy-go/internal/proxy/routes_test.go
git commit -m "feat(proxy): resolveModel + alias listing + 404 (R2/R5)"
```

---

### Task 8: Anthropic shim - alias resolution

**Satisfies:** R2 (shim parity), R3 (lock interplay via scheduler), R5 (log lines)

**Files:**
- Modify: `proxy-go/internal/proxy/anthropic.go`
- Test: `proxy-go/internal/proxy/anthropic_test.go` (create if absent; else add to server_test.go)

- [ ] **Step 1: Resolve the alias in Serve**

In `Serve`, replace the `var preset *Preset` + `if model != "" { ... preset = ... }` block with:

```go
	var preset *Preset
	var route *Route
	if model != "" {
		var err error
		preset, route, err = s.resolveModel(model)
		if err != nil {
			status = 404
			anthropicErr(w, 404, "invalid_request_error", err.Error())
			return
		}
	}
```

and change the acquire call to carry `Route: route`:

```go
	res := s.sched.Acquire(r.Context(), AcquireRequest{Mode: "llm", Preset: preset, Route: route, RawModel: model})
```

(The ServeAs rewrite already keys off `res.ServeAs`, which now carries the resident/head model ID for alias grants - no further change to `upModel` is needed.)

- [ ] **Step 2: Write the shim test**

```go
func TestAnthropicAliasResolves(t *testing.T) {
	ts := startServerWithRoutes(t, newFakeOrch("llm"),
		newTestStore(t, map[string]string{"a": tomlA, "b": tomlB}),
		&State{Mode: "llm", Model: "a"}, ServerConfig{}, routesForServer)
	code, body := doPost(t, ts.URL+"/v1/messages", map[string]any{
		"model": "auto:default", "max_tokens": 5,
		"messages": []any{map[string]any{"role": "user", "content": "hi"}},
	})
	if code != 200 {
		t.Fatalf("anthropic alias: %d %#v", code, body)
	}
	if body["role"] != "assistant" {
		t.Fatalf("anthropic alias response wrong: %#v", body)
	}
}

func TestAnthropicUnknownAlias404(t *testing.T) {
	ts := startServerWithRoutes(t, newFakeOrch("llm"),
		newTestStore(t, map[string]string{"a": tomlA, "b": tomlB}),
		&State{Mode: "llm", Model: "a"}, ServerConfig{}, routesForServer)
	code, body := doPost(t, ts.URL+"/v1/messages", map[string]any{
		"model": "auto:nope", "max_tokens": 5,
		"messages": []any{map[string]any{"role": "user", "content": "hi"}},
	})
	if code != 404 {
		t.Fatalf("anthropic unknown alias: want 404, got %d %#v", code, body)
	}
}
```

Note: the fake orchestrator never runs a real llama-server, so `/v1/messages` streaming is bypassed in these tests; the non-stream `translateResponse` path returns a `message` object. If the shim forces streaming, confirm by reading `Serve`; the test above asserts the 404 path and the resolve path separately.

- [ ] **Step 3: Run the shim tests**

Run: `cd proxy-go && go test ./internal/proxy -run 'TestAnthropic' -race -count=1`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add proxy-go/internal/proxy/anthropic.go proxy-go/internal/proxy/anthropic_test.go
git commit -m "feat(proxy): alias resolution in anthropic shim (R2)"
```

---

### Task 9: wiring - main.go, compose.yaml, routes.toml, smoke

**Satisfies:** R1 (file path + mount), R5 (observability wiring)

**Files:**
- Modify: `proxy-go/cmd/proxy/main.go`
- Modify: `compose.yaml`
- Create: `routes.toml`
- Modify: `tests/hurl/proxy-go-smoke.hurl`

- [ ] **Step 1: Wire routes + scheduler VRAM in main.go**

After the existing `vramLimit`/`vramReserve` env reads, add:

```go
	routesFile := envStr("LLMC_ROUTES_FILE", "/routes.toml")
```

After `store, err := proxy.NewPresetStore(presetsDir)`, add:

```go
	routes, err := proxy.NewRouteStore(routesFile)
	if err != nil {
		log.Fatalf("routes: %v", err)
	}
```

Pass VRAM into the scheduler config and routes into the server:

```go
	sched, err := proxy.NewScheduler(proxy.SchedulerConfig{
		StatePath:     filepath.Join(stateDir, "active.toml"),
		AssetsDir:     assetsDir,
		DrainGrace:    drainGrace,
		LockTTL:       lockTTL,
		HealthTimeout: healthTimeout,
		VRAMLimitGB:   vramLimit,
		VRAMReserveGB: vramReserve,
	}, orch, store, logf)
	...
	server := proxy.NewServer(sched, store, routes, proxy.ServerConfig{
		VRAMLimitGB:   vramLimit,
		VRAMReserveGB: vramReserve,
	}, logf)
```

- [ ] **Step 2: Mount routes.toml in compose.yaml**

In the `model-proxy-go` service's `volumes:` list, after the `./models:/presets:ro` line, add:

```yaml
      - ./routes.toml:/routes.toml:ro
```

- [ ] **Step 3: Write routes.toml at repo root**

```toml
# Task-routing alias table. model: "auto" -> default, "auto:<name>" -> <name>.
# chain is ordered by preference; empty chain = "whatever is resident".

[default]
description = "general chat: big model if loaded, small otherwise"
chain = ["qwen38", "qwen35-9b"]

[code]
description = "coding tasks"
chain = ["qwen3-coder", "qwen38"]

[cheap]
description = "cheap tasks: 9b then summarizer"
chain = ["qwen35-9b", "summarizer"]

[resident]
description = "never swap: whatever is loaded"
chain = []
```

- [ ] **Step 4: Add smoke entries**

Append to `tests/hurl/proxy-go-smoke.hurl` (before the ephemeral-preset section at the end):

```hurl
# Task routing: unknown alias 404, resident alias served in place.
POST {{base}}/v1/chat/completions
Content-Type: application/json
{"model":"auto:nope","messages":[{"role":"user","content":"hi"}],"max_tokens":5,"stream":false}
HTTP 404

POST {{base}}/v1/chat/completions
Content-Type: application/json
{"model":"auto","messages":[{"role":"user","content":"hi"}],"max_tokens":5,"stream":false}
HTTP 200
[Asserts]
jsonpath "$.choices[0].message.role" == "assistant"
```

- [ ] **Step 5: Build the image + run unit suite**

Run: `cd proxy-go && go build ./... && go test ./... -race -count=1`
Expected: PASS.

- [ ] **Step 6: Redeploy and smoke (needs the stack up)**

Run: `make build-proxy-go && make restart` then `make smoke-proxy-go`
Expected: smoke passes (note the Makefile smoke target points at `127.0.0.1:11435`; confirm the port matches the running proxy or adjust the `base` variable).

- [ ] **Step 7: Commit**

```bash
git add proxy-go/cmd/proxy/main.go compose.yaml routes.toml tests/hurl/proxy-go-smoke.hurl
git commit -m "feat(proxy): wire routes.toml + VRAM into scheduler, mount + smoke (R1/R5)"
```

---

## Self-review

**Spec coverage:** R1 -> Task 1 (load/validate/live-reload) + Task 9 (path + mount); R2 -> Tasks 2-4 (scheduler) + 6-8 (handler/shim); R3 -> Task 3 + 4; R4 -> Task 3 + 5; R5 -> Task 6-7 (listing) + 3/8 (log lines) + 9 (smoke). All five covered.

**Placeholder scan:** Every code step shows complete code; the one `grep`-first instruction (Task 6 Step 1) is because the exact `Server` struct field names were not read verbatim in this session - the step tells the worker to confirm, then shows the target shape. No TBD/TODO/"similar to Task N".

**Type consistency:** `Route`, `RouteStore`, `routeAliasName`, `aliasError`, `containsPreset`, `acquireRoute`, `greedyServe`, `resolveModel`, `prepLLM` returning `error`, `startServerWithRoutes`, `containsString`, `routesForServer`, `tomlA`/`tomlB` fixtures - all defined before first use in a later task. `AcquireRequest.Route` (Task 2) is consumed by Tasks 3-8. `SchedulerConfig.VRAMLimitGB` (Task 2) consumed in Task 3.

**Residual risk (named, not hidden):** BurntSushi/toml decodes a TOML array of strings into `[]interface{}` when the target is `any` - the `v.([]any)` assertion in `parseRoutes` relies on this. Task 1 Step 2 verifies it against a real decode before any other task depends on it; if the concrete type differs, the fix is local to `parseRoutes`.
