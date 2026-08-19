package proxy

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

// ── fake orchestrator ────────────────────────────────────────────────────

// fakeOrch is the test double for Orchestrator. doSwap runs on its own
// goroutine, so every accessor is mutex-guarded.
type fakeOrch struct {
	mu         sync.Mutex
	mode       string
	llamaCalls []*Preset
	comfyCalls int
	trainCalls int
}

func newFakeOrch(mode string) *fakeOrch { return &fakeOrch{mode: mode} }

func (f *fakeOrch) CurrentMode() string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.mode
}

func (f *fakeOrch) SpawnLlama(p *Preset) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.llamaCalls = append(f.llamaCalls, p)
	f.mode = "llm"
	return nil
}

func (f *fakeOrch) SpawnComfyUI() error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.comfyCalls++
	f.mode = "comfyui"
	return nil
}

func (f *fakeOrch) SpawnTrain() error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.trainCalls++
	f.mode = "train"
	return nil
}

func (f *fakeOrch) WaitHealthy(GpuService, time.Duration) bool { return true }
func (f *fakeOrch) EnsurePresetAssets(*Preset, string) error   { return nil }

func (f *fakeOrch) setMode(m string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.mode = m
}

func (f *fakeOrch) llamaCount() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return len(f.llamaCalls)
}

func (f *fakeOrch) llamaNames() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	out := make([]string, len(f.llamaCalls))
	for i, p := range f.llamaCalls {
		out[i] = p.Name
	}
	return out
}

func (f *fakeOrch) comfyCount() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.comfyCalls
}

func (f *fakeOrch) spawnTotal() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return len(f.llamaCalls) + f.comfyCalls + f.trainCalls
}

// ── fixtures ─────────────────────────────────────────────────────────────

const tomlA = `name = "Alpha"
vram_gb = 10.0
capabilities = ["vision"]

[model]
repo = "org/alpha"
file = "alpha.gguf"
`

const tomlB = `name = "Beta"
vram_gb = 12.0

[model]
repo = "org/beta"
file = "beta.gguf"
`

const tomlBCode = `name = "Beta"
vram_gb = 12.0
capabilities = ["code"]

[model]
repo = "org/beta"
file = "beta.gguf"
`

// a with an mmproj asset: HasVision() (driving /v1/models meta.capabilities)
// is set by the mmproj spec, not the capabilities list.
const tomlAVisionMMProj = `name = "Alpha"
vram_gb = 10.0
capabilities = ["vision"]

[model]
repo = "org/alpha"
file = "alpha.gguf"

[mmproj]
url = "https://huggingface.co/org/alpha/resolve/main/mmproj.gguf"
`

func newTestStore(t *testing.T, files map[string]string) *PresetStore {
	t.Helper()
	dir := t.TempDir()
	for name, body := range files {
		if err := os.WriteFile(filepath.Join(dir, name+".toml"), []byte(body), 0o644); err != nil {
			t.Fatalf("writing preset %s: %v", name, err)
		}
	}
	st, err := NewPresetStore(dir)
	if err != nil {
		t.Fatalf("NewPresetStore: %v", err)
	}
	return st
}

// ── scheduler plumbing ───────────────────────────────────────────────────

type schedTest struct {
	s     *Scheduler
	orch  *fakeOrch
	state string // state file path
}

func startSched(t *testing.T, orch *fakeOrch, store *PresetStore, st *State, cfg SchedulerConfig) *schedTest {
	t.Helper()
	if cfg.StatePath == "" {
		cfg.StatePath = filepath.Join(t.TempDir(), "state.toml")
	}
	if cfg.DrainGrace == 0 {
		cfg.DrainGrace = 3 * time.Second
	}
	if cfg.LockTTL == 0 {
		cfg.LockTTL = 900 * time.Second
	}
	if cfg.HealthTimeout == 0 {
		cfg.HealthTimeout = time.Second
	}
	if st != nil {
		if err := SaveState(cfg.StatePath, st); err != nil {
			t.Fatalf("SaveState: %v", err)
		}
	}
	s, err := NewScheduler(cfg, orch, store, func(string, ...any) {})
	if err != nil {
		t.Fatalf("NewScheduler: %v", err)
	}
	go s.Run()
	t.Cleanup(s.Close)
	return &schedTest{s: s, orch: orch, state: cfg.StatePath}
}

func (e *schedTest) loadedState(t *testing.T) *State {
	t.Helper()
	st, err := LoadState(e.state)
	if err != nil {
		t.Fatalf("LoadState: %v", err)
	}
	return st
}

// waitFor polls cond on a 5ms tick until it holds or the deadline passes.
func waitFor(t *testing.T, d time.Duration, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(d)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatalf("timed out after %s waiting for %s", d, what)
}

func bodyOwners(r LockResult) []string {
	owners, _ := r.Body["lock_owners"].([]string)
	return owners
}

func bodyPosition(r LockResult) int {
	pos, _ := r.Body["position"].(int)
	return pos
}

func ctx() context.Context { return context.Background() }

// ── 1. VRAM budget ───────────────────────────────────────────────────────

func TestVRAMBudget(t *testing.T) {
	p := &Preset{Name: "alpha", DisplayName: "Alpha", VRAMGB: 10.0,
		Model: ModelSpec{Repo: "org/alpha", File: "alpha.gguf"}}

	if ok, _ := CheckVRAMBudget(p, 24, 4); !ok {
		t.Fatal("10GB preset should fit in 24-4=20GB")
	}
	if ok, _ := CheckVRAMBudget(p, 14, 4); !ok {
		t.Fatal("10GB preset should fit exactly in 14-4=10GB")
	}
	ok, msg := CheckVRAMBudget(p, 13.9, 4)
	if ok {
		t.Fatal("10GB preset should not fit in 13.9-4=9.9GB")
	}
	if !strings.Contains(msg, "VRAM") || !strings.Contains(msg, "Alpha") {
		t.Fatalf("over-budget message should name the model and mention VRAM, got: %s", msg)
	}
}

// ── 2. system message merge ──────────────────────────────────────────────

func TestNeedsSystemMerge(t *testing.T) {
	sys := func(s string) map[string]any { return map[string]any{"role": "system", "content": s} }
	user := map[string]any{"role": "user", "content": "hi"}

	if needsSystemMerge([]any{sys("a"), user}) {
		t.Fatal("single system at index 0 must not merge")
	}
	if !needsSystemMerge([]any{sys("a"), sys("b"), user}) {
		t.Fatal("two system messages must merge")
	}
	if !needsSystemMerge([]any{user, sys("a")}) {
		t.Fatal("system after user must merge")
	}
	if needsSystemMerge([]any{user, map[string]any{"role": "assistant", "content": "yo"}}) {
		t.Fatal("no system messages at all must not merge")
	}
}

func TestMergeSystemMessages(t *testing.T) {
	sys := func(s string) map[string]any { return map[string]any{"role": "system", "content": s} }
	user := map[string]any{"role": "user", "content": "hi"}

	merged := mergeSystemMessages([]any{sys("A"), user, sys("B")})
	if len(merged) != 2 {
		t.Fatalf("expected 2 messages after merge, got %d", len(merged))
	}
	m0, ok := merged[0].(map[string]any)
	if !ok || m0["role"] != "system" || m0["content"] != "A\n\nB" {
		t.Fatalf("merged system wrong: %#v", merged[0])
	}
	m1, ok := merged[1].(map[string]any)
	if !ok || m1["role"] != "user" || m1["content"] != "hi" {
		t.Fatalf("user message should survive at index 1: %#v", merged[1])
	}

	// Multimodal content blocks: only text parts survive the merge.
	mmSys := map[string]any{
		"role": "system",
		"content": []any{
			map[string]any{"type": "text", "text": "A"},
			map[string]any{"type": "image_url", "image_url": "x"},
			map[string]any{"type": "text", "text": "B"},
		},
	}
	out := mergeSystemMessages([]any{mmSys, user})
	if m, ok := out[0].(map[string]any); !ok || m["content"] != "A\n\nB" {
		t.Fatalf("multimodal merge should keep text parts only: %#v", out[0])
	}
}

// ── 3. preset lookup ─────────────────────────────────────────────────────

func TestPresetStoreByName(t *testing.T) {
	store := newTestStore(t, map[string]string{"a": tomlA, "b": tomlB})

	byID := store.ByName("alpha") // model id (filename stem of the .gguf)
	if byID == nil {
		t.Fatal("ByName(\"alpha\") (model id) should resolve")
	}
	if byID.Name != "a" || byID.DisplayName != "Alpha" {
		t.Fatalf("resolved preset wrong: %+v", byID)
	}
	if byID.ModelID() != "alpha" {
		t.Fatalf("ModelID: got %q", byID.ModelID())
	}

	byStem := store.ByName("a") // preset file stem
	if byStem == nil || byStem.ModelID() != "alpha" {
		t.Fatalf("ByName(\"a\") (preset stem) should resolve, got %#v", byStem)
	}

	if got := store.ByName("nope"); got != nil {
		t.Fatalf("unknown model should be nil, got %#v", got)
	}
}

// ── 4. resident grant (no spawn) ─────────────────────────────────────────

func TestResidentGrantNoSpawn(t *testing.T) {
	e := startSched(t, newFakeOrch("llm"), newTestStore(t, map[string]string{"a": tomlA, "b": tomlB}),
		&State{Mode: "llm", Model: "a"}, SchedulerConfig{})
	pA := e.s.presets.ByName("a")

	res := e.s.Acquire(ctx(), AcquireRequest{Mode: "llm", Preset: pA})
	if !res.OK || !res.Granted || res.Key != "a" {
		t.Fatalf("resident acquire: %+v", res)
	}
	defer e.s.Release(res.Key)
	if got := e.orch.llamaCount(); got != 0 {
		t.Fatalf("resident grant must not spawn, SpawnLlama called %d times", got)
	}
}

// ── 5. swap to different model ───────────────────────────────────────────

func TestSwapToDifferentModel(t *testing.T) {
	e := startSched(t, newFakeOrch("llm"), newTestStore(t, map[string]string{"a": tomlA, "b": tomlB}),
		&State{Mode: "llm", Model: "a"}, SchedulerConfig{})
	pB := e.s.presets.ByName("b")

	res := e.s.Acquire(ctx(), AcquireRequest{Mode: "llm", Preset: pB})
	if !res.OK || !res.Granted || res.Key != "b" {
		t.Fatalf("swap acquire: %+v", res)
	}
	defer e.s.Release(res.Key)

	if got := e.orch.llamaNames(); len(got) != 1 || got[0] != "b" {
		t.Fatalf("expected exactly one SpawnLlama(b), got %v", got)
	}
	if st := e.loadedState(t); st.Model != "b" {
		t.Fatalf("state file model should be b, got %q", st.Model)
	}
}

// ── 6. R2: drain sequencing ──────────────────────────────────────────────

func TestDrainSequencing(t *testing.T) {
	e := startSched(t, newFakeOrch("llm"), newTestStore(t, map[string]string{"a": tomlA, "b": tomlB}),
		&State{Mode: "llm", Model: "a"}, SchedulerConfig{})
	pA, pB := e.s.presets.ByName("a"), e.s.presets.ByName("b")

	resA := e.s.Acquire(ctx(), AcquireRequest{Mode: "llm", Preset: pA})
	if !resA.OK || resA.Key != "a" {
		t.Fatalf("first acquire: %+v", resA)
	}

	resCh := make(chan AcqResult, 1)
	go func() { resCh <- e.s.Acquire(ctx(), AcquireRequest{Mode: "llm", Preset: pB}) }()

	// The pending swap must be registered before we check for no-spawn.
	waitFor(t, 2*time.Second, "pending swap to register", func() bool {
		return e.s.Status().Switching
	})
	time.Sleep(150 * time.Millisecond)
	if got := e.orch.llamaCount(); got != 0 {
		t.Fatalf("swap must not start while model a is in-flight, SpawnLlama called %d times", got)
	}

	// Release A; the drain completes and the queued swap may proceed.
	e.s.Release(resA.Key)
	select {
	case res := <-resCh:
		if !res.OK || !res.Granted || res.Key != "b" {
			t.Fatalf("queued acquire: %+v", res)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("queued swap did not grant after release")
	}
	if got := e.orch.llamaNames(); len(got) != 1 || got[0] != "b" {
		t.Fatalf("expected SpawnLlama(b) after drain, got %v", got)
	}
}

// ── 7. drain grace timeout ───────────────────────────────────────────────

func TestDrainGraceTimeoutForcesSwap(t *testing.T) {
	cfg := SchedulerConfig{DrainGrace: 300 * time.Millisecond}
	e := startSched(t, newFakeOrch("llm"), newTestStore(t, map[string]string{"a": tomlA, "b": tomlB}),
		&State{Mode: "llm", Model: "a"}, cfg)
	pA, pB := e.s.presets.ByName("a"), e.s.presets.ByName("b")

	resA := e.s.Acquire(ctx(), AcquireRequest{Mode: "llm", Preset: pA})
	if !resA.OK {
		t.Fatalf("first acquire: %+v", resA)
	}

	resCh := make(chan AcqResult, 1)
	go func() { resCh <- e.s.Acquire(ctx(), AcquireRequest{Mode: "llm", Preset: pB}) }()
	waitFor(t, 2*time.Second, "pending swap to register", func() bool {
		return e.s.Status().Switching
	})

	// A is never released: the grace timeout must force the swap anyway.
	select {
	case res := <-resCh:
		if !res.OK || !res.Granted || res.Key != "b" {
			t.Fatalf("grace-timeout acquire: %+v", res)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("drain grace timeout did not force the swap")
	}
	if got := e.orch.llamaNames(); len(got) != 1 || got[0] != "b" {
		t.Fatalf("expected SpawnLlama(b) after grace, got %v", got)
	}
}

// ── 8. lock basics: 409 / 202 FIFO ───────────────────────────────────────

func TestLock409ThenQueueThenGrant(t *testing.T) {
	e := startSched(t, newFakeOrch("llm"), newTestStore(t, map[string]string{"a": tomlA, "b": tomlB}),
		&State{Mode: "llm", Model: "a"}, SchedulerConfig{})

	r := e.s.Lock("a", false, "A", false)
	if r.Status != 200 || r.Body["locked"] != "a" || len(bodyOwners(r)) != 1 || bodyOwners(r)[0] != "A" {
		t.Fatalf("lock a/A: %d %#v", r.Status, r.Body)
	}

	r = e.s.Lock("b", false, "B", false)
	if r.Status != 409 {
		t.Fatalf("different model without wait: want 409, got %d", r.Status)
	}
	if err, _ := r.Body["error"].(string); !strings.Contains(err, "wait") {
		t.Fatalf("409 body should mention wait, got %#v", r.Body)
	}

	r = e.s.Lock("b", false, "B", true)
	if r.Status != 202 || r.Body["queued"] != true || bodyPosition(r) != 1 {
		t.Fatalf("queued lock: %d %#v", r.Status, r.Body)
	}

	if r = e.s.Unlock("A"); r.Status != 200 || r.Body["locked"] != nil {
		t.Fatalf("unlock: %d %#v", r.Status, r.Body)
	}

	// Free again; B polls and gets the model (it is the queue head).
	r = e.s.Lock("b", false, "B", true)
	if r.Status != 200 || r.Body["locked"] != "b" || len(bodyOwners(r)) != 1 || bodyOwners(r)[0] != "B" {
		t.Fatalf("B polling after free: %d %#v", r.Status, r.Body)
	}
	if got := e.s.Status().LockQueue; len(got) != 0 {
		t.Fatalf("B should be removed from the queue on grant, queue: %#v", got)
	}
}

func TestLockFIFOOrdering(t *testing.T) {
	e := startSched(t, newFakeOrch("llm"), newTestStore(t, map[string]string{"a": tomlA, "b": tomlB}),
		&State{Mode: "llm", Model: "a"}, SchedulerConfig{})

	if r := e.s.Lock("a", false, "A", false); r.Status != 200 {
		t.Fatalf("lock a/A: %d %#v", r.Status, r.Body)
	}
	if r := e.s.Lock("b", false, "B", true); r.Status != 202 || bodyPosition(r) != 1 {
		t.Fatalf("B queue: %d %#v", r.Status, r.Body)
	}
	if r := e.s.Lock("c", false, "C", true); r.Status != 202 || bodyPosition(r) != 2 {
		t.Fatalf("C queue: %d %#v", r.Status, r.Body)
	}

	if r := e.s.Unlock("A"); r.Status != 200 {
		t.Fatalf("unlock A: %d %#v", r.Status, r.Body)
	}

	// C polls before B: the head is B->b, so C stays queued at position 2.
	if r := e.s.Lock("c", false, "C", true); r.Status != 202 || bodyPosition(r) != 2 {
		t.Fatalf("C polling (not head): %d %#v", r.Status, r.Body)
	}
	// B polls: head is B->b, grant.
	if r := e.s.Lock("b", false, "B", true); r.Status != 200 || r.Body["locked"] != "b" || bodyOwners(r)[0] != "B" {
		t.Fatalf("B polling (head): %d %#v", r.Status, r.Body)
	}
	// C's entry survives, still queued.
	if got := e.s.Status().LockQueue; len(got) != 1 {
		t.Fatalf("C should remain queued: %#v", got)
	}
}

func TestLockSameModelBypassesFIFO(t *testing.T) {
	e := startSched(t, newFakeOrch("llm"), newTestStore(t, map[string]string{"a": tomlA, "b": tomlB}),
		&State{Mode: "llm", Model: "a"}, SchedulerConfig{})

	if r := e.s.Lock("a", false, "A", false); r.Status != 200 {
		t.Fatalf("lock a/A: %d %#v", r.Status, r.Body)
	}
	if r := e.s.Lock("b", false, "B", true); r.Status != 202 {
		t.Fatalf("B queue: %d %#v", r.Status, r.Body)
	}
	// C wants the already-locked model: joining must bypass the FIFO gate.
	r := e.s.Lock("a", false, "C", false)
	if r.Status != 200 {
		t.Fatalf("join locked model: %d %#v", r.Status, r.Body)
	}
	owners := bodyOwners(r)
	if len(owners) != 2 || owners[0] != "A" || owners[1] != "C" {
		t.Fatalf("owners should be [A C]: %#v", owners)
	}
}

func TestLockQueueIdempotentAndUnlockRemoves(t *testing.T) {
	e := startSched(t, newFakeOrch("llm"), newTestStore(t, map[string]string{"a": tomlA, "b": tomlB}),
		&State{Mode: "llm", Model: "a"}, SchedulerConfig{})

	if r := e.s.Lock("a", false, "A", false); r.Status != 200 {
		t.Fatalf("lock a/A: %d %#v", r.Status, r.Body)
	}
	if r := e.s.Lock("b", false, "B", true); r.Status != 202 || bodyPosition(r) != 1 {
		t.Fatalf("B queue: %d %#v", r.Status, r.Body)
	}
	// Re-enqueue by the same owner: position and queue length unchanged.
	if r := e.s.Lock("b", false, "B", true); r.Status != 202 || bodyPosition(r) != 1 {
		t.Fatalf("B requeue: %d %#v", r.Status, r.Body)
	}
	if got := e.s.Status().LockQueue; len(got) != 1 {
		t.Fatalf("re-enqueue must not duplicate, queue: %#v", got)
	}
	// Owner changes target model: same position, model updated in place.
	if r := e.s.Lock("c", false, "B", true); r.Status != 202 || bodyPosition(r) != 1 {
		t.Fatalf("B requeue to c: %d %#v", r.Status, r.Body)
	}
	if got := e.s.Status().LockQueue; len(got) != 1 || got[0].Model != "c" {
		t.Fatalf("queue entry should update model to c: %#v", got)
	}
	// Unlock removes the owner from the queue entirely.
	if r := e.s.Unlock("B"); r.Status != 200 {
		t.Fatalf("unlock B: %d %#v", r.Status, r.Body)
	}
	if got := e.s.Status().LockQueue; len(got) != 0 {
		t.Fatalf("unlock should drop the queue entry: %#v", got)
	}
}

// ── 9. lock TTL ──────────────────────────────────────────────────────────

func TestLockTTLExpiry(t *testing.T) {
	e := startSched(t, newFakeOrch("llm"), newTestStore(t, map[string]string{"a": tomlA, "b": tomlB}),
		&State{Mode: "llm", Model: "a"}, SchedulerConfig{LockTTL: time.Second})

	if r := e.s.Lock("a", false, "A", false); r.Status != 200 {
		t.Fatalf("lock a/A: %d %#v", r.Status, r.Body)
	}
	// LockExpiresAt is unix SECONDS (LockTTL=1s), so expiry can only be
	// observed once the integer clock rolls one second past the granted
	// value - worst case ~2s of wall time.
	time.Sleep(2100 * time.Millisecond)

	// A's slot expired lazily; B can now take a different model.
	if r := e.s.Lock("b", false, "B", false); r.Status != 200 || r.Body["locked"] != "b" || bodyOwners(r)[0] != "B" {
		t.Fatalf("lock after TTL expiry: %d %#v", r.Status, r.Body)
	}
}

// ── 10. R3: serve-in-place ───────────────────────────────────────────────

func TestServeInPlaceWhenResidentHasCapability(t *testing.T) {
	e := startSched(t, newFakeOrch("llm"), newTestStore(t, map[string]string{"a": tomlA, "b": tomlB}),
		&State{Mode: "llm", Model: "a"}, SchedulerConfig{})
	pB := e.s.presets.ByName("b")

	res := e.s.Acquire(ctx(), AcquireRequest{Mode: "llm", Preset: pB, Capability: "vision"})
	if !res.OK || !res.Granted || res.ServeAs != "alpha" || res.Key != "a" {
		t.Fatalf("serve-in-place: %+v", res)
	}
	defer e.s.Release(res.Key)
	if got := e.orch.llamaCount(); got != 0 {
		t.Fatalf("serve-in-place must not spawn, got %d", got)
	}
}

func TestServeInPlaceUnderLock(t *testing.T) {
	e := startSched(t, newFakeOrch("llm"), newTestStore(t, map[string]string{"a": tomlA, "b": tomlB}),
		&State{Mode: "llm", Model: "a"}, SchedulerConfig{})
	pB := e.s.presets.ByName("b")

	if r := e.s.Lock("a", false, "A", false); r.Status != 200 {
		t.Fatalf("lock: %d %#v", r.Status, r.Body)
	}
	t.Cleanup(func() { e.s.Unlock("A") })

	res := e.s.Acquire(ctx(), AcquireRequest{Mode: "llm", Preset: pB, Capability: "vision"})
	if !res.OK || !res.Granted || res.ServeAs != "alpha" {
		t.Fatalf("serve-in-place under lock: %+v", res)
	}
	e.s.Release(res.Key)
	if got := e.orch.llamaCount(); got != 0 {
		t.Fatalf("serve-in-place under lock must not spawn, got %d", got)
	}

	// Same preset request but no capability: no serve-in-place, and the
	// locked model is not the requested one, so 422 model_unavailable.
	res = e.s.Acquire(ctx(), AcquireRequest{Mode: "llm", Preset: pB})
	if res.OK || res.Status != 422 || res.ErrType != "model_unavailable" {
		t.Fatalf("locked + different preset + no capability: %+v", res)
	}
}

// ── 11. capability resolution ────────────────────────────────────────────

func TestCapabilityResolution(t *testing.T) {
	t.Run("no preset advertises the capability", func(t *testing.T) {
		e := startSched(t, newFakeOrch("llm"), newTestStore(t, map[string]string{"a": tomlA, "b": tomlB}),
			&State{Mode: "llm", Model: "a"}, SchedulerConfig{})
		res := e.s.Acquire(ctx(), AcquireRequest{Mode: "llm", Capability: "code"})
		if res.OK || res.Status != 422 {
			t.Fatalf("want 422, got %+v", res)
		}
		if !strings.Contains(res.ErrMsg, "no preset advertises capability") {
			t.Fatalf("message should name the capability gap: %q", res.ErrMsg)
		}
	})

	t.Run("swap to the preset that has the capability", func(t *testing.T) {
		e := startSched(t, newFakeOrch("idle"), newTestStore(t, map[string]string{"a": tomlA, "b": tomlBCode}),
			nil, SchedulerConfig{})
		res := e.s.Acquire(ctx(), AcquireRequest{Mode: "llm", Capability: "code"})
		if !res.OK || !res.Granted || res.Key != "b" {
			t.Fatalf("capability swap: %+v", res)
		}
		e.s.Release(res.Key)
		if got := e.orch.llamaNames(); len(got) != 1 || got[0] != "b" {
			t.Fatalf("expected SpawnLlama(b), got %v", got)
		}
	})
}

// ── 12. mode swap ────────────────────────────────────────────────────────

func TestModeSwapToComfyUI(t *testing.T) {
	e := startSched(t, newFakeOrch("idle"), newTestStore(t, map[string]string{"a": tomlA, "b": tomlB}),
		nil, SchedulerConfig{})

	res := e.s.Acquire(ctx(), AcquireRequest{Mode: "comfyui"})
	if !res.OK || !res.Granted || res.Key != "comfyui" {
		t.Fatalf("comfyui acquire: %+v", res)
	}
	e.s.Release(res.Key)
	if got := e.orch.comfyCount(); got != 1 {
		t.Fatalf("expected one SpawnComfyUI, got %d", got)
	}
	if st := e.loadedState(t); st.Mode != "comfyui" {
		t.Fatalf("state mode should be comfyui, got %q", st.Mode)
	}
}

func TestModeSwapBlockedByLock(t *testing.T) {
	e := startSched(t, newFakeOrch("llm"), newTestStore(t, map[string]string{"a": tomlA, "b": tomlB}),
		&State{Mode: "llm", Model: "a"}, SchedulerConfig{})
	if r := e.s.Lock("a", false, "A", false); r.Status != 200 {
		t.Fatalf("lock: %d %#v", r.Status, r.Body)
	}

	res := e.s.Acquire(ctx(), AcquireRequest{Mode: "comfyui"})
	if res.OK || res.Status != 503 {
		t.Fatalf("mode swap under lock: want 503, got %+v", res)
	}
	if got := e.orch.comfyCount(); got != 0 {
		t.Fatalf("blocked swap must not spawn comfyui, got %d", got)
	}
}

// ── 13. persistence across restart ───────────────────────────────────────

func TestLockPersistsAcrossRestart(t *testing.T) {
	dir := t.TempDir()
	statePath := filepath.Join(dir, "state.toml")
	cfg := SchedulerConfig{StatePath: statePath, LockTTL: 900 * time.Second,
		DrainGrace: 3 * time.Second, HealthTimeout: time.Second}
	store := newTestStore(t, map[string]string{"a": tomlA, "b": tomlB})
	logf := func(string, ...any) {}

	s1, err := NewScheduler(cfg, newFakeOrch("llm"), store, logf)
	if err != nil {
		t.Fatalf("NewScheduler #1: %v", err)
	}
	go s1.Run()
	if r := s1.Lock("a", false, "A", false); r.Status != 200 {
		t.Fatalf("lock on s1: %d %#v", r.Status, r.Body)
	}
	s1.Close()

	s2, err := NewScheduler(cfg, newFakeOrch("llm"), store, logf)
	if err != nil {
		t.Fatalf("NewScheduler #2: %v", err)
	}
	go s2.Run()
	t.Cleanup(s2.Close)

	snap := s2.Status()
	if snap.Locked != "a" || len(snap.LockOwners) != 1 || snap.LockOwners[0] != "A" {
		t.Fatalf("lock should survive restart: %+v", snap)
	}
	if snap.Mode != "llm" {
		t.Fatalf("mode should be llm, got %q", snap.Mode)
	}
	if st := loadStateAt(t, statePath); st.Mode != "llm" || st.Locked != "a" {
		t.Fatalf("state file should carry mode=llm + lock: %+v", st)
	}
}

func loadStateAt(t *testing.T, path string) *State {
	t.Helper()
	st, err := LoadState(path)
	if err != nil {
		t.Fatalf("LoadState: %v", err)
	}
	return st
}

// ── 14. passthrough (unknown model) ──────────────────────────────────────

func TestPassthroughUnknownModel(t *testing.T) {
	e := startSched(t, newFakeOrch("llm"), newTestStore(t, map[string]string{"a": tomlA, "b": tomlB}),
		&State{Mode: "llm", Model: "a"}, SchedulerConfig{})

	res := e.s.Acquire(ctx(), AcquireRequest{Mode: "llm", RawModel: "some-unknown"})
	if !res.OK || !res.Granted || res.ServeAs != "" {
		t.Fatalf("passthrough unknown model: %+v", res)
	}
	e.s.Release(res.Key)

	res = e.s.Acquire(ctx(), AcquireRequest{Mode: "llm", RawModel: ""})
	if !res.OK || !res.Granted {
		t.Fatalf("passthrough empty model: %+v", res)
	}
	e.s.Release(res.Key)

	// Verbatim request of the resident model is also passthrough.
	res = e.s.Acquire(ctx(), AcquireRequest{Mode: "llm", RawModel: "a"})
	if !res.OK || !res.Granted {
		t.Fatalf("verbatim resident model: %+v", res)
	}
	e.s.Release(res.Key)

	// Under a lock, an unknown model cannot be served by the locked model.
	if r := e.s.Lock("a", false, "A", false); r.Status != 200 {
		t.Fatalf("lock: %d %#v", r.Status, r.Body)
	}
	res = e.s.Acquire(ctx(), AcquireRequest{Mode: "llm", RawModel: "some-unknown"})
	if res.OK || res.Status != 422 {
		t.Fatalf("unknown model under lock: want 422, got %+v", res)
	}
}
