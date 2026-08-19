// Scheduler: single-goroutine event loop owning all mutable proxy state
// (lock owners, FIFO queue, in-flight counts, pending swap). Architecture
// adopted from llama-swap's internal/router: handlers send events, swaps run
// as fire-and-forget goroutines reporting back on the same channel, and
// in-flight grants are counted on the loop before the handler forwards - so
// the "request between lock-release and in-flight increment" race is
// structurally impossible.
//
// Adds over llama-swap: drain-before-swap with a grace deadline (R2),
// capability serve-in-place routing (R3), the lock/FIFO-queue API with TTL
// (R1), and docker-container lifecycle (not raw processes).
package proxy

import (
	"context"
	"fmt"
	"sort"
	"time"
)

type SchedulerConfig struct {
	StatePath     string
	AssetsDir     string
	DrainGrace    time.Duration
	LockTTL       time.Duration
	HealthTimeout time.Duration
}

// AcquireRequest is one request for GPU capacity.
type AcquireRequest struct {
	Mode       string  // llm | comfyui | train
	Preset     *Preset // resolved llm target; nil = passthrough/unknown/current
	RawModel   string  // model field as requested (for passthrough + errors)
	Capability string  // X-LLM-Capability / cap:<name> hint
}

// AcqResult is the loop's verdict. When Granted is true the caller MUST
// Release(Key) exactly once when its forwarded request completes.
type AcqResult struct {
	OK       bool
	Status   int
	ErrMsg   string
	ErrType  string
	ServeAs  string // non-empty: rewrite the body model field to this (serve-in-place)
	Key      string // in-flight key
	Granted  bool
	Dequeued bool
}

// LockResult carries an HTTP-ready payload for lock/unlock calls.
type LockResult struct {
	Status int
	Body   map[string]any
}

type QueuePayload struct {
	Owner    string `json:"owner"`
	Model    string `json:"model"`
	Position int    `json:"position"`
}

type StatusSnapshot struct {
	Mode       string
	Switching  bool
	Model      string
	Locked     string
	LockOwners []string
	LockQueue  []QueuePayload
	Inflight   map[string]int
}

// ── events ──────────────────────────────────────────────────────────

type evAcquire struct {
	req   AcquireRequest
	reply chan AcqResult
}

type evRelease struct{ key string }
type evGrantTaken struct{ ev *evAcquire }
type evAbandon struct{ ev *evAcquire }
type evDrainTimeout struct{ gen int64 }

type evSwapDone struct {
	gen int64
	err error
}

type evLock struct {
	model      string
	useCurrent bool // {"lock": true} - pin the currently-active model
	owner      string
	wait       bool
	reply      chan LockResult
}

type evUnlock struct {
	owner string // empty = clear all owners
	reply chan LockResult
}

type evStatus struct{ reply chan StatusSnapshot }

type pendingSwap struct {
	gen      int64
	mode     string
	preset   *Preset // llm target; nil for comfy/train
	drainKey string
	started  bool
	waiters  []*evAcquire
}

// ── Scheduler ───────────────────────────────────────────────────────

type Scheduler struct {
	cfg     SchedulerConfig
	orch    Orchestrator
	presets *PresetStore
	logf    func(string, ...any)

	events chan any
	done   chan struct{}

	// Loop-owned state (never touched off the loop goroutine):
	st            *State
	inflight      map[string]int
	pending       *pendingSwap
	deferred      []*evAcquire
	grantedWaiter map[*evAcquire]string // waiter -> in-flight key, until evGrantTaken/evAbandon
	gen           int64
}

func NewScheduler(cfg SchedulerConfig, orch Orchestrator, presets *PresetStore, logf func(string, ...any)) (*Scheduler, error) {
	if logf == nil {
		logf = func(string, ...any) {}
	}
	st, err := LoadState(cfg.StatePath)
	if err != nil {
		return nil, err
	}
	s := &Scheduler{
		cfg:           cfg,
		orch:          orch,
		presets:       presets,
		logf:          logf,
		events:        make(chan any),
		done:          make(chan struct{}),
		st:            st,
		inflight:      map[string]int{},
		grantedWaiter: map[*evAcquire]string{},
	}
	// Reconcile on-disk mode vs what Docker says is running: the running
	// container wins (proxy may have crashed mid-swap).
	actual := orch.CurrentMode()
	if actual != st.Mode && !(actual == "idle" && st.Mode == "idle") {
		s.logf("state reconciliation: on-disk mode=%q, running=%q - trusting running container", st.Mode, actual)
		st.Mode = actual
		s.persist()
	}
	now := time.Now()
	s.expireLockIfNeeded(now)
	s.pruneQueue(now)
	return s, nil
}

func (s *Scheduler) Run()   { s.loop() }
func (s *Scheduler) Close() { close(s.done) }

func (s *Scheduler) loop() {
	for {
		select {
		case ev := <-s.events:
			s.handle(ev)
		case <-s.done:
			return
		}
	}
}

func (s *Scheduler) handle(ev any) {
	switch e := ev.(type) {
	case *evAcquire:
		s.handleAcquire(e)
	case *evRelease:
		if s.inflight[e.key] > 0 {
			s.inflight[e.key]--
		}
		s.maybeStartSwap()
	case *evGrantTaken:
		delete(s.grantedWaiter, e.ev)
	case *evAbandon:
		s.handleAbandon(e.ev)
	case *evDrainTimeout:
		if s.pending != nil && s.pending.gen == e.gen && !s.pending.started {
			n := s.inflight[s.pending.drainKey]
			s.logf("drain grace %s elapsed with %d still in-flight on %q - swapping anyway", s.cfg.DrainGrace, n, s.pending.drainKey)
			s.startSwap()
		}
	case *evSwapDone:
		s.handleSwapDone(e)
	case *evLock:
		s.handleLock(e)
	case *evUnlock:
		s.handleUnlock(e)
	case *evStatus:
		e.reply <- s.snapshot()
	}
}

// ── public API (handlers) ───────────────────────────────────────────

// Acquire asks for GPU capacity, blocking until granted/rejected or ctx
// cancel. On cancel the wait is abandoned on the loop (a grant already in
// flight is unwound so in-flight counts never leak).
func (s *Scheduler) Acquire(ctx context.Context, req AcquireRequest) AcqResult {
	ev := &evAcquire{req: req, reply: make(chan AcqResult, 1)}
	s.events <- ev
	select {
	case res := <-ev.reply:
		if res.Granted {
			s.events <- &evGrantTaken{ev: ev}
		}
		return res
	case <-ctx.Done():
		s.events <- &evAbandon{ev: ev}
		return AcqResult{OK: false, Status: 499, ErrMsg: "client disconnected", ErrType: "client_closed"}
	}
}

func (s *Scheduler) Release(key string) {
	if key == "" {
		return
	}
	s.events <- &evRelease{key: key}
}

func (s *Scheduler) Lock(model string, useCurrent bool, owner string, wait bool) LockResult {
	ev := &evLock{model: model, useCurrent: useCurrent, owner: owner, wait: wait, reply: make(chan LockResult, 1)}
	s.events <- ev
	return <-ev.reply
}

func (s *Scheduler) Unlock(owner string) LockResult {
	ev := &evUnlock{owner: owner, reply: make(chan LockResult, 1)}
	s.events <- ev
	return <-ev.reply
}

func (s *Scheduler) Status() StatusSnapshot {
	ev := &evStatus{reply: make(chan StatusSnapshot, 1)}
	s.events <- ev
	return <-ev.reply
}

// ── acquire paths ───────────────────────────────────────────────────

func (s *Scheduler) handleAcquire(ev *evAcquire) {
	now := time.Now()
	s.expireLockIfNeeded(now)
	s.pruneQueue(now)

	if s.pending != nil {
		// A swap is draining or running. Join if this request targets the
		// same swap; defer otherwise (reprocessed after swap completion).
		if s.pending.matches(ev.req) {
			s.pending.waiters = append(s.pending.waiters, ev)
		} else {
			s.deferred = append(s.deferred, ev)
		}
		return
	}
	if ev.req.Mode != "llm" {
		s.acquireMode(ev)
		return
	}
	s.acquireLLM(ev, now)
}

func (p *pendingSwap) matches(req AcquireRequest) bool {
	if req.Mode != p.mode {
		return false
	}
	if p.mode == "llm" {
		return p.preset != nil && req.Preset != nil && req.Preset.Name == p.preset.Name
	}
	return true
}

// residentKey is the in-flight key requests are counted under right now.
func (s *Scheduler) residentKey() string {
	if s.st.Mode == "llm" {
		if s.st.Model != "" {
			return s.st.Model
		}
		return "llm"
	}
	return s.st.Mode
}

// residentPreset is the preset currently loaded, nil if unknown/deleted.
func (s *Scheduler) residentPreset() *Preset {
	if s.st.Mode != "llm" || s.st.Model == "" {
		return nil
	}
	return s.presets.ByName(s.st.Model)
}

func (s *Scheduler) grant(ev *evAcquire, key, serveAs string) {
	s.inflight[key]++
	ev.reply <- AcqResult{OK: true, Granted: true, Key: key, ServeAs: serveAs}
}

func (s *Scheduler) reject(ev *evAcquire, status int, errType, msg string) {
	ev.reply <- AcqResult{OK: false, Status: status, ErrType: errType, ErrMsg: msg}
}

// tryServeInPlace grants on the resident model when it advertises the
// requested capability (R3) - the swap is skipped entirely.
func (s *Scheduler) tryServeInPlace(ev *evAcquire) bool {
	if ev.req.Capability == "" {
		return false
	}
	resident := s.residentPreset()
	if resident == nil || !resident.HasCapability(ev.req.Capability) {
		return false
	}
	s.logf("serve-in-place: request for %q handled by resident %q (capability %q, no swap)",
		firstNonEmpty(ev.req.RawModel, "cap:"+ev.req.Capability), resident.Name, ev.req.Capability)
	s.grant(ev, resident.Name, resident.ModelID())
	return true
}

func (s *Scheduler) acquireLLM(ev *evAcquire, now time.Time) {
	p := ev.req.Preset
	locked := s.st.Locked != ""

	if locked {
		if p != nil && p.Name != s.st.Locked {
			if s.tryServeInPlace(ev) {
				return
			}
			s.reject(ev, 422, "model_unavailable", fmt.Sprintf(
				"model lock active on %q: refusing to swap to %q (POST /mode {\"lock\": false} to unlock)",
				s.st.Locked, p.Name))
			return
		}
		if p == nil && ev.req.RawModel != "" && ev.req.RawModel != s.st.Locked {
			if s.tryServeInPlace(ev) {
				return
			}
			s.reject(ev, 422, "model_unavailable", fmt.Sprintf(
				"model lock active on %q: rejecting unknown model %q (passthrough would silently run on the locked model)",
				s.st.Locked, ev.req.RawModel))
			return
		}
		// Request names the locked preset, the locked model verbatim, or
		// carries no model: allowed. Swap to the locked preset if it is not
		// actually resident yet (lock pins a name, requests drive the swap).
		if p == nil {
			if rp := s.residentPreset(); rp != nil {
				s.refreshLockExpiry(now)
				s.grant(ev, s.residentKey(), "")
			} else if s.st.Mode == "llm" {
				s.refreshLockExpiry(now)
				s.grant(ev, s.residentKey(), "")
			} else {
				s.swapOrJoin(ev, "llm", s.presets.ByName(s.st.Locked))
			}
			return
		}
		if s.st.Mode == "llm" && s.st.Model == p.Name {
			s.refreshLockExpiry(now)
			s.grant(ev, p.Name, "")
			return
		}
		s.swapOrJoin(ev, "llm", p)
		return
	}

	// Unlocked.
	if p != nil {
		if s.st.Mode == "llm" && s.st.Model == p.Name {
			s.grant(ev, p.Name, "")
			return
		}
		if s.tryServeInPlace(ev) {
			return
		}
		s.swapOrJoin(ev, "llm", p)
		return
	}
	if ev.req.Capability != "" {
		if s.tryServeInPlace(ev) {
			return
		}
		target := s.resolveCapability(ev.req.Capability)
		if target == nil {
			s.reject(ev, 422, "model_unavailable",
				fmt.Sprintf("no preset advertises capability %q and no model was specified", ev.req.Capability))
			return
		}
		if s.st.Mode == "llm" && s.st.Model == target.Name {
			s.grant(ev, target.Name, "")
			return
		}
		s.logf("capability %q resolved to preset %q (no resident match)", ev.req.Capability, target.Name)
		s.swapOrJoin(ev, "llm", target)
		return
	}
	// Passthrough: unknown/empty model, llama-server decides.
	if s.st.Mode == "llm" {
		s.grant(ev, s.residentKey(), "")
		return
	}
	preset := s.presets.ByName(s.st.Model)
	if preset == nil {
		s.reject(ev, 503, "server_error",
			"cannot enter LLM mode without an active preset; POST /mode with {\"mode\":\"llm\", \"model\":\"<preset>\"}")
		return
	}
	s.swapOrJoin(ev, "llm", preset)
}

func (s *Scheduler) acquireMode(ev *evAcquire) {
	mode := ev.req.Mode
	if s.st.Locked != "" {
		s.reject(ev, 503, "server_error", fmt.Sprintf(
			"model lock active on %q: refusing to leave llm mode for %q (POST /mode {\"lock\": false} to unlock)",
			s.st.Locked, mode))
		return
	}
	if s.st.Mode == mode {
		s.grant(ev, mode, "")
		return
	}
	s.swapOrJoin(ev, mode, nil)
}

// resolveCapability picks a preset advertising cap (deterministic: first by
// sorted model ID).
func (s *Scheduler) resolveCapability(cap string) *Preset {
	all := s.presets.All()
	ids := make([]string, 0, len(all))
	for id, p := range all {
		if p.HasCapability(cap) {
			ids = append(ids, id)
		}
	}
	if len(ids) == 0 {
		return nil
	}
	sort.Strings(ids)
	return all[ids[0]]
}

// ── swap machinery ──────────────────────────────────────────────────

func (s *Scheduler) swapOrJoin(ev *evAcquire, mode string, preset *Preset) {
	if mode == "llm" && preset == nil {
		s.reject(ev, 503, "server_error", "no preset resolved for llm swap")
		return
	}
	s.gen++
	drainKey := ""
	if s.st.Mode != "idle" {
		drainKey = s.residentKey()
	}
	s.pending = &pendingSwap{gen: s.gen, mode: mode, preset: preset, drainKey: drainKey, waiters: []*evAcquire{ev}}
	target := mode
	if preset != nil {
		target = preset.Name
	}
	s.logf("swap requested: %s -> %s (draining %q, grace %s)", s.st.Mode, target, drainKey, s.cfg.DrainGrace)
	gen := s.gen
	time.AfterFunc(s.cfg.DrainGrace, func() {
		select {
		case s.events <- &evDrainTimeout{gen: gen}:
		case <-s.done:
		}
	})
	s.maybeStartSwap()
}

func (s *Scheduler) maybeStartSwap() {
	if s.pending == nil || s.pending.started {
		return
	}
	if s.pending.drainKey != "" && s.inflight[s.pending.drainKey] > 0 {
		return // still draining
	}
	s.startSwap()
}

func (s *Scheduler) startSwap() {
	p := s.pending
	p.started = true
	target := p.mode
	if p.preset != nil {
		target = p.preset.Name
	}
	s.logf("swap starting: %s -> %s", s.st.Mode, target)
	go s.doSwap(p.gen, p.mode, p.preset)
}

func (s *Scheduler) doSwap(gen int64, mode string, preset *Preset) {
	var err error
	switch mode {
	case "llm":
		if e := s.orch.EnsurePresetAssets(preset, s.cfg.AssetsDir); e != nil {
			err = e
			break
		}
		if e := s.orch.SpawnLlama(preset); e != nil {
			err = e
			break
		}
		if !s.orch.WaitHealthy(LlamaService, s.cfg.HealthTimeout) {
			err = &OrchestratorError{Msg: fmt.Sprintf("timeout loading %s", preset.DisplayName)}
		}
	case "comfyui":
		if e := s.orch.SpawnComfyUI(); e != nil {
			err = e
			break
		}
		if !s.orch.WaitHealthy(ComfyUIService, s.cfg.HealthTimeout) {
			err = &OrchestratorError{Msg: "timeout waiting for comfyui healthcheck"}
		}
	case "train":
		if e := s.orch.SpawnTrain(); e != nil {
			err = e
			break
		}
		if !s.orch.WaitHealthy(TrainService, s.cfg.HealthTimeout) {
			err = &OrchestratorError{Msg: "timeout waiting for lora-train healthcheck"}
		}
	default:
		err = &OrchestratorError{Msg: fmt.Sprintf("unknown swap target mode %q", mode)}
	}
	select {
	case s.events <- &evSwapDone{gen: gen, err: err}:
	case <-s.done:
	}
}

func (s *Scheduler) handleSwapDone(ev *evSwapDone) {
	if s.pending == nil || s.pending.gen != ev.gen {
		s.logf("stale swap completion (gen %d) ignored", ev.gen)
		return
	}
	p := s.pending
	s.pending = nil
	target := p.mode
	if p.preset != nil {
		target = p.preset.Name
	}
	if ev.err != nil {
		s.logf("swap to %s failed: %v", target, ev.err)
		for _, w := range p.waiters {
			s.reject(w, 503, "server_error", ev.err.Error())
		}
	} else {
		s.st.Mode = p.mode
		if p.mode == "llm" && p.preset != nil {
			s.st.Model = p.preset.Name
		}
		s.persist()
		s.logf("swap complete: mode=%s model=%s", s.st.Mode, s.st.Model)
		key := p.mode
		serveAs := ""
		if p.preset != nil {
			key = p.preset.Name
		}
		for _, w := range p.waiters {
			s.grantedWaiter[w] = key
			s.inflight[key]++
			w.reply <- AcqResult{OK: true, Granted: true, Key: key, ServeAs: serveAs}
		}
	}
	// Reprocess requests that arrived mid-swap for a different target.
	deferred := s.deferred
	s.deferred = nil
	for _, ev := range deferred {
		s.handleAcquire(ev)
	}
}

func (s *Scheduler) handleAbandon(ev *evAcquire) {
	if key, ok := s.grantedWaiter[ev]; ok {
		// Granted but never taken: unwind the in-flight count.
		delete(s.grantedWaiter, ev)
		if s.inflight[key] > 0 {
			s.inflight[key]--
		}
		s.maybeStartSwap()
		return
	}
	if s.pending != nil {
		for i, w := range s.pending.waiters {
			if w == ev {
				s.pending.waiters = append(s.pending.waiters[:i], s.pending.waiters[i+1:]...)
				return
			}
		}
	}
	for i, w := range s.deferred {
		if w == ev {
			s.deferred = append(s.deferred[:i], s.deferred[i+1:]...)
			return
		}
	}
}

// ── lock / queue (R1) ───────────────────────────────────────────────

func (s *Scheduler) queuePayload() []QueuePayload {
	out := make([]QueuePayload, 0, len(s.st.LockQueue))
	for i, e := range s.st.LockQueue {
		out = append(out, QueuePayload{Owner: e.Owner, Model: e.Model, Position: i + 1})
	}
	return out
}

func (s *Scheduler) lockBody() map[string]any {
	return map[string]any{"locked": nilIfEmpty(s.st.Locked), "lock_owners": s.st.SortedOwners()}
}

func (s *Scheduler) handleLock(ev *evLock) {
	now := time.Now()
	s.expireLockIfNeeded(now)
	s.pruneQueue(now)

	lockName := ev.model
	if ev.useCurrent {
		lockName = s.st.Model
		if lockName == "" {
			ev.reply <- LockResult{400, map[string]any{"error": "no active model to lock"}}
			return
		}
	}
	owner := ev.owner
	if owner == "" {
		owner = "default"
	}

	free := len(s.st.LockOwners) == 0
	same := s.st.Locked == lockName
	// FIFO gate on a FREE lock: only the queue head may take it, and only
	// for the model it queued for. Joining the already-locked model's owner
	// set is never gated (no eviction risk).
	behindQueue := free && len(s.st.LockQueue) > 0 &&
		(s.st.LockQueue[0].Owner != owner || s.st.LockQueue[0].Model != lockName)

	if !same && (!free || behindQueue) {
		var why string
		if s.st.Locked != "" {
			why = fmt.Sprintf("model lock active on %q (owners: %v)", s.st.Locked, s.st.SortedOwners())
		} else {
			head := s.st.LockQueue[0]
			why = fmt.Sprintf("lock free but queue head is %s->%s", head.Owner, head.Model)
		}
		if !ev.wait {
			ev.reply <- LockResult{409, map[string]any{
				"error":       why + "; refusing to hijack - retry with wait to join the FIFO queue",
				"locked":      nilIfEmpty(s.st.Locked),
				"lock_owners": s.st.SortedOwners(),
				"lock_queue":  s.queuePayload(),
			}}
			return
		}
		pos := s.enqueue(owner, lockName, now)
		s.persist()
		s.logf("lock queue: %s waits for %q at position %d", owner, lockName, pos)
		ev.reply <- LockResult{202, map[string]any{
			"queued":      true,
			"position":    pos,
			"locked":      nilIfEmpty(s.st.Locked),
			"lock_owners": s.st.SortedOwners(),
			"lock_queue":  s.queuePayload(),
		}}
		return
	}

	if s.st.Model != lockName {
		s.logf("warning: locking %q while active model is %q - lock pins a model that is not running", lockName, s.st.Model)
	}
	s.st.Locked = lockName
	found := false
	for _, o := range s.st.LockOwners {
		if o == owner {
			found = true
			break
		}
	}
	if !found {
		s.st.LockOwners = append(s.st.LockOwners, owner)
	}
	// Acquiring drops any queue entry the requester held.
	s.removeFromQueue(owner)
	s.st.LockExpiresAt = now.Add(s.cfg.LockTTL).Unix()
	s.persist()
	s.logf("model lock set: %s (owner %s, TTL %s)", lockName, owner, s.cfg.LockTTL)
	ev.reply <- LockResult{200, s.lockBody()}
}

func (s *Scheduler) handleUnlock(ev *evUnlock) {
	if ev.owner != "" {
		owners := s.st.LockOwners[:0]
		for _, o := range s.st.LockOwners {
			if o != ev.owner {
				owners = append(owners, o)
			}
		}
		s.st.LockOwners = owners
		// Unlocking also abandons any queued wait this owner had.
		s.removeFromQueue(ev.owner)
		if len(s.st.LockOwners) == 0 {
			s.st.Locked = ""
			s.st.LockExpiresAt = 0
		}
	} else {
		s.st.Locked = ""
		s.st.LockOwners = nil
		s.st.LockExpiresAt = 0
	}
	s.persist()
	s.logf("model lock cleared")
	ev.reply <- LockResult{200, s.lockBody()}
}

// enqueue idempotently places owner in the queue; a re-enqueue updates the
// model in place (keeps position). Returns 1-based position.
func (s *Scheduler) enqueue(owner, model string, now time.Time) int {
	for i, e := range s.st.LockQueue {
		if e.Owner == owner {
			s.st.LockQueue[i].Model = model
			s.st.LockQueue[i].TS = now.Unix()
			return i + 1
		}
	}
	s.st.LockQueue = append(s.st.LockQueue, QueueEntry{Owner: owner, Model: model, TS: now.Unix()})
	return len(s.st.LockQueue)
}

func (s *Scheduler) removeFromQueue(owner string) {
	out := s.st.LockQueue[:0]
	for _, e := range s.st.LockQueue {
		if e.Owner != owner {
			out = append(out, e)
		}
	}
	s.st.LockQueue = out
}

func (s *Scheduler) expireLockIfNeeded(now time.Time) {
	if s.st.Locked != "" && s.st.LockExpiresAt > 0 && now.Unix() > s.st.LockExpiresAt {
		s.logf("model lock on %q expired (TTL %s); clearing", s.st.Locked, s.cfg.LockTTL)
		s.st.Locked = ""
		s.st.LockOwners = nil
		s.st.LockExpiresAt = 0
		s.persist()
	}
}

func (s *Scheduler) refreshLockExpiry(now time.Time) {
	if s.st.Locked != "" {
		s.st.LockExpiresAt = now.Add(s.cfg.LockTTL).Unix()
	}
}

// pruneQueue drops queue entries whose owner stopped polling (TTL elapsed).
func (s *Scheduler) pruneQueue(now time.Time) {
	ttl := int64(s.cfg.LockTTL.Seconds())
	out := s.st.LockQueue[:0]
	for _, e := range s.st.LockQueue {
		if e.TS+ttl >= now.Unix() {
			out = append(out, e)
		} else {
			s.logf("lock queue: dropping stale entry %s->%s", e.Owner, e.Model)
		}
	}
	s.st.LockQueue = out
}

// ── status / persistence ────────────────────────────────────────────

func (s *Scheduler) snapshot() StatusSnapshot {
	return StatusSnapshot{
		Mode:       s.st.Mode,
		Switching:  s.pending != nil,
		Model:      s.st.Model,
		Locked:     s.st.Locked,
		LockOwners: s.st.SortedOwners(),
		LockQueue:  s.queuePayload(),
		Inflight:   copyMap(s.inflight),
	}
}

func (s *Scheduler) persist() {
	if err := SaveState(s.cfg.StatePath, s.st); err != nil {
		s.logf("failed to persist state: %v (continuing in-memory)", err)
	}
}

func nilIfEmpty(v string) any {
	if v == "" {
		return nil
	}
	return v
}

func copyMap(in map[string]int) map[string]int {
	out := make(map[string]int, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

func firstNonEmpty(a, b string) string {
	if a != "" {
		return a
	}
	return b
}
