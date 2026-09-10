package proxy

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"sync"
	"time"
)

// WedgeWatchdog targets the 2026-09-10 ninfer context-materialization wedge
// (Neroued/ninfer#184, open upstream): a client disconnect while the engine
// is materializing a long-context request leaves the sole
// --max-concurrency 1 slot stuck forever - the SSE heartbeat cannot run
// during materialization, so the disconnect is never noticed, and the
// engine's own cancellation path explicitly no-ops while a context
// transaction is open. The HTTP listener itself goes unresponsive; a bare
// `docker restart ninfer_server` did NOT clear it in the 2026-09-10
// reproduction (see the ninfer-wedge-pattern memory).
//
// There is no signal for "materializing" in ninfer's API, so this cannot
// detect the wedge directly. Instead: a client_gone on the ninfer engine is
// a wedge SUSPICION. Wait a grace period (materialization for a request that
// was going to finish anyway should clear well within it), then probe with
// a plain GET /health. If the probe times out, treat it as confirmed and
// report to the scheduler via the same NoteUpstreamDead path connection-
// level upstream deaths already use - that flips scheduler state to idle,
// so the *next* acquire does a full stop+remove+create respawn
// (DockerOrchestrator.spawnCmd -> stopGPU + CreateAndStart), not a bare
// restart.
//
// Disabled by default: set LLMC_NINFER_WEDGE_WATCHDOG=1 to enable. Whether
// the respawn path alone (without also restarting model_proxy_go) actually
// clears a real wedge is UNVERIFIED - the 2026-09-10 session's only
// confirmed recovery was a full stack restart, and it isn't known whether
// that was necessity or habit. Verify against a deliberately reproduced
// wedge before relying on this in place of manual intervention.
type WedgeWatchdog struct {
	enabled      bool
	ninferHealth string // e.g. "http://ninfer-server:8080/health"
	grace        time.Duration
	probeTimeout time.Duration
	reporter     upstreamDeathReporter
	logf         func(string, ...any)
	httpClient   *http.Client

	mu      sync.Mutex
	pending bool // a grace timer is already in flight; NoteClientGone coalesces bursts
}

// upstreamDeathReporter is the slice of *Scheduler this package needs -
// narrow on purpose so tests can fake it without a real scheduler+orchestrator.
type upstreamDeathReporter interface {
	NoteUpstreamDead(mode, key string)
}

// NewWedgeWatchdog reads LLMC_NINFER_WEDGE_WATCHDOG for the enable gate.
// reporter is typically the *Scheduler; logf may be nil.
func NewWedgeWatchdog(reporter upstreamDeathReporter, logf func(string, ...any)) *WedgeWatchdog {
	enabled := os.Getenv("LLMC_NINFER_WEDGE_WATCHDOG") == "1"
	return &WedgeWatchdog{
		enabled:      enabled,
		ninferHealth: fmt.Sprintf("http://%s:%d%s", NinferService.Hostname, NinferService.InternalPort, NinferService.HealthPath),
		grace:        20 * time.Second,
		probeTimeout: 10 * time.Second,
		reporter:     reporter,
		logf:         logf,
		httpClient:   &http.Client{},
	}
}

func (w *WedgeWatchdog) log(format string, args ...any) {
	if w.logf != nil {
		w.logf(format, args...)
	}
}

// NoteClientGone records a client disconnect on a request served by the
// ninfer engine. Call sites: the " client_gone" branches in
// forwardTo/handleMessages, gated on the engine actually being ninfer - a
// disconnect against llama.cpp (which has no known wedge class) should not
// spend a goroutine on this.
//
// key/mode are threaded straight into NoteUpstreamDead if the probe fails,
// matching the existing upstream_dead call site's staleness guards (a report
// naming a model that already drained is ignored by the scheduler).
func (w *WedgeWatchdog) NoteClientGone(mode, key string) {
	if w == nil || !w.enabled {
		return
	}
	w.mu.Lock()
	if w.pending {
		w.mu.Unlock()
		return
	}
	w.pending = true
	w.mu.Unlock()

	go w.graceThenProbe(mode, key)
}

func (w *WedgeWatchdog) graceThenProbe(mode, key string) {
	defer func() {
		w.mu.Lock()
		w.pending = false
		w.mu.Unlock()
	}()
	time.Sleep(w.grace)
	if w.probeHealthy() {
		return
	}
	w.log("wedge_watchdog: ninfer /health did not respond within %s after a client_gone + %s grace - "+
		"suspected context-materialization wedge (Neroued/ninfer#184), reporting upstream-dead so the next acquire respawns",
		w.probeTimeout, w.grace)
	w.reporter.NoteUpstreamDead(mode, key)
}

// probeHealthy is the false-positive risk this whole mechanism carries: at
// max_concurrency=1 a legitimately busy engine (a large-but-finite prefill on
// a still-healthy container, if the health handler shares the request loop)
// can look identical to a wedge for the duration of the probe. The grace
// period is the mitigation, not a guarantee.
func (w *WedgeWatchdog) probeHealthy() bool {
	ctx, cancel := context.WithTimeout(context.Background(), w.probeTimeout)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, w.ninferHealth, nil)
	if err != nil {
		return false
	}
	resp, err := w.httpClient.Do(req)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode == 200
}
