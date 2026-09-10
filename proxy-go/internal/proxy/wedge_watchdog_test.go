package proxy

import (
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// fakeReporter is the upstreamDeathReporter test double - no scheduler, no
// orchestrator, just a record of calls.
type fakeReporter struct {
	mu    sync.Mutex
	calls []struct{ mode, key string }
}

func (f *fakeReporter) NoteUpstreamDead(mode, key string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.calls = append(f.calls, struct{ mode, key string }{mode, key})
}

func (f *fakeReporter) callCount() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return len(f.calls)
}

// newTestWatchdog builds a watchdog pointed at a test health server, with the
// grace/probe timers shrunk to millisecond scale so tests don't sleep 20s.
func newTestWatchdog(t *testing.T, healthURL string, reporter upstreamDeathReporter) *WedgeWatchdog {
	t.Helper()
	return &WedgeWatchdog{
		enabled:      true,
		ninferHealth: healthURL,
		grace:        10 * time.Millisecond,
		probeTimeout: 200 * time.Millisecond,
		reporter:     reporter,
		httpClient:   &http.Client{},
	}
}

// A healthy engine after the grace period: no report. This is the common
// case - a client cancelling a request the engine was going to finish fine.
func TestWedgeWatchdogHealthyProbeDoesNotReport(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(200)
	}))
	defer srv.Close()
	reporter := &fakeReporter{}
	w := newTestWatchdog(t, srv.URL, reporter)

	w.NoteClientGone("llm", "qwen3.8-27b-nvfp4")
	time.Sleep(100 * time.Millisecond) // grace (10ms) + probe roundtrip

	if n := reporter.callCount(); n != 0 {
		t.Errorf("NoteUpstreamDead called %d times, want 0 (engine answered healthy)", n)
	}
}

// The engine stops answering (simulating the HTTP listener going
// unresponsive during materialization): probe times out, watchdog reports.
func TestWedgeWatchdogUnresponsiveProbeReports(t *testing.T) {
	// close(block) must run BEFORE srv.Close() or Close() deadlocks waiting
	// for the handler goroutine that is waiting on block - defers are LIFO,
	// so this one is declared last to run first.
	block := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		<-block // never responds within the probe's timeout
	}))
	defer srv.Close()
	defer close(block)
	reporter := &fakeReporter{}
	w := newTestWatchdog(t, srv.URL, reporter)
	w.probeTimeout = 30 * time.Millisecond // keep the test fast

	w.NoteClientGone("llm", "qwen3.8-27b-nvfp4")
	time.Sleep(200 * time.Millisecond) // grace + probe timeout + margin

	if n := reporter.callCount(); n != 1 {
		t.Fatalf("NoteUpstreamDead called %d times, want 1", n)
	}
	if got := reporter.calls[0]; got.mode != "llm" || got.key != "qwen3.8-27b-nvfp4" {
		t.Errorf("reported (mode=%q key=%q), want (llm, qwen3.8-27b-nvfp4)", got.mode, got.key)
	}
}

// A non-200 status is not healthy either.
func TestWedgeWatchdogErrorStatusReports(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(500)
	}))
	defer srv.Close()
	reporter := &fakeReporter{}
	w := newTestWatchdog(t, srv.URL, reporter)

	w.NoteClientGone("llm", "x")
	time.Sleep(100 * time.Millisecond)

	if n := reporter.callCount(); n != 1 {
		t.Errorf("NoteUpstreamDead called %d times, want 1 (health returned 500)", n)
	}
}

// A disabled watchdog (the production default) must not spawn a goroutine
// or ever report, no matter how the probe would have resolved.
func TestWedgeWatchdogDisabledIsNoop(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(500)
	}))
	defer srv.Close()
	reporter := &fakeReporter{}
	w := newTestWatchdog(t, srv.URL, reporter)
	w.enabled = false

	w.NoteClientGone("llm", "x")
	time.Sleep(100 * time.Millisecond)

	if n := reporter.callCount(); n != 0 {
		t.Errorf("NoteUpstreamDead called %d times, want 0 (watchdog disabled)", n)
	}
}

// A nil watchdog (e.g. a test Server built without NewServer) must not panic -
// call sites use s.watchdog.NoteClientGone(...) unconditionally.
func TestWedgeWatchdogNilReceiverIsSafe(t *testing.T) {
	var w *WedgeWatchdog
	w.NoteClientGone("llm", "x") // must not panic
}

// A burst of client_gone events while a grace timer is already running must
// coalesce into a single probe, not one per event - otherwise a flapping
// client during a real wedge would fire many concurrent probes/reports.
func TestWedgeWatchdogCoalescesBurst(t *testing.T) {
	var probes int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&probes, 1)
		w.WriteHeader(200)
	}))
	defer srv.Close()
	reporter := &fakeReporter{}
	w := newTestWatchdog(t, srv.URL, reporter)
	w.grace = 50 * time.Millisecond

	for i := 0; i < 5; i++ {
		w.NoteClientGone("llm", "x")
		time.Sleep(5 * time.Millisecond)
	}
	time.Sleep(150 * time.Millisecond)

	if got := atomic.LoadInt32(&probes); got != 1 {
		t.Errorf("probed %d times for one burst, want 1", got)
	}
}
