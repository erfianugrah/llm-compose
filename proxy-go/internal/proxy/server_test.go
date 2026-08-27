package proxy

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"
)

// tomlBig exceeds the default 24-4=20GB budget.
const tomlBig = `name = "Big"
vram_gb = 99.0

[model]
repo = "org/big"
file = "big.gguf"
`

// startServer wires a scheduler + HTTP server with a fake orchestrator.
// Default ServerConfig is 24GB / 4GB reserve so the 10/12GB fixtures pass
// the VRAM gate (0,0 would reject every preset).
func startServer(t *testing.T, orch *fakeOrch, store *PresetStore, st *State, cfg ServerConfig) *httptest.Server {
	return startServerWithRoutes(t, orch, store, st, cfg, "")
}

func doGet(t *testing.T, url string) (int, map[string]any) {
	t.Helper()
	resp, err := http.Get(url)
	if err != nil {
		t.Fatalf("GET %s: %v", url, err)
	}
	defer resp.Body.Close()
	return readJSON(t, resp, url)
}

func doPost(t *testing.T, url string, body any) (int, map[string]any) {
	t.Helper()
	raw, err := json.Marshal(body)
	if err != nil {
		t.Fatalf("marshal body: %v", err)
	}
	return rawDo(t, "POST", url, string(raw))
}

func rawDo(t *testing.T, method, url, raw string) (int, map[string]any) {
	t.Helper()
	req, err := http.NewRequest(method, url, bytes.NewReader([]byte(raw)))
	if err != nil {
		t.Fatalf("new request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("%s %s: %v", method, url, err)
	}
	defer resp.Body.Close()
	return readJSON(t, resp, url)
}

func readJSON(t *testing.T, resp *http.Response, url string) (int, map[string]any) {
	t.Helper()
	var m map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&m); err != nil && resp.StatusCode != 204 {
		t.Fatalf("%s: status %d: bad JSON: %v", url, resp.StatusCode, err)
	}
	return resp.StatusCode, m
}

// errMsg handles both {"error":"str"} and {"error":{"message":...}} shapes.
func errMsg(m map[string]any) string {
	if e, ok := m["error"].(string); ok {
		return e
	}
	if e, ok := m["error"].(map[string]any); ok {
		if msg, ok := e["message"].(string); ok {
			return msg
		}
	}
	return ""
}

// ── 15. health ───────────────────────────────────────────────────────────

func TestHealth(t *testing.T) {
	ts := startServer(t, newFakeOrch("llm"),
		newTestStore(t, map[string]string{"a": tomlA, "b": tomlB}),
		&State{Mode: "llm", Model: "a"}, ServerConfig{})
	code, body := doGet(t, ts.URL+"/health")
	if code != 200 {
		t.Fatalf("health: %d %#v", code, body)
	}
	if body["status"] != "ok" || body["mode"] != "llm" {
		t.Fatalf("health body: %#v", body)
	}
}

// ── 16. models listing ───────────────────────────────────────────────────

func TestModels(t *testing.T) {
	ts := startServer(t, newFakeOrch("llm"),
		newTestStore(t, map[string]string{"a": tomlAVisionMMProj, "b": tomlB}),
		&State{Mode: "llm", Model: "a"}, ServerConfig{})
	code, body := doGet(t, ts.URL+"/v1/models")
	if code != 200 {
		t.Fatalf("models: %d %#v", code, body)
	}
	if body["object"] != "list" {
		t.Fatalf("expected object list: %#v", body["object"])
	}
	data, ok := body["data"].([]any)
	if !ok || len(data) != 2 {
		t.Fatalf("expected 2 models: %#v", body["data"])
	}
	byID := map[string]map[string]any{}
	for _, d := range data {
		e := d.(map[string]any)
		byID[e["id"].(string)] = e
	}
	alpha := byID["alpha"]
	if alpha == nil {
		t.Fatalf("alpha entry missing: %v", byID)
	}
	if alpha["object"] != "model" || alpha["owned_by"] != "local" {
		t.Fatalf("alpha envelope: %#v", alpha)
	}
	meta := alpha["meta"].(map[string]any)
	if meta["name"] != "Alpha" || meta["preset"] != "a" || meta["vram_gb"] != 10.0 {
		t.Fatalf("alpha meta: %#v", meta)
	}
	if meta["loaded"] != true {
		t.Fatalf("alpha should be loaded (resident): %#v", meta)
	}
	if meta["capabilities"].(map[string]any)["vision"] != true {
		t.Fatalf("alpha should have vision (mmproj set): %#v", meta["capabilities"])
	}
	beta := byID["beta"]
	if beta == nil {
		t.Fatalf("beta entry missing: %v", byID)
	}
	bmeta := beta["meta"].(map[string]any)
	if bmeta["loaded"] != false || bmeta["capabilities"].(map[string]any)["vision"] != false {
		t.Fatalf("beta meta: %#v", bmeta)
	}
}

// ── 17. mode get / status ────────────────────────────────────────────────

func TestModeGet(t *testing.T) {
	ts := startServer(t, newFakeOrch("llm"),
		newTestStore(t, map[string]string{"a": tomlA, "b": tomlB}),
		&State{Mode: "llm", Model: "a", Locked: "a", LockOwners: []string{"A"}}, ServerConfig{})
	code, body := doGet(t, ts.URL+"/mode")
	if code != 200 {
		t.Fatalf("mode get: %d %#v", code, body)
	}
	if body["mode"] != "llm" || body["switching"] != false || body["model"] != "a" {
		t.Fatalf("mode get body: %#v", body)
	}
	if body["locked"] != "a" {
		t.Fatalf("locked: %#v", body["locked"])
	}
	owners, _ := body["lock_owners"].([]any)
	if len(owners) != 1 || owners[0] != "A" {
		t.Fatalf("lock_owners: %#v", body["lock_owners"])
	}
}

func TestStatus(t *testing.T) {
	ts := startServer(t, newFakeOrch("llm"),
		newTestStore(t, map[string]string{"a": tomlA, "b": tomlB}),
		&State{Mode: "llm", Model: "a", Locked: "a", LockOwners: []string{"A"}}, ServerConfig{})
	code, body := doGet(t, ts.URL+"/status")
	if code != 200 || body["mode"] != "llm" || body["model"] != "a" || body["locked"] != "a" {
		t.Fatalf("status: %d %#v", code, body)
	}
	if _, ok := body["inflight"]; !ok {
		t.Fatalf("status should include inflight: %#v", body)
	}
}

// ── 18. mode post: preset switch / lock management ──────────────────────

func TestModePostSwitch(t *testing.T) {
	orch := newFakeOrch("llm")
	ts := startServer(t, orch,
		newTestStore(t, map[string]string{"a": tomlA, "b": tomlB}),
		&State{Mode: "llm", Model: "a"}, ServerConfig{})

	// Switch to a different known preset: 200 + switched.
	code, body := doPost(t, ts.URL+"/mode", map[string]any{"mode": "llm", "model": "beta"})
	if code != 200 || body["switched"] != true || body["mode"] != "llm" || body["model"] != "b" {
		t.Fatalf("switch to beta: %d %#v", code, body)
	}
	if got := orch.llamaNames(); len(got) != 1 || got[0] != "b" {
		t.Fatalf("expected SpawnLlama(b): %v", got)
	}

	// Switch GPU mode (no model).
	code, body = doPost(t, ts.URL+"/mode", map[string]any{"mode": "comfyui"})
	if code != 200 || body["mode"] != "comfyui" || body["switched"] != true {
		t.Fatalf("switch to comfyui: %d %#v", code, body)
	}
}

func TestModePostErrors(t *testing.T) {
	ts := startServer(t, newFakeOrch("llm"),
		newTestStore(t, map[string]string{"a": tomlA, "b": tomlB, "big": tomlBig}),
		&State{Mode: "llm", Model: "a"}, ServerConfig{})

	if code, body := doPost(t, ts.URL+"/mode", map[string]any{"mode": "llm", "model": "nope"}); code != 404 || !strings.Contains(errMsg(body), "unknown preset") {
		t.Fatalf("unknown preset: %d %#v", code, body)
	}
	// 99GB > 24-4=20: VRAM gate.
	code, body := doPost(t, ts.URL+"/mode", map[string]any{"mode": "llm", "model": "big"})
	if code != 422 || !strings.Contains(errMsg(body), "VRAM") || !strings.Contains(errMsg(body), "Big") {
		t.Fatalf("vram over: %d %#v", code, body)
	}
	if code, body := doPost(t, ts.URL+"/mode", map[string]any{"mode": "bogus"}); code != 400 || !strings.Contains(errMsg(body), "invalid mode") {
		t.Fatalf("invalid mode: %d %#v", code, body)
	}
	if code, body := rawDo(t, "POST", ts.URL+"/mode", "not json"); code != 400 || !strings.Contains(errMsg(body), "invalid JSON") {
		t.Fatalf("bad json: %d %#v", code, body)
	}
}

func TestModePostLockManagement(t *testing.T) {
	ts := startServer(t, newFakeOrch("llm"),
		newTestStore(t, map[string]string{"a": tomlA, "b": tomlB}),
		&State{Mode: "llm", Model: "a"}, ServerConfig{})

	// Lock a named preset via /mode.
	code, body := doPost(t, ts.URL+"/mode", map[string]any{"lock": "a", "owner": "A"})
	if code != 200 || body["locked"] != "a" {
		t.Fatalf("mode lock a: %d %#v", code, body)
	}
	// Lock the current model (owner B joins A's owner set).
	code, body = doPost(t, ts.URL+"/mode", map[string]any{"lock": true, "owner": "B"})
	if code != 200 || body["locked"] != "a" {
		t.Fatalf("mode lock current: %d %#v", code, body)
	}
	owners, _ := body["lock_owners"].([]any)
	if len(owners) != 2 {
		t.Fatalf("two owners expected: %#v", body["lock_owners"])
	}
	// Unknown preset: 404 like the Python server.
	code, body = doPost(t, ts.URL+"/mode", map[string]any{"lock": "nope"})
	if code != 404 || !strings.Contains(errMsg(body), "unknown preset") {
		t.Fatalf("mode lock unknown: %d %#v", code, body)
	}
	// One owner leaves: the lock still stands.
	code, body = doPost(t, ts.URL+"/mode", map[string]any{"lock": nil, "owner": "A"})
	if code != 200 || body["locked"] != "a" {
		t.Fatalf("partial unlock: %d %#v", code, body)
	}
	// Last owner leaves: lock released.
	code, body = doPost(t, ts.URL+"/mode", map[string]any{"lock": nil, "owner": "B"})
	if code != 200 || body["locked"] != nil {
		t.Fatalf("mode unlock: %d %#v", code, body)
	}
}

// ── 19. /lock + /unlock aliases ──────────────────────────────────────────

func TestLockAliases(t *testing.T) {
	ts := startServer(t, newFakeOrch("llm"),
		newTestStore(t, map[string]string{"a": tomlA, "b": tomlB}),
		&State{Mode: "llm", Model: "a"}, ServerConfig{})

	code, body := doPost(t, ts.URL+"/lock", map[string]any{"lock": "a", "owner": "A"})
	if code != 200 || body["locked"] != "a" {
		t.Fatalf("lock: %d %#v", code, body)
	}
	owners, _ := body["lock_owners"].([]any)
	if len(owners) != 1 || owners[0] != "A" {
		t.Fatalf("lock owners: %#v", body["lock_owners"])
	}

	if code, body = rawDo(t, "POST", ts.URL+"/lock", "{}"); code != 400 || !strings.Contains(errMsg(body), "missing model") {
		t.Fatalf("missing model: %d %#v", code, body)
	}
	if code, body = rawDo(t, "POST", ts.URL+"/lock", "not json"); code != 400 || !strings.Contains(errMsg(body), "invalid JSON") {
		t.Fatalf("lock bad json: %d %#v", code, body)
	}

	code, body = doPost(t, ts.URL+"/unlock", map[string]any{"owner": "A"})
	if code != 200 || body["locked"] != nil {
		t.Fatalf("unlock: %d %#v", code, body)
	}
}

// ── 20. CORS preflight ───────────────────────────────────────────────────

func TestCORSPreflight(t *testing.T) {
	ts := startServer(t, newFakeOrch("llm"),
		newTestStore(t, map[string]string{"a": tomlA, "b": tomlB}),
		&State{Mode: "llm", Model: "a"}, ServerConfig{})

	req, err := http.NewRequest("OPTIONS", ts.URL+"/v1/models", nil)
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Origin", "http://app.test")
	req.Header.Set("Access-Control-Request-Method", "POST")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != 204 {
		t.Fatalf("OPTIONS: want 204, got %d", resp.StatusCode)
	}
	if got := resp.Header.Get("Access-Control-Allow-Origin"); got != "http://app.test" {
		t.Fatalf("allow-origin: %q", got)
	}
	if got := resp.Header.Get("Access-Control-Allow-Methods"); got != "POST" {
		t.Fatalf("allow-methods: %q", got)
	}
	if got := resp.Header.Get("Access-Control-Max-Age"); got != "86400" {
		t.Fatalf("max-age: %q", got)
	}
}

// ── 21. unknown route ────────────────────────────────────────────────────

func TestTokenizeRoute(t *testing.T) {
	orch := newFakeOrch("llm")
	ts := startServer(t, orch,
		newTestStore(t, map[string]string{"a": tomlA, "b": tomlB}),
		&State{Mode: "llm", Model: "a"}, ServerConfig{})
	// POST /tokenize triggers a swap if model is already 'a' but we force a POST
	// to trigger the orchestrator. If the orchestrator succeeds, it tries to
	// forward to the (nonexistent) llama-server, resulting in a 502.
	code, body := doPost(t, ts.URL+"/tokenize", map[string]any{
		"content": "hello",
	})
	if code != 502 {
		t.Fatalf("tokenize: want 502, got %d %#v", code, body)
	}
	if e := body["error"].(map[string]any); e["type"] != "server_error" ||
		!strings.Contains(e["message"].(string), "upstream error") {
		t.Fatalf("502 body: %#v", e)
	}
}

// ── 22. forwarding ───────────────────────────────────────────────────────

func TestForwardInactive503(t *testing.T) {
	ts := startServer(t, newFakeOrch("llm"),
		newTestStore(t, map[string]string{"a": tomlA, "b": tomlB}),
		&State{Mode: "llm", Model: "a"}, ServerConfig{})
	// comfyui is not active: a read-only request must 503, not swap.
	code, body := doGet(t, ts.URL+"/comfyui/")
	if code != 503 {
		t.Fatalf("inactive forward: want 503, got %d %#v", code, body)
	}
	e := body["error"].(map[string]any)
	if e["type"] != "service_inactive" || !strings.Contains(e["message"].(string), "not active") {
		t.Fatalf("503 body: %#v", e)
	}
}

func TestForwardPassthroughUpstream502(t *testing.T) {
	// llama-server (127.x DNS name) does not exist in the test env, so an
	// otherwise-valid forward must surface the dial failure as a 502.
	ts := startServer(t, newFakeOrch("llm"),
		newTestStore(t, map[string]string{"a": tomlA, "b": tomlB}),
		&State{Mode: "llm", Model: "a"}, ServerConfig{})
	code, body := doGet(t, ts.URL+"/metrics")
	if code != 502 {
		t.Fatalf("passthrough: want 502, got %d %#v", code, body)
	}
	if e := body["error"].(map[string]any); e["type"] != "server_error" ||
		!strings.Contains(e["message"].(string), "upstream error") {
		t.Fatalf("502 body: %#v", e)
	}
}

func TestForwardSwapThen502(t *testing.T) {
	// A chat completion is a swap trigger: the swap to beta happens first,
	// then the forward to the (nonexistent) llama-server fails with 502.
	orch := newFakeOrch("llm")
	ts := startServer(t, orch,
		newTestStore(t, map[string]string{"a": tomlA, "b": tomlB}),
		&State{Mode: "llm", Model: "a"}, ServerConfig{})
	code, body := doPost(t, ts.URL+"/v1/chat/completions", map[string]any{
		"model":    "beta",
		"messages": []any{map[string]any{"role": "user", "content": "hi"}},
	})
	if code != 502 {
		t.Fatalf("swap forward: want 502, got %d %#v", code, body)
	}
	if got := orch.llamaNames(); len(got) != 1 || got[0] != "b" {
		t.Fatalf("swap must have happened before the forward, got %v", got)
	}
}

func TestForwardVRAM422(t *testing.T) {
	// The VRAM gate fires before the scheduler sees the request.
	ts := startServer(t, newFakeOrch("llm"),
		newTestStore(t, map[string]string{"a": tomlA, "b": tomlB, "big": tomlBig}),
		&State{Mode: "llm", Model: "a"}, ServerConfig{})
	code, body := doPost(t, ts.URL+"/v1/chat/completions", map[string]any{
		"model":    "big",
		"messages": []any{map[string]any{"role": "user", "content": "hi"}},
	})
	if code != 422 {
		t.Fatalf("vram gate: want 422, got %d %#v", code, body)
	}
	e := body["error"].(map[string]any)
	if e["type"] != "model_unavailable" || !strings.Contains(e["message"].(string), "VRAM") {
		t.Fatalf("422 body: %#v", e)
	}
}

// TestForwardAccessLog: every proxied request must emit a start line on
// arrival and a done line with status + duration + outcome note. The start
// line is what distinguishes "request never reached the proxy" from
// "request in flight" during incident forensics (2026-08-22 pi abort: the
// proxy logged nothing for the dying request and the blind spot cost an
// hour).
func TestForwardAccessLog(t *testing.T) {
	var mu sync.Mutex
	var lines []string
	logf := func(f string, a ...any) {
		mu.Lock()
		lines = append(lines, fmt.Sprintf(f, a...))
		mu.Unlock()
	}

	statePath := t.TempDir() + "/state.toml"
	if err := SaveState(statePath, &State{Mode: "llm", Model: "a"}); err != nil {
		t.Fatalf("SaveState: %v", err)
	}
	store := newTestStore(t, map[string]string{"a": tomlA, "b": tomlB})
	scfg := SchedulerConfig{StatePath: statePath, DrainGrace: 3 * time.Second,
		LockTTL: 900 * time.Second, HealthTimeout: time.Second}
	s, err := NewScheduler(scfg, newFakeOrch("llm"), store, logf)
	if err != nil {
		t.Fatalf("NewScheduler: %v", err)
	}
	go s.Run()
	t.Cleanup(s.Close)
	ts := httptest.NewServer(NewServer(s, store, ServerConfig{VRAMLimitGB: 24, VRAMReserveGB: 4}, logf))
	t.Cleanup(ts.Close)

	// Upstream llama-server does not exist in the test env: 502 after swap.
	code, _ := doPost(t, ts.URL+"/v1/chat/completions", map[string]any{
		"model":    "beta",
		"messages": []any{map[string]any{"role": "user", "content": "hi"}},
	})
	if code != 502 {
		t.Fatalf("want 502, got %d", code)
	}

	mu.Lock()
	defer mu.Unlock()
	joined := "\n" + strings.Join(lines, "\n")
	if !strings.Contains(joined, "req start POST /v1/chat/completions model=beta") {
		t.Fatalf("missing start line, got:\n%s", joined)
	}
	if !strings.Contains(joined, "req done POST /v1/chat/completions model=beta status=502") {
		t.Fatalf("missing done line, got:\n%s", joined)
	}
	if !strings.Contains(joined, "upstream_dead") {
		t.Fatalf("done line must note upstream death, got:\n%s", joined)
	}
}
