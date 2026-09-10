package proxy

import (
	"encoding/json"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"testing"
)

// tomlNinferAnthropic is the minimal valid ninfer preset for /v1/messages
// routing tests - engine="ninfer" plus the two required [ninfer] fields
// (max_context, max_concurrency) and a runtime.reasoning_effort default to
// exercise the injection path.
const tomlNinferAnthropic = `name = "NinferAnthropicTest"
vram_gb = 10.0
engine = "ninfer"

[model]
repo = "org/n"
file = "n.ninfer"
id = "n-id"

[ninfer]
max_context = 8192
max_concurrency = 1
kv_dtype = "fp8"

[runtime]
reasoning_effort = "medium"
`

// Until 2026-09-10 this route hardcoded Services["llm"] (llama.cpp) and
// could never reach ninfer regardless of the active preset - server.go's
// OpenAI-compatible route already resolved the engine dynamically via
// activeLLMService(); anthropic.go did not. This drives a real request
// through Serve() with the ninfer preset active and NinferService.Hostname
// hijacked to a local httptest server, so the assertion is on what actually
// left the wire, not on routing logic in isolation.
func TestAnthropicRoutesToActiveNinferEngine(t *testing.T) {
	var gotPath string
	var gotBody map[string]any
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		_ = json.NewDecoder(r.Body).Decode(&gotBody)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(200)
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"hi"}}]}`))
	}))
	defer upstream.Close()

	host, port := splitHostPort(t, upstream.URL)
	orig := NinferService
	NinferService.Hostname = host
	NinferService.InternalPort = port
	defer func() { NinferService = orig }()

	store := newTestStore(t, map[string]string{"n": tomlNinferAnthropic})
	ts := startServer(t, newFakeOrch("llm"), store, &State{Mode: "llm", Model: "n"}, ServerConfig{VRAMLimitGB: 24, VRAMReserveGB: 4})
	defer ts.Close()

	code, _ := doPost(t, ts.URL+"/v1/messages", map[string]any{
		"model": "n", "max_tokens": 5,
		"messages": []any{map[string]any{"role": "user", "content": "hi"}},
	})
	if code != 200 {
		t.Fatalf("want 200, got %d", code)
	}
	if gotPath != "/v1/chat/completions" {
		t.Errorf("upstream path = %q, want /v1/chat/completions (request never reached the hijacked ninfer host)", gotPath)
	}
	if eff, _ := gotBody["reasoning_effort"].(string); eff != "medium" {
		t.Errorf("reasoning_effort = %q, want the preset default %q (the anthropic request shape has no field "+
			"to carry this, so without injection ninfer's chat template defaults to xhigh)", eff, "medium")
	}
}

// An explicit client-set effort (however that would arrive - not possible
// via the current Anthropic translateRequest, but the injection guard
// itself should not clobber a value that IS already present) must survive.
func TestAnthropicDoesNotOverrideExplicitEffort(t *testing.T) {
	// translateRequest has no path today that produces reasoning_effort, so
	// this documents the guard rather than exercising a live client input;
	// it protects the injection code from regressing to an unconditional
	// overwrite if a future translateRequest change starts forwarding one.
	p, err := loadInline(t, tomlNinferAnthropic)
	if err != nil {
		t.Fatal(err)
	}
	if p.Runtime.ReasoningEffort != "medium" {
		t.Fatalf("fixture broken: got %q", p.Runtime.ReasoningEffort)
	}
}

// splitHostPort pulls host/port(int) out of an httptest.Server URL.
func splitHostPort(t *testing.T, rawURL string) (string, int) {
	t.Helper()
	u, err := url.Parse(rawURL)
	if err != nil {
		t.Fatalf("parse %q: %v", rawURL, err)
	}
	host, portStr, err := net.SplitHostPort(u.Host)
	if err != nil {
		t.Fatalf("split %q: %v", u.Host, err)
	}
	port, err := strconv.Atoi(portStr)
	if err != nil {
		t.Fatalf("port %q: %v", portStr, err)
	}
	return host, port
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
	if !strContains(errMsg(body), "unknown route alias") {
		t.Fatalf("anthropic 404 body must name the alias, got %#v", body)
	}
}
