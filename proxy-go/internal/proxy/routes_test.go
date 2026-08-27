package proxy

import (
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"
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

// ── 1. loader ────────────────────────────────────────────────────────

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

// ── 2. server-level routing ──────────────────────────────────────────

const routesForServer = `
[default]
chain = ["a", "b"]

[resident]
chain = []
`

// startServerWithRoutes is startServer + a routes file (delegated from
// server_test.go's startServer). NewServer takes a routes store here.
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

func strContains(haystack, needle string) bool {
	for i := 0; i+len(needle) <= len(haystack); i++ {
		if haystack[i:i+len(needle)] == needle {
			return true
		}
	}
	return false
}

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
	if !strContains(errMsg(body), "unknown route alias") {
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
		mm, ok := m.(map[string]any)
		if !ok {
			continue
		}
		meta, _ := mm["meta"].(map[string]any)
		alias, _ := meta["alias"].(bool)
		if !alias {
			continue
		}
		id, _ := mm["id"].(string)
		found[id] = true
		if meta["loaded"] != true {
			t.Fatalf("alias %v should be loaded (resident a in chain / empty chain), meta: %#v", id, meta)
		}
	}
	if !found["auto"] || !found["auto:resident"] {
		t.Fatalf("expected auto + auto:resident aliases, got %v", found)
	}
}
