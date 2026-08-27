package proxy

import "testing"

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
