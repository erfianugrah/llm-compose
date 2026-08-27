# Slice 2: server.go + anthropic.go + main.go wiring

`routes.go` and the `AcquireRequest.Route` field and the scheduler logic
(acquireRoute/greedyServe/containsPreset) are ALREADY DONE and committed.
Do NOT touch routes.go or scheduler.go. Your job is ONLY the HTTP wiring.

## 1. server.go: NewServer gains a routes param

Current signature (4 args):

```go
func NewServer(sched *Scheduler, presets *PresetStore, cfg ServerConfig, logf func(string, ...any)) *Server {
	return &Server{sched: sched, presets: presets, cfg: cfg, logf: logf, anth: NewAnthropicTranslator(logf)}
}
```

Change to 5 args and add the field:

```go
type Server struct {
	sched   *Scheduler
	presets *PresetStore
	routes  *RouteStore
	cfg     ServerConfig
	logf    func(string, ...any)
	anth    *AnthropicTranslator
}

func NewServer(sched *Scheduler, presets *PresetStore, routes *RouteStore, cfg ServerConfig, logf func(string, ...any)) *Server {
	return &Server{sched: sched, presets: presets, routes: routes, cfg: cfg, logf: logf, anth: NewAnthropicTranslator(logf)}
}
```

## 2. server.go: resolveModel

Add this method:

```go
// resolveModel turns the raw model field into (preset, route, err).
// err is non-nil (an *aliasError) only for an unknown alias.
func (s *Server) resolveModel(model string) (*Preset, *Route, error) {
	if alias := routeAliasName(model); alias != "" {
		if err := s.routes.Reload(); err != nil {
			s.log(fmt.Sprintf("routes reload failed: %v", err))
		}
		r := s.routes.ByName(alias)
		if r == nil {
			return nil, nil, &aliasError{name: alias}
		}
		return nil, r, nil
	}
	if err := s.presets.Reload(); err != nil {
		s.log(fmt.Sprintf("preset reload failed: %v", err))
	}
	return s.presets.ByName(model), nil, nil
}
```

## 3. server.go: prepLLM returns error

Replace the whole prepLLM function body with:

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

## 4. server.go: forward handles the 404

In `forward`, the block currently reads:

```go
	if mode == "llm" && body != nil {
		var payload map[string]any
		if err := json.Unmarshal(body, &payload); err == nil {
			s.prepLLM(r, &req, payload)
		}
```

Change the `s.prepLLM(...)` call to check the error:

```go
	if mode == "llm" && body != nil {
		var payload map[string]any
		if err := json.Unmarshal(body, &payload); err == nil {
			if err := s.prepLLM(r, &req, payload); err != nil {
				errPlain(w, 404, err.Error(), 404, "unknown_route")
				return
			}
		}
```

(The VRAM gate block that follows stays unchanged.)

## 5. server.go: handleModels lists aliases

In `handleModels`, the function currently reloads presets, builds
`data := []map[string]any{}`, loops over `s.presets.All()`, then jsonReply.

After the existing preset loop, BEFORE `jsonReply(w, 200, ...)`, add:

```go
	if s.routes != nil {
		if err := s.routes.Reload(); err != nil {
			s.log(fmt.Sprintf("routes reload failed: %v", err))
		}
	}
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

## 6. anthropic.go: Serve resolves aliases

Replace this block:

```go
	var preset *Preset
	if model != "" {
		if err := s.presets.Reload(); err != nil {
			a.logf("anthropic: preset reload failed: %v", err)
		}
		preset = s.presets.ByName(model)
	}
```

with:

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

And change the Acquire call to carry Route:

```go
	res := s.sched.Acquire(r.Context(), AcquireRequest{Mode: "llm", Preset: preset, RawModel: model})
```

to:

```go
	res := s.sched.Acquire(r.Context(), AcquireRequest{Mode: "llm", Preset: preset, Route: route, RawModel: model})
```

## 7. main.go: wire routes + VRAM

After `store, err := proxy.NewPresetStore(presetsDir)` (and its error check),
add:

```go
	routesFile := envStr("LLMC_ROUTES_FILE", "/routes.toml")
	routes, err := proxy.NewRouteStore(routesFile)
	if err != nil {
		log.Fatalf("routes: %v", err)
	}
```

Add VRAM to the SchedulerConfig (after HealthTimeout):

```go
		VRAMLimitGB:   vramLimit,
		VRAMReserveGB: vramReserve,
```

And change the NewServer call to pass routes (5 args):

```go
	server := proxy.NewServer(sched, store, routes, proxy.ServerConfig{
		VRAMLimitGB:   vramLimit,
		VRAMReserveGB: vramReserve,
	}, logf)
```

## Verify

After every edit run: `cd proxy-go && go build ./...`

When all 3 files are done, run the FULL contract:
`cd proxy-go && go test ./... -race -count=1`

It must PASS. The committed contract tests (routes_test.go, scheduler_test.go,
server_test.go, anthropic_test.go) are the source of truth - do not modify them.
