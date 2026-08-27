// HTTP surface: /health, /v1/models, /mode (lock/unlock/switch), /lock,
// /unlock, /status aliases, /metrics, and transparent forwarding to the GPU
// services with SSE flushing. Handlers are thin; decisions live on the
// scheduler loop.
package proxy

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"strconv"
	"strings"
	"time"
)

type Server struct {
	sched   *Scheduler
	presets *PresetStore
	routes  *RouteStore
	cfg     ServerConfig
	logf    func(string, ...any)
	anth    *AnthropicTranslator
}

type ServerConfig struct {
	VRAMLimitGB   float64
	VRAMReserveGB float64
}

var hopByHop = map[string]bool{
	"host": true, "connection": true, "transfer-encoding": true,
	"content-length": true, "keep-alive": true, "proxy-authenticate": true,
	"proxy-authorization": true, "te": true, "trailer": true, "upgrade": true,
}

// upstreamClient has no total timeout (long SSE streams must survive); only
// the dial is bounded. See forwardTo.
var upstreamClient = &http.Client{
	Transport: &http.Transport{
		DialContext:           (&net.Dialer{Timeout: 10 * time.Second}).DialContext,
		ResponseHeaderTimeout: 0, // headers can legitimately take a whole generation
		IdleConnTimeout:       90 * time.Second,
	},
}

func NewServer(sched *Scheduler, presets *PresetStore, routes *RouteStore, cfg ServerConfig, logf func(string, ...any)) *Server {
	return &Server{sched: sched, presets: presets, routes: routes, cfg: cfg, logf: logf, anth: NewAnthropicTranslator(logf)}
}

func (s *Server) log(msg string) {
	if s.logf != nil {
		s.logf("%s", msg)
	}
}

func jsonReply(w http.ResponseWriter, status int, payload any) {
	body, err := json.Marshal(payload)
	if err != nil {
		w.WriteHeader(500)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Content-Length", strconv.Itoa(len(body)))
	w.WriteHeader(status)
	w.Write(body)
}

var errPlain = func(w http.ResponseWriter, status int, msg string, code int, typ string) {
	jsonReply(w, status, map[string]any{
		"error": map[string]any{"message": msg, "type": typ, "code": code},
	})
}

// ── routing ─────────────────────────────────────────────────────────

func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	path := r.URL.Path
	switch {
	case path == "/health" && r.Method == "GET":
		s.handleHealth(w)
	case path == "/v1/models" && r.Method == "GET":
		s.handleModels(w)
	case path == "/v1/presets" && r.Method == "POST":
		s.handlePresetRegister(w, r)
	case strings.HasPrefix(path, "/v1/presets/") && r.Method == "DELETE":
		s.handlePresetDelete(w, r)
	case path == "/mode" && r.Method == "GET":
		s.handleModeGet(w)
	case path == "/mode" && r.Method == "POST":
		s.handleModePost(w, r)
	case (path == "/lock" || path == "/unlock") && r.Method == "POST":
		s.handleLockAlias(w, r, path == "/lock")
	case path == "/status" && r.Method == "GET":
		s.handleStatus(w)
	case path == "/v1/messages" && r.Method == "POST":
		s.anth.Serve(s, w, r)
	case r.Method == "OPTIONS":
		s.handleOptions(w, r)
	default:
		s.forward(w, r)
	}
}

func (s *Server) handleHealth(w http.ResponseWriter) {
	snap := s.sched.Status()
	if snap.Switching {
		jsonReply(w, 200, map[string]any{"status": "switching", "mode": snap.Mode})
		return
	}
	jsonReply(w, 200, map[string]any{"status": "ok", "mode": snap.Mode})
}

func (s *Server) handleStatus(w http.ResponseWriter) {
	snap := s.sched.Status()
	jsonReply(w, 200, map[string]any{
		"mode":             snap.Mode,
		"switching":        snap.Switching,
		"model":            snap.Model,
		"locked":           nilIfEmpty(snap.Locked),
		"lock_owners":      snap.LockOwners,
		"lock_queue":       snap.LockQueue,
		"lock_expires_at":  nilIfZero(snap.LockExpiresAt),
		"lock_ttl_seconds": snap.LockTTLSec,
		"inflight":         snap.Inflight,
	})
}

func (s *Server) handleModeGet(w http.ResponseWriter) {
	snap := s.sched.Status()
	jsonReply(w, 200, map[string]any{
		"mode":             snap.Mode,
		"switching":        snap.Switching,
		"model":            nilIfEmpty(snap.Model),
		"locked":           nilIfEmpty(snap.Locked),
		"lock_owners":      snap.LockOwners,
		"lock_queue":       snap.LockQueue,
		"lock_expires_at":  nilIfZero(snap.LockExpiresAt),
		"lock_ttl_seconds": snap.LockTTLSec,
	})
}

func (s *Server) handleModels(w http.ResponseWriter) {
	if err := s.presets.Reload(); err != nil {
		s.log(fmt.Sprintf("preset reload failed: %v", err))
	}
	snap := s.sched.Status()
	data := []map[string]any{}
	for id, p := range s.presets.All() {
		desc := p.Description
		if i := strings.Index(desc, "\n"); i >= 0 {
			desc = desc[:i]
		}
		caps := map[string]any{"vision": p.HasVision()}
		data = append(data, map[string]any{
			"id":       id,
			"object":   "model",
			"created":  0,
			"owned_by": "local",
			"meta": map[string]any{
				"description":     desc,
				"capabilities":    caps,
				"capability_list": p.Capabilities,
				"name":            p.DisplayName,
				"preset":          p.Name,
				"loaded":          p.Name == snap.Model,
				"context":         p.Runtime.ContextSize,
				"reasoning":       p.Runtime.Reasoning == "on",
				"vram_gb":         p.VRAMGB,
				"mode":            snap.Mode,
			},
		})
	}
	if s.routes != nil {
		if err := s.routes.Reload(); err != nil {
			s.log(fmt.Sprintf("routes reload failed: %v", err))
		}
	}
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
	jsonReply(w, 200, map[string]any{"object": "list", "data": data})
}

// ── /mode + /lock + /unlock ─────────────────────────────────────────

func (s *Server) handleLockAlias(w http.ResponseWriter, r *http.Request, isLock bool) {
	body := readBody(r)
	var payload struct {
		Model *string `json:"lock"`
		Owner string  `json:"owner"`
		Wait  bool    `json:"wait"`
	}
	if len(body) > 0 {
		if err := json.Unmarshal(body, &payload); err != nil {
			jsonReply(w, 400, map[string]any{"error": "invalid JSON"})
			return
		}
	}
	if isLock {
		if payload.Model == nil {
			jsonReply(w, 400, map[string]any{"error": "missing model"})
			return
		}
		res := s.lockModel(w, *payload.Model, payload.Owner, payload.Wait)
		jsonReply(w, res.Status, res.Body)
		return
	}
	res := s.sched.Unlock(payload.Owner)
	jsonReply(w, res.Status, res.Body)
}

func (s *Server) handleModePost(w http.ResponseWriter, r *http.Request) {
	body := readBody(r)
	var payload map[string]any
	if err := json.Unmarshal(body, &payload); err != nil {
		jsonReply(w, 400, map[string]any{"error": "invalid JSON"})
		return
	}

	// Lock renewal first: {"renew": true, "owner": X} heartbeats the TTL.
	if rawRenew, has := payload["renew"]; has {
		renew, ok := rawRenew.(bool)
		if !ok || !renew {
			jsonReply(w, 400, map[string]any{"error": "renew must be true"})
			return
		}
		owner, _ := payload["owner"].(string)
		res := s.sched.RenewLock(owner)
		jsonReply(w, res.Status, res.Body)
		return
	}

	// Lock management: {"lock": <preset|true|false|null>} (+owner,+wait)
	if rawLock, has := payload["lock"]; has {
		owner, _ := payload["owner"].(string)
		wait, _ := payload["wait"].(bool)
		if rawLock == nil || rawLock == false {
			res := s.sched.Unlock(owner)
			jsonReply(w, res.Status, res.Body)
			return
		}
		if rawLock == true {
			res := s.sched.Lock("", true, owner, false)
			jsonReply(w, res.Status, res.Body)
			return
		}
		name := fmt.Sprint(rawLock)
		// Pre-flight: lock must name a known preset (404 like Python).
		if err := s.presets.Reload(); err != nil {
			s.log(fmt.Sprintf("preset reload failed: %v", err))
		}
		if s.presets.ByName(name) != nil {
			res := s.sched.Lock(s.presets.ByName(name).Name, false, owner, wait)
			jsonReply(w, res.Status, res.Body)
			return
		}
		jsonReply(w, 404, map[string]any{"error": fmt.Sprintf("unknown preset %q", name)})
		return
	}

	target, _ := payload["mode"].(string)
	if !ValidMode(target) {
		jsonReply(w, 400, map[string]any{"error": fmt.Sprintf("invalid mode %q; must be one of [comfyui llm train]", target)})
		return
	}
	reqModel, _ := payload["model"].(string)

	var req AcquireRequest
	req.Mode = target
	if target == "llm" && reqModel != "" {
		if err := s.presets.Reload(); err != nil {
			s.log(fmt.Sprintf("preset reload failed: %v", err))
		}
		p := s.presets.ByName(reqModel)
		if p == nil {
			jsonReply(w, 404, map[string]any{"error": fmt.Sprintf("unknown preset %q", reqModel)})
			return
		}
		if !s.checkVRAM(w, p) {
			return
		}
		req.Preset = p
		req.RawModel = reqModel
	}

	res := s.sched.Acquire(r.Context(), req)
	if !res.OK {
		status := res.Status
		if status == 0 {
			status = 503
		}
		jsonReply(w, status, map[string]any{"error": res.ErrMsg, "mode": s.sched.Status().Mode})
		return
	}
	s.sched.Release(res.Key) // /mode switch grants nothing real; drop the count
	status := 200
	jsonReply(w, status, map[string]any{
		"mode":     s.sched.Status().Mode,
		"model":    nilIfEmpty(s.sched.Status().Model),
		"switched": true,
	})
}

func (s *Server) lockModel(w http.ResponseWriter, name string, owner string, wait bool) LockResult {
	return s.sched.Lock(name, false, owner, wait)
}

// ── forwarding ──────────────────────────────────────────────────────

var swapTriggerMethods = map[string]bool{"POST": true, "PUT": true, "PATCH": true, "DELETE": true}

func (s *Server) forward(w http.ResponseWriter, r *http.Request) {
	mode, targetPath, ok := classify(r.URL.Path)
	if !ok {
		jsonReply(w, 404, map[string]any{"error": fmt.Sprintf("unknown route %q", r.URL.Path)})
		return
	}

	var body []byte
	if swapTriggerMethods[r.Method] {
		body = readBody(r)
	}

	req := AcquireRequest{Mode: mode}
	if mode == "llm" && body != nil {
		var payload map[string]any
		if err := json.Unmarshal(body, &payload); err == nil {
			if err := s.prepLLM(r, &req, payload); err != nil {
				errPlain(w, 404, err.Error(), 404, "unknown_route")
				return
			}
		}
		// VRAM gate before the scheduler sees the request (422 parity).
		if req.Preset != nil && !s.checkVRAMBody(req.Preset) {
			errPlain(w, 422, vramMsg(req.Preset, s.cfg.VRAMLimitGB, s.cfg.VRAMReserveGB), 422, "model_unavailable")
			return
		}
	}

	if !swapTriggerMethods[r.Method] {
		// Read-only: never triggers a swap; 503 if the mode isn't active.
		snap := s.sched.Status()
		active := snap.Mode == mode
		if mode == "llm" && snap.Mode == "llm" {
			active = true
		}
		if !active {
			errPlain(w, 503, fmt.Sprintf(
				"%s service is not active (current mode: %s). Switch with POST /mode {\"mode\":\"%s\"}.",
				mode, snap.Mode, mode), 503, "service_inactive")
			return
		}
		s.forwardTo(w, r, mode, targetPath, body, "")
		return
	}

	res := s.sched.Acquire(r.Context(), req)
	if !res.OK {
		status := res.Status
		if status == 0 {
			status = 503
		}
		errPlain(w, status, res.ErrMsg, status, firstNonEmpty(res.ErrType, "server_error"))
		return
	}
	defer s.sched.Release(res.Key)

	// llm JSON body handling: system-merge + serve-in-place rewrite.
	if mode == "llm" && body != nil {
		var payload map[string]any
		if json.Unmarshal(body, &payload) == nil {
			if messages, ok := payload["messages"].([]any); ok && needsSystemMerge(messages) {
				payload["messages"] = mergeSystemMessages(messages)
			}
			if res.ServeAs != "" {
				payload["model"] = res.ServeAs
			}
			if newBody, err := json.Marshal(payload); err == nil {
				body = newBody
			}
		}
	}
	s.forwardTo(w, r, mode, targetPath, body, res.Key)
}

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

func (s *Server) checkVRAM(w http.ResponseWriter, p *Preset) bool {
	if ok, msg := CheckVRAMBudget(p, s.cfg.VRAMLimitGB, s.cfg.VRAMReserveGB); !ok {
		jsonReply(w, 422, map[string]any{"error": msg})
		return false
	}
	return true
}

func (s *Server) checkVRAMBody(p *Preset) bool {
	ok, _ := CheckVRAMBudget(p, s.cfg.VRAMLimitGB, s.cfg.VRAMReserveGB)
	return ok
}

func vramMsg(p *Preset, limit, reserve float64) string {
	_, msg := CheckVRAMBudget(p, limit, reserve)
	return msg
}

// classify maps URL prefix to GPU mode + stripped path (parity with Python).
func classify(path string) (mode, target string, ok bool) {
	if strings.HasPrefix(path, "/v1/") {
		return "llm", path, true
	}
	if path == "/metrics" || path == "/tokenize" || path == "/detokenize" {
		return "llm", path, true
	}
	if strings.HasPrefix(path, "/comfyui") {
		stripped := strings.TrimPrefix(path, "/comfyui")
		return "comfyui", orRoot(stripped), true
	}
	if strings.HasPrefix(path, "/train") {
		stripped := strings.TrimPrefix(path, "/train")
		return "train", orRoot(stripped), true
	}
	return "", path, false
}

func orRoot(p string) string {
	if p == "" {
		return "/"
	}
	return p
}

// forwardTo proxies to the GPU backend, streaming SSE chunk-by-chunk. On
// upstream mid-stream death it injects an error event (an unattended agent
// must not consume a truncated stream as a full answer). A connection-level
// failure (the container died out-of-band) is reported to the scheduler so
// the next acquire respawns instead of 502-looping; key identifies the
// grant so stale reports from an already-drained model are ignored.
// peekModel best-effort extracts the "model" field from a JSON request body.
// Access logging only; routing never depends on it.
func peekModel(body []byte) string {
	var v struct {
		Model string `json:"model"`
	}
	if json.Unmarshal(body, &v) != nil {
		return ""
	}
	return v.Model
}

func orDash(s string) string {
	if s == "" {
		return "-"
	}
	return s
}

// forwardTo proxies to the GPU backend, streaming SSE chunk-by-chunk. On
// upstream mid-stream death it injects an error event (an unattended agent
// must not consume a truncated stream as a full answer). A connection-level
// failure (the container died out-of-band) is reported to the scheduler so
// the next acquire respawns instead of 502-looping; key identifies the
// grant so stale reports from an already-drained model are ignored.
//
// Every request logs a start line on entry and a done line on exit. The
// start line matters: a request queued at a busy llama.cpp slot sits silent
// for minutes, and without it there is no way to tell "request never
// arrived" from "request in flight" (2026-08-22 pi-abort forensics).
func (s *Server) forwardTo(w http.ResponseWriter, r *http.Request, mode, targetPath string, body []byte, key string) {
	start := time.Now()
	status := 0
	note := ""
	model := peekModel(body)
	s.log(fmt.Sprintf("req start %s %s model=%s body=%dB", r.Method, targetPath, orDash(model), len(body)))
	defer func() {
		s.log(fmt.Sprintf("req done %s %s model=%s status=%d dur=%s%s",
			r.Method, targetPath, orDash(model), status, time.Since(start).Round(time.Millisecond), note))
	}()
	svc, ok := Services[mode]
	if !ok {
		status = 500
		jsonReply(w, 500, map[string]any{"error": "unknown mode"})
		return
	}
	url := fmt.Sprintf("http://%s:%d%s", svc.Hostname, svc.InternalPort, targetPath)
	ctx := r.Context()
	var upstream *http.Request
	var err error
	if body != nil {
		upstream, err = http.NewRequestWithContext(ctx, r.Method, url, bytes.NewReader(body))
	} else {
		upstream, err = http.NewRequestWithContext(ctx, r.Method, url, nil)
	}
	if err != nil {
		status = 502
		note = " conn_init"
		errPlain(w, 502, fmt.Sprintf("connection failed: %v", err), 502, "server_error")
		return
	}
	for k, vs := range r.Header {
		if hopByHop[strings.ToLower(k)] {
			continue
		}
		for _, v := range vs {
			upstream.Header.Add(k, v)
		}
	}
	// No total timeout: the Python proxy used a per-read timeout, so hours-long
	// SSE streams were fine there; http.Client.Timeout is wall-clock total and
	// would kill them. Stall detection is client-cancel (ctx) + container
	// healthcheck; the dial is the only bounded phase.
	resp, err := upstreamClient.Do(upstream)
	if err != nil {
		// Client disconnects also surface here; only a live client means the
		// upstream itself is gone.
		status = 502
		if r.Context().Err() == nil {
			note = " upstream_dead"
			s.sched.NoteUpstreamDead(mode, key)
		} else {
			note = " client_gone"
		}
		errPlain(w, 502, fmt.Sprintf("upstream error: %v", err), 502, "server_error")
		return
	}
	defer resp.Body.Close()

	isStream := strings.Contains(resp.Header.Get("Content-Type"), "text/event-stream")
	for k, vs := range resp.Header {
		lk := strings.ToLower(k)
		if lk == "transfer-encoding" || lk == "connection" {
			continue
		}
		for _, v := range vs {
			w.Header().Add(k, v)
		}
	}
	w.WriteHeader(resp.StatusCode)
	status = resp.StatusCode

	buf := make([]byte, 4096)
	flush := w.(http.Flusher)
	for {
		n, readErr := resp.Body.Read(buf)
		if n > 0 {
			if _, werr := w.Write(buf[:n]); werr != nil {
				note = " client_disconnect_midstream"
				return
			}
			if isStream {
				flush.Flush()
			}
		}
		if readErr != nil {
			if readErr != io.EOF {
				note = " upstream_died_midstream"
				s.log(fmt.Sprintf("upstream %s died mid-stream: %v", svc.Hostname, readErr))
				if isStream {
					w.Write([]byte(`data: {"error":{"message":"upstream terminated mid-stream","type":"upstream_error"}}` + "\n\n"))
					flush.Flush()
				}
			}
			return
		}
	}
}

// ── small helpers ───────────────────────────────────────────────────

func readBody(r *http.Request) []byte {
	if r.Body == nil {
		return nil
	}
	defer r.Body.Close()
	body, _ := io.ReadAll(io.LimitReader(r.Body, 64<<20))
	return body
}

// ephemeralPresetBody is the JSON body for POST /v1/presets.
type ephemeralPresetBody struct {
	Name         string   `json:"name"`
	DisplayName  string   `json:"display_name"`
	VRAMGB       float64  `json:"vram_gb"`
	Capabilities []string `json:"capabilities"`
	Model        struct {
		Repo string `json:"repo"`
		File string `json:"file"`
	} `json:"model"`
	Runtime struct {
		ContextSize   int `json:"context_size"`
		ParallelSlots int `json:"parallel_slots"`
	} `json:"runtime"`
	// Base preset to inherit the full runtime from (sampling, spec decode,
	// reasoning, etc.) with context_size/parallel_slots overridden. Set by the
	// sweep so the variant runs the base's real config, not bare defaults.
	InheritFrom string `json:"inherit_from"`
}

// handlePresetRegister registers an in-memory (ephemeral) preset - the
// context sweep's throwaway variants. Never touches disk; dropped on restart.
func (s *Server) handlePresetRegister(w http.ResponseWriter, r *http.Request) {
	var b ephemeralPresetBody
	if err := json.Unmarshal(readBody(r), &b); err != nil {
		jsonReply(w, 400, map[string]any{"error": "invalid JSON"})
		return
	}
	if b.Name == "" || b.Model.File == "" || b.Model.Repo == "" {
		jsonReply(w, 422, map[string]any{"error": "name, model.repo and model.file are required"})
		return
	}
	rt := defaultRuntime()
	if b.InheritFrom != "" {
		if bp := s.presets.ByName(b.InheritFrom); bp != nil {
			rt = bp.Runtime // inherit full runtime (sampling, spec, reasoning)
		}
	}
	if b.Runtime.ContextSize > 0 {
		rt.ContextSize = b.Runtime.ContextSize
	}
	if b.Runtime.ParallelSlots > 0 {
		rt.ParallelSlots = b.Runtime.ParallelSlots
	}
	p := &Preset{
		Name: b.Name, DisplayName: firstNonEmpty(b.DisplayName, b.Name),
		VRAMGB: b.VRAMGB, Capabilities: b.Capabilities,
		Model: ModelSpec{Repo: b.Model.Repo, File: b.Model.File},
		Runtime: rt,
	}
	if b.InheritFrom != "" {
		if bp := s.presets.ByName(b.InheritFrom); bp != nil {
			p.MMProj = bp.MMProj     // vision asset
			p.Template = bp.Template // chat template
			if p.VRAMGB == 0 {
				p.VRAMGB = bp.VRAMGB
			}
			if len(p.Capabilities) == 0 {
				p.Capabilities = bp.Capabilities
			}
		}
	}
	if err := s.presets.RegisterEphemeral(p); err != nil {
		jsonReply(w, 409, map[string]any{"error": err.Error()})
		return
	}
	s.log(fmt.Sprintf("ephemeral preset registered: %s (model_id %s)", p.Name, p.ModelID()))
	jsonReply(w, 201, map[string]any{"registered": p.Name, "model_id": p.ModelID()})
}

// handlePresetDelete removes an ephemeral preset by model ID.
func (s *Server) handlePresetDelete(w http.ResponseWriter, r *http.Request) {
	modelID := strings.TrimPrefix(r.URL.Path, "/v1/presets/")
	if modelID == "" {
		jsonReply(w, 400, map[string]any{"error": "model id required"})
		return
	}
	s.presets.DeleteEphemeral(modelID)
	s.log(fmt.Sprintf("ephemeral preset deleted: %s", modelID))
	w.WriteHeader(204)
}

func (s *Server) handleOptions(w http.ResponseWriter, r *http.Request) {
	origin := r.Header.Get("Origin")
	if origin == "" {
		origin = "*"
	}
	reqMethod := r.Header.Get("Access-Control-Request-Method")
	if reqMethod == "" {
		reqMethod = "GET, POST, OPTIONS"
	}
	reqHeaders := r.Header.Get("Access-Control-Request-Headers")
	if reqHeaders == "" {
		reqHeaders = "Content-Type, Authorization"
	}
	w.Header().Set("Access-Control-Allow-Origin", origin)
	w.Header().Set("Access-Control-Allow-Methods", reqMethod)
	w.Header().Set("Access-Control-Allow-Headers", reqHeaders)
	w.Header().Set("Access-Control-Max-Age", "86400")
	w.WriteHeader(204)
}

// CheckVRAMBudget mirrors the Python budget check.
func CheckVRAMBudget(p *Preset, limitGB, reserveGB float64) (bool, string) {
	available := limitGB - reserveGB
	if p.VRAMGB > available {
		return false, fmt.Sprintf(
			"Model %q needs ~%gGB VRAM for weights, but only %gGB available after reserving %gGB for KV cache + compute buffer (total VRAM: %gGB). Use a smaller quant.",
			p.DisplayName, p.VRAMGB, available, reserveGB, limitGB)
	}
	return true, ""
}

// needsSystemMerge / mergeSystemMessages: Qwen templates reject multiple or
// out-of-position system messages; collapse them (parity with Python).
func needsSystemMerge(messages []any) bool {
	sysCount := 0
	for _, m := range messages {
		if md, ok := m.(map[string]any); ok && md["role"] == "system" {
			sysCount++
		}
	}
	if sysCount >= 2 {
		return true
	}
	for i, m := range messages {
		md, ok := m.(map[string]any)
		if !ok || md["role"] != "system" || i == 0 {
			continue
		}
		prev, ok := messages[i-1].(map[string]any)
		if !ok || prev["role"] != "system" {
			return true
		}
	}
	return false
}

func mergeSystemMessages(messages []any) []any {
	parts := []string{}
	others := []any{}
	for _, m := range messages {
		md, ok := m.(map[string]any)
		if !ok {
			others = append(others, m)
			continue
		}
		if md["role"] != "system" {
			others = append(others, m)
			continue
		}
		switch content := md["content"].(type) {
		case string:
			if strings.TrimSpace(content) != "" {
				parts = append(parts, content)
			}
		case []any:
			for _, chunk := range content {
				if cm, ok := chunk.(map[string]any); ok && cm["type"] == "text" {
					if t, ok := cm["text"].(string); ok {
						parts = append(parts, t)
					}
				}
			}
		}
	}
	if len(parts) == 0 {
		return others
	}
	merged := map[string]any{"role": "system", "content": strings.Join(parts, "\n\n")}
	return append([]any{merged}, others...)
}
