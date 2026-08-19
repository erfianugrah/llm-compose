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

func NewServer(sched *Scheduler, presets *PresetStore, cfg ServerConfig, logf func(string, ...any)) *Server {
	return &Server{sched: sched, presets: presets, cfg: cfg, logf: logf, anth: NewAnthropicTranslator(logf)}
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
		"mode":        snap.Mode,
		"switching":   snap.Switching,
		"model":       snap.Model,
		"locked":      nilIfEmpty(snap.Locked),
		"lock_owners": snap.LockOwners,
		"lock_queue":  snap.LockQueue,
		"inflight":    snap.Inflight,
	})
}

func (s *Server) handleModeGet(w http.ResponseWriter) {
	snap := s.sched.Status()
	jsonReply(w, 200, map[string]any{
		"mode":        snap.Mode,
		"switching":   snap.Switching,
		"model":       nilIfEmpty(snap.Model),
		"locked":      nilIfEmpty(snap.Locked),
		"lock_owners": snap.LockOwners,
		"lock_queue":  snap.LockQueue,
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

	// Lock management first: {"lock": <preset|true|false|null>} (+owner,+wait)
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
			s.prepLLM(r, &req, payload)
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
		s.forwardTo(w, r, mode, targetPath, body)
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
	s.forwardTo(w, r, mode, targetPath, body)
}

func (s *Server) prepLLM(r *http.Request, req *AcquireRequest, payload map[string]any) {
	model, _ := payload["model"].(string)
	cap := r.Header.Get("X-LLM-Capability")
	if strings.HasPrefix(model, "cap:") {
		cap = strings.TrimPrefix(model, "cap:")
		model = ""
	}
	if err := s.presets.Reload(); err != nil {
		s.log(fmt.Sprintf("preset reload failed: %v", err))
	}
	if p := s.presets.ByName(model); p != nil {
		req.Preset = p
	}
	req.RawModel = model
	req.Capability = cap
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
	if path == "/metrics" {
		return "llm", "/metrics", true
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
// must not consume a truncated stream as a full answer).
func (s *Server) forwardTo(w http.ResponseWriter, r *http.Request, mode, targetPath string, body []byte) {
	svc, ok := Services[mode]
	if !ok {
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

	buf := make([]byte, 4096)
	flush := w.(http.Flusher)
	for {
		n, readErr := resp.Body.Read(buf)
		if n > 0 {
			if _, werr := w.Write(buf[:n]); werr != nil {
				return // client disconnected mid-stream
			}
			if isStream {
				flush.Flush()
			}
		}
		if readErr != nil {
			if readErr != io.EOF {
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
