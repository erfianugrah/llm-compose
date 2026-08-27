// anthropic.go: Anthropic Messages API shim (POST /v1/messages) so
// Anthropic-native clients (Claude Code via ANTHROPIC_BASE_URL) can talk to
// the OpenAI-compatible llama-server upstream behind this proxy.
//
// Request: Anthropic Messages -> OpenAI chat.completions.
// Response: OpenAI JSON / SSE -> Anthropic message + event stream.
package proxy

import (
	"bufio"
	"bytes"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

type AnthropicTranslator struct {
	logf func(string, ...any)
}

func NewAnthropicTranslator(logf func(string, ...any)) *AnthropicTranslator {
	if logf == nil {
		logf = func(string, ...any) {}
	}
	return &AnthropicTranslator{logf: logf}
}

// anthropicErr writes an Anthropic-shaped error body.
func anthropicErr(w http.ResponseWriter, status int, typ, msg string) {
	jsonReply(w, status, map[string]any{
		"type": "error", "error": map[string]any{"type": typ, "message": msg},
	})
}

// writeAnthEvent emits one Anthropic SSE event and flushes it.
func writeAnthEvent(w http.ResponseWriter, f http.Flusher, eventType string, payload any) {
	data, err := json.Marshal(payload)
	if err != nil {
		data = []byte(`{"type":"error","error":{"type":"api_error","message":"internal encode error"}}`)
	}
	fmt.Fprintf(w, "event: %s\ndata: %s\n\n", eventType, data)
	f.Flush()
}

func randHex(n int) string {
	b := make([]byte, n)
	if _, err := rand.Read(b); err != nil {
		return strings.Repeat("0", n*2)
	}
	return hex.EncodeToString(b)
}

// toInt64 coerces JSON numbers (and friends) to int64, 0 on failure.
func toInt64(v any) int64 {
	switch n := v.(type) {
	case float64:
		return int64(n)
	case int:
		return int64(n)
	case int64:
		return n
	case json.Number:
		if i, err := n.Int64(); err == nil {
			return i
		}
	case bool:
		if n {
			return 1
		}
	}
	return 0
}

// anthropicStopReason maps an OpenAI finish_reason to an Anthropic stop_reason.
func anthropicStopReason(fr string) string {
	switch fr {
	case "tool_calls":
		return "tool_use"
	case "length":
		return "max_tokens"
	default: // stop, content_filter, or anything unknown
		return "end_turn"
	}
}

// ── request translation ─────────────────────────────────────────────

func (a *AnthropicTranslator) systemToText(sys any) string {
	switch v := sys.(type) {
	case string:
		return v
	case []any:
		parts := []string{}
		for _, b := range v {
			if m, ok := b.(map[string]any); ok && m["type"] == "text" {
				if t, ok := m["text"].(string); ok && t != "" {
					parts = append(parts, t)
				}
			}
		}
		return strings.Join(parts, "\n\n")
	}
	return ""
}

// toolResultText flattens Anthropic tool_result content to a string.
func (a *AnthropicTranslator) toolResultText(content any) string {
	switch v := content.(type) {
	case string:
		return v
	case []any:
		parts := []string{}
		for _, b := range v {
			m, ok := b.(map[string]any)
			if !ok {
				continue
			}
			if m["type"] == "text" {
				if t, ok := m["text"].(string); ok {
					parts = append(parts, t)
				}
			} else {
				a.logf("anthropic: dropping unsupported tool_result block type %v", m["type"])
			}
		}
		return strings.Join(parts, "\n")
	}
	return ""
}

// translateBlocks converts one message's Anthropic content blocks into
// OpenAI message(s). tool_result blocks (user) split out into separate
// {"role":"tool"} messages, which OpenAI requires.
func (a *AnthropicTranslator) translateBlocks(role string, blocks []any, out *[]any) {
	items := []any{}
	var calls []any
	var toolMsgs []any
	for _, b := range blocks {
		block, ok := b.(map[string]any)
		if !ok {
			continue
		}
		switch block["type"] {
		case "text":
			if t, ok := block["text"].(string); ok {
				items = append(items, map[string]any{"type": "text", "text": t})
			}
		case "image":
			src, _ := block["source"].(map[string]any)
			mt, _ := src["media_type"].(string)
			data, _ := src["data"].(string)
			if mt == "" {
				mt = "image/png"
			}
			items = append(items, map[string]any{"type": "image_url", "image_url": map[string]any{"url": fmt.Sprintf("data:%s;base64,%s", mt, data)}})
		case "tool_use":
			id, _ := block["id"].(string)
			if id == "" {
				id = "toolu_" + randHex(8)
			}
			name, _ := block["name"].(string)
			argsJSON, err := json.Marshal(block["input"])
			if err != nil || len(argsJSON) == 0 {
				argsJSON = []byte("{}")
			}
			calls = append(calls, map[string]any{"id": id, "type": "function", "function": map[string]any{"name": name, "arguments": string(argsJSON)}})
		case "tool_result":
			id, _ := block["tool_use_id"].(string)
			toolMsgs = append(toolMsgs, map[string]any{
				"role":         "tool",
				"tool_call_id": id,
				"content":      a.toolResultText(block["content"]),
			})
		default:
			a.logf("anthropic: dropping unsupported content block type %v in %s message", block["type"], role)
		}
	}
	var content any = ""
	if len(items) > 0 {
		content = items
	}
	*out = append(*out, map[string]any{"role": role, "content": content})
	if len(calls) > 0 {
		if msg, ok := (*out)[len(*out)-1].(map[string]any); ok {
			msg["tool_calls"] = calls
		}
	}
	*out = append(*out, toolMsgs...)
}

// translateRequest builds the OpenAI chat.completions payload (model field
// is filled in by Serve after scheduler resolution).
func (a *AnthropicTranslator) translateRequest(req map[string]any) map[string]any {
	payload := map[string]any{}
	msgs := []any{}
	if sys, ok := req["system"]; ok {
		if text := a.systemToText(sys); text != "" {
			msgs = append(msgs, map[string]any{"role": "system", "content": text})
		}
	}
	rawMsgs, _ := req["messages"].([]any)
	for _, rm := range rawMsgs {
		md, ok := rm.(map[string]any)
		if !ok {
			continue
		}
		role, _ := md["role"].(string)
		if role != "user" && role != "assistant" {
			a.logf("anthropic: dropping message with unsupported role %q", role)
			continue
		}
		switch c := md["content"].(type) {
		case string:
			msgs = append(msgs, map[string]any{"role": role, "content": c})
		case []any:
			a.translateBlocks(role, c, &msgs)
		default:
			msgs = append(msgs, map[string]any{"role": role, "content": ""})
		}
	}
	payload["messages"] = msgs

	if rawTools, ok := req["tools"].([]any); ok {
		tools := []any{}
		for _, rt := range rawTools {
			td, ok := rt.(map[string]any)
			if !ok {
				continue
			}
			fn := map[string]any{}
			if n, ok := td["name"].(string); ok {
				fn["name"] = n
			}
			if d, ok := td["description"].(string); ok && d != "" {
				fn["description"] = d
			}
			if sch, ok := td["input_schema"]; ok {
				fn["parameters"] = sch
			}
			tools = append(tools, map[string]any{"type": "function", "function": fn})
		}
		if len(tools) > 0 {
			payload["tools"] = tools
		}
	}

	if tc, ok := req["tool_choice"].(map[string]any); ok {
		switch tc["type"] {
		case "auto":
			payload["tool_choice"] = "auto"
		case "any":
			payload["tool_choice"] = "required"
		case "tool":
			if name, _ := tc["name"].(string); name != "" {
				payload["tool_choice"] = map[string]any{"type": "function", "function": map[string]any{"name": name}}
			}
		}
	}

	for _, k := range []string{"max_tokens", "temperature", "top_p", "top_k"} {
		if v, ok := req[k]; ok && v != nil {
			payload[k] = v
		}
	}
	if ss, ok := req["stop_sequences"].([]any); ok {
		strs := []any{}
		for _, x := range ss {
			if t, ok := x.(string); ok {
				strs = append(strs, t)
			}
		}
		if len(strs) > 0 {
			payload["stop"] = strs
		}
	}

	stream, _ := req["stream"].(bool)
	payload["stream"] = stream
	if stream {
		payload["stream_options"] = map[string]any{"include_usage": true}
	}
	return payload
}

// ── response translation ────────────────────────────────────────────

func (a *AnthropicTranslator) toolUseBlock(content []any, tc any) []any {
	tcd, ok := tc.(map[string]any)
	if !ok {
		return content
	}
	id, _ := tcd["id"].(string)
	if id == "" {
		id = "toolu_" + randHex(8)
	}
	name := ""
	var args any = map[string]any{}
	if f, ok := tcd["function"].(map[string]any); ok {
		name, _ = f["name"].(string)
		rawArgs, _ := f["arguments"].(string)
		if rawArgs != "" {
			var parsed any
			if err := json.Unmarshal([]byte(rawArgs), &parsed); err != nil {
				a.logf("anthropic: unparseable tool arguments %q: %v", rawArgs, err)
			} else {
				args = parsed
			}
		}
	}
	return append(content, map[string]any{"type": "tool_use", "id": id, "name": name, "input": args})
}

// translateResponse converts one OpenAI chat.completion into an Anthropic
// message object. reqModel is echoed back in the "model" field.
func (a *AnthropicTranslator) translateResponse(reqModel string, om map[string]any) map[string]any {
	content := []any{}
	stopReason := "end_turn"
	usage := map[string]any{"input_tokens": 0, "output_tokens": 0}
	if u, ok := om["usage"].(map[string]any); ok {
		usage["input_tokens"] = toInt64(u["prompt_tokens"])
		usage["output_tokens"] = toInt64(u["completion_tokens"])
	}
	choices, _ := om["choices"].([]any)
	if len(choices) > 0 {
		if ch, ok := choices[0].(map[string]any); ok {
			if fr, ok := ch["finish_reason"].(string); ok {
				stopReason = anthropicStopReason(fr)
			}
			if msg, ok := ch["message"].(map[string]any); ok {
				if text, ok := msg["content"].(string); ok && text != "" {
					content = append(content, map[string]any{"type": "text", "text": text})
				}
				if tcs, ok := msg["tool_calls"].([]any); ok {
					for _, tc := range tcs {
						content = a.toolUseBlock(content, tc)
					}
				}
			}
		}
	}
	return map[string]any{
		"id": "msg_" + randHex(16), "type": "message", "role": "assistant",
		"model": firstNonEmpty(reqModel, "unknown"), "content": content,
		"stop_reason": stopReason, "stop_sequence": nil, "usage": usage,
	}
}

// upstreamError renders a non-2xx upstream body as an Anthropic error.
func (a *AnthropicTranslator) upstreamError(w http.ResponseWriter, status int, body []byte) {
	msg := strings.TrimSpace(string(body))
	var om map[string]any
	if json.Unmarshal(body, &om) == nil {
		switch e := om["error"].(type) {
		case map[string]any:
			if m, ok := e["message"].(string); ok {
				msg = m
			}
		case string:
			msg = e
		}
	}
	if len(msg) > 500 {
		msg = msg[:500] + "..."
	}
	if msg == "" {
		msg = fmt.Sprintf("upstream returned HTTP %d", status)
	}
	typ := "api_error"
	if status >= 400 && status < 500 {
		typ = "invalid_request_error"
	}
	anthropicErr(w, status, typ, msg)
}

// ── streaming ───────────────────────────────────────────────────────

type streamState struct {
	logf      func(string, ...any)
	model     string
	msgID     string
	started   bool
	sentStop  bool
	blockIdx  int            // currently open Anthropic block index, -1 = none
	blockKind map[int]string // open Anthropic block index -> "text"|"tool"
	toolIdx   map[int]int    // OpenAI tool_calls[].index -> Anthropic block index
	inTokens  int64
	outTokens int64
}

func (st *streamState) ensureStarted(w http.ResponseWriter, f http.Flusher) {
	if st.started {
		return
	}
	st.started = true
	msg := map[string]any{
		"id": st.msgID, "type": "message", "role": "assistant", "model": st.model,
		"content": []any{}, "stop_reason": nil, "stop_sequence": nil,
		"usage": map[string]any{"input_tokens": st.inTokens, "output_tokens": 0},
	}
	writeAnthEvent(w, f, "message_start", map[string]any{"type": "message_start", "message": msg})
}

func (st *streamState) closeOpenBlock(w http.ResponseWriter, f http.Flusher) {
	if st.blockIdx < 0 {
		return
	}
	idx := st.blockIdx
	delete(st.blockKind, idx)
	st.blockIdx = -1
	writeAnthEvent(w, f, "content_block_stop", map[string]any{"type": "content_block_stop", "index": idx})
}

func (st *streamState) finish(reason string, w http.ResponseWriter, f http.Flusher) {
	if st.sentStop {
		return
	}
	st.sentStop = true
	if !st.started {
		st.ensureStarted(w, f)
	}
	st.closeOpenBlock(w, f)
	writeAnthEvent(w, f, "message_delta", map[string]any{
		"type": "message_delta", "delta": map[string]any{"stop_reason": anthropicStopReason(reason), "stop_sequence": nil},
		"usage": map[string]any{"output_tokens": st.outTokens},
	})
	writeAnthEvent(w, f, "message_stop", map[string]any{"type": "message_stop"})
}

// handleChunk converts one OpenAI SSE chunk into Anthropic events.
func (st *streamState) handleChunk(w http.ResponseWriter, f http.Flusher, chunk map[string]any) {
	if u, ok := chunk["usage"].(map[string]any); ok {
		if v := toInt64(u["prompt_tokens"]); v > 0 {
			st.inTokens = v
		}
		if v := toInt64(u["completion_tokens"]); v > 0 {
			st.outTokens = v
		}
	}
	choices, _ := chunk["choices"].([]any)
	if len(choices) == 0 {
		return
	}
	ch, ok := choices[0].(map[string]any)
	if !ok {
		return
	}
	if fr, ok := ch["finish_reason"].(string); ok && fr != "" {
		st.finish(fr, w, f)
		return
	}
	delta, ok := ch["delta"].(map[string]any)
	if !ok {
		return
	}
	st.ensureStarted(w, f)

	if t, ok := delta["content"].(string); ok && t != "" {
		if st.blockKind[st.blockIdx] != "text" {
			st.closeOpenBlock(w, f)
			st.blockIdx++
			st.blockKind[st.blockIdx] = "text"
			writeAnthEvent(w, f, "content_block_start", map[string]any{"type": "content_block_start", "index": st.blockIdx, "content_block": map[string]any{"type": "text", "text": ""}})
		}
		writeAnthEvent(w, f, "content_block_delta", map[string]any{"type": "content_block_delta", "index": st.blockIdx, "delta": map[string]any{"type": "text_delta", "text": t}})
	}

	tcs, _ := delta["tool_calls"].([]any)
	for _, tcr := range tcs {
		tc, ok := tcr.(map[string]any)
		if !ok {
			continue
		}
		oidx := int(toInt64(tc["index"]))
		block, known := st.toolIdx[oidx]
		if !known {
			st.closeOpenBlock(w, f)
			block = st.blockIdx + 1
			st.blockIdx = block
			st.toolIdx[oidx] = block
			st.blockKind[block] = "tool"
			id, _ := tc["id"].(string)
			if id == "" {
				id = "toolu_" + randHex(8)
			}
			name := ""
			if fn, ok := tc["function"].(map[string]any); ok {
				name, _ = fn["name"].(string)
			}
			writeAnthEvent(w, f, "content_block_start", map[string]any{"type": "content_block_start", "index": block, "content_block": map[string]any{"type": "tool_use", "id": id, "name": name, "input": map[string]any{}}})
		}
		if fn, ok := tc["function"].(map[string]any); ok {
			if frag, ok := fn["arguments"].(string); ok && frag != "" {
				writeAnthEvent(w, f, "content_block_delta", map[string]any{"type": "content_block_delta", "index": block, "delta": map[string]any{"type": "input_json_delta", "partial_json": frag}})
			}
		}
	}
}

// serveStream consumes the upstream SSE body and emits Anthropic events.
func (st *streamState) serveStream(w http.ResponseWriter, resp *http.Response, errPrefix string) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		anthropicErr(w, 500, "api_error", "streaming not supported")
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Cache-Control", "no-cache")
	w.WriteHeader(200)

	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "[DONE]" {
			st.finish("end_turn", w, flusher)
			return
		}
		var chunk map[string]any
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			st.logf("%s: skipping unparseable SSE data: %v", errPrefix, err)
			continue
		}
		st.handleChunk(w, flusher, chunk)
	}
	if err := sc.Err(); err != nil && err != io.EOF {
		st.logf("%s: upstream died mid-stream: %v", errPrefix, err)
		writeAnthEvent(w, flusher, "error", map[string]any{"type": "error", "error": map[string]any{"type": "api_error", "message": fmt.Sprintf("upstream terminated mid-stream: %v", err)}})
		return
	}
	st.finish("end_turn", w, flusher)
}

// ── route handler ───────────────────────────────────────────────────

// Serve handles POST /v1/messages: parse the Anthropic request, translate
// to OpenAI, run the scheduler, forward, and translate the response back.
func (a *AnthropicTranslator) Serve(s *Server, w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	status := 0
	note := ""
	body := readBody(r)
	var req map[string]any
	defer func() {
		model, _ := req["model"].(string)
		a.logf("req done POST /v1/messages model=%s status=%d dur=%s%s",
			orDash(model), status, time.Since(start).Round(time.Millisecond), note)
	}()
	if len(body) == 0 {
		status = 400
		anthropicErr(w, 400, "invalid_request_error", "empty request body")
		return
	}
	if err := json.Unmarshal(body, &req); err != nil {
		status = 400
		anthropicErr(w, 400, "invalid_request_error", fmt.Sprintf("invalid JSON: %v", err))
		return
	}
	model, _ := req["model"].(string)
	a.logf("req start POST /v1/messages model=%s body=%dB", orDash(model), len(body))
	payload := a.translateRequest(req)

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
	if preset != nil {
		if ok, msg := CheckVRAMBudget(preset, s.cfg.VRAMLimitGB, s.cfg.VRAMReserveGB); !ok {
			status = 422
			anthropicErr(w, 422, "overloaded_error", msg)
			return
		}
	}

	res := s.sched.Acquire(r.Context(), AcquireRequest{Mode: "llm", Preset: preset, Route: route, RawModel: model})
	if !res.OK {
		status = res.Status
		if status == 0 {
			status = 503
		}
		typ := "api_error"
		if status == 400 || status == 404 {
			typ = "invalid_request_error"
		}
		anthropicErr(w, status, typ, firstNonEmpty(res.ErrMsg, "scheduler rejected the request"))
		return
	}
	defer s.sched.Release(res.Key)

	// Upstream model: scheduler override > preset id > raw passthrough.
	upModel := res.ServeAs
	if upModel == "" && preset != nil {
		upModel = preset.ModelID()
	}
	if upModel == "" {
		upModel = model
	}
	payload["model"] = upModel

	svc, ok := Services["llm"]
	if !ok {
		status = 500
		anthropicErr(w, 500, "api_error", "llm service not configured")
		return
	}
	outBody, err := json.Marshal(payload)
	if err != nil {
		status = 500
		anthropicErr(w, 500, "api_error", "internal encode error")
		return
	}
	url := fmt.Sprintf("http://%s:%d/v1/chat/completions", svc.Hostname, svc.InternalPort)
	upReq, err := http.NewRequestWithContext(r.Context(), http.MethodPost, url, bytes.NewReader(outBody))
	if err != nil {
		status = 502
		note = " conn_init"
		anthropicErr(w, 502, "api_error", fmt.Sprintf("connection failed: %v", err))
		return
	}
	upReq.Header.Set("Content-Type", "application/json")
	// Claude Code authenticates with x-api-key; the upstream wants Bearer.
	if auth := r.Header.Get("Authorization"); auth != "" {
		upReq.Header.Set("Authorization", auth)
	} else if key := r.Header.Get("x-api-key"); key != "" {
		upReq.Header.Set("Authorization", "Bearer "+key)
	}

	client := upstreamClient // no total timeout; see forwardTo
	resp, err := client.Do(upReq)
	if err != nil {
		status = 502
		if r.Context().Err() == nil {
			note = " upstream_dead"
			s.sched.NoteUpstreamDead("llm", res.Key)
		} else {
			note = " client_gone"
		}
		anthropicErr(w, 502, "api_error", fmt.Sprintf("upstream error: %v", err))
		return
	}
	defer resp.Body.Close()

	stream, _ := req["stream"].(bool)
	if stream && resp.StatusCode == 200 &&
		strings.Contains(resp.Header.Get("Content-Type"), "text/event-stream") {
		status = 200
		note = " stream"
		st := &streamState{
			logf:      a.logf,
			model:     firstNonEmpty(model, "unknown"),
			msgID:     "msg_" + randHex(16),
			blockIdx:  -1,
			blockKind: map[int]string{},
			toolIdx:   map[int]int{},
		}
		st.serveStream(w, resp, "anthropic")
		return
	}

	// Non-stream (or a non-SSE upstream answer to a stream request).
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		status = 502
		anthropicErr(w, 502, "api_error", fmt.Sprintf("upstream read error: %v", err))
		return
	}
	if resp.StatusCode != 200 {
		status = resp.StatusCode
		note = " upstream_error"
		a.upstreamError(w, resp.StatusCode, respBody)
		return
	}
	var om map[string]any
	if err := json.Unmarshal(respBody, &om); err != nil {
		status = 502
		anthropicErr(w, 502, "api_error", fmt.Sprintf("upstream returned invalid JSON: %v", err))
		return
	}
	status = 200
	jsonReply(w, 200, a.translateResponse(model, om))
}
