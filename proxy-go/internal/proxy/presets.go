// TOML preset loader - schema mirror of llmc/presets.py.
//
// Schema:
//
//	name = "..."                  # human-readable (shown in UIs)
//	description = "..."           # one-line summary
//	vram_gb = 20.2                # weights-only estimate (VRAM budget gate)
//	capabilities = ["vision"]     # optional flat list (serve-in-place routing)
//	[model]
//	repo = "org/name"             # HuggingFace repo
//	file = "name.gguf"            # -> model ID = file minus .gguf
//	[mmproj] / [template]         # optional; url xor file
//	[runtime]                     # all optional, defaults below
//	[bench]                       # optional {tokenizer, tags}
//
// Unknown keys are rejected (catch typos), matching the Python loader.
package proxy

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/BurntSushi/toml"
)

// PresetError is raised on any schema violation.
type PresetError struct{ Msg string }

func (e *PresetError) Error() string { return e.Msg }

func presetErr(format string, args ...any) *PresetError {
	return &PresetError{Msg: fmt.Sprintf(format, args...)}
}

type ModelSpec struct {
	Repo string `toml:"repo"`
	File string `toml:"file"`
	// IDOverride is the served alias when the artifact filename cannot supply
	// one. NInfer artifacts are ".ninfer" and their served id is chosen at
	// serve time (--model-id), so those presets state it outright.
	IDOverride string `toml:"id"`
}

// ID is the OpenAI-API model ID: the explicit override, else the artifact
// filename minus its suffix.
func (m ModelSpec) ID() string {
	if m.IDOverride != "" {
		return m.IDOverride
	}
	return strings.TrimSuffix(strings.TrimSuffix(m.File, ".gguf"), ".ninfer")
}

// AssetSpec is an optional asset (mmproj or template): auto-download (URL)
// or pre-placed (File), never both.
type AssetSpec struct {
	URL  string `toml:"url"`
	File string `toml:"file"`
}

func (a AssetSpec) IsSet() bool { return a.URL != "" || a.File != "" }

// Filename is the name to use in /models: the explicit file override or the
// auto-derived <preset><suffix>.
func (a AssetSpec) Filename(presetName, suffix string) string {
	if a.File != "" {
		return a.File
	}
	if a.URL != "" {
		return presetName + suffix
	}
	return ""
}

type RuntimeSpec struct {
	Reasoning       string   `toml:"reasoning"` // "on" | "off" | ""
	ContextSize     int      `toml:"context_size"`
	ParallelSlots   int      `toml:"parallel_slots"`
	Temperature     float64  `toml:"temperature"`
	TopP            float64  `toml:"top_p"`
	TopK            int      `toml:"top_k"`
	MinP            float64  `toml:"min_p"`
	PresencePenalty *float64 `toml:"presence_penalty"`
	RepeatPenalty   *float64 `toml:"repeat_penalty"`
	SpecType        string   `toml:"spec_type"`
	SpecNgramNMin   *int     `toml:"spec_ngram_n_min"`
	SpecNgramNMax   *int     `toml:"spec_ngram_n_max"`
	SpecNgramNMatch *int     `toml:"spec_ngram_n_match"`
	ReasoningEffort string   `toml:"reasoning_effort"` // xhigh|high|medium|low
	// llama.cpp --reasoning-budget: -1 unrestricted, 0 stop thinking at once,
	// N>0 a hard token cap. reasoning_effort lowers the AVERAGE thinking
	// spend; only this bounds the tail.
	ReasoningBudget *int `toml:"reasoning_budget"`
}

func defaultRuntime() RuntimeSpec {
	return RuntimeSpec{
		ContextSize:   65536,
		ParallelSlots: 1,
		Temperature:   1.0,
		TopP:          0.95,
		TopK:          64,
		MinP:          0.0,
	}
}

type BenchSpec struct {
	Tokenizer string `toml:"tokenizer"`
	Tags      string `toml:"tags"`
}

// Engines the proxy can start. Absent in a preset means EngineLlama, so every
// pre-existing preset keeps working untouched.
const (
	EngineLlama  = "llama"
	EngineNinfer = "ninfer"
)

// NinferSpec holds ninfer-serve flags. Defaults are the configuration
// measured in docs/plans/2026-09-06-ninfer-nvfp4-spike.md section 6.
//
// HostStateSlots / HostKVMib default to 0 because pinned-host cudaMallocHost
// OOMs under WSL2 Docker Desktop; no decode cost was observed at 1-2 lanes.
// They are pointers so a configured 0 is distinguishable from "unset".
type NinferSpec struct {
	MaxContext            int    `toml:"max_context"`
	MaxConcurrency        int    `toml:"max_concurrency"`
	KVDtype               string `toml:"kv_dtype"`
	Spec                  string `toml:"spec"`
	DraftTokens           *int   `toml:"draft_tokens"`
	LMHeadDraft           bool   `toml:"lm_head_draft"`
	PreserveThinking      bool   `toml:"preserve_thinking"`
	Vision                bool   `toml:"vision"`
	HostStateSlots        *int   `toml:"host_state_slots"`
	HostKVMib             *int   `toml:"host_kv_mib"`
	DeviceStateSlots      *int   `toml:"device_state_slots"`
	DefaultThinkingBudget *int   `toml:"default_thinking_budget"`
	KVCapacity            string `toml:"kv_capacity"`
}

type Preset struct {
	Name         string      `toml:"-"` // filename stem
	DisplayName  string      `toml:"name"`
	Description  string      `toml:"description"`
	VRAMGB       float64     `toml:"vram_gb"`
	Capabilities []string    `toml:"capabilities"`
	Model        ModelSpec   `toml:"model"`
	MMProj       AssetSpec   `toml:"mmproj"`
	Template     AssetSpec   `toml:"template"`
	Runtime      RuntimeSpec `toml:"runtime"`
	Bench        BenchSpec   `toml:"bench"`
	Engine       string      `toml:"engine"`
	Ninfer       *NinferSpec `toml:"ninfer"` // non-nil iff Engine == EngineNinfer
}

func (p *Preset) ModelID() string { return p.Model.ID() }

func (p *Preset) MMProjFilename() string   { return p.MMProj.Filename(p.Name, "-mmproj.gguf") }
func (p *Preset) TemplateFilename() string { return p.Template.Filename(p.Name, "-template.jinja") }
func (p *Preset) HasVision() bool          { return p.MMProj.IsSet() }

func (p *Preset) HasCapability(cap string) bool {
	for _, c := range p.Capabilities {
		if c == cap {
			return true
		}
	}
	return false
}

// rawPreset mirrors the TOML document for strict unknown-key detection.
type rawPreset struct {
	Name         string            `toml:"name"`
	Description  string            `toml:"description"`
	VRAMGB       *float64          `toml:"vram_gb"`
	Capabilities []string          `toml:"capabilities"`
	Model        map[string]string `toml:"model"`
	MMProj       map[string]string `toml:"mmproj"`
	Template     map[string]string `toml:"template"`
	Runtime      map[string]any    `toml:"runtime"`
	Bench        map[string]string `toml:"bench"`
	Engine       string            `toml:"engine"`
	Ninfer       map[string]any    `toml:"ninfer"`
}

// ninferKeys is the strictness net for [ninfer]; the typed decode is a second
// pass into NinferSpec.
var ninferKeys = map[string]bool{
	"max_context": true, "max_concurrency": true, "kv_dtype": true,
	"spec": true, "draft_tokens": true, "lm_head_draft": true,
	"preserve_thinking": true, "vision": true, "host_state_slots": true,
	"host_kv_mib": true, "device_state_slots": true,
	"default_thinking_budget": true, "kv_capacity": true,
}

// runtimeKeys lists the allowed [runtime] keys (typed decode is done via a
// second pass into RuntimeSpec, so this is the strictness net).
var runtimeKeys = map[string]bool{
	"reasoning": true, "context_size": true, "parallel_slots": true,
	"temperature": true, "top_p": true, "top_k": true, "min_p": true,
	"presence_penalty": true, "repeat_penalty": true,
	"spec_type": true, "spec_ngram_n_min": true, "spec_ngram_n_max": true,
	"spec_ngram_n_match": true, "reasoning_effort": true,
	"reasoning_budget": true,
}

// LoadPreset loads and validates one preset TOML. Name = filename stem.
func LoadPreset(path string) (*Preset, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, presetErr("%s: %v", path, err)
	}
	var raw rawPreset
	if _, err := toml.Decode(string(data), &raw); err != nil {
		return nil, presetErr("%s: invalid TOML: %v", path, err)
	}

	if raw.Name == "" {
		return nil, presetErr("%s: missing required key 'name'", path)
	}
	if raw.VRAMGB == nil {
		return nil, presetErr("%s: missing required key 'vram_gb'", path)
	}
	if raw.Model == nil {
		return nil, presetErr("%s: [model] table missing", path)
	}
	for _, k := range []string{"repo", "file"} {
		if strings.TrimSpace(raw.Model[k]) == "" {
			return nil, presetErr("%s: model.%s is required and must be non-empty", path, k)
		}
	}
	for k := range raw.Model {
		if k != "repo" && k != "file" && k != "id" {
			return nil, presetErr("%s:model: unknown key %q", path, k)
		}
	}
	for _, section := range []struct {
		name string
		m    map[string]string
	}{
		{"mmproj", raw.MMProj}, {"template", raw.Template}, {"bench", raw.Bench},
	} {
		for k := range section.m {
			ok := (section.name != "bench" && (k == "url" || k == "file")) ||
				(section.name == "bench" && (k == "tokenizer" || k == "tags"))
			if !ok {
				return nil, presetErr("%s:%s: unknown key %q", path, section.name, k)
			}
		}
	}
	for k := range raw.Runtime {
		if !runtimeKeys[k] {
			return nil, presetErr("%s:runtime: unknown key %q", path, k)
		}
	}
	for _, c := range raw.Capabilities {
		if strings.TrimSpace(c) == "" {
			return nil, presetErr("%s: capabilities entries must be non-empty strings", path)
		}
	}

	loadAsset := func(section string, m map[string]string) (AssetSpec, error) {
		a := AssetSpec{URL: strings.TrimSpace(m["url"]), File: strings.TrimSpace(m["file"])}
		if a.URL != "" && a.File != "" {
			return a, presetErr("%s: %s: 'url' and 'file' are mutually exclusive", path, section)
		}
		return a, nil
	}
	mmproj, err := loadAsset("mmproj", raw.MMProj)
	if err != nil {
		return nil, err
	}
	tmpl, err := loadAsset("template", raw.Template)
	if err != nil {
		return nil, err
	}

	// Typed [runtime] decode over defaults. Re-encode the raw map through
	// toml primitives: simplest correct path is decoding the whole doc into
	// a RuntimeSpec-carrying struct.
	var typed struct {
		Runtime RuntimeSpec `toml:"runtime"`
	}
	rt := defaultRuntime()
	typed.Runtime = rt
	if _, err := toml.Decode(string(data), &typed); err != nil {
		return nil, presetErr("%s: [runtime] type error: %v", path, err)
	}
	rt = typed.Runtime
	if rt.Reasoning != "" && rt.Reasoning != "on" && rt.Reasoning != "off" {
		return nil, presetErr("%s: runtime.reasoning: must be 'on' or 'off', got %q", path, rt.Reasoning)
	}
	if rt.ReasoningEffort != "" {
		switch rt.ReasoningEffort {
		case "xhigh", "high", "medium", "low":
		default:
			return nil, presetErr("%s: runtime.reasoning_effort: unexpected value %q", path, rt.ReasoningEffort)
		}
	}
	if rt.ReasoningBudget != nil && *rt.ReasoningBudget < -1 {
		return nil, presetErr("%s: runtime.reasoning_budget: must be -1, 0 or a positive token count, got %d",
			path, *rt.ReasoningBudget)
	}

	name := strings.TrimSuffix(filepath.Base(path), filepath.Ext(path))
	engine := raw.Engine
	if engine == "" {
		engine = EngineLlama
	}
	if engine != EngineLlama && engine != EngineNinfer {
		return nil, presetErr("%s: engine must be %q or %q, got %q",
			path, EngineLlama, EngineNinfer, engine)
	}
	if raw.Ninfer != nil && engine != EngineNinfer {
		return nil, presetErr("%s: a [ninfer] section requires engine = %q; "+
			"this preset declares engine = %q", path, EngineNinfer, engine)
	}
	for k := range raw.Ninfer {
		if !ninferKeys[k] {
			return nil, presetErr("%s:ninfer: unknown key %q", path, k)
		}
	}
	var ninfer *NinferSpec
	if engine == EngineNinfer {
		var wrapper struct {
			Ninfer NinferSpec `toml:"ninfer"`
		}
		if _, err := toml.Decode(string(data), &wrapper); err != nil {
			return nil, presetErr("%s: [ninfer] type error: %v", path, err)
		}
		ninfer = &wrapper.Ninfer
		if ninfer.MaxContext <= 0 {
			return nil, presetErr("%s: ninfer.max_context must be positive", path)
		}
		if ninfer.MaxConcurrency < 1 || ninfer.MaxConcurrency > 8 {
			return nil, presetErr("%s: ninfer.max_concurrency must be 1..8, got %d",
				path, ninfer.MaxConcurrency)
		}
	}

	return &Preset{
		Name:         name,
		DisplayName:  raw.Name,
		Description:  strings.TrimSpace(raw.Description),
		VRAMGB:       *raw.VRAMGB,
		Capabilities: raw.Capabilities,
		Model: ModelSpec{
			Repo:       raw.Model["repo"],
			File:       raw.Model["file"],
			IDOverride: strings.TrimSpace(raw.Model["id"]),
		},
		MMProj:   mmproj,
		Template: tmpl,
		Runtime:  rt,
		Engine:   engine,
		Ninfer:   ninfer,
		Bench:    BenchSpec{Tokenizer: raw.Bench["tokenizer"], Tags: raw.Bench["tags"]},
	}, nil
}

// PresetStore is the preset registry, keyed by model ID for OpenAI-API
// compatibility. Reload() rescans the directory so a new TOML is pickable
// without a proxy restart (parity with the Python live-reload).
type PresetStore struct {
	Dir string

	// Loaded via Reload; guarded by the scheduler loop in practice, but the
	// HTTP layer reads it concurrently - treated as immutable snapshots.
	presets map[string]*Preset
	// ephemeral is the in-memory overlay (registered via the API, e.g. the
	// context sweep's throwaway presets). Survives Reload (not on disk),
	// dropped on proxy restart. TOML presets win on model_id collision.
	ephemeral map[string]*Preset
}

func NewPresetStore(dir string) (*PresetStore, error) {
	s := &PresetStore{Dir: dir, ephemeral: map[string]*Preset{}}
	return s, s.Reload()
}

func (s *PresetStore) Reload() error {
	entries, err := filepath.Glob(filepath.Join(s.Dir, "*.toml"))
	if err != nil {
		return presetErr("presets glob failed: %v", err)
	}
	if fi, err := os.Stat(s.Dir); err != nil || !fi.IsDir() {
		return presetErr("presets directory not found: %s", s.Dir)
	}
	sort.Strings(entries)
	out := map[string]*Preset{}
	for _, path := range entries {
		p, err := LoadPreset(path)
		if err != nil {
			return err
		}
		if prev, dup := out[p.ModelID()]; dup {
			return presetErr("duplicate model_id %q: %s.toml and %s.toml", p.ModelID(), prev.Name, p.Name)
		}
		out[p.ModelID()] = p
	}
	s.presets = out
	return nil
}

// All returns the current snapshot keyed by model ID (TOML + ephemeral).
func (s *PresetStore) All() map[string]*Preset {
	out := make(map[string]*Preset, len(s.presets)+len(s.ephemeral))
	for k, v := range s.presets {
		out[k] = v
	}
	for k, v := range s.ephemeral {
		if _, taken := out[k]; !taken { // TOML wins on collision
			out[k] = v
		}
	}
	return out
}

// Names returns the sorted model IDs (for startup logging).
func (s *PresetStore) Names() []string {
	all := s.All()
	out := make([]string, 0, len(all))
	for id := range all {
		out = append(out, id)
	}
	sort.Strings(out)
	return out
}

// ByName looks up by model ID or preset name (filename stem). Ephemeral
// presets are matched on their explicit Name too.
func (s *PresetStore) ByName(name string) *Preset {
	all := s.All()
	if p, ok := all[name]; ok {
		return p
	}
	for _, p := range all {
		if p.Name == name {
			return p
		}
	}
	return nil
}

// RegisterEphemeral adds an in-memory preset. Rejects a model_id collision
// with an existing TOML preset (TOML is authoritative). Idempotent for an
// identical re-register.
func (s *PresetStore) RegisterEphemeral(p *Preset) error {
	if p == nil || p.ModelID() == "" {
		return presetErr("ephemeral preset needs a model file")
	}
	if _, dup := s.presets[p.ModelID()]; dup {
		return presetErr("model_id %q collides with an on-disk preset", p.ModelID())
	}
	s.ephemeral[p.ModelID()] = p
	return nil
}

// DeleteEphemeral removes an ephemeral preset by model ID (no-op if absent).
func (s *PresetStore) DeleteEphemeral(modelID string) {
	delete(s.ephemeral, modelID)
}

// Env renders the preset as the environment variables expected by the
// llama-server image entrypoint (parity with preset_to_env in Python).
func (p *Preset) Env() map[string]string {
	env := map[string]string{
		"MODEL_REPO":     p.Model.Repo,
		"MODEL_FILE":     p.Model.File,
		"MMPROJ_FILE":    p.MMProjFilename(),
		"TEMPLATE_FILE":  p.TemplateFilename(),
		"CONTEXT_SIZE":   fmt.Sprintf("%d", p.Runtime.ContextSize),
		"PARALLEL_SLOTS": fmt.Sprintf("%d", p.Runtime.ParallelSlots),
		"TEMPERATURE":    fmt.Sprintf("%g", p.Runtime.Temperature),
		"TOP_P":          fmt.Sprintf("%g", p.Runtime.TopP),
		"TOP_K":          fmt.Sprintf("%d", p.Runtime.TopK),
		"MIN_P":          fmt.Sprintf("%g", p.Runtime.MinP),
	}
	if p.Runtime.Reasoning != "" {
		env["REASONING"] = p.Runtime.Reasoning
	}
	if p.Runtime.PresencePenalty != nil {
		env["PRESENCE_PENALTY"] = fmt.Sprintf("%g", *p.Runtime.PresencePenalty)
	}
	if p.Runtime.RepeatPenalty != nil {
		env["REPEAT_PENALTY"] = fmt.Sprintf("%g", *p.Runtime.RepeatPenalty)
	}
	if p.Runtime.SpecType != "" {
		env["SPEC_TYPE"] = p.Runtime.SpecType
	}
	if p.Runtime.SpecNgramNMin != nil {
		env["SPEC_NGRAM_N_MIN"] = fmt.Sprintf("%d", *p.Runtime.SpecNgramNMin)
	}
	if p.Runtime.SpecNgramNMax != nil {
		env["SPEC_NGRAM_N_MAX"] = fmt.Sprintf("%d", *p.Runtime.SpecNgramNMax)
	}
	if p.Runtime.SpecNgramNMatch != nil {
		env["SPEC_NGRAM_N_MATCH"] = fmt.Sprintf("%d", *p.Runtime.SpecNgramNMatch)
	}
	if p.Runtime.ReasoningEffort != "" {
		// Compact JSON: the entrypoint word-splits ${VAR:+--flag $VAR}.
		env["CHAT_TEMPLATE_KWARGS"] = fmt.Sprintf(`{"reasoning_effort":%q}`, p.Runtime.ReasoningEffort)
	}
	if p.Runtime.ReasoningBudget != nil {
		env["REASONING_BUDGET"] = fmt.Sprintf("%d", *p.Runtime.ReasoningBudget)
	}
	return env
}
