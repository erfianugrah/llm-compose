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
}

// ID is the OpenAI-API model ID: GGUF filename minus ".gguf".
func (m ModelSpec) ID() string { return strings.TrimSuffix(m.File, ".gguf") }

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
}

// runtimeKeys lists the allowed [runtime] keys (typed decode is done via a
// second pass into RuntimeSpec, so this is the strictness net).
var runtimeKeys = map[string]bool{
	"reasoning": true, "context_size": true, "parallel_slots": true,
	"temperature": true, "top_p": true, "top_k": true, "min_p": true,
	"presence_penalty": true, "repeat_penalty": true,
	"spec_type": true, "spec_ngram_n_min": true, "spec_ngram_n_max": true,
	"spec_ngram_n_match": true, "reasoning_effort": true,
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
		if k != "repo" && k != "file" {
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

	name := strings.TrimSuffix(filepath.Base(path), filepath.Ext(path))
	return &Preset{
		Name:         name,
		DisplayName:  raw.Name,
		Description:  strings.TrimSpace(raw.Description),
		VRAMGB:       *raw.VRAMGB,
		Capabilities: raw.Capabilities,
		Model:        ModelSpec{Repo: raw.Model["repo"], File: raw.Model["file"]},
		MMProj:       mmproj,
		Template:     tmpl,
		Runtime:      rt,
		Bench:        BenchSpec{Tokenizer: raw.Bench["tokenizer"], Tags: raw.Bench["tags"]},
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
}

func NewPresetStore(dir string) (*PresetStore, error) {
	s := &PresetStore{Dir: dir}
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

// All returns the current snapshot keyed by model ID.
func (s *PresetStore) All() map[string]*Preset { return s.presets }

// Names returns the sorted model IDs (for startup logging).
func (s *PresetStore) Names() []string {
	out := make([]string, 0, len(s.presets))
	for id := range s.presets {
		out = append(out, id)
	}
	sort.Strings(out)
	return out
}

// ByName looks up by model ID or preset name (filename stem).
func (s *PresetStore) ByName(name string) *Preset {
	if p, ok := s.presets[name]; ok {
		return p
	}
	for _, p := range s.presets {
		if p.Name == name {
			return p
		}
	}
	return nil
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
	return env
}
