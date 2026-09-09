package proxy

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const stagedNinferPreset = "../../../models/qwen38-ninfer.toml"

func loadStagedNinfer(t *testing.T) *Preset {
	t.Helper()
	p, err := LoadPreset(stagedNinferPreset)
	if err != nil {
		t.Fatalf("load ninfer preset: %v", err)
	}
	return p
}

// A preset names the engine that serves it; absent means llama.cpp, so every
// pre-existing preset keeps working untouched. Presets that declare
// engine="ninfer" are excluded and asserted separately - counted here so the
// exclusion cannot quietly empty the set.
func TestEngineDefaultsToLlama(t *testing.T) {
	llama, ninfer := 0, 0
	for _, path := range mustGlob(t, "../../../models/*.toml") {
		p, err := LoadPreset(path)
		if err != nil {
			t.Fatalf("%s: %v", filepath.Base(path), err)
		}
		switch p.Engine {
		case EngineNinfer:
			ninfer++
		case EngineLlama:
			llama++
		default:
			t.Errorf("%s: engine = %q, want %q or %q",
				filepath.Base(path), p.Engine, EngineLlama, EngineNinfer)
		}
	}
	if llama == 0 {
		t.Error("no llama presets found - the default is untested")
	}
	if ninfer == 0 {
		t.Error("no ninfer preset found - the engine field is untested against a real preset")
	}
}

func mustGlob(t *testing.T, pattern string) []string {
	t.Helper()
	paths, err := filepath.Glob(pattern)
	if err != nil || len(paths) == 0 {
		t.Fatalf("glob %q: %v (%d hits)", pattern, err, len(paths))
	}
	return paths
}

func TestNinferPresetParses(t *testing.T) {
	p := loadStagedNinfer(t)
	if p.Engine != EngineNinfer {
		t.Fatalf("engine = %q, want %q", p.Engine, EngineNinfer)
	}
	if p.Ninfer == nil {
		t.Fatal("Ninfer section is nil")
	}
	// 252928 is NInfer's own published ceiling for this artifact on a 5090
	// (010ba88); the arch limit 262144 does not fit after weights.
	if p.Ninfer.MaxContext != 252928 {
		t.Errorf("max_context = %d", p.Ninfer.MaxContext)
	}
	if p.Ninfer.KVDtype != "fp8" || p.Ninfer.Spec != "mtp" || p.Ninfer.DraftTokens == nil || *p.Ninfer.DraftTokens != 3 {
		t.Errorf("flags wrong: %+v", p.Ninfer)
	}
	if !p.Ninfer.LMHeadDraft || !p.Ninfer.PreserveThinking || !p.Ninfer.Vision {
		t.Errorf("bools wrong: %+v", p.Ninfer)
	}
}

// NInfer artifacts carry no .gguf name to derive an id from, so the preset
// states it outright. Without this the served alias would silently become the
// artifact filename and every client's `model` parameter would stop matching.
func TestExplicitModelIDWins(t *testing.T) {
	p := loadStagedNinfer(t)
	if got := p.ModelID(); got != "qwen3.8-27b-nvfp4" {
		t.Errorf("ModelID() = %q", got)
	}
}

func TestNinferSuffixStrippedWhenNoExplicitID(t *testing.T) {
	m := ModelSpec{File: "qwen3_8_27b_nvfp4.ninfer"}
	if got := m.ID(); got != "qwen3_8_27b_nvfp4" {
		t.Errorf("ID() = %q", got)
	}
}

func TestUnknownEngineRejected(t *testing.T) {
	_, err := loadInline(t, `name="x"
vram_gb=5
engine="vllm"
[model]
repo="r"
file="f.gguf"`)
	if err == nil || !strings.Contains(err.Error(), "engine") {
		t.Fatalf("want engine error, got %v", err)
	}
}

func TestNinferSectionOnLlamaPresetRejected(t *testing.T) {
	_, err := loadInline(t, `name="x"
vram_gb=5
[model]
repo="r"
file="f.gguf"
[ninfer]
max_context=1024`)
	if err == nil || !strings.Contains(err.Error(), "ninfer") {
		t.Fatalf("want ninfer/engine mismatch error, got %v", err)
	}
}

func TestUnknownNinferKeyRejected(t *testing.T) {
	_, err := loadInline(t, `name="x"
vram_gb=5
engine="ninfer"
[model]
repo="r"
file="f.ninfer"
id="x-id"
[ninfer]
bogus_flag=1`)
	if err == nil || !strings.Contains(err.Error(), "unknown key") {
		t.Fatalf("want unknown-key error, got %v", err)
	}
}

// ── argv builder ────────────────────────────────────────────────────────

func TestNinferCommand(t *testing.T) {
	p := loadStagedNinfer(t)
	argv, err := NinferCommand(p)
	if err != nil {
		t.Fatalf("NinferCommand: %v", err)
	}
	if argv[0] != "ninfer-serve" {
		t.Errorf("argv[0] = %q", argv[0])
	}
	if argv[1] != "/models/qwen3_8_27b_nvfp4.ninfer" {
		t.Errorf("artifact path = %q", argv[1])
	}
	want := map[string]string{
		"--model-id":           "qwen3.8-27b-nvfp4",
		"--host":               "0.0.0.0",
		"--max-context":        "252928",
		"--max-concurrency":    "1",
		"--kv-dtype":           "fp8",
		"--spec":               "mtp",
		"--draft-tokens":       "3",
		"--host-state-slots":   "0",
		"--host-kv-mib":        "0",
		"--device-state-slots": "0",
	}
	for flag, value := range want {
		got, ok := argvValue(argv, flag)
		if !ok {
			t.Errorf("%s missing", flag)
			continue
		}
		if got != value {
			t.Errorf("%s = %q, want %q", flag, got, value)
		}
	}
	for _, flag := range []string{"--lm-head-draft", "--preserve-thinking", "--vision"} {
		if !argvHas(argv, flag) {
			t.Errorf("%s missing", flag)
		}
	}
	for _, a := range argv {
		if a == "<nil>" || a == "" {
			t.Errorf("argv contains an empty/nil rendering: %v", argv)
		}
	}
}

// host_state_slots = 0 is meaningful (the WSL2 pinned-host workaround), so it
// must survive any zero-value-means-unset check.
func TestZeroIsNotTreatedAsUnset(t *testing.T) {
	p := loadStagedNinfer(t)
	argv, _ := NinferCommand(p)
	if got, ok := argvValue(argv, "--host-state-slots"); !ok || got != "0" {
		t.Errorf("--host-state-slots = %q (ok=%v)", got, ok)
	}
}

func TestUnsetOptionalsOmitted(t *testing.T) {
	p := loadStagedNinfer(t)
	p.Ninfer.Spec = ""
	p.Ninfer.DraftTokens = nil
	p.Ninfer.Vision = false
	argv, _ := NinferCommand(p)
	for _, flag := range []string{"--spec", "--draft-tokens", "--vision"} {
		if argvHas(argv, flag) {
			t.Errorf("%s should be omitted", flag)
		}
	}
}

func TestNinferCommandRefusesLlamaPreset(t *testing.T) {
	p, err := LoadPreset("../../../models/qwen38.toml")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := NinferCommand(p); err == nil {
		t.Fatal("want error for a llama preset")
	}
}

// ── service routing ─────────────────────────────────────────────────────

// Two engines share mode "llm", so resolving the service from the mode alone
// would forward every request to llama-server even while ninfer holds the GPU.
func TestLLMServiceForEngine(t *testing.T) {
	llama, err := LoadPreset("../../../models/qwen38.toml")
	if err != nil {
		t.Fatal(err)
	}
	ninfer := loadStagedNinfer(t)

	if got := LLMServiceFor(llama); got.Name != LlamaService.Name {
		t.Errorf("llama preset -> %q", got.Name)
	}
	if got := LLMServiceFor(ninfer); got.Name != NinferService.Name {
		t.Errorf("ninfer preset -> %q", got.Name)
	}
	if LlamaService.Hostname == NinferService.Hostname {
		t.Error("the two engines must have distinct hostnames or forwarding cannot distinguish them")
	}
	if NinferService.Mode != "llm" {
		t.Errorf("ninfer mode = %q, want llm", NinferService.Mode)
	}
}

func argvValue(argv []string, flag string) (string, bool) {
	for i, a := range argv {
		if a == flag && i+1 < len(argv) {
			return argv[i+1], true
		}
	}
	return "", false
}

func argvHas(argv []string, flag string) bool {
	for _, a := range argv {
		if a == flag {
			return true
		}
	}
	return false
}

// loadInline writes a TOML body to a temp file and loads it, so schema
// rejections can be asserted without shipping broken presets.
func loadInline(t *testing.T, body string) (*Preset, error) {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "inline.toml")
	if err := writeFile(path, body); err != nil {
		t.Fatal(err)
	}
	return LoadPreset(path)
}

func writeFile(path, body string) error {
	return os.WriteFile(path, []byte(body), 0o644)
}

// End-to-end through the scheduler: a ninfer preset must reach SpawnNinfer.
// The isolated LLMServiceFor test cannot catch a doSwap that still calls
// SpawnLlama directly, which is what the code did before this change.
func TestSchedulerRoutesNinferPresetToNinferSpawn(t *testing.T) {
	orch := newFakeOrch("idle")
	if err := orch.SpawnLLM(loadStagedNinfer(t)); err != nil {
		t.Fatalf("SpawnLLM: %v", err)
	}
	if len(orch.ninferCalls) != 1 {
		t.Errorf("ninferCalls = %d, want 1", len(orch.ninferCalls))
	}
	if len(orch.llamaCalls) != 0 {
		t.Errorf("a ninfer preset started llama.cpp (%d calls)", len(orch.llamaCalls))
	}
}

func TestSchedulerRoutesLlamaPresetToLlamaSpawn(t *testing.T) {
	orch := newFakeOrch("idle")
	p, err := LoadPreset("../../../models/qwen38.toml")
	if err != nil {
		t.Fatal(err)
	}
	if err := orch.SpawnLLM(p); err != nil {
		t.Fatal(err)
	}
	if len(orch.llamaCalls) != 1 || len(orch.ninferCalls) != 0 {
		t.Errorf("llama=%d ninfer=%d", len(orch.llamaCalls), len(orch.ninferCalls))
	}
}

// ── outbound model rewrite ──────────────────────────────────────────────

// NInfer validates the request's `model` against its served alias and 404s on
// a mismatch, where llama-server ignores the field entirely. So a client
// addressing the preset by stem ("qwen38-ninfer") swaps correctly and then
// gets model_not_found from the engine. Observed live 2026-09-07.
func TestRewriteModelSetsServedAlias(t *testing.T) {
	body := []byte(`{"model":"qwen38-ninfer","max_tokens":256,"temperature":0.7}`)
	out := rewriteModel(body, "qwen3.8-27b-nvfp4")
	if got := peekModel(out); got != "qwen3.8-27b-nvfp4" {
		t.Errorf("model = %q", got)
	}
}

// Other fields must survive byte-exact: round-tripping through float64 would
// quietly rewrite sampling params.
func TestRewriteModelPreservesOtherFields(t *testing.T) {
	body := []byte(`{"model":"x","temperature":0.70,"top_p":0.95,"n":1}`)
	out := rewriteModel(body, "served")
	for _, frag := range []string{`"temperature":0.70`, `"top_p":0.95`, `"n":1`} {
		if !strings.Contains(string(out), frag) {
			t.Errorf("lost %s from %s", frag, out)
		}
	}
}

func TestRewriteModelLeavesNonJSONAlone(t *testing.T) {
	for _, body := range [][]byte{nil, {}, []byte("not json"), []byte(`[1,2]`)} {
		out := rewriteModel(body, "served")
		if string(out) != string(body) {
			t.Errorf("body %q was altered to %q", body, out)
		}
	}
}

func TestRewriteModelAddsFieldWhenAbsent(t *testing.T) {
	out := rewriteModel([]byte(`{"max_tokens":8}`), "served")
	if got := peekModel(out); got != "served" {
		t.Errorf("model = %q, want served", got)
	}
}

// ── per-request effort injection ────────────────────────────────────────

// For NInfer, thinking effort is a per-request field, not a serve flag. A
// client that sends none gets the chat template's default, which is xhigh and
// produced 30k+ token thinking traces for a single tool call (2026-09-07). So
// the preset declares the effort and the proxy injects it when absent.
func TestInjectIfAbsentAddsEffort(t *testing.T) {
	out := injectIfAbsent([]byte(`{"model":"m","max_tokens":8}`), "reasoning_effort", "medium")
	var got map[string]any
	if err := json.Unmarshal(out, &got); err != nil {
		t.Fatal(err)
	}
	if got["reasoning_effort"] != "medium" {
		t.Errorf("reasoning_effort = %v", got["reasoning_effort"])
	}
}

// An explicit client choice must win: the proxy sets a default, not a policy.
func TestInjectIfAbsentRespectsExplicitValue(t *testing.T) {
	out := injectIfAbsent([]byte(`{"reasoning_effort":"xhigh"}`), "reasoning_effort", "medium")
	var got map[string]any
	json.Unmarshal(out, &got)
	if got["reasoning_effort"] != "xhigh" {
		t.Errorf("overwrote the client's value: %v", got["reasoning_effort"])
	}
}

// "none" disables thinking and is a legitimate explicit choice, so it must not
// be treated as absent.
func TestInjectIfAbsentRespectsNone(t *testing.T) {
	out := injectIfAbsent([]byte(`{"reasoning_effort":"none"}`), "reasoning_effort", "medium")
	var got map[string]any
	json.Unmarshal(out, &got)
	if got["reasoning_effort"] != "none" {
		t.Errorf("overwrote none: %v", got["reasoning_effort"])
	}
}

func TestInjectIfAbsentLeavesNonJSONAlone(t *testing.T) {
	for _, body := range [][]byte{nil, {}, []byte("not json"), []byte(`[1]`)} {
		if out := injectIfAbsent(body, "reasoning_effort", "medium"); string(out) != string(body) {
			t.Errorf("body %q altered to %q", body, out)
		}
	}
}

func TestInjectIfAbsentPreservesOtherFields(t *testing.T) {
	out := injectIfAbsent([]byte(`{"temperature":0.70,"top_p":0.95}`), "reasoning_effort", "medium")
	for _, frag := range []string{`"temperature":0.70`, `"top_p":0.95`} {
		if !strings.Contains(string(out), frag) {
			t.Errorf("lost %s from %s", frag, out)
		}
	}
}

// The engine rejects "high" with reasoning_effort_not_supported, so a preset
// that would inject it must fail at load rather than at request time.
func TestNinferPresetRejectsHighEffort(t *testing.T) {
	_, err := loadInline(t, `name="x"
vram_gb=5
engine="ninfer"
[model]
repo="r"
file="f.ninfer"
id="x-id"
[runtime]
reasoning_effort="high"
[ninfer]
max_context=1024
max_concurrency=1`)
	if err == nil || !strings.Contains(err.Error(), "reasoning_effort") {
		t.Fatalf("want a reasoning_effort rejection, got %v", err)
	}
}

func TestNinferPresetAcceptsMediumEffort(t *testing.T) {
	p, err := loadInline(t, `name="x"
vram_gb=5
engine="ninfer"
[model]
repo="r"
file="f.ninfer"
id="x-id"
[runtime]
reasoning_effort="medium"
[ninfer]
max_context=1024
max_concurrency=1`)
	if err != nil {
		t.Fatalf("medium must be accepted: %v", err)
	}
	if p.Runtime.ReasoningEffort != "medium" {
		t.Errorf("effort = %q", p.Runtime.ReasoningEffort)
	}
}

// The shipped preset must actually carry the effort, or the injection is dead
// code and clients silently get xhigh.
func TestShippedNinferPresetDeclaresEffort(t *testing.T) {
	p := loadStagedNinfer(t)
	if p.Runtime.ReasoningEffort == "" {
		t.Fatal("models/qwen38-ninfer.toml must declare runtime.reasoning_effort")
	}
	if p.Runtime.ReasoningEffort == "high" {
		t.Fatal("high is rejected by the engine")
	}
}

// ── engine-aware display metadata ───────────────────────────────────────

// The CLI reads these from the proxy when it is up, so fixing only the Python
// side left `llmc models` still showing a ninfer preset as context 65536 and
// vision "no" (2026-09-07).
func TestEffectiveContextIsEngineAware(t *testing.T) {
	llama, err := LoadPreset("../../../models/qwen38.toml")
	if err != nil {
		t.Fatal(err)
	}
	if llama.EffectiveContext() != llama.Runtime.ContextSize {
		t.Errorf("llama context = %d, want %d", llama.EffectiveContext(), llama.Runtime.ContextSize)
	}
	n := loadStagedNinfer(t)
	if n.EffectiveContext() != n.Ninfer.MaxContext {
		t.Errorf("ninfer context = %d, want %d", n.EffectiveContext(), n.Ninfer.MaxContext)
	}
	if n.EffectiveContext() == n.Runtime.ContextSize {
		t.Error("ninfer context must not fall back to the llama runtime default")
	}
}

func TestHasVisionIsEngineAware(t *testing.T) {
	n := loadStagedNinfer(t)
	if !n.Ninfer.Vision {
		t.Fatal("fixture must have vision on")
	}
	if !n.HasVision() {
		t.Error("ninfer vision comes from the serve flag, not an mmproj asset")
	}
	n.Ninfer.Vision = false
	if n.HasVision() {
		t.Error("vision off must report off")
	}
}

// pi's defaultThinkingLevel is "high", which this engine's chat template
// rejects outright (it exposes none|low|medium|xhigh). Once the model is
// registered under the llama-server provider by the dynamic-registration
// extension, pi sends "high" and every request 400s - so the model is
// unusable from pi without an explicit :medium suffix. Injecting only when
// ABSENT does not help; an explicit unsupported value has to be coerced.
// Found 2026-09-07 by actually calling `pi --model llama-server/qwen38-ninfer`.
func TestCoerceEffortReplacesUnsupportedValues(t *testing.T) {
	out := coerceEffort([]byte(`{"reasoning_effort":"high","max_tokens":8}`), "medium")
	var got map[string]any
	if err := json.Unmarshal(out, &got); err != nil {
		t.Fatal(err)
	}
	if got["reasoning_effort"] != "medium" {
		t.Errorf("reasoning_effort = %v, want medium", got["reasoning_effort"])
	}
	if got["max_tokens"] == nil {
		t.Error("other fields must survive")
	}
}

// Values the template accepts are the client's business and must pass through.
func TestCoerceEffortLeavesSupportedValuesAlone(t *testing.T) {
	for _, eff := range []string{"none", "low", "medium", "xhigh"} {
		out := coerceEffort([]byte(`{"reasoning_effort":"`+eff+`"}`), "medium")
		var got map[string]any
		json.Unmarshal(out, &got)
		if got["reasoning_effort"] != eff {
			t.Errorf("%s was rewritten to %v", eff, got["reasoning_effort"])
		}
	}
}

func TestCoerceEffortIgnoresAbsentField(t *testing.T) {
	body := []byte(`{"max_tokens":8}`)
	if out := coerceEffort(body, "medium"); string(out) != string(body) {
		t.Errorf("absent effort must be left to injectIfAbsent, got %s", out)
	}
}

func TestCoerceEffortLeavesNonJSONAlone(t *testing.T) {
	for _, body := range [][]byte{nil, {}, []byte("nope"), []byte(`[1]`)} {
		if out := coerceEffort(body, "medium"); string(out) != string(body) {
			t.Errorf("body %q altered to %q", body, out)
		}
	}
}

// The default image must be registry-qualified. It was `ninfer:local`, which
// exists only where it was built - and on 2026-09-07 Docker reclaimed it under
// disk pressure once the last container referencing it was removed, so a swap
// failed with "image not found" and the proxy could not recover. A pullable
// default turns that into a pull. Commit-pinned on purpose: upstream is young
// enough that tracking a moving tag risks a silent engine change.
func TestNinferDefaultImageIsPullable(t *testing.T) {
	img := NinferService.Image
	if !strings.Contains(img, "/") {
		t.Errorf("default image %q has no registry path, so it cannot be pulled", img)
	}
	if strings.HasSuffix(img, ":local") {
		t.Errorf("default image %q is a local-only tag", img)
	}
	if !strings.Contains(img, "-") {
		t.Errorf("default image %q looks unpinned; prefer a commit-pinned tag", img)
	}
}
