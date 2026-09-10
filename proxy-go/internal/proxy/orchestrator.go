// Orchestrator: GPU service lifecycle on top of the Docker client.
// Mirror of llmc/orchestrator.py - three mutually exclusive GPU services
// (llama-server, comfyui, lora-train) found via the llmc.mode label.
package proxy

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

type GpuService struct {
	Name         string // container name
	Hostname     string // hostname on the user-defined network
	Mode         string // llm | comfyui | train
	Image        string
	InternalPort int
	HealthPath   string
}

const (
	GPULabel     = "llmc.mode"
	ServiceLabel = "llmc.service"
)

var (
	LlamaService   = GpuService{Name: "llama_server", Hostname: "llama-server", Mode: "llm", Image: envOr("LLMC_LLAMA_IMAGE", "erfianugrah/llama-server:cuda12.8-sm120"), InternalPort: 8080, HealthPath: "/health"}
	ComfyUIService = GpuService{Name: "comfyui", Hostname: "comfyui", Mode: "comfyui", Image: envOr("LLMC_COMFYUI_IMAGE", "erfianugrah/comfyui:cuda12.8-sm120"), InternalPort: 8188, HealthPath: "/system_stats"}
	TrainService   = GpuService{Name: "lora_train", Hostname: "lora-train", Mode: "train", Image: envOr("LLMC_TRAIN_IMAGE", "erfianugrah/lora-train:latest"), InternalPort: 8787, HealthPath: "/health"}
)

// NinferService serves the same mode as llama.cpp: both are the LLM and the
// GPU holds one workload at a time. It is deliberately NOT in Services (which
// maps mode -> service 1:1); resolve it from the preset via LLMServiceFor.
var NinferService = GpuService{Name: "ninfer_server", Hostname: "ninfer-server", Mode: "llm", Image: envOr("LLMC_NINFER_IMAGE", "erfianugrah/ninfer:cuda13.1-sm120a-487f897"), InternalPort: 8080, HealthPath: "/health"}

// LLMServiceFor returns the container that serves this preset. Two engines
// share mode "llm", so the mode alone cannot decide - the preset does.
func LLMServiceFor(p *Preset) GpuService {
	if p != nil && p.Engine == EngineNinfer {
		return NinferService
	}
	return LlamaService
}

// NinferCommand builds the ninfer-serve argv for a ninfer preset. Unlike the
// llama image (whose ENTRYPOINT assembles a command line from environment
// variables), ninfer-serve is argv-driven, so the proxy owns flag rendering.
func NinferCommand(p *Preset) ([]string, error) {
	if p.Engine != EngineNinfer {
		return nil, &PresetError{Msg: fmt.Sprintf(
			"NinferCommand: preset %q declares engine %q", p.Name, p.Engine)}
	}
	if p.Ninfer == nil {
		return nil, &PresetError{Msg: fmt.Sprintf("NinferCommand: preset %q has no [ninfer] section", p.Name)}
	}
	n := p.Ninfer
	argv := []string{
		"ninfer-serve",
		"/models/" + p.Model.File,
		"--model-id", p.ModelID(),
		"--host", "0.0.0.0",
		"--port", strconv.Itoa(NinferService.InternalPort),
		"--max-context", strconv.Itoa(n.MaxContext),
		"--max-concurrency", strconv.Itoa(n.MaxConcurrency),
		"--kv-dtype", n.KVDtype,
	}
	if n.Spec != "" {
		argv = append(argv, "--spec", n.Spec)
	}
	if n.KVCapacity != "" {
		argv = append(argv, "--kv-capacity", n.KVCapacity)
	}
	// Pointer checks, not zero checks: 0 is meaningful for the slot flags
	// (the WSL2 pinned-host workaround) and must still be emitted.
	for _, f := range []struct {
		flag string
		val  *int
	}{
		{"--draft-tokens", n.DraftTokens},
		{"--host-state-slots", n.HostStateSlots},
		{"--host-kv-mib", n.HostKVMib},
		{"--device-state-slots", n.DeviceStateSlots},
		{"--default-thinking-budget", n.DefaultThinkingBudget},
		{"--prefill-chunk", n.PrefillChunk},
		{"--max-pending-requests", n.MaxPendingRequests},
		{"--pending-timeout-ms", n.PendingTimeoutMs},
	} {
		if f.val != nil {
			argv = append(argv, f.flag, strconv.Itoa(*f.val))
		}
	}
	for _, f := range []struct {
		flag string
		on   bool
	}{
		{"--lm-head-draft", n.LMHeadDraft},
		{"--preserve-thinking", n.PreserveThinking},
		{"--vision", n.Vision},
	} {
		if f.on {
			argv = append(argv, f.flag)
		}
	}
	return argv, nil
}

var Services = map[string]GpuService{
	"llm":     LlamaService,
	"comfyui": ComfyUIService,
	"train":   TrainService,
}

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

// OrchestratorError surfaces Docker failures to the client (503s).
type OrchestratorError struct{ Msg string }

func (e *OrchestratorError) Error() string { return e.Msg }

// Orchestrator is the scheduler's port to Docker. An interface so scheduler
// tests drive a fake (parity with Python's MagicMock orchestrator tests).
type Orchestrator interface {
	CurrentMode() string
	SpawnLlama(p *Preset) error
	// SpawnLLM dispatches on the preset's engine. Prefer it over SpawnLlama:
	// a ninfer preset handed to SpawnLlama would start the wrong engine.
	SpawnLLM(p *Preset) error
	SpawnNinfer(p *Preset) error
	SpawnComfyUI() error
	SpawnTrain() error
	WaitHealthy(svc GpuService, timeout time.Duration) bool
	EnsurePresetAssets(p *Preset, assetsDir string) error
}

// DockerOrchestrator is the production Orchestrator.
type DockerOrchestrator struct {
	Client  *DockerClient
	Network string
	Volumes *VolumeRegistry
}

func (o *DockerOrchestrator) CurrentMode() string {
	containers, err := o.Client.ListByLabel(GPULabel)
	if err != nil {
		return "idle"
	}
	for _, c := range containers {
		if c.State != "running" {
			continue
		}
		if mode := c.Labels[GPULabel]; Services[mode].Name != "" {
			return mode
		}
	}
	return "idle"
}

func (o *DockerOrchestrator) stopGPU() error {
	containers, err := o.Client.ListByLabel(GPULabel)
	if err != nil {
		return err
	}
	for _, c := range containers {
		if c.State == "running" {
			if err := o.Client.Stop(c.ID, 10); err != nil {
				return err
			}
		}
		if err := o.Client.Remove(c.ID); err != nil {
			return err
		}
	}
	return nil
}

func (o *DockerOrchestrator) resolveBinds(mounts map[string]BindSpec) (map[string]BindSpec, error) {
	out := map[string]BindSpec{}
	for name, spec := range mounts {
		host, err := o.Volumes.DeviceFor(name)
		if err != nil {
			return nil, err
		}
		out[host] = spec
	}
	return out, nil
}

func (o *DockerOrchestrator) spawn(svc GpuService, env map[string]string, mounts map[string]BindSpec, shmMB int64, ports map[string]string) error {
	return o.spawnCmd(svc, env, mounts, shmMB, ports, nil)
}

func (o *DockerOrchestrator) spawnCmd(svc GpuService, env map[string]string, mounts map[string]BindSpec, shmMB int64, ports map[string]string, cmd []string) error {
	if err := o.stopGPU(); err != nil {
		return &OrchestratorError{Msg: fmt.Sprintf("stopping GPU services: %v", err)}
	}
	// Defense in depth: remove any name-conflict container left without the
	// label (crashed run, older proxy) so create doesn't 409.
	if existing, err := o.Client.GetByName(svc.Name); err == nil && existing != nil {
		_ = o.Client.Remove(existing.ID)
	}
	binds, err := o.resolveBinds(mounts)
	if err != nil {
		return &OrchestratorError{Msg: err.Error()}
	}
	err = o.Client.CreateAndStart(CreateSpec{
		Image:     svc.Image,
		Name:      svc.Name,
		Hostname:  svc.Hostname,
		Env:       env,
		Binds:     binds,
		Network:   o.Network,
		Labels:    map[string]string{ServiceLabel: svc.Hostname, GPULabel: svc.Mode},
		ShmSize:   shmMB << 20,
		PortBinds: ports,
		Cmd:       cmd,
		GPU:       true,
	})
	if err != nil {
		return &OrchestratorError{Msg: fmt.Sprintf("starting %s: %v", svc.Name, err)}
	}
	return nil
}

func (o *DockerOrchestrator) SpawnLlama(p *Preset) error {
	return o.spawn(LlamaService, p.Env(), map[string]BindSpec{
		"llmc-llama-cache":  {Bind: "/root/.cache", Mode: "rw"},
		"llmc-llama-models": {Bind: "/models", Mode: "rw"},
	}, 2048, nil)
}

// SpawnNinfer starts ninfer-serve for the given preset. The artifact dir is
// mounted read-only: ninfer only reads its .ninfer file, and unlike the llama
// flow there is nothing to download into the mount at spawn time.
func (o *DockerOrchestrator) SpawnNinfer(p *Preset) error {
	argv, err := NinferCommand(p)
	if err != nil {
		return &OrchestratorError{Msg: err.Error()}
	}
	return o.spawnCmd(NinferService, nil, map[string]BindSpec{
		"llmc-ninfer-models": {Bind: "/models", Mode: "ro"},
	}, 2048, nil, argv)
}

// SpawnLLM starts whichever engine the preset names. Callers that reached for
// SpawnLlama should use this, so a ninfer preset cannot silently start
// llama.cpp with a config it does not understand.
func (o *DockerOrchestrator) SpawnLLM(p *Preset) error {
	if p != nil && p.Engine == EngineNinfer {
		return o.SpawnNinfer(p)
	}
	return o.SpawnLlama(p)
}

func (o *DockerOrchestrator) SpawnComfyUI() error {
	return o.spawn(ComfyUIService, nil, map[string]BindSpec{
		"llmc-comfyui-models":       {Bind: "/app/ComfyUI/models", Mode: "rw"},
		"llmc-comfyui-output":       {Bind: "/app/ComfyUI/output", Mode: "rw"},
		"llmc-comfyui-input":        {Bind: "/app/ComfyUI/input", Mode: "rw"},
		"llmc-comfyui-custom-nodes": {Bind: "/app/ComfyUI/custom_nodes", Mode: "rw"},
		"llmc-comfyui-user":         {Bind: "/app/ComfyUI/user", Mode: "rw"},
	}, 4096, map[string]string{"8188/tcp": "8188"})
}

func (o *DockerOrchestrator) SpawnTrain() error {
	return o.spawn(TrainService, map[string]string{
		"TRAIN_PORT":      "8787",
		"DATA_DIR":        "/data",
		"CHECKPOINTS_DIR": "/checkpoints",
	}, map[string]BindSpec{
		"llmc-training-data":  {Bind: "/data", Mode: "rw"},
		"llmc-comfyui-models": {Bind: "/models", Mode: "ro"},
		"llmc-comfyui-loras":  {Bind: "/loras", Mode: "rw"},
	}, 4096, nil)
}

// WaitHealthy polls the service health endpoint until 200 or timeout.
func (o *DockerOrchestrator) WaitHealthy(svc GpuService, timeout time.Duration) bool {
	deadline := time.Now().Add(timeout)
	url := fmt.Sprintf("http://%s:%d%s", svc.Hostname, svc.InternalPort, svc.HealthPath)
	client := &http.Client{Timeout: 3 * time.Second}
	for time.Now().Before(deadline) {
		resp, err := client.Get(url)
		if err == nil {
			io.Copy(io.Discard, resp.Body)
			resp.Body.Close()
			if resp.StatusCode == 200 {
				return true
			}
		}
		time.Sleep(2 * time.Second)
	}
	return false
}

// EnsurePresetAssets downloads mmproj/template assets if missing.
func (o *DockerOrchestrator) EnsurePresetAssets(p *Preset, assetsDir string) error {
	if p.MMProj.URL != "" {
		if _, err := ensureAsset(assetsDir, p.MMProjFilename(), p.MMProj.URL); err != nil {
			return err
		}
	}
	if p.Template.URL != "" {
		if _, err := ensureAsset(assetsDir, p.TemplateFilename(), p.Template.URL); err != nil {
			return err
		}
	}
	return nil
}

// ensureAsset downloads url to assetsDir/filename if missing (tmp + rename).
func ensureAsset(assetsDir, filename, url string) (string, error) {
	if err := os.MkdirAll(assetsDir, 0o755); err != nil {
		return "", err
	}
	dest := filepath.Join(assetsDir, filename)
	if _, err := os.Stat(dest); err == nil {
		return dest, nil
	}
	tmp := dest + ".tmp"
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return "", err
	}
	req.Header.Set("User-Agent", "llmc/2.0")
	client := &http.Client{Timeout: 300 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return "", &OrchestratorError{Msg: fmt.Sprintf("downloading %s: %v", filename, err)}
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return "", &OrchestratorError{Msg: fmt.Sprintf("downloading %s: HTTP %d", filename, resp.StatusCode)}
	}
	f, err := os.Create(tmp)
	if err != nil {
		return "", err
	}
	if _, err := io.Copy(f, resp.Body); err != nil {
		f.Close()
		os.Remove(tmp)
		return "", &OrchestratorError{Msg: fmt.Sprintf("downloading %s: %v", filename, err)}
	}
	f.Close()
	if err := os.Rename(tmp, dest); err != nil {
		return "", err
	}
	return dest, nil
}

// LoadedLlamaModel returns the preset name of the model llama-server
// currently has loaded ("" if unknown/unreachable). Used at startup to fill
// state.model after a state-loss restart, avoiding a pointless swap.
func (o *DockerOrchestrator) LoadedLlamaModel(presets *PresetStore) string {
	client := &http.Client{Timeout: 5 * time.Second}
	resp, err := client.Get(fmt.Sprintf("http://%s:%d/v1/models", LlamaService.Hostname, LlamaService.InternalPort))
	if err != nil {
		return ""
	}
	defer resp.Body.Close()
	var body struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil || len(body.Data) == 0 {
		return ""
	}
	// llama-server reports the GGUF path (e.g. /models/Qwen3.8-27B-Q4_K_M.gguf)
	base := strings.TrimSuffix(filepath.Base(body.Data[0].ID), ".gguf")
	if p := presets.ByName(base); p != nil {
		return p.Name
	}
	return ""
}

// ValidMode reports whether name is a switchable GPU mode.
func ValidMode(name string) bool {
	_, ok := Services[name]
	return ok
}

// ServiceFor returns the GpuService for a mode.
func ServiceFor(mode string) (GpuService, bool) {
	s, ok := Services[mode]
	return s, ok
}
