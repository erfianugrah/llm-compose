// Orchestrator: GPU service lifecycle on top of the Docker client.
// Mirror of llmc/orchestrator.py - three mutually exclusive GPU services
// (llama-server, comfyui, lora-train) found via the llmc.mode label.
package proxy

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
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
