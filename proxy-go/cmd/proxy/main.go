// Command proxy runs the llmc model proxy: an OpenAI/Anthropic
// compatibility gateway in front of the GPU llama-server fleet.
package main

import (
	"context"
	"log"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"sort"
	"strconv"
	"syscall"
	"time"

	"github.com/erfianugrah/llm-compose/proxy-go/internal/proxy"
)

func main() {
	logf := func(f string, a ...any) { log.Printf("[llmc-proxy] "+f, a...) }

	port := envStr("LLMC_PROXY_PORT", "11434")
	presetsDir := envStr("LLMC_PRESETS_DIR", "/presets")
	stateDir := envStr("LLMC_STATE_DIR", "/state")
	assetsDir := envStr("LLMC_ASSETS_DIR", "/assets")
	volumesTOML := envStr("LLMC_VOLUMES_TOML", "/volumes.toml")
	vramLimit := envFloat("LLMC_VRAM_LIMIT_GB", 32)
	vramReserve := envFloat("LLMC_VRAM_RESERVE_GB", 6)
	network := envStr("LLMC_NETWORK", "llmc")
	healthTimeout := time.Duration(envInt("LLMC_HEALTH_TIMEOUT", 900)) * time.Second
	drainGrace := time.Duration(envInt("LLMC_DRAIN_GRACE_S", 60)) * time.Second
	lockTTL := time.Duration(envInt("LLMC_LOCK_TTL_S", 900)) * time.Second

	volumes, err := proxy.LoadVolumes(volumesTOML)
	if err != nil {
		log.Fatalf("load volumes: %v", err)
	}

	client := proxy.NewDockerClient("/var/run/docker.sock")
	orch := &proxy.DockerOrchestrator{Client: client, Network: network, Volumes: volumes}

	store, err := proxy.NewPresetStore(presetsDir)
	if err != nil {
		log.Fatalf("preset store: %v", err)
	}

	routesFile := envStr("LLMC_ROUTES_FILE", "/routes.toml")
	routes, err := proxy.NewRouteStore(routesFile)
	if err != nil {
		log.Fatalf("routes: %v", err)
	}

	sched, err := proxy.NewScheduler(proxy.SchedulerConfig{
		StatePath:     filepath.Join(stateDir, "active.toml"),
		AssetsDir:     assetsDir,
		DrainGrace:    drainGrace,
		LockTTL:       lockTTL,
		HealthTimeout: healthTimeout,
		VRAMLimitGB:   vramLimit,
		VRAMReserveGB: vramReserve,
	}, orch, store, logf)
	if err != nil {
		log.Fatalf("scheduler: %v", err)
	}
	go sched.Run()

	server := proxy.NewServer(sched, store, routes, proxy.ServerConfig{
		VRAMLimitGB:   vramLimit,
		VRAMReserveGB: vramReserve,
	}, logf)

	// Startup log.
	names := store.Names()
	sort.Strings(names)
	status := sched.Status()
	logf("listening on :%s", port)
	logf("docker network: %s", network)
	logf("presets: %v", names)
	logf("active mode=%s model=%s", status.Mode, status.Model)
	logf("vram budget: %.0f GB total, %.0f GB reserved -> %.0f GB max model weight",
		vramLimit, vramReserve, vramLimit-vramReserve)

	srv := &http.Server{Addr: ":" + port, Handler: server}

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGTERM, syscall.SIGINT)
	defer stop()

	errCh := make(chan error, 1)
	go func() {
		logf("http server starting on :%s", port)
		errCh <- srv.ListenAndServe()
	}()

	select {
	case <-ctx.Done():
		logf("shutdown signal received")
	case err := <-errCh:
		if err != nil && err != http.ErrServerClosed {
			log.Fatalf("http server: %v", err)
		}
	}

	shutCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := srv.Shutdown(shutCtx); err != nil {
		logf("http shutdown: %v", err)
	}
	sched.Close()
	logf("shutdown complete")
}

func envStr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func envInt(key string, def int) int {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
		log.Printf("[llmc-proxy] invalid %s=%q, using default %d", key, v, def)
	}
	return def
}

func envFloat(key string, def float64) float64 {
	if v := os.Getenv(key); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			return f
		}
		log.Printf("[llmc-proxy] invalid %s=%q, using default %v", key, v, def)
	}
	return def
}
