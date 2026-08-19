// Docker Engine API client over the unix socket - stdlib net/http only
// (house convention; drawbridge does the same). Covers exactly the
// operations the orchestrator needs: list-by-label, inspect, stop, remove,
// create+start with GPU device requests, and image presence via create
// errors.
package proxy

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"strings"
	"time"
)

type DockerError struct {
	StatusCode int
	Msg        string
}

func (e *DockerError) Error() string { return e.Msg }

func dockerErr(status int, msg string) *DockerError {
	return &DockerError{StatusCode: status, Msg: msg}
}

// IsNotFound reports a 404 from the daemon (missing container or image).
func IsNotFound(err error) bool {
	var de *DockerError
	if asDockerErr(err, &de) {
		return de.StatusCode == 404
	}
	return false
}

func asDockerErr(err error, out **DockerError) bool {
	for err != nil {
		if de, ok := err.(*DockerError); ok {
			*out = de
			return true
		}
		if u, ok := err.(interface{ Unwrap() error }); ok {
			err = u.Unwrap()
			continue
		}
		break
	}
	return false
}

type DockerClient struct {
	hc *http.Client
}

func NewDockerClient(socketPath string) *DockerClient {
	return &DockerClient{hc: &http.Client{
		Timeout: 60 * time.Second,
		Transport: &http.Transport{
			DialContext: func(ctx context.Context, _, _ string) (net.Conn, error) {
				var d net.Dialer
				return d.DialContext(ctx, "unix", socketPath)
			},
		},
	}}
}

func (d *DockerClient) do(method, path string, body any, out any) error {
	var rdr io.Reader
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			return err
		}
		rdr = bytes.NewReader(b)
	}
	req, err := http.NewRequest(method, "http://docker"+path, rdr)
	if err != nil {
		return err
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := d.hc.Do(req)
	if err != nil {
		return fmt.Errorf("docker %s %s: %w", method, path, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		var e struct {
			Message string `json:"message"`
		}
		msg := strings.TrimSpace(string(b))
		if json.Unmarshal(b, &e) == nil && e.Message != "" {
			msg = e.Message
		}
		return dockerErr(resp.StatusCode, fmt.Sprintf("docker %s %s: %d %s", method, path, resp.StatusCode, msg))
	}
	if out != nil {
		return json.NewDecoder(resp.Body).Decode(out)
	}
	io.Copy(io.Discard, resp.Body)
	return nil
}

type ContainerSummary struct {
	ID     string            `json:"Id"`
	Names  []string          `json:"Names"`
	Image  string            `json:"Image"`
	State  string            `json:"State"`
	Labels map[string]string `json:"Labels"`
}

// ListByLabel returns containers (all states) carrying the given label key.
// An empty label lists all containers.
func (d *DockerClient) ListByLabel(label string) ([]ContainerSummary, error) {
	q := url.Values{}
	q.Set("all", "1")
	if label != "" {
		fb, _ := json.Marshal(map[string][]string{"label": {label}})
		q.Set("filters", string(fb))
	}
	var out []ContainerSummary
	if err := d.do("GET", "/containers/json?"+q.Encode(), nil, &out); err != nil {
		return nil, err
	}
	return out, nil
}

// Stop stops a container (SIGTERM, then SIGKILL after timeoutSec).
func (d *DockerClient) Stop(id string, timeoutSec int) error {
	err := d.do("POST", fmt.Sprintf("/containers/%s/stop?t=%d", id, timeoutSec), nil, nil)
	if IsNotFound(err) || (err != nil && strings.Contains(err.Error(), "304")) {
		return nil // already stopped
	}
	return err
}

// Remove force-removes a container. Missing container is not an error.
func (d *DockerClient) Remove(id string) error {
	err := d.do("DELETE", "/containers/"+id+"?force=true", nil, nil)
	if IsNotFound(err) {
		return nil
	}
	return err
}

// GetByName returns the container with the exact name, or nil.
func (d *DockerClient) GetByName(name string) (*ContainerSummary, error) {
	all, err := d.ListByLabel("") // cheap enough at this scale; filtered below
	if err != nil {
		return nil, err
	}
	for _, c := range all {
		for _, n := range c.Names {
			if strings.TrimPrefix(n, "/") == name {
				cp := c
				return &cp, nil
			}
		}
	}
	return nil, nil
}

// CreateSpec is the subset of the Engine create body the proxy uses.
type CreateSpec struct {
	Image     string
	Name      string
	Hostname  string
	Env       map[string]string
	Binds     map[string]BindSpec // host path -> bind
	Network   string
	Labels    map[string]string
	ShmSize   int64
	PortBinds map[string]string // containerPort/tcp -> host loopback port
	GPU       bool
}

type BindSpec struct {
	Bind string
	Mode string
}

type createBody struct {
	Image      string            `json:"Image"`
	Hostname   string            `json:"Hostname"`
	Env        []string          `json:"Env"`
	Labels     map[string]string `json:"Labels"`
	HostConfig hostConfig        `json:"HostConfig"`
}

type hostConfig struct {
	Binds          []string          `json:"Binds,omitempty"`
	NetworkMode    string            `json:"NetworkMode,omitempty"`
	ShmSize        int64             `json:"ShmSize,omitempty"`
	RestartPolicy  restartPolicy     `json:"RestartPolicy"`
	LogConfig      logConfig         `json:"LogConfig"`
	DeviceRequests []deviceRequest   `json:"DeviceRequests,omitempty"`
	PortBindings   map[string][]port `json:"PortBindings,omitempty"`
}

type restartPolicy struct {
	Name string `json:"Name"`
}

type logConfig struct {
	Type   string            `json:"Type"`
	Config map[string]string `json:"Config"`
}

type deviceRequest struct {
	Count        int        `json:"Count"`
	Capabilities [][]string `json:"Capabilities"`
}

type port struct {
	HostIP   string `json:"HostIp"`
	HostPort string `json:"HostPort"`
}

// CreateAndStart creates and starts the container. A 404 on create means the
// image is missing - surfaced verbatim so the proxy can tell the caller to
// build/pull.
func (d *DockerClient) CreateAndStart(spec CreateSpec) error {
	body := createBody{
		Image:    spec.Image,
		Hostname: spec.Hostname,
		Labels:   spec.Labels,
		HostConfig: hostConfig{
			NetworkMode:   spec.Network,
			ShmSize:       spec.ShmSize,
			RestartPolicy: restartPolicy{Name: "unless-stopped"},
			LogConfig: logConfig{
				Type:   "json-file",
				Config: map[string]string{"max-size": "50m", "max-file": "3"},
			},
		},
	}
	for k, v := range spec.Env {
		body.Env = append(body.Env, k+"="+v)
	}
	for host, b := range spec.Binds {
		body.HostConfig.Binds = append(body.HostConfig.Binds, host+":"+b.Bind+":"+b.Mode)
	}
	if spec.GPU {
		body.HostConfig.DeviceRequests = []deviceRequest{{Count: -1, Capabilities: [][]string{{"gpu"}}}}
	}
	if len(spec.PortBinds) > 0 {
		body.HostConfig.PortBindings = map[string][]port{}
		for cp, hp := range spec.PortBinds {
			body.HostConfig.PortBindings[cp] = []port{{HostIP: "127.0.0.1", HostPort: hp}}
		}
	}
	var created struct {
		ID string `json:"Id"`
	}
	if err := d.do("POST", "/containers/create?name="+url.QueryEscape(spec.Name), body, &created); err != nil {
		if IsNotFound(err) {
			return dockerErr(404, fmt.Sprintf("image %q not found. Run `make build` or `make pull` to fetch it.", spec.Image))
		}
		return err
	}
	return d.do("POST", "/containers/"+created.ID+"/start", nil, nil)
}
