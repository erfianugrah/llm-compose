// Bind-mount path registry - mirror of llmc/volumes.py. The proxy resolves
// logical volume names to host paths when spawning GPU services (the Docker
// daemon binds host paths directly; no named-volume indirection).
package proxy

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/BurntSushi/toml"
)

type VolumeError struct{ Msg string }

func (e *VolumeError) Error() string { return e.Msg }

type VolumeRegistry struct {
	Root    string
	Volumes map[string]string // logical name -> absolute host path
}

var volumeNameRe = regexp.MustCompile(`^[a-zA-Z0-9][a-zA-Z0-9_.-]*$`)

var envRe = regexp.MustCompile(`\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)`)

// expandEnv expands $VAR/${VAR} and ~. Fails loud on unset vars.
func expandEnv(value string) (string, error) {
	missing := ""
	out := envRe.ReplaceAllStringFunc(value, func(m string) string {
		sub := envRe.FindStringSubmatch(m)
		name := sub[1]
		if name == "" {
			name = sub[2]
		}
		v, ok := os.LookupEnv(name)
		if !ok {
			missing = name
			return m
		}
		return v
	})
	if missing != "" {
		return "", &VolumeError{Msg: fmt.Sprintf("unresolved environment variable ${%s} in %q", missing, value)}
	}
	if strings.HasPrefix(out, "~/") {
		home, err := os.UserHomeDir()
		if err != nil {
			return "", err
		}
		out = filepath.Join(home, out[2:])
	}
	return out, nil
}

func LoadVolumes(path string) (*VolumeRegistry, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, &VolumeError{Msg: fmt.Sprintf("%s: %v", path, err)}
	}
	var doc struct {
		Root    string `toml:"root"`
		Volumes map[string]struct {
			Path string `toml:"path"`
		} `toml:"volumes"`
	}
	if _, err := toml.Decode(string(data), &doc); err != nil {
		return nil, &VolumeError{Msg: fmt.Sprintf("%s: invalid TOML: %v", path, err)}
	}
	if len(doc.Volumes) == 0 {
		return nil, &VolumeError{Msg: fmt.Sprintf("%s: at least one volume required", path)}
	}
	root := doc.Root
	if root == "" {
		root = "${HOME}/docker-volumes"
	}
	root, err = expandEnv(root)
	if err != nil {
		return nil, err
	}
	reg := &VolumeRegistry{Root: root, Volumes: map[string]string{}}
	for name, spec := range doc.Volumes {
		if !volumeNameRe.MatchString(name) {
			return nil, &VolumeError{Msg: fmt.Sprintf("%s: invalid volume name %q", path, name)}
		}
		if strings.TrimSpace(spec.Path) == "" {
			return nil, &VolumeError{Msg: fmt.Sprintf("%s: volumes.%s.path must be non-empty", path, name)}
		}
		p, err := expandEnv(spec.Path)
		if err != nil {
			return nil, err
		}
		if !filepath.IsAbs(p) {
			p = filepath.Join(root, p)
		}
		reg.Volumes[name] = p
	}
	return reg, nil
}

// DeviceFor resolves a logical name to a host path, creating the directory
// (the daemon refuses to auto-create bind sources). Passes through values
// that are already absolute host paths.
func (r *VolumeRegistry) DeviceFor(name string) (string, error) {
	if strings.HasPrefix(name, "/") {
		return name, nil
	}
	p, ok := r.Volumes[name]
	if !ok {
		return "", &VolumeError{Msg: fmt.Sprintf("unknown volume %q", name)}
	}
	if err := os.MkdirAll(p, 0o755); err != nil {
		return "", err
	}
	return p, nil
}
