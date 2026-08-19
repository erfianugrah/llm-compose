// Proxy state - persisted atomically to active.toml (schema-compatible with
// llmc/state.py; the Go proxy adds lock_expires_at and lock_queue, which the
// Python loader was taught to tolerate for shared-file cutover).
//
// Writes are tmp-file + fsync + rename. The scheduler loop is the only
// in-process writer, so no locking here; the file lock story across
// processes is "last writer wins", same as the Python proxy.
package proxy

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"time"

	"github.com/BurntSushi/toml"
)

// StateError is raised when the state file is malformed.
type StateError struct{ Msg string }

func (e *StateError) Error() string { return e.Msg }

var validModes = map[string]bool{"llm": true, "comfyui": true, "train": true, "idle": true}

type QueueEntry struct {
	Owner string `toml:"owner"`
	Model string `toml:"model"`
	TS    int64  `toml:"ts"`
}

type State struct {
	Mode          string       `toml:"mode"`
	Model         string       `toml:"model,omitempty"`
	Locked        string       `toml:"locked,omitempty"`
	LockOwners    []string     `toml:"lock_owners,omitempty"`
	LockExpiresAt int64        `toml:"lock_expires_at,omitempty"`
	LockQueue     []QueueEntry `toml:"lock_queue,omitempty"`
	UpdatedAt     int64        `toml:"updated_at"`
}

func (s *State) validate() error {
	if !validModes[s.Mode] {
		return &StateError{Msg: fmt.Sprintf("invalid mode %q", s.Mode)}
	}
	return nil
}

// LoadState reads the state file; a missing file yields idle state.
// Unknown keys are tolerated (forward/backward compat with the Python proxy).
func LoadState(path string) (*State, error) {
	data, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return &State{Mode: "idle"}, nil
	}
	if err != nil {
		return nil, err
	}
	st := &State{Mode: "idle"}
	if _, err := toml.Decode(string(data), st); err != nil {
		return nil, &StateError{Msg: fmt.Sprintf("%s: invalid TOML: %v", path, err)}
	}
	if err := st.validate(); err != nil {
		return nil, err
	}
	return st, nil
}

// SaveState atomically writes state (tmp + fsync + rename).
func SaveState(path string, st *State) error {
	if st.UpdatedAt == 0 {
		st.UpdatedAt = time.Now().Unix()
	}
	if err := st.validate(); err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	tmp := path + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	enc := toml.NewEncoder(f)
	if err := enc.Encode(st); err != nil {
		f.Close()
		os.Remove(tmp)
		return err
	}
	if err := f.Sync(); err != nil {
		f.Close()
		os.Remove(tmp)
		return err
	}
	if err := f.Close(); err != nil {
		os.Remove(tmp)
		return err
	}
	return os.Rename(tmp, path)
}

// SortedOwners returns the lock owners in deterministic order (the Python
// proxy always serves them sorted).
func (s *State) SortedOwners() []string {
	out := append([]string(nil), s.LockOwners...)
	sort.Strings(out)
	return out
}
