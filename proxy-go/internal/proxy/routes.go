// Task-routing alias table: maps an alias name ("default", "code", ...) to
// an ordered chain of presets. Mirrors the preset loader's strictness
// (unknown keys rejected) but chain-vs-preset resolution is deferred to
// request time (presets reload independently).
// See docs/specs/2026-08-24-task-routing.md.
package proxy

import (
	"fmt"
	"os"
	"strings"

	"github.com/BurntSushi/toml"
)

// RouteError is raised on any routes-file schema violation.
type RouteError struct{ Msg string }

func (e *RouteError) Error() string { return e.Msg }

func routeErr(format string, args ...any) *RouteError {
	return &RouteError{Msg: fmt.Sprintf(format, args...)}
}

// Route is one alias: an ordered chain of preset names/model IDs.
type Route struct {
	Name        string   // alias name (table key)
	Description string   // optional one-line summary
	Chain       []string // ordered by preference; empty = "whatever is resident"
}

// parseRoutes decodes a routes.toml document into a map of alias -> Route.
func parseRoutes(data []byte) (map[string]*Route, error) {
	var raw map[string]map[string]any
	if _, err := toml.Decode(string(data), &raw); err != nil {
		return nil, routeErr("routes: invalid TOML: %v", err)
	}
	out := map[string]*Route{}
	for name, table := range raw {
		if strings.Contains(name, ":") {
			return nil, routeErr("routes: alias %q must not contain ':'", name)
		}
		r := &Route{Name: name}
		for k, v := range table {
			switch k {
			case "description":
				d, ok := v.(string)
				if !ok {
					return nil, routeErr("routes: alias %q: description must be a string", name)
				}
				r.Description = d
			case "chain":
				arr, ok := v.([]any)
				if !ok {
					return nil, routeErr("routes: alias %q: chain must be an array of strings", name)
				}
				for _, e := range arr {
					s, ok := e.(string)
					if !ok {
						return nil, routeErr("routes: alias %q: chain entries must be strings", name)
					}
					r.Chain = append(r.Chain, s)
				}
			default:
				return nil, routeErr("routes: alias %q: unknown key %q", name, k)
			}
		}
		out[name] = r
	}
	return out, nil
}

// RouteStore is the routes registry. Reload() rescans the file so an edit
// is picked up without a proxy restart (parity with preset live-reload).
type RouteStore struct {
	Path   string
	routes map[string]*Route
}

func NewRouteStore(path string) (*RouteStore, error) {
	s := &RouteStore{Path: path, routes: map[string]*Route{}}
	if err := s.Reload(); err != nil {
		return nil, err
	}
	return s, nil
}

// Reload reads and reparses the routes file. A missing file yields zero
// aliases (no error); a malformed file returns an error and keeps the last
// good snapshot (or empty on first load).
func (s *RouteStore) Reload() error {
	data, err := os.ReadFile(s.Path)
	if os.IsNotExist(err) {
		s.routes = map[string]*Route{}
		return nil
	}
	if err != nil {
		return routeErr("routes: %v", err)
	}
	routes, err := parseRoutes(data)
	if err != nil {
		return err
	}
	s.routes = routes
	return nil
}

func (s *RouteStore) ByName(name string) *Route { return s.routes[name] }

func (s *RouteStore) All() map[string]*Route {
	out := make(map[string]*Route, len(s.routes))
	for k, v := range s.routes {
		out[k] = v
	}
	return out
}

// routeAliasName maps a model field to an alias name: "auto" -> "default",
// "auto:<name>" -> "<name>", anything else -> "" (not an alias).
func routeAliasName(model string) string {
	if model == "auto" {
		return "default"
	}
	if strings.HasPrefix(model, "auto:") {
		return strings.TrimPrefix(model, "auto:")
	}
	return ""
}

// aliasError marks an unknown alias (handler maps it to 404).
type aliasError struct{ name string }

func (e *aliasError) Error() string {
	return fmt.Sprintf("unknown route alias %q", e.name)
}
