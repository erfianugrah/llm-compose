package proxy

import "testing"

// Ephemeral presets: in-memory overlay over the TOML store. Never on disk,
// dropped on restart. TOML (on-disk) wins on model_id collision.

func TestEphemeralRegisterAndLookup(t *testing.T) {
	s := newTestStore(t, map[string]string{"a": tomlA})
	eph := &Preset{Name: "ctx-sweep-131072", DisplayName: "Sweep", VRAMGB: 5,
		Model: ModelSpec{Repo: "org/x", File: "ctx-sweep-131072.gguf"}}
	if err := s.RegisterEphemeral(eph); err != nil {
		t.Fatalf("register: %v", err)
	}
	if got := s.ByName("ctx-sweep-131072"); got == nil || got.Name != "ctx-sweep-131072" {
		t.Fatalf("ByName did not find ephemeral: %+v", got)
	}
	if got := s.ByName("ctx-sweep-131072.gguf"[:0]+""); got != nil {
		_ = got
	}
	// model_id lookup too (file stem)
	if got := s.ByName(eph.ModelID()); got == nil {
		t.Fatalf("ByName(model_id) did not find ephemeral")
	}
	if _, ok := s.All()[eph.ModelID()]; !ok {
		t.Fatalf("All() missing ephemeral")
	}
	// Names includes it
	found := false
	for _, n := range s.Names() {
		if n == eph.ModelID() {
			found = true
		}
	}
	if !found {
		t.Fatalf("Names() missing ephemeral: %v", s.Names())
	}
}

func TestEphemeralCollisionWithTOMLRejected(t *testing.T) {
	s := newTestStore(t, map[string]string{"a": tomlA})
	p := s.ByName("a")
	dup := &Preset{Name: "a", Model: ModelSpec{Repo: "org/x", File: p.Model.File}}
	if err := s.RegisterEphemeral(dup); err == nil {
		t.Fatalf("expected collision rejection")
	}
}

func TestEphemeralDelete(t *testing.T) {
	s := newTestStore(t, map[string]string{"a": tomlA})
	eph := &Preset{Name: "tmp", Model: ModelSpec{Repo: "o/r", File: "tmp.gguf"}}
	_ = s.RegisterEphemeral(eph)
	if s.ByName("tmp") == nil {
		t.Fatal("not registered")
	}
	s.DeleteEphemeral(eph.ModelID())
	if s.ByName("tmp") != nil {
		t.Fatal("not deleted")
	}
	if _, ok := s.All()[eph.ModelID()]; ok {
		t.Fatal("All() still has it")
	}
}

func TestEphemeralSurvivesTReload(t *testing.T) {
	s := newTestStore(t, map[string]string{"a": tomlA})
	eph := &Preset{Name: "tmp", Model: ModelSpec{Repo: "o/r", File: "tmp.gguf"}}
	_ = s.RegisterEphemeral(eph)
	if err := s.Reload(); err != nil {
		t.Fatalf("reload: %v", err)
	}
	if s.ByName("tmp") == nil {
		t.Fatal("ephemeral lost on Reload - should survive (in-memory overlay)")
	}
}
