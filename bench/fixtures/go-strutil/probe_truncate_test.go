package strutil

import "testing"

// T1 probe: Truncate(s, n) keeps at most n RUNES (unicode-safe), no suffix.
func TestTruncate(t *testing.T) {
	cases := []struct {
		in   string
		n    int
		want string
	}{
		{"hello world", 5, "hello"},
		{"short", 10, "short"},
		{"héllo wörld", 3, "hél"},
		{"", 4, ""},
	}
	for _, c := range cases {
		if got := Truncate(c.in, c.n); got != c.want {
			t.Errorf("Truncate(%q, %d) = %q, want %q", c.in, c.n, got, c.want)
		}
	}
}
