package strutil

import "testing"

func TestSplitBasic(t *testing.T) {
	if got := Split("a,b,c", ","); len(got) != 3 || got[0] != "a" || got[2] != "c" {
		t.Errorf("Split(a,b,c, comma) = %v", got)
	}
}

func TestSplitEmptySep(t *testing.T) {
	if got := Split("abc", ""); len(got) != 1 || got[0] != "abc" {
		t.Errorf("Split(abc, empty) = %v", got)
	}
}

func TestSplitRepeated(t *testing.T) {
	if got := Split("a--b--c", "--"); len(got) != 3 {
		t.Errorf("Split(a--b--c, --) = %v", got)
	}
}
