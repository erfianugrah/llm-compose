package strutil

import "testing"

func TestReverse(t *testing.T) {
	if got := Reverse("hello"); got != "olleh" {
		t.Errorf("Reverse(hello) = %q, want olleh", got)
	}
}

func TestIsPalindrome(t *testing.T) {
	if !IsPalindrome("level") {
		t.Error("IsPalindrome(level) = false, want true")
	}
	if IsPalindrome("hello") {
		t.Error("IsPalindrome(hello) = true, want false")
	}
}
