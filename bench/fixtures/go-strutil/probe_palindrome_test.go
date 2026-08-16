package strutil

import "testing"

// T2 probe: IsPalindrome must be case-insensitive per the package contract.
func TestIsPalindromeCaseInsensitive(t *testing.T) {
	if !IsPalindrome("Level") {
		t.Error("IsPalindrome(Level) = false, want true")
	}
	if !IsPalindrome("RaceCar") {
		t.Error("IsPalindrome(RaceCar) = false, want true")
	}
}
