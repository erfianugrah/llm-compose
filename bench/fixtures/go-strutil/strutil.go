package strutil

import "strings"

// Reverse returns s with its runes in reverse order.
func Reverse(s string) string {
	r := []rune(s)
	for i, j := 0, len(r)-1; i < j; i, j = i+1, j-1 {
		r[i], r[j] = r[j], r[i]
	}
	return string(r)
}

// IsPalindrome reports whether s reads the same forwards and backwards.
// BUG: comparison is case-sensitive; the package contract is case-insensitive.
func IsPalindrome(s string) bool {
	return s == Reverse(s)
}

// Split splits s around each instance of sep.
func Split(s, sep string) []string {
	if sep == "" {
		return []string{s}
	}
	return strings.Split(s, sep)
}
