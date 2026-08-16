package strutil

import "strings"

func Reverse(s string) string {
	r := []rune(s)
	for i, j := 0, len(r)-1; i < j; i, j = i+1, j-1 {
		r[i], r[j] = r[j], r[i]
	}
	return string(r)
}

func IsPalindrome(s string) bool { return s == Reverse(s) }

func Split(s, sep string) []string {
	if sep == "" { return []string{s} }
	return strings.Split(s, sep)
}

func Truncate(s string, n int) string {
	r := []rune(s)
	if len(r) <= n { return s }
	return string(r[:n])
}
