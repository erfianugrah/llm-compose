package strutil

import "strings"

func Reverse(s string) string {
	r := []rune(s)
	for i, j := 0, len(r)-1; i < j; i, j = i+1, j-1 {
		r[i], r[j] = r[j], r[i]
	}
	return string(r)
}

func IsPalindrome(s string) bool {
	return strings.EqualFold(s, Reverse(s))
}

func Split(s, sep string) []string {
	if sep == "" { return []string{s} }
	return strings.Split(s, sep)
}
