package main

import "testing"

func TestVersionIsSet(t *testing.T) {
	if Version == "" {
		t.Fatal("expected Version to be a non-empty string")
	}
}
