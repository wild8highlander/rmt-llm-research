package main

import (
	"encoding/json"
	"fmt"
	"os"
	"testing"
)

func TestAll3DExperiments(t *testing.T) {
	params := map[string]interface{}{
		"hessian_grid_size":              16,
		"trajectory_points":              32,
		"pca_components":                 3,
		"curvature_neighbors":            6,
		"attention_flow_3d_resolution":   16,
		"parameter_space_grid":           8,
		"spectral_surface_layers":        4,
		"n_agents":                       3,
		"n_rounds":                       3,
		"ncrit_threshold":                96.0,
		"theta_b_deg":                    7.07,
		"seed":                           42,
		"temperature":                    "inf",
		"max_tokens":                     "+inf",
	}
	combined := runAll3D(params)
	if combined == nil {
		t.Fatal("runAll3D returned nil")
	}
	exps, ok := combined["experiments"].([]map[string]interface{})
	if !ok {
		t.Fatalf("experiments slice missing: %T", combined["experiments"])
	}
	if len(exps) != 9 {
		t.Fatalf("expected 9 experiments, got %d", len(exps))
	}
	research, ok := combined["3d_research"].(map[string]interface{})
	if !ok {
		t.Fatalf("3d_research map missing")
	}
	if len(research) != 9 {
		t.Fatalf("expected 9 3d_research entries, got %d", len(research))
	}
	for i, e := range exps {
		if errMsg, ok := e["error"]; ok {
			t.Errorf("experiment %d (id=%v) errored: %v", i, e["id"], errMsg)
			continue
		}
		fmt.Printf("  [%v] %v — %vs\n", e["id"], e["name"], e["elapsed_seconds"])
	}
	// Marshal one result to JSON to ensure validity
	b, err := json.MarshalIndent(combined, "", "  ")
	if err != nil {
		t.Fatalf("JSON marshal failed: %v", err)
	}
	// Write to /tmp for inspection
	os.WriteFile("/tmp/go_3d_combined.json", b, 0644)
	fmt.Printf("Combined JSON: %d bytes written to /tmp/go_3d_combined.json\n", len(b))
}

func TestClampInf(t *testing.T) {
	cases := []struct {
		name     string
		input    interface{}
		expected float64
	}{
		{"nil", nil, 7.0},
		{"inf_str", "inf", 1e6},
		{"plus_inf_str", "+inf", 1e6},
		{"infinity_str", "infinity", 1e6},
		{"neg_inf_str", "-inf", -1e6},
		{"nan_str", "nan", 7.0},
		{"empty_str", "", 7.0},
		{"int", 42, 42.0},
		{"float", 3.14, 3.14},
		{"string_float", "2.5", 2.5},
		{"bool_true", true, 1.0},
		{"bool_false", false, 0.0},
	}
	for _, c := range cases {
		got := clampInf(c.input, 7.0, 1e6)
		if got != c.expected {
			t.Errorf("clampInf(%v) = %v, want %v (%s)", c.input, got, c.expected, c.name)
		}
	}
}

func TestJacobiEigen(t *testing.T) {
	// Diagonal 2x2 matrix
	m := [][]float64{{3.0, 0.0}, {0.0, 1.0}}
	vals, vecs := jacobiEigen(m)
	if len(vals) != 2 {
		t.Fatalf("expected 2 eigenvalues, got %d", len(vals))
	}
	if vals[0] != 1.0 || vals[1] != 3.0 {
		t.Errorf("eigenvalues = %v, want [1, 3]", vals)
	}
	if len(vecs) != 2 {
		t.Fatalf("expected 2 eigenvectors")
	}
}

func TestSoftmax(t *testing.T) {
	x := []float64{1.0, 2.0, 3.0}
	s := softmax(x)
	sum := 0.0
	for _, v := range s {
		sum += v
	}
	if sum < 0.99 || sum > 1.01 {
		t.Errorf("softmax doesn't sum to 1: %v (sum=%v)", s, sum)
	}
}
