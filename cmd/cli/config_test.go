package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func writeConfig(t *testing.T, contents string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "custom.yaml")
	if err := os.WriteFile(path, []byte(contents), 0600); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestReadRuntimeConfigHonorsNestedIntegrations(t *testing.T) {
	path := writeConfig(t, `server:
  host: "127.0.0.1"
  http_port: 8123
  grpc_port: 9123
integrations:
  lrs:
    enabled: false
    endpoint: "http://localhost:4000/xapi/"
  opencode:
    enabled: false
    local_path: "/tmp/example-research-docs"
  neuralblitz:
    enabled: false
    backend: cpu
processing:
  context_max_size: 4096
  query_timeout: 45s
  context_cleanup_interval: 5m
`)
	cfg, err := readRuntimeConfig(path)
	if err != nil {
		t.Fatalf("readRuntimeConfig: %v", err)
	}
	if cfg.Server.HTTPPort != 8123 || cfg.Server.GRPCPort != 9123 {
		t.Fatalf("ignored selected server ports: %+v", cfg.Server)
	}
	if cfg.LRS.Enabled || cfg.LRS.Endpoint != "http://localhost:4000/xapi/" {
		t.Fatalf("ignored LRS configuration: %+v", cfg.LRS)
	}
	if cfg.Opencode.Enabled || cfg.Opencode.LocalPath != "/tmp/example-research-docs" {
		t.Fatalf("ignored workspace configuration: %+v", cfg.Opencode)
	}
	if cfg.NeuralBlitz.Enabled {
		t.Fatalf("ignored neuralblitz disable flag: %+v", cfg.NeuralBlitz)
	}
	if cfg.Processing.ContextMaxSize != 4096 ||
		cfg.Processing.QueryTimeout != 45*time.Second ||
		cfg.Processing.ContextCleanupInterval != 5*time.Minute {
		t.Fatalf("incorrect duration or context settings: %+v", cfg.Processing)
	}
}

func TestReadRuntimeConfigRejectsMissingAndMalformedFiles(t *testing.T) {
	if _, err := readRuntimeConfig(filepath.Join(t.TempDir(), "missing.yaml")); err == nil {
		t.Fatal("missing config must not silently return defaults")
	}
	path := writeConfig(t, "server: [unclosed\n")
	if _, err := readRuntimeConfig(path); err == nil {
		t.Fatal("malformed YAML must not silently return defaults")
	}
	badDuration := writeConfig(t, "processing:\n  query_timeout: definitely-not-a-duration\n")
	if _, err := readRuntimeConfig(badDuration); err == nil {
		t.Fatal("invalid time duration must fail")
	}
}

func TestGeneratedConfigDoesNotEnableExternalIntegrations(t *testing.T) {
	path := filepath.Join(t.TempDir(), "generated.yaml")
	if err := createDefaultConfig(path); err != nil {
		t.Fatalf("createDefaultConfig: %v", err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(data), "jwt_secret:") {
		t.Fatal("generated configuration must not contain a placeholder signing secret")
	}
	cfg, err := readRuntimeConfig(path)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.LRS.Enabled || cfg.Opencode.Enabled {
		t.Fatalf("external services must be opt-in: LRS=%v Opencode=%v", cfg.LRS.Enabled, cfg.Opencode.Enabled)
	}
	if cfg.Server.Host != "127.0.0.1" {
		t.Fatalf("generated configuration must not expose a network listener: %s", cfg.Server.Host)
	}
}

func TestServerCommandDoesNotClaimToStartUnsupportedListener(t *testing.T) {
	if serverCmd.RunE == nil {
		t.Fatal("server command must return an explicit error instead of blocking")
	}
	err := serverCmd.RunE(serverCmd, nil)
	if err == nil || !strings.Contains(err.Error(), "not implemented") {
		t.Fatalf("expected explicit unsupported-server error; got %v", err)
	}
}
