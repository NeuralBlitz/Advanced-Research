package unified

import (
	"context"
	"testing"

	"github.com/sirupsen/logrus"
)

// A disabled-integration configuration must support local status inspection
// without requiring an LRS server, a Git workspace, or a computation backend.
func TestDisabledIntegrationsCanInitializeAndReportStatusOffline(t *testing.T) {
	cfg := DefaultConfig()
	cfg.LRS.Enabled = false
	cfg.Opencode.Enabled = false
	cfg.NeuralBlitz.Enabled = false
	cfg.Security.JWTSecret = ""

	system, err := NewSystem(cfg, logrus.New())
	if err != nil {
		t.Fatalf("offline initialization failed: %v", err)
	}
	defer func() {
		if err := system.Close(context.Background()); err != nil {
			t.Errorf("offline shutdown failed: %v", err)
		}
	}()

	status, err := system.GetSystemStatus(context.Background())
	if err != nil {
		t.Fatalf("offline status failed: %v", err)
	}
	for _, key := range []string{"lrs", "opencode", "neuralblitz"} {
		if status.IntegrationStatus[key] {
			t.Errorf("%s should remain disabled", key)
		}
	}
	if status.ActiveSessions != 0 {
		t.Errorf("expected no active sessions, got %d", status.ActiveSessions)
	}
}

func TestDisabledIntegrationsRemainDisabledAfterCreatingSession(t *testing.T) {
	cfg := DefaultConfig()
	cfg.LRS.Enabled = false
	cfg.Opencode.Enabled = false
	cfg.NeuralBlitz.Enabled = false
	system, err := NewSystem(cfg, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer system.Close(context.Background())
	sessionID, err := system.CreateSession(context.Background(), "offline-user", ModeResearch, nil)
	if err != nil {
		t.Fatal(err)
	}
	if sessionID == "" {
		t.Fatal("expected a session ID")
	}
	status, err := system.GetSystemStatus(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if status.ActiveSessions != 1 {
		t.Fatalf("expected one session, got %d", status.ActiveSessions)
	}
	for _, name := range []string{"lrs", "opencode", "neuralblitz"} {
		if status.IntegrationStatus[name] {
			t.Fatalf("session creation unexpectedly enabled %s", name)
		}
	}
}
