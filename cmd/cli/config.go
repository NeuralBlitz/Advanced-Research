package main

import (
	"fmt"

	"github.com/spf13/viper"

	"github.com/advanced-research/go/pkg/unified"
)

// readRuntimeConfig reads the supported runtime settings from a YAML file.
// Integrations use the documented nested "integrations" section, rather than
// the top-level layout of unified.Config. Omitted settings retain defaults.
func readRuntimeConfig(path string) (*unified.Config, error) {
	if path == "" {
		return nil, fmt.Errorf("configuration path must not be empty")
	}

	reader := viper.New()
	reader.SetConfigFile(path)
	if err := reader.ReadInConfig(); err != nil {
		return nil, fmt.Errorf("read configuration %q: %w", path, err)
	}

	cfg := unified.DefaultConfig()
	sections := []struct {
		key    string
		target interface{}
	}{
		{"server", &cfg.Server},
		{"integrations.lrs", cfg.LRS},
		{"integrations.opencode", cfg.Opencode},
		{"integrations.neuralblitz", cfg.NeuralBlitz},
		{"processing", &cfg.Processing},
		{"security", &cfg.Security},
	}
	for _, section := range sections {
		if !reader.IsSet(section.key) {
			continue
		}
		if err := reader.UnmarshalKey(section.key, section.target); err != nil {
			return nil, fmt.Errorf("invalid %s configuration: %w", section.key, err)
		}
	}
	return cfg, nil
}
