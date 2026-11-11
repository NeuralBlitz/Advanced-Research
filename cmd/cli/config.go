package main

import (
	"fmt"
	"strconv"
	"time"

	"github.com/spf13/viper"

	"github.com/advanced-research/go/pkg/unified"
	"github.com/advanced-research/go/pkg/neuralblitz"
)

// readRuntimeConfig reads the documented nested YAML layout. Explicit key
// mapping avoids relying on mapstructure's inference for snake_case names,
// and validates duration syntax instead of silently accepting zero values.
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
	type setting struct {
		key string
		assign func(string) error
	}
	stringSetting := func(key string, target *string) setting {
		return setting{key, func(value string) error { *target = value; return nil }}
	}
	boolSetting := func(key string, target *bool) setting {
		return setting{key, func(value string) error {
			parsed, err := strconv.ParseBool(value)
			if err != nil { return err }
			*target = parsed
			return nil
		}}
	}
	intSetting := func(key string, target *int) setting {
		return setting{key, func(value string) error {
			parsed, err := strconv.Atoi(value)
			if err != nil { return err }
			*target = parsed
			return nil
		}}
	}
	floatSetting := func(key string, target *float64) setting {
		return setting{key, func(value string) error {
			parsed, err := strconv.ParseFloat(value, 64)
			if err != nil { return err }
			*target = parsed
			return nil
		}}
	}
	durationSetting := func(key string, target *time.Duration) setting {
		return setting{key, func(value string) error {
			parsed, err := time.ParseDuration(value)
			if err != nil { return err }
			*target = parsed
			return nil
		}}
	}

	settings := []setting{
		stringSetting("server.host", &cfg.Server.Host),
		intSetting("server.http_port", &cfg.Server.HTTPPort),
		intSetting("server.grpc_port", &cfg.Server.GRPCPort),
		boolSetting("integrations.lrs.enabled", &cfg.LRS.Enabled),
		stringSetting("integrations.lrs.endpoint", &cfg.LRS.Endpoint),
		stringSetting("integrations.lrs.username", &cfg.LRS.Username),
		stringSetting("integrations.lrs.password", &cfg.LRS.Password),
		durationSetting("integrations.lrs.timeout", &cfg.LRS.Timeout),
		intSetting("integrations.lrs.retry_attempts", &cfg.LRS.RetryAttempts),
		intSetting("integrations.lrs.batch_size", &cfg.LRS.BatchSize),
		boolSetting("integrations.opencode.enabled", &cfg.Opencode.Enabled),
		stringSetting("integrations.opencode.api_key", &cfg.Opencode.APIKey),
		stringSetting("integrations.opencode.workspace", &cfg.Opencode.Workspace),
		stringSetting("integrations.opencode.git_repo", &cfg.Opencode.GitRepo),
		stringSetting("integrations.opencode.local_path", &cfg.Opencode.LocalPath),
		durationSetting("integrations.opencode.timeout", &cfg.Opencode.Timeout),
		boolSetting("integrations.opencode.auto_commit", &cfg.Opencode.AutoCommit),
		stringSetting("integrations.opencode.author_name", &cfg.Opencode.AuthorName),
		stringSetting("integrations.opencode.author_email", &cfg.Opencode.AuthorEmail),
		boolSetting("integrations.neuralblitz.enabled", &cfg.NeuralBlitz.Enabled),
		intSetting("integrations.neuralblitz.device_id", &cfg.NeuralBlitz.DeviceID),
		floatSetting("integrations.neuralblitz.memory_fraction", &cfg.NeuralBlitz.MemoryFraction),
		intSetting("integrations.neuralblitz.tensor_ops_threads", &cfg.NeuralBlitz.TensorOpsThreads),
		intSetting("integrations.neuralblitz.max_batch_size", &cfg.NeuralBlitz.MaxBatchSize),
		durationSetting("integrations.neuralblitz.timeout", &cfg.NeuralBlitz.Timeout),
		intSetting("processing.max_concurrent_queries", &cfg.Processing.MaxConcurrentQueries),
		durationSetting("processing.query_timeout", &cfg.Processing.QueryTimeout),
		durationSetting("processing.pipeline_timeout", &cfg.Processing.PipelineTimeout),
		intSetting("processing.context_max_size", &cfg.Processing.ContextMaxSize),
		durationSetting("processing.context_cleanup_interval", &cfg.Processing.ContextCleanupInterval),
		stringSetting("security.jwt_secret", &cfg.Security.JWTSecret),
		durationSetting("security.token_expiry", &cfg.Security.TokenExpiry),
		durationSetting("security.refresh_token_expiry", &cfg.Security.RefreshTokenExpiry),
		intSetting("security.bcrypt_cost", &cfg.Security.BCryptCost),
		boolSetting("security.rate_limit_enabled", &cfg.Security.RateLimitEnabled),
		intSetting("security.requests_per_minute", &cfg.Security.RequestsPerMinute),
		boolSetting("security.rate_limit.enabled", &cfg.Security.RateLimitEnabled),
		intSetting("security.rate_limit.requests_per_minute", &cfg.Security.RequestsPerMinute),
	}
	for _, entry := range settings {
		if !reader.IsSet(entry.key) { continue }
		if err := entry.assign(reader.GetString(entry.key)); err != nil {
			return nil, fmt.Errorf("invalid %s: %w", entry.key, err)
		}
	}
	if reader.IsSet("integrations.neuralblitz.backend") {
		// The typed backend value is filled below after validating its string.
		backend := reader.GetString("integrations.neuralblitz.backend")
		switch backend {
		case "cpu", "cuda", "mps":
			cfg.NeuralBlitz.Backend = neuralblitz.BackendType(backend)
		default:
			return nil, fmt.Errorf("invalid integrations.neuralblitz.backend: %q", backend)
		}
	}
	return cfg, nil
}
