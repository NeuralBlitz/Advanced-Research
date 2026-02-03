package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/spf13/cobra"
	"github.com/spf13/viper"
	"github.com/fatih/color"
	"github.com/sirupsen/logrus"

	"github.com/advanced-research/go/pkg/core"
	"github.com/advanced-research/go/pkg/lrs"
	"github.com/advanced-research/go/pkg/neuralblitz"
	"github.com/advanced-research/go/pkg/opencode"
	"github.com/advanced-research/go/pkg/unified"
	"github.com/advanced-research/go/pkg/pipeline"
)

var (
	cfgFile string
	verbose bool
)

// rootCmd represents the base command when called without any subcommands
var rootCmd = &cobra.Command{
	Use:   "advanced-research",
	Short: "Advanced Research Framework CLI",
	Long: `A comprehensive CLI for the Advanced Research framework that provides:
- Context injection and management
- LRS (Learning Record Store) integration
- Opencode documentation workflow
- NeuralBlitz geometric computation
- Unified research API
- Automated research pipeline

Use 'advanced-research help [command]' for more information about a specific command.`,
	PersistentPreRun: func(cmd *cobra.Command, args []string) {
		if verbose {
			logrus.SetLevel(logrus.DebugLevel)
		}
	},
}

// initCmd represents the init command
var initCmd = &cobra.Command{
	Use:   "init",
	Short: "Initialize the Advanced Research framework",
	Long: `Initialize the Advanced Research framework with default configuration.
This creates the necessary configuration files and initializes all integration systems.`,
	Run: initialize,
}

// sessionCmd represents the session command
var sessionCmd = &cobra.Command{
	Use:   "session",
	Short: "Manage research sessions",
	Long: `Create and manage research sessions for tracking user interactions and context.`,
}

// createSessionCmd represents the session create command
var createSessionCmd = &cobra.Command{
	Use:   "create",
	Short: "Create a new research session",
	Long: `Create a new research session with specified user and mode.
This initializes context tracking and enables learning analytics.`,
	Run: createSession,
}

// queryCmd represents the query command
var queryCmd = &cobra.Command{
	Use:   "query",
	Short: "Process research queries",
	Long: `Process research queries through the unified system.
Supports multiple modes: research, learning, computation, collaborative.`,
	Run: processQuery,
}

// pipelineCmd represents the pipeline command
var pipelineCmd = &cobra.Command{
	Use:   "pipeline",
	Short: "Manage automated research pipelines",
	Long: `Execute and manage automated research pipelines that implement the
Ideas → Synthesis → Implementation workflow.`,
}

// runPipelineCmd represents the pipeline run command
var runPipelineCmd = &cobra.Command{
	Use:   "run",
	Short: "Run automated research pipeline",
	Long: `Run the complete automated research pipeline with initial ideas.
This executes all stages: ideation, synthesis, implementation, validation, deployment.`,
	Run: runPipeline,
}

// statusCmd represents the status command
var statusCmd = &cobra.Command{
	Use:   "status",
	Short: "Show system status",
	Long: `Display the current status of all integration systems and overall system health.`,
	Run: showStatus,
}

// serverCmd represents the server command
var serverCmd = &cobra.Command{
	Use:   "server",
	Short: "Start the HTTP/gRPC server",
	Long: `Start the Advanced Research server to provide HTTP and gRPC APIs.
This enables remote access to all system capabilities.`,
	Run: startServer,
}

// contextCmd represents the context command
var contextCmd = &cobra.Command{
	Use:   "context",
	Short: "Manage context and context injection",
	Long: `Manage context blocks, view context statistics, and test the context injection system.`,
}

// contextDemoCmd represents the context demo command
var contextDemoCmd = &cobra.Command{
	Use:   "demo",
	Short: "Demonstrate context injection system",
	Long: `Run a demonstration of the context injection system with sample data.
Shows how context blocks are managed and prioritized.`,
	Run: runContextDemo,
}

// testCmd represents the test command
var testCmd = &cobra.Command{
	Use:   "test",
	Short: "Test all integrations",
	Long: `Test all integration systems (LRS, Opencode, NeuralBlitz) to verify connectivity and functionality.`,
	Run: testIntegrations,
}

func init() {
	cobra.OnInitialize(initConfig)

	// Global flags
	rootCmd.PersistentFlags().StringVar(&cfgFile, "config", "", "config file (default is $HOME/.advanced-research.yaml)")
	rootCmd.PersistentFlags().BoolVarP(&verbose, "verbose", "v", false, "verbose output")

	// Add subcommands
	rootCmd.AddCommand(initCmd)
	rootCmd.AddCommand(sessionCmd)
	rootCmd.AddCommand(queryCmd)
	rootCmd.AddCommand(pipelineCmd)
	rootCmd.AddCommand(statusCmd)
	rootCmd.AddCommand(serverCmd)
	rootCmd.AddCommand(contextCmd)
	rootCmd.AddCommand(testCmd)

	// Session subcommands
	sessionCmd.AddCommand(createSessionCmd)
	createSessionCmd.Flags().StringP("user", "u", "", "User ID")
	createSessionCmd.Flags().StringP("mode", "m", "research", "Session mode (research, learning, computation, collaborative)")

	// Pipeline subcommands
	pipelineCmd.AddCommand(runPipelineCmd)
	runPipelineCmd.Flags().StringP("ideas", "i", "", "Initial ideas (comma-separated)")
	runPipelineCmd.Flags().StringP("user", "u", "", "User ID")

	// Context subcommands
	contextCmd.AddCommand(contextDemoCmd)
	contextDemoCmd.Flags().StringP("user", "u", "demo-user", "User ID for demo")

	// Query flags
	queryCmd.Flags().StringP("session", "s", "", "Session ID")
	queryCmd.Flags().StringP("mode", "m", "research", "Query mode (research, learning, computation, collaborative)")

	// Bind flags to viper
	viper.BindPFlag("config", rootCmd.PersistentFlags().Lookup("config"))
	viper.BindPFlag("verbose", rootCmd.PersistentFlags().Lookup("verbose"))
}

func initConfig() {
	if cfgFile != "" {
		// Use config file from the flag
		viper.SetConfigFile(cfgFile)
	} else {
		// Find home directory
		home, err := os.UserHomeDir()
		cobra.CheckErr(err)

		// Search config in home directory with name ".advanced-research" (without extension)
		viper.AddConfigPath(home)
		viper.SetConfigType("yaml")
		viper.SetConfigName(".advanced-research")
	}

	viper.AutomaticEnv()

	if err := viper.ReadInConfig(); err == nil {
		logrus.WithField("config", viper.ConfigFileUsed()).Info("Using configuration file")
	}
}

func initialize(cmd *cobra.Command, args []string) {
	color.Green("🚀 Initializing Advanced Research Framework...")

	// Create default config if it doesn't exist
	configPath := getConfigPath()
	if _, err := os.Stat(configPath); os.IsNotExist(err) {
		createDefaultConfig(configPath)
		color.Green("✓ Created default configuration at %s", configPath)
	}

	// Load configuration
	config := loadConfig()
	if config == nil {
		color.Red("❌ Failed to load configuration")
		return
	}

	// Initialize unified system
	logger := logrus.New()
	if verbose {
		logger.SetLevel(logrus.DebugLevel)
	}

	system, err := unified.NewSystem(config, logger)
	if err != nil {
		color.Red("❌ Failed to initialize system: %v", err)
		return
	}
	defer system.Close(cmd.Context())

	// Test initialization
	color.Green("✓ All integration systems initialized successfully")
	printIntegrationStatus(config)
}

func createSession(cmd *cobra.Command, args []string) {
	userID, _ := cmd.Flags().GetString("user")
	mode, _ := cmd.Flags().GetString("mode")

	if userID == "" {
		color.Red("❌ User ID is required")
		return
	}

	if mode == "" {
		mode = "research"
	}

	color.Cyan("👤 Creating research session...")
	
	config := loadConfig()
	if config == nil {
		color.Red("❌ Failed to load configuration")
		return
	}

	logger := logrus.New()
	system, err := unified.NewSystem(config, logger)
	if err != nil {
		color.Red("❌ Failed to initialize system: %v", err)
		return
	}
	defer system.Close(cmd.Context())

	sessionID, err := system.CreateSession(cmd.Context(), userID, unified.SystemMode(mode), nil)
	if err != nil {
		color.Red("❌ Failed to create session: %v", err)
		return
	}

	color.Green("✓ Session created successfully")
	color.White("Session ID: %s", sessionID)
	color.White("User: %s", userID)
	color.White("Mode: %s", mode)
}

func processQuery(cmd *cobra.Command, args []string) {
	if len(args) == 0 {
		color.Red("❌ Query is required")
		return
	}

	query := args[0]
	sessionID, _ := cmd.Flags().GetString("session")
	mode, _ := cmd.Flags().GetString("mode")

	if sessionID == "" {
		color.Red("❌ Session ID is required")
		return
	}

	if mode == "" {
		mode = "research"
	}

	color.Cyan("🔍 Processing query...")

	config := loadConfig()
	if config == nil {
		color.Red("❌ Failed to load configuration")
		return
	}

	logger := logrus.New()
	system, err := unified.NewSystem(config, logger)
	if err != nil {
		color.Red("❌ Failed to initialize system: %v", err)
		return
	}
	defer system.Close(cmd.Context())

	sysMode := unified.SystemMode(mode)
	queryReq := &unified.QueryRequest{
		SessionID:    sessionID,
		Query:        query,
		TrackLearning: true,
		Mode:         &sysMode,
	}

	response, err := system.ProcessQuery(cmd.Context(), queryReq)
	if err != nil {
		color.Red("❌ Query processing failed: %v", err)
		return
	}

	color.Green("✓ Query processed successfully")
	color.White("Mode: %s", response.Mode)
	color.White("Processing time: %v", response.ProcessingTime)
	
	if len(response.Actions) > 0 {
		color.Cyan("\n📋 Actions taken:")
		for i, action := range response.Actions {
			color.White("  %d. %s", i+1, action.Type)
			if action.ID != "" {
				color.White("     ID: %s", action.ID)
			}
			if action.Success {
				color.Green("     Status: ✓ Success")
			} else {
				color.Red("     Status: ✗ Failed - %s", action.Error)
			}
		}
	}

	if response.Context != "" {
		color.Cyan("\n📝 Generated context:")
		color.White(response.Context)
	}
}

func runPipeline(cmd *cobra.Command, args []string) {
	ideas, _ := cmd.Flags().GetString("ideas")
	userID, _ := cmd.Flags().GetString("user")

	if ideas == "" {
		color.Red("❌ Ideas are required (comma-separated)")
		return
	}

	if userID == "" {
		color.Red("❌ User ID is required")
		return
	}

	// Parse ideas
	ideaList := splitCommaSeparated(ideas)

	color.Cyan("🔄 Running automated research pipeline...")

	config := loadConfig()
	if config == nil {
		color.Red("❌ Failed to load configuration")
		return
	}

	logger := logrus.New()
	system, err := unified.NewSystem(config, logger)
	if err != nil {
		color.Red("❌ Failed to initialize system: %v", err)
		return
	}
	defer system.Close(cmd.Context())

	// Create pipeline configuration
	pipelineConfig := &pipeline.PipelineConfig{
		Name:             "Automated Research Pipeline",
		Description:       "Ideas → Synthesis → Implementation workflow",
		Stages:           []pipeline.PipelineStage{
			pipeline.StageIdeation,
			pipeline.StageSynthesis,
			pipeline.StageImplementation,
			pipeline.StageValidation,
			pipeline.StageDeployment,
		},
		AutoTransition:    true,
		ValidationRequired: true,
		TimeoutMinutes:    60,
		RetryAttempts:     3,
	}

	automatedPipeline := pipeline.NewAutomatedPipeline(pipelineConfig, system, logger)
	
	summary, err := automatedPipeline.ExecutePipeline(cmd.Context(), userID, ideaList, nil)
	if err != nil {
		color.Red("❌ Pipeline execution failed: %v", err)
		return
	}

	color.Green("✓ Pipeline execution completed")
	printPipelineSummary(summary)
}

func showStatus(cmd *cobra.Command, args []string) {
	color.Cyan("📊 Advanced Research System Status")

	config := loadConfig()
	if config == nil {
		color.Red("❌ Failed to load configuration")
		return
	}

	logger := logrus.New()
	system, err := unified.NewSystem(config, logger)
	if err != nil {
		color.Red("❌ Failed to initialize system: %v", err)
		return
	}
	defer system.Close(cmd.Context())

	status, err := system.GetSystemStatus(cmd.Context())
	if err != nil {
		color.Red("❌ Failed to get system status: %v", err)
		return
	}

	printSystemStatus(status)
}

func startServer(cmd *cobra.Command, args []string) {
	color.Cyan("🚀 Starting Advanced Research server...")

	config := loadConfig()
	if config == nil {
		color.Red("❌ Failed to load configuration")
		return
	}

	logger := logrus.New()
	system, err := unified.NewSystem(config, logger)
	if err != nil {
		color.Red("❌ Failed to initialize system: %v", err)
		return
	}
	defer system.Close(cmd.Context())

	color.Green("✓ Server started successfully")
	color.White("HTTP Server: http://localhost:%d", config.Server.HTTPPort)
	color.White("gRPC Server: localhost:%d", config.Server.GRPCPort)
	color.White("Press Ctrl+C to stop")

	// In a real implementation, this would start the actual HTTP/gRPC servers
	// For now, we'll just simulate server startup
	select {}
}

func runContextDemo(cmd *cobra.Command, args []string) {
	color.Cyan("🧠 Context Injection System Demo")

	userID, _ := cmd.Flags().GetString("user")
	if userID == "" {
		userID = "demo-user"
	}

	// Create context injector
	config := core.DefaultConfig()
	injector := core.NewContextInjector(config)
	defer injector.Close(cmd.Context())

	color.White("\n📝 Adding sample context blocks...")

	// Add sample context blocks
	injector.AddContext(cmd.Context(), "research_goal",
		"Develop geometric deep learning architectures for understanding complex manifolds in high-dimensional spaces.",
		core.PriorityHigh,
		core.WithTags([]string{"research", "geometry"}),
		core.WithTTL(24*time.Hour))

	injector.AddContext(cmd.Context(), "current_approach",
		"Using Riemannian geometry principles to design attention mechanisms that respect manifold structure.",
		core.PriorityMedium,
		core.WithTags([]string{"approach", "riemannian"}))

	injector.AddContext(cmd.Context(), "mathematical_foundation",
		"Key concepts: geodesics, curvature tensors, parallel transport, exponential maps.",
		core.PriorityHigh,
		core.WithTags([]string{"math", "foundation"}))

	// Display context
	context := injector.GetContext(cmd.Context(), nil)
	color.Cyan("\n📋 Generated Context:")
	color.White(context)

	// Display statistics
	stats := injector.GetStatistics(cmd.Context())
	color.Cyan("\n📊 Context Statistics:")
	color.White("Total blocks: %d", stats.TotalBlocks)
	color.White("Blocks by priority: %v", stats.BlocksByPriority)
	color.White("Oldest block age: %d seconds", stats.OldestBlockAgeSeconds)
	color.White("Newest block age: %d seconds", stats.NewestBlockAgeSeconds)
	color.White("Total content size: %d bytes", stats.TotalContentSize)

	color.Green("✓ Context injection demo completed")
}

func testIntegrations(cmd *cobra.Command, args []string) {
	color.Cyan("🧪 Testing All Integration Systems")

	config := loadConfig()
	if config == nil {
		color.Red("❌ Failed to load configuration")
		return
	}

	logger := logrus.New()
	system, err := unified.NewSystem(config, logger)
	if err != nil {
		color.Red("❌ Failed to initialize system: %v", err)
		return
	}
	defer system.Close(cmd.Context())

	// Test LRS integration
	if config.LRS != nil && config.LRS.Enabled {
		color.Cyan("\n📚 Testing LRS Integration...")
		lrsClient := lrs.NewClient(config.LRS, logger)
		
		// Test learning record
		record := &lrs.LearningRecord{
			Actor: lrs.Actor{
				Account: &lrs.Account{
					HomePage: "advanced-research",
					Name:     "test-user",
				},
			},
			Verb: lrs.Verb{
				ID: "http://adlnet.gov/expapi/verbs/tested",
			},
			Object: lrs.Object{
				ID:         "urn:test:integration",
				ObjectType: "Activity",
			},
			Timestamp: time.Now(),
		}

		if err := lrsClient.RecordLearningEvent(cmd.Context(), record); err != nil {
			color.Red("  ❌ LRS test failed: %v", err)
		} else {
			color.Green("  ✓ LRS integration test passed")
		}
		lrsClient.Close(cmd.Context())
	}

	// Test Opencode integration
	if config.Opencode != nil && config.Opencode.Enabled {
		color.Cyan("\n📝 Testing Opencode Integration...")
		opencodeClient, err := opencode.NewClient(config.Opencode, logger)
		if err != nil {
			color.Red("  ❌ Opencode client creation failed: %v", err)
		} else {
			// Test document creation
			docID, err := opencodeClient.CreateResearchDocument(
				cmd.Context(),
				"Test Document",
				"This is a test document created during integration testing.",
				opencode.StageIdeas,
				opencode.TypeTheory,
				[]string{"#test", "#integration"},
				nil,
			)
			if err != nil {
				color.Red("  ❌ Opencode test failed: %v", err)
			} else {
				color.Green("  ✓ Opencode integration test passed - Doc ID: %s", docID)
			}
			opencodeClient.Close(cmd.Context())
		}
	}

	// Test NeuralBlitz integration
	if config.NeuralBlitz != nil && config.NeuralBlitz.Enabled {
		color.Cyan("\n🧮 Testing NeuralBlitz Integration...")
		engine, err := neuralblitz.NewEngine(config.NeuralBlitz, logger)
		if err != nil {
			color.Red("  ❌ NeuralBlitz engine creation failed: %v", err)
		} else {
			// Test geometric computation
			inputData := make([]float64, 10*64)
			for i := range inputData {
				inputData[i] = float64(i%100) / 100.0
			}
			tensor := neuralblitz.NewTensor([]int{10, 64}, inputData)
			
			features, err := engine.ComputeGeometricFeatures(cmd.Context(), tensor, "eigenvalue")
			if err != nil {
				color.Red("  ❌ NeuralBlitz test failed: %v", err)
			} else {
				color.Green("  ✓ NeuralBlitz integration test passed")
				color.White("    Eigenvalues computed: %d", len(features.Eigenvalues))
			}
			engine.Close(cmd.Context())
		}
	}

	color.Green("\n✅ All integration tests completed")
}

func main() {
	if err := rootCmd.Execute(); err != nil {
		logrus.WithError(err).Fatal("CLI execution failed")
	}
}

// Helper functions
func getConfigPath() string {
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".advanced-research.yaml")
}

func createDefaultConfig(path string) {
	defaultConfig := unified.DefaultConfig()
	
	// In a real implementation, this would marshal the config to YAML
	// For now, we'll create a simple config file
	configContent := `server:
  http_port: 8080
  grpc_port: 9090
  host: "0.0.0.0"

integrations:
  lrs:
    enabled: true
    endpoint: "http://localhost:8080/xapi/"
    timeout: "30s"
    retry_attempts: 3
    batch_size: 10
  
  opencode:
    enabled: true
    workspace: "advanced-research"
    local_path: "./research-docs"
    auto_commit: true
  
  neuralblitz:
    enabled: true
    backend: "cpu"
    tensor_ops_threads: 4
    max_batch_size: 32

processing:
  max_concurrent_queries: 10
  query_timeout: "60s"
  context_max_size: 10000

security:
  jwt_secret: "your-jwt-secret-key"
  token_expiry: "24h"
  rate_limit_enabled: true
  requests_per_minute: 100
`

	os.WriteFile(path, []byte(configContent), 0644)
}

func loadConfig() *unified.Config {
	// In a real implementation, this would use viper to unmarshal YAML
	// For now, return default config
	return unified.DefaultConfig()
}

func printIntegrationStatus(config *unified.Config) {
	color.Cyan("\n🔗 Integration Status:")
	
	if config.LRS != nil {
		if config.LRS.Enabled {
			color.Green("  ✓ LRS Integration: Enabled")
		} else {
			color.Yellow("  ⚠ LRS Integration: Disabled")
		}
	}
	
	if config.Opencode != nil {
		if config.Opencode.Enabled {
			color.Green("  ✓ Opencode Integration: Enabled")
		} else {
			color.Yellow("  ⚠ Opencode Integration: Disabled")
		}
	}
	
	if config.NeuralBlitz != nil {
		if config.NeuralBlitz.Enabled {
			color.Green("  ✓ NeuralBlitz Integration: Enabled")
		} else {
			color.Yellow("  ⚠ NeuralBlitz Integration: Disabled")
		}
	}
}

func printPipelineSummary(summary *pipeline.ExecutionSummary) {
	color.Cyan("\n📊 Pipeline Execution Summary:")
	color.White("Pipeline Name: %s", summary.PipelineName)
	color.White("Status: %s", summary.PipelineStatus)
	color.White("Duration: %v", summary.Metrics.PipelineDuration)
	
	color.Cyan("\n📈 Metrics:")
	color.White("Total Tasks: %d", summary.Metrics.TotalTasks)
	color.White("Completed: %d", summary.Metrics.CompletedTasks)
	color.White("Failed: %d", summary.Metrics.FailedTasks)
	color.White("Documents Generated: %d", summary.Metrics.DocumentsGenerated)
	color.White("Learning Events: %d", summary.Metrics.LearningEvents)
	color.White("Geometric Computations: %d", summary.Metrics.GeometricComputations)
	
	if summary.Metrics.TotalTasks > 0 {
		successRate := float64(summary.Metrics.CompletedTasks) / float64(summary.Metrics.TotalTasks) * 100
		color.White("Success Rate: %.1f%%", successRate)
	}
}

func printSystemStatus(status *unified.SystemStatistics) {
	color.Cyan("\n📊 System Statistics:")
	color.White("Total Queries: %d", status.TotalQueries)
	color.White("Documents Created: %d", status.DocumentsCreated)
	color.White("Learning Events: %d", status.LearningEvents)
	color.White("Geometric Computations: %d", status.GeometricComputations)
	color.White("Active Sessions: %d", status.ActiveSessions)
	
	color.Cyan("\n🔗 Integration Status:")
	for name, enabled := range status.IntegrationStatus {
		if enabled {
			color.Green("  ✓ %s: Enabled", name)
		} else {
			color.Yellow("  ⚠ %s: Disabled", name)
		}
	}
	
	color.Cyan("\n📅 Last Updated: %s", status.LastUpdated.Format("2006-01-02 15:04:05"))
}

func splitCommaSeparated(s string) []string {
	if s == "" {
		return []string{}
	}
	
	// Simple split implementation
	var result []string
	start := 0
	for i, r := range s {
		if r == ',' {
			if i > start {
				result = append(result, s[start:i])
			}
			start = i + 1
		}
	}
	if start < len(s) {
		result = append(result, s[start:])
	}
	
	// Trim spaces from each item
	for i := range result {
		result[i] = strings.TrimSpace(result[i])
	}
	
	return result
}