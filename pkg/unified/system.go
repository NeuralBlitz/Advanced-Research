package unified

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/sirupsen/logrus"

	"github.com/advanced-research/go/pkg/core"
	"github.com/advanced-research/go/pkg/lrs"
	"github.com/advanced-research/go/pkg/neuralblitz"
	"github.com/advanced-research/go/pkg/opencode"
)

// SystemMode represents the operational mode of the unified system
type SystemMode string

const (
	ModeResearch      SystemMode = "research"
	ModeLearning      SystemMode = "learning"
	ModeComputation   SystemMode = "computation"
	ModeCollaborative SystemMode = "collaborative"
)

// UserContext represents the context for a user session
type UserContext struct {
	UserID        string                 `json:"user_id"`
	SessionID     string                 `json:"session_id"`
	Mode          SystemMode             `json:"mode"`
	CurrentStage  *opencode.ResearchStage `json:"current_stage,omitempty"`
	Preferences   map[string]interface{} `json:"preferences"`
	CreatedAt     time.Time              `json:"created_at"`
	LastActivity  time.Time              `json:"last_activity"`
}

// QueryRequest represents a query request to the unified system
type QueryRequest struct {
	SessionID    string                 `json:"session_id"`
	Query        string                 `json:"query"`
	ContextData  map[string]interface{} `json:"context_data,omitempty"`
	Mode         *SystemMode            `json:"mode,omitempty"`
	TrackLearning bool                  `json:"track_learning"`
}

// QueryResponse represents the response from a query
type QueryResponse struct {
	Mode              SystemMode             `json:"mode"`
	Query             string                 `json:"query"`
	Actions           []QueryAction          `json:"actions"`
	Context           string                 `json:"context,omitempty"`
	ProcessingTime     time.Duration          `json:"processing_time"`
	Metadata          map[string]interface{} `json:"metadata"`
}

// QueryAction represents an action taken during query processing
type QueryAction struct {
	Type        string                 `json:"type"`
	ID          string                 `json:"id,omitempty"`
	Data        map[string]interface{} `json:"data,omitempty"`
	Timestamp   time.Time              `json:"timestamp"`
	Success     bool                   `json:"success"`
	Error       string                 `json:"error,omitempty"`
}

// SystemStatistics represents system-wide statistics
type SystemStatistics struct {
	TotalQueries            int64                  `json:"total_queries"`
	DocumentsCreated        int64                  `json:"documents_created"`
	LearningEvents         int64                  `json:"learning_events"`
	GeometricComputations  int64                  `json:"geometric_computations"`
	ActiveSessions         int                     `json:"active_sessions"`
	IntegrationStatus     map[string]bool         `json:"integration_status"`
	WorkflowReport        *opencode.WorkflowReport `json:"workflow_report,omitempty"`
	LastUpdated           time.Time               `json:"last_updated"`
}

// Config represents the unified system configuration
type Config struct {
	Server      core.Config         `yaml:"server" json:"server"`
	LRS         *lrs.Config         `yaml:"lrs" json:"lrs"`
	Opencode    *opencode.Config    `yaml:"opencode" json:"opencode"`
	NeuralBlitz *neuralblitz.Config `yaml:"neuralblitz" json:"neuralblitz"`
	Processing  ProcessingConfig     `yaml:"processing" json:"processing"`
	Security    SecurityConfig      `yaml:"security" json:"security"`
}

// ProcessingConfig represents processing-related configuration
type ProcessingConfig struct {
	MaxConcurrentQueries  int           `yaml:"max_concurrent_queries" json:"max_concurrent_queries"`
	QueryTimeout         time.Duration `yaml:"query_timeout" json:"query_timeout"`
	PipelineTimeout      time.Duration `yaml:"pipeline_timeout" json:"pipeline_timeout"`
	ContextMaxSize      int           `yaml:"context_max_size" json:"context_max_size"`
	ContextCleanupInterval time.Duration `yaml:"context_cleanup_interval" json:"context_cleanup_interval"`
}

// SecurityConfig represents security configuration
type SecurityConfig struct {
	JWTSecret           string        `yaml:"jwt_secret" json:"jwt_secret"`
	TokenExpiry         time.Duration `yaml:"token_expiry" json:"token_expiry"`
	RefreshTokenExpiry  time.Duration `yaml:"refresh_token_expiry" json:"refresh_token_expiry"`
	BCryptCost          int           `yaml:"bcrypt_cost" json:"bcrypt_cost"`
	RateLimitEnabled    bool          `yaml:"rate_limit_enabled" json:"rate_limit_enabled"`
	RequestsPerMinute   int           `yaml:"requests_per_minute" json:"requests_per_minute"`
}

// DefaultConfig returns the default configuration
func DefaultConfig() *Config {
	return &Config{
		Server:      *core.DefaultConfig(),
		LRS:         lrs.DefaultConfig(),
		Opencode:    opencode.DefaultConfig(),
		NeuralBlitz: neuralblitz.DefaultConfig(),
		Processing: ProcessingConfig{
			MaxConcurrentQueries:  10,
			QueryTimeout:         60 * time.Second,
			PipelineTimeout:      30 * time.Minute,
			ContextMaxSize:      10000,
			ContextCleanupInterval: time.Hour,
		},
		Security: SecurityConfig{
			JWTSecret:           "your-jwt-secret-key",
			TokenExpiry:         24 * time.Hour,
			RefreshTokenExpiry:  168 * time.Hour, // 7 days
			BCryptCost:          12,
			RateLimitEnabled:    true,
			RequestsPerMinute:   100,
		},
	}
}

// System represents the unified research system
type System struct {
	config          *Config
	logger          *logrus.Logger
	contextInjector  *core.ContextInjector
	lrsClient       *lrs.Client
	opencodeClient  *opencode.Client
	neuralBlitz     *neuralblitz.Engine
	activeSessions  map[string]*UserContext
	stats           SystemStatistics
	mutex           sync.RWMutex
}

// NewSystem creates a new unified research system
func NewSystem(config *Config, logger *logrus.Logger) (*System, error) {
	if config == nil {
		config = DefaultConfig()
	}

	if logger == nil {
		logger = logrus.New()
	}

	system := &System{
		config:         config,
		logger:         logger,
		activeSessions: make(map[string]*UserContext),
		stats: SystemStatistics{
			IntegrationStatus: make(map[string]bool),
			LastUpdated:    time.Now(),
		},
	}

	// Initialize core components
	var err error

	// Initialize context injector
	system.contextInjector = core.NewContextInjector(&config.Processing)

	// Initialize LRS client
	system.lrsClient, err = lrs.NewClient(config.LRS, logger)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize LRS client: %w", err)
	}

	// Initialize Opencode client
	system.opencodeClient, err = opencode.NewClient(config.Opencode, logger)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize Opencode client: %w", err)
	}

	// Initialize NeuralBlitz engine
	system.neuralBlitz, err = neuralblitz.NewEngine(config.NeuralBlitz, logger)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize NeuralBlitz engine: %w", err)
	}

	// Update integration status
	system.stats.IntegrationStatus["lrs"] = config.LRS.Enabled
	system.stats.IntegrationStatus["opencode"] = config.Opencode.Enabled
	system.stats.IntegrationStatus["neuralblitz"] = config.NeuralBlitz.Enabled

	logger.Info("Unified research system initialized successfully")
	return system, nil
}

// CreateSession creates a new user session
func (s *System) CreateSession(ctx context.Context, userID string, mode SystemMode, preferences map[string]interface{}) (string, error) {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	sessionID := fmt.Sprintf("session_%d_%s", time.Now().Unix(), userID)
	
	userContext := &UserContext{
		UserID:       userID,
		SessionID:    sessionID,
		Mode:         mode,
		Preferences:   preferences,
		CreatedAt:     time.Now(),
		LastActivity:  time.Now(),
	}

	s.activeSessions[sessionID] = userContext
	s.stats.ActiveSessions = len(s.activeSessions)

	s.logger.WithFields(logrus.Fields{
		"session_id": sessionID,
		"user_id":    userID,
		"mode":       mode,
	}).Info("Created user session")

	// Record session creation in LRS
	if s.lrsClient != nil && s.lrsClient.Enabled() {
		actor := lrs.Actor{
			Account: &lrs.Account{
				HomePage: "advanced-research",
				Name:     userID,
			},
			ObjectType: "Agent",
		}

		sessionObject := lrs.Object{
			ID:         fmt.Sprintf("urn:session:%s", sessionID),
			ObjectType: "Activity",
			Name: &map[string]string{
				"en-US": fmt.Sprintf("Session - %s", mode),
			},
		}

		record := &lrs.LearningRecord{
			Actor:     actor,
			Verb:      lrs.Verb{ID: "http://adlnet.gov/expapi/verbs/initialized", Display: map[string]string{"en-US": "initialized"}},
			Object:    sessionObject,
			Timestamp: time.Now(),
		}

		if err := s.lrsClient.RecordLearningEvent(ctx, record); err != nil {
			s.logger.WithError(err).Warn("Failed to record session creation in LRS")
		}
	}

	return sessionID, nil
}

// ProcessQuery processes a user query through the unified system
func (s *System) ProcessQuery(ctx context.Context, req *QueryRequest) (*QueryResponse, error) {
	startTime := time.Now()

	// Validate session
	s.mutex.RLock()
	userContext, exists := s.activeSessions[req.SessionID]
	if !exists {
		s.mutex.RUnlock()
		return nil, fmt.Errorf("invalid session ID: %s", req.SessionID)
	}

	// Use request mode or session mode
	mode := req.Mode
	if mode == nil {
		mode = &userContext.Mode
	}
	s.mutex.RUnlock()

	// Update last activity
	s.mutex.Lock()
	userContext.LastActivity = time.Now()
	s.mutex.Unlock()

	// Add query to context
	if s.contextInjector != nil {
		priority := core.PriorityMedium
		if isUrgentQuery(req.Query) {
			priority = core.PriorityHigh
		}

		err := s.contextInjector.AddContext(ctx,
			fmt.Sprintf("query_%d", s.stats.TotalQueries+1),
			req.Query,
			priority,
			core.WithTags([]string{"query", string(*mode)}),
		)
		if err != nil {
			s.logger.WithError(err).Warn("Failed to add query to context")
		}
	}

	s.stats.TotalQueries++

	// Process query based on mode
	var response *QueryResponse
	var err error

	switch *mode {
	case ModeResearch:
		response, err = s.processResearchQuery(ctx, req, userContext)
	case ModeLearning:
		response, err = s.processLearningQuery(ctx, req, userContext)
	case ModeComputation:
		response, err = s.processComputationQuery(ctx, req, userContext)
	case ModeCollaborative:
		response, err = s.processCollaborativeQuery(ctx, req, userContext)
	default:
		return nil, fmt.Errorf("unsupported mode: %s", *mode)
	}

	if err != nil {
		s.logger.WithError(err).WithField("query", req.Query).Error("Query processing failed")
		return nil, err
	}

	response.ProcessingTime = time.Since(startTime)
	response.Metadata = map[string]interface{}{
		"session_id": req.SessionID,
		"timestamp":  time.Now().Format(time.RFC3339),
	}

	return response, nil
}

// processResearchQuery handles research-specific queries
func (s *System) processResearchQuery(ctx context.Context, req *QueryRequest, userContext *UserContext) (*QueryResponse, error) {
	response := &QueryResponse{
		Mode:    ModeResearch,
		Query:    req.Query,
		Actions:  make([]QueryAction, 0),
		Metadata: make(map[string]interface{}),
	}

	// Check if query is about creating documents
	if isCreateDocumentQuery(req.Query) {
		if s.opencodeClient != nil {
			stage := extractResearchStage(req.Query)
			docType := extractDocumentType(req.Query)

			docID, err := s.opencodeClient.CreateResearchDocument(
				ctx,
				extractTitleFromQuery(req.Query),
				req.Query,
				stage,
				docType,
				[]string{"auto-generated", "query-driven"},
				map[string]interface{}{
					"query_id": s.stats.TotalQueries,
					"mode":     "research",
				},
			)
			if err != nil {
				response.Actions = append(response.Actions, QueryAction{
					Type:      "document_creation_failed",
					Timestamp: time.Now(),
					Success:   false,
					Error:     err.Error(),
				})
			} else {
				response.Actions = append(response.Actions, QueryAction{
					Type:  "document_created",
					ID:    docID,
					Data: map[string]interface{}{
						"stage":    stage,
						"doc_type": docType,
					},
					Timestamp: time.Now(),
					Success:   true,
				})
				s.stats.DocumentsCreated++
			}
		}
	}

	// Search for relevant documents if requested
	if isSearchQuery(req.Query) {
		if s.opencodeClient != nil {
			searchTerm := extractSearchTerm(req.Query)
			docs, err := s.opencodeClient.SearchDocuments(ctx, searchTerm, nil, nil, nil)
			if err != nil {
				s.logger.WithError(err).Warn("Document search failed")
			} else {
				docData := make([]map[string]interface{}, len(docs))
				for i, doc := range docs {
					docData[i] = map[string]interface{}{
						"id":    doc.ID,
						"title": doc.Title,
						"stage": doc.Stage,
					}
				}
				response.Actions = append(response.Actions, QueryAction{
					Type:  "documents_found",
					Data:  map[string]interface{}{"documents": docData},
					Timestamp: time.Now(),
					Success: true,
				})
			}
		}
	}

	// Include context if available
	if s.contextInjector != nil {
		context := s.contextInjector.GetContext(ctx, nil)
		response.Context = context
	}

	return response, nil
}

// processLearningQuery handles learning-specific queries
func (s *System) processLearningQuery(ctx context.Context, req *QueryRequest, userContext *UserContext) (*QueryResponse, error) {
	response := &QueryResponse{
		Mode:    ModeLearning,
		Query:    req.Query,
		Actions:  make([]QueryAction, 0),
		Metadata: make(map[string]interface{}),
	}

	// Record learning interaction if enabled
	if req.TrackLearning && s.lrsClient != nil {
		actor := lrs.Actor{
			Account: &lrs.Account{
				HomePage: "advanced-research",
				Name:     userContext.UserID,
			},
			ObjectType: "Agent",
		}

		object := lrs.Object{
			ID:         fmt.Sprintf("urn:query:%d", s.stats.TotalQueries),
			ObjectType: "Activity",
			Name: &map[string]string{
				"en-US": fmt.Sprintf("Learning Query - %s", req.Query[:50]),
			},
		}

		record := &lrs.LearningRecord{
			Actor:     actor,
			Verb:      lrs.Verb{ID: "http://adlnet.gov/expapi/verbs/asked", Display: map[string]string{"en-US": "asked"}},
			Object:    object,
			Timestamp: time.Now(),
		}

		if err := s.lrsClient.RecordLearningEvent(ctx, record); err != nil {
			s.logger.WithError(err).Warn("Failed to record learning event")
		} else {
			s.stats.LearningEvents++
			response.Actions = append(response.Actions, QueryAction{
				Type:  "learning_recorded",
				Timestamp: time.Now(),
				Success: true,
			})
		}
	}

	// Provide learning context
	if s.contextInjector != nil {
		context := s.contextInjector.GetContext(ctx, nil)
		response.Context = context
	}

	return response, nil
}

// processComputationQuery handles computation-specific queries
func (s *System) processComputationQuery(ctx context.Context, req *QueryRequest, userContext *UserContext) (*QueryResponse, error) {
	response := &QueryResponse{
		Mode:    ModeComputation,
		Query:    req.Query,
		Actions:  make([]QueryAction, 0),
		Metadata: make(map[string]interface{}),
	}

	// Handle geometric computation requests
	if isGeometricComputationQuery(req.Query) && s.neuralBlitz != nil {
		// Create mock input tensor
		inputData := make([]float64, 10*64) // 10 samples, 64 dimensions
		for i := range inputData {
			inputData[i] = float64(i%100) / 100.0 // Simple mock data
		}
		inputTensor := neuralblitz.NewTensor([]int{10, 64}, inputData)

		features, err := s.neuralBlitz.ComputeGeometricFeatures(ctx, inputTensor, "eigenvalue")
		if err != nil {
			response.Actions = append(response.Actions, QueryAction{
				Type:      "geometric_computation_failed",
				Timestamp: time.Now(),
				Success:   false,
				Error:     err.Error(),
			})
		} else {
			response.Actions = append(response.Actions, QueryAction{
				Type: "geometric_computed",
				Data: map[string]interface{}{
					"features": features,
				},
				Timestamp: time.Now(),
				Success:   true,
			})
			s.stats.GeometricComputations++
		}
	}

	return response, nil
}

// processCollaborativeQuery handles collaborative queries that span multiple modes
func (s *System) processCollaborativeQuery(ctx context.Context, req *QueryRequest, userContext *UserContext) (*QueryResponse, error) {
	response := &QueryResponse{
		Mode:    ModeCollaborative,
		Query:    req.Query,
		Actions:  make([]QueryAction, 0),
		Metadata: make(map[string]interface{}),
	}

	// Process across all systems
	researchResp, _ := s.processResearchQuery(ctx, req, userContext)
	learningResp, _ := s.processLearningQuery(ctx, req, userContext)
	computationResp, _ := s.processComputationQuery(ctx, req, userContext)

	// Combine actions
	response.Actions = append(response.Actions, researchResp.Actions...)
	response.Actions = append(response.Actions, learningResp.Actions...)
	response.Actions = append(response.Actions, computationResp.Actions...)

	response.Metadata = map[string]interface{}{
		"research_actions":      researchResp.Actions,
		"learning_actions":      learningResp.Actions,
		"computation_actions":   computationResp.Actions,
		"combined_mode":        true,
	}

	return response, nil
}

// GetSystemStatus returns the current system status
func (s *System) GetSystemStatus(ctx context.Context) (*SystemStatistics, error) {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	stats := s.stats
	stats.ActiveSessions = len(s.activeSessions)
	stats.LastUpdated = time.Now()

	// Get workflow report from opencode
	if s.opencodeClient != nil {
		workflowReport := s.opencodeClient.GetProjectStructure(ctx)
		// Convert to appropriate format for response
		stats.WorkflowReport = &opencode.WorkflowReport{
			TotalDocuments:     len(workflowReport.ResearchAreas),
			TotalAreas:        len(workflowReport.ResearchAreas),
			TotalWorkflows:    len(workflowReport.Workflows),
			TotalSemanticTags: len(workflowReport.SemanticTags),
		}
	}

	return &stats, nil
}

// Close shuts down the unified system
func (s *System) Close(ctx context.Context) error {
	s.logger.Info("Shutting down unified research system")

	var errors []error

	// Close components
	if s.lrsClient != nil {
		if err := s.lrsClient.Close(ctx); err != nil {
			errors = append(errors, fmt.Errorf("LRS client close error: %w", err))
		}
	}

	if s.opencodeClient != nil {
		if err := s.opencodeClient.Close(ctx); err != nil {
			errors = append(errors, fmt.Errorf("Opencode client close error: %w", err))
		}
	}

	if s.neuralBlitz != nil {
		if err := s.neuralBlitz.Close(ctx); err != nil {
			errors = append(errors, fmt.Errorf("NeuralBlitz engine close error: %w", err))
		}
	}

	if s.contextInjector != nil {
		if err := s.contextInjector.Close(); err != nil {
			errors = append(errors, fmt.Errorf("Context injector close error: %w", err))
		}
	}

	if len(errors) > 0 {
		return fmt.Errorf("multiple errors occurred during shutdown: %v", errors)
	}

	s.logger.Info("Unified research system shutdown complete")
	return nil
}

// Helper functions
func isUrgentQuery(query string) bool {
	urgentWords := []string{"urgent", "critical", "emergency", "asap"}
	queryLower := strings.ToLower(query)
	for _, word := range urgentWords {
		if strings.Contains(queryLower, word) {
			return true
		}
	}
	return false
}

func isCreateDocumentQuery(query string) bool {
	return strings.Contains(strings.ToLower(query), "create") && 
		(strings.Contains(strings.ToLower(query), "document") || strings.Contains(strings.ToLower(query), "doc"))
}

func isSearchQuery(query string) bool {
	return strings.Contains(strings.ToLower(query), "search") || strings.Contains(strings.ToLower(query), "find")
}

func isGeometricComputationQuery(query string) bool {
	return strings.Contains(strings.ToLower(query), "geometric") && 
		(strings.Contains(strings.ToLower(query), "compute") || strings.Contains(strings.ToLower(query), "features"))
}

func extractResearchStage(query string) opencode.ResearchStage {
	queryLower := strings.ToLower(query)
	if strings.Contains(queryLower, "idea") || strings.Contains(queryLower, "theory") {
		return opencode.StageIdeas
	} else if strings.Contains(queryLower, "synthesis") || strings.Contains(queryLower, "framework") {
		return opencode.StageSynthesis
	} else if strings.Contains(queryLower, "implementation") || strings.Contains(queryLower, "experiment") {
		return opencode.StageImplementation
	}
	return opencode.StageIdeas
}

func extractDocumentType(query string) opencode.DocumentType {
	queryLower := strings.ToLower(query)
	if strings.Contains(queryLower, "theory") {
		return opencode.TypeTheory
	} else if strings.Contains(queryLower, "framework") {
		return opencode.TypeFramework
	} else if strings.Contains(queryLower, "experiment") {
		return opencode.TypeExperiment
	} else if strings.Contains(queryLower, "code") {
		return opencode.TypeCode
	}
	return opencode.TypeTheory
}

func extractTitleFromQuery(query string) string {
	// Simple extraction - in practice would use NLP
	if len(query) > 50 {
		return query[:47] + "..."
	}
	return query
}

func extractSearchTerm(query string) string {
	// Simple extraction - in practice would use NLP
	words := strings.Fields(query)
	for i, word := range words {
		if strings.ToLower(word) == "search" && i+1 < len(words) {
			return strings.Join(words[i+1:], " ")
		} else if strings.ToLower(word) == "find" && i+1 < len(words) {
			return strings.Join(words[i+1:], " ")
		}
	}
	return query
}