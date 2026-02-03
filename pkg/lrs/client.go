package lrs

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/sirupsen/logrus"
)

// LearningRecord represents an xAPI statement for learning analytics
type LearningRecord struct {
	ID        string                 `json:"id,omitempty"`
	Actor     Actor                 `json:"actor"`
	Verb      Verb                  `json:"verb"`
	Object    Object                `json:"object"`
	Result    *Result               `json:"result,omitempty"`
	Timestamp time.Time             `json:"timestamp"`
	Stored    time.Time             `json:"stored"`
	Authority *Authority            `json:"authority,omitempty"`
	Version   string                `json:"version,omitempty"`
	Attachments []Attachment        `json:"attachments,omitempty"`
}

// Actor represents the entity performing the action
type Actor struct {
	Account  *Account      `json:"account,omitempty"`
	Mbox     *string       `json:"mbox,omitempty"`
	MboxSha1 *string       `json:"mbox_sha1,omitempty"`
	OpenID   *string       `json:"openid,omitempty"`
	ObjectType string      `json:"objectType,omitempty"`
	Name     *string       `json:"name,omitempty"`
}

// Account represents an account-based actor
type Account struct {
	HomePage string `json:"homePage"`
	Name     string `json:"name"`
}

// Verb represents the action being performed
type Verb struct {
	ID         string            `json:"id"`
	Display    map[string]string `json:"display,omitempty"`
}

// Object represents the object of the learning activity
type Object struct {
	ID           string            `json:"id"`
	ObjectType   string            `json:"objectType"`
	Name         *map[string]string `json:"name,omitempty"`
	Description  *map[string]string `json:"description,omitempty"`
	Definition   *ObjectDefinition `json:"definition,omitempty"`
}

// ObjectDefinition provides additional object details
type ObjectDefinition struct {
	Name        *map[string]string `json:"name,omitempty"`
	Description *map[string]string `json:"description,omitempty"`
	Type        string            `json:"type,omitempty"`
	InteractionType string         `json:"interactionType,omitempty"`
	Extensions  map[string]interface{} `json:"extensions,omitempty"`
}

// Result represents the outcome of the activity
type Result struct {
	Completion *bool              `json:"completion,omitempty"`
	Success    *bool              `json:"success,omitempty"`
	Response   *string            `json:"response,omitempty"`
	Duration   *string            `json:"duration,omitempty"`
	Score      *Score             `json:"score,omitempty"`
	Extensions map[string]interface{} `json:"extensions,omitempty"`
}

// Score represents a numeric result
type Score struct {
	Scaled    *float64 `json:"scaled,omitempty"`
	Raw       *float64 `json:"raw,omitempty"`
	Min       *float64 `json:"min,omitempty"`
	Max       *float64 `json:"max,omitempty"`
}

// Authority represents the authority that stored the statement
type Authority struct {
	ObjectType string      `json:"objectType"`
	Name      *string     `json:"name,omitempty"`
	Mbox      *string     `json:"mbox,omitempty"`
	Account   *Account    `json:"account,omitempty"`
}

// Attachment represents attached resources
type Attachment struct {
	UsageType    *string            `json:"usageType,omitempty"`
	Display      *map[string]string  `json:"display,omitempty"`
	Description  *map[string]string  `json:"description,omitempty"`
	ContentType  *string            `json:"contentType,omitempty"`
	Length       *int               `json:"length,omitempty"`
	SHA2         *string            `json:"sha2,omitempty"`
	URL          *url.URL           `json:"url,omitempty"`
}

// Config represents the LRS client configuration
type Config struct {
	Endpoint     string        `yaml:"endpoint" json:"endpoint"`
	Username     string        `yaml:"username" json:"username"`
	Password     string        `yaml:"password" json:"password"`
	Timeout      time.Duration `yaml:"timeout" json:"timeout"`
	RetryAttempts int          `yaml:"retry_attempts" json:"retry_attempts"`
	BatchSize    int          `yaml:"batch_size" json:"batch_size"`
	Enabled      bool          `yaml:"enabled" json:"enabled"`
}

// DefaultConfig returns the default LRS configuration
func DefaultConfig() *Config {
	return &Config{
		Timeout:       30 * time.Second,
		RetryAttempts: 3,
		BatchSize:     10,
		Enabled:       true,
	}
}

// Client represents an LRS client
type Client struct {
	config      *Config
	httpClient  *http.Client
	logger      *logrus.Logger
	batchBuffer []*LearningRecord
	mutex       sync.RWMutex
}

// NewClient creates a new LRS client
func NewClient(config *Config, logger *logrus.Logger) (*Client, error) {
	if config == nil {
		config = DefaultConfig()
	}

	if logger == nil {
		logger = logrus.New()
	}

	if !config.Enabled {
		return &Client{
			config:     config,
			logger:     logger,
		}, nil
	}

	// Validate configuration
	if config.Endpoint == "" {
		return nil, fmt.Errorf("LRS endpoint is required")
	}

	// Ensure endpoint ends with /
	if !strings.HasSuffix(config.Endpoint, "/") {
		config.Endpoint += "/"
	}

	// Add xAPI endpoint path
	if !strings.Contains(config.Endpoint, "/xapi/") {
		config.Endpoint += "xapi/"
	}

	httpClient := &http.Client{
		Timeout: config.Timeout,
	}

	client := &Client{
		config:      config,
		httpClient:  httpClient,
		logger:      logger,
		batchBuffer: make([]*LearningRecord, 0),
	}

	return client, nil
}

// RecordLearningEvent records a single learning event
func (c *Client) RecordLearningEvent(ctx context.Context, record *LearningRecord) error {
	if !c.config.Enabled {
		return nil
	}

	// Set timestamp if not provided
	if record.Timestamp.IsZero() {
		record.Timestamp = time.Now()
	}

	// Set stored timestamp
	record.Stored = time.Now()

	// Add to batch buffer
	c.mutex.Lock()
	c.batchBuffer = append(c.batchBuffer, record)
	shouldFlush := len(c.batchBuffer) >= c.config.BatchSize
	c.mutex.Unlock()

	if shouldFlush {
		return c.FlushRecords(ctx)
	}

	return nil
}

// RecordContextInteraction records a context interaction event
func (c *Client) RecordContextInteraction(ctx context.Context, userID, contextKey, interactionType, outcome string, blockMetadata map[string]string) error {
	actor := Actor{
		Account: &Account{
			HomePage: "advanced-research",
			Name:     userID,
		},
		ObjectType: "Agent",
	}

	verb := Verb{
		ID: fmt.Sprintf("http://adlnet.gov/expapi/verbs/%s", interactionType),
		Display: map[string]string{
			"en-US": interactionType,
		},
	}

	obj := Object{
		ID:         fmt.Sprintf("urn:context:%s", contextKey),
		ObjectType: "Activity",
		Name: &map[string]string{
			"en-US": contextKey,
		},
		Description: &map[string]string{
			"en-US": fmt.Sprintf("Context block: %s", contextKey),
		},
		Definition: &ObjectDefinition{
			Type: "http://adlnet.gov/expapi/activities/cmi.interaction",
		},
	}

	// Add metadata as extensions
	if len(blockMetadata) > 0 {
		if obj.Definition.Extensions == nil {
			obj.Definition.Extensions = make(map[string]interface{})
		}
		for k, v := range blockMetadata {
			obj.Definition.Extensions[k] = v
		}
	}

	record := &LearningRecord{
		Actor:     actor,
		Verb:      verb,
		Object:    obj,
		Timestamp: time.Now(),
	}

	// Add result if outcome provided
	if outcome != "" {
		success := outcome == "success"
		completion := true
		record.Result = &Result{
			Completion: &completion,
			Success:    &success,
			Response:   &outcome,
		}
	}

	return c.RecordLearningEvent(ctx, record)
}

// RecordConversation records a conversation message
func (c *Client) RecordConversation(ctx context.Context, userID, role, content string, metadata map[string]string) error {
	actor := Actor{
		Account: &Account{
			HomePage: "advanced-research",
			Name:     userID,
		},
		ObjectType: "Agent",
	}

	verb := Verb{
		ID: "http://adlnet.gov/expapi/verbs/communicated",
		Display: map[string]string{
			"en-US": "communicated",
		},
	}

	obj := Object{
		ID:         fmt.Sprintf("urn:conversation:%d", time.Now().Unix()),
		ObjectType: "Activity",
		Name: &map[string]string{
			"en-US": fmt.Sprintf("Conversation - %s", role),
		},
		Description: &map[string]string{
			"en-US": content[:min(100, len(content))],
		},
		Definition: &ObjectDefinition{
			Type: "http://adlnet.gov/expapi/activities/conversation",
		},
	}

	// Add role and metadata as extensions
	if obj.Definition.Extensions == nil {
		obj.Definition.Extensions = make(map[string]interface{})
	}
	obj.Definition.Extensions["role"] = role
	obj.Definition.Extensions["message_length"] = len(content)

	for k, v := range metadata {
		obj.Definition.Extensions[k] = v
	}

	record := &LearningRecord{
		Actor:     actor,
		Verb:      verb,
		Object:    obj,
		Timestamp: time.Now(),
	}

	return c.RecordLearningEvent(ctx, record)
}

// FlushRecords flushes the batch buffer to the LRS
func (c *Client) FlushRecords(ctx context.Context) error {
	if !c.config.Enabled {
		return nil
	}

	c.mutex.Lock()
	if len(c.batchBuffer) == 0 {
		c.mutex.Unlock()
		return nil
	}

	// Copy buffer and clear it
	records := make([]*LearningRecord, len(c.batchBuffer))
	copy(records, c.batchBuffer)
	c.batchBuffer = c.batchBuffer[:0]
	c.mutex.Unlock()

	c.logger.WithField("count", len(records)).Info("Flushing learning records to LRS")

	// Send records in batches
	for i := 0; i < len(records); i += c.config.BatchSize {
		end := i + c.config.BatchSize
		if end > len(records) {
			end = len(records)
		}

		batch := records[i:end]
		if err := c.sendBatch(ctx, batch); err != nil {
			c.logger.WithError(err).WithField("batch_size", len(batch)).Error("Failed to send batch")
			return err
		}
	}

	return nil
}

// sendBatch sends a batch of records to the LRS
func (c *Client) sendBatch(ctx context.Context, records []*LearningRecord) error {
	// Convert to JSON
	jsonData, err := json.Marshal(records)
	if err != nil {
		return fmt.Errorf("failed to marshal records: %w", err)
	}

	// Create request
	req, err := http.NewRequestWithContext(ctx, "POST", c.config.Endpoint+"statements", bytes.NewBuffer(jsonData))
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	// Set headers
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Experience-API-Version", "1.0.3")

	// Add basic auth if credentials provided
	if c.config.Username != "" && c.config.Password != "" {
		req.SetBasicAuth(c.config.Username, c.config.Password)
	}

	// Send request with retries
	var lastErr error
	for attempt := 0; attempt < c.config.RetryAttempts; attempt++ {
		if attempt > 0 {
			// Exponential backoff
			backoff := time.Duration(1<<uint(attempt-1)) * time.Second
			c.logger.WithField("attempt", attempt+1).WithField("backoff", backoff).Warn("Retrying LRS request")
			time.Sleep(backoff)
		}

		resp, err := c.httpClient.Do(req)
		if err != nil {
			lastErr = err
			continue
		}

		// Check response
		if resp.StatusCode >= 200 && resp.StatusCode < 300 {
			resp.Body.Close()
			return nil
		}

		// Log error response
		body, _ := readResponseBody(resp.Body)
		resp.Body.Close()
		lastErr = fmt.Errorf("LRS returned status %d: %s", resp.StatusCode, string(body))
	}

	return lastErr
}

// GetStatements retrieves statements from the LRS
func (c *Client) GetStatements(ctx context.Context, params map[string]string) (*StatementsResponse, error) {
	if !c.config.Enabled {
		return nil, fmt.Errorf("LRS is disabled")
	}

	// Build URL with parameters
	u, err := url.Parse(c.config.Endpoint + "statements")
	if err != nil {
		return nil, fmt.Errorf("failed to parse URL: %w", err)
	}

	q := u.Query()
	for k, v := range params {
		q.Set(k, v)
	}
	u.RawQuery = q.Encode()

	// Create request
	req, err := http.NewRequestWithContext(ctx, "GET", u.String(), nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	// Add basic auth if credentials provided
	if c.config.Username != "" && c.config.Password != "" {
		req.SetBasicAuth(c.config.Username, c.config.Password)
	}

	// Send request
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		body, _ := readResponseBody(resp.Body)
		return nil, fmt.Errorf("LRS returned status %d: %s", resp.StatusCode, string(body))
	}

	// Parse response
	var statementsResponse StatementsResponse
	if err := json.NewDecoder(resp.Body).Decode(&statementsResponse); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &statementsResponse, nil
}

// StatementsResponse represents the response from the statements endpoint
type StatementsResponse struct {
	Statements []LearningRecord `json:"statements"`
	MoreURL    *string          `json:"more,omitempty"`
}

// Close flushes any pending records and closes the client
func (c *Client) Close(ctx context.Context) error {
	return c.FlushRecords(ctx)
}

// Helper functions
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func readResponseBody(body *http.Response) ([]byte, error) {
	defer body.Close()
	buf := new(bytes.Buffer)
	_, err := buf.ReadFrom(body)
	return buf.Bytes(), err
}