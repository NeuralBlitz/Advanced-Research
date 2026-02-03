package context

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"
)

// Priority defines the priority level of context blocks
type Priority int

const (
	PriorityLow Priority = iota + 1
	PriorityMedium
	PriorityHigh
	PriorityCritical
)

// String returns the string representation of Priority
func (p Priority) String() string {
	switch p {
	case PriorityLow:
		return "low"
	case PriorityMedium:
		return "medium"
	case PriorityHigh:
		return "high"
	case PriorityCritical:
		return "critical"
	default:
		return "unknown"
	}
}

// ContextBlock represents a single block of context with metadata
type ContextBlock struct {
	Key        string            `json:"key"`
	Content    string            `json:"content"`
	Priority   Priority          `json:"priority"`
	Timestamp  time.Time         `json:"timestamp"`
	Expires    *time.Time        `json:"expires,omitempty"`
	Tags       []string          `json:"tags"`
	Metadata   map[string]string `json:"metadata"`
	mutex      sync.RWMutex
}

// IsExpired returns true if the context block has expired
func (cb *ContextBlock) IsExpired() bool {
	cb.mutex.RLock()
	defer cb.mutex.RUnlock()
	
	if cb.Expires == nil {
		return false
	}
	return time.Now().After(*cb.Expires)
}

// AgeSeconds returns the age of the context block in seconds
func (cb *ContextBlock) AgeSeconds() int64 {
	cb.mutex.RLock()
	defer cb.mutex.RUnlock()
	return int64(time.Since(cb.Timestamp).Seconds())
}

// AddTag adds a tag to the context block
func (cb *ContextBlock) AddTag(tag string) {
	cb.mutex.Lock()
	defer cb.mutex.Unlock()
	
	for _, existingTag := range cb.Tags {
		if existingTag == tag {
			return
		}
	}
	cb.Tags = append(cb.Tags, tag)
}

// HasTag returns true if the context block has the specified tag
func (cb *ContextBlock) HasTag(tag string) bool {
	cb.mutex.RLock()
	defer cb.mutex.RUnlock()
	
	for _, existingTag := range cb.Tags {
		if existingTag == tag {
			return true
		}
	}
	return false
}

// ConversationMessage represents a message in the conversation history
type ConversationMessage struct {
	Role      string            `json:"role"`
	Content   string            `json:"content"`
	Timestamp time.Time         `json:"timestamp"`
	Metadata  map[string]string `json:"metadata"`
}

// ContextStatistics provides statistics about the context injector
type ContextStatistics struct {
	TotalBlocks           int                  `json:"total_blocks"`
	TotalMessages         int                  `json:"total_messages"`
	BlocksByPriority      map[string]int       `json:"blocks_by_priority"`
	OldestBlockAgeSeconds int64                `json:"oldest_block_age_seconds"`
	NewestBlockAgeSeconds int64                `json:"newest_block_age_seconds"`
	TotalContentSize      int64                `json:"total_content_size"`
	TotalTags            int                  `json:"total_tags"`
	UniqueTags           []string             `json:"unique_tags"`
}

// Config represents the configuration for the context injector
type Config struct {
	MaxContextSize        int           `yaml:"max_context_size" json:"max_context_size"`
	CleanupInterval       time.Duration `yaml:"cleanup_interval" json:"cleanup_interval"`
	DefaultTTL           time.Duration `yaml:"default_ttl" json:"default_ttl"`
	MaxConversationLength int           `yaml:"max_conversation_length" json:"max_conversation_length"`
	EnableCompression    bool          `yaml:"enable_compression" json:"enable_compression"`
}

// DefaultConfig returns the default configuration
func DefaultConfig() *Config {
	return &Config{
		MaxContextSize:        10000,
		CleanupInterval:       time.Hour,
		DefaultTTL:           24 * time.Hour,
		MaxConversationLength: 100,
		EnableCompression:    false,
	}
}

// ContextInjector is the main context injection system
type ContextInjector struct {
	config              *Config
	contextBlocks       map[string]*ContextBlock
	conversationHistory  []ConversationMessage
	mutex               sync.RWMutex
	cleanupTicker       *time.Ticker
	done                chan struct{}
}

// NewContextInjector creates a new context injector with the given configuration
func NewContextInjector(config *Config) *ContextInjector {
	if config == nil {
		config = DefaultConfig()
	}

	ci := &ContextInjector{
		config:             config,
		contextBlocks:       make(map[string]*ContextBlock),
		conversationHistory:  make([]ConversationMessage, 0),
		cleanupTicker:       time.NewTicker(config.CleanupInterval),
		done:               make(chan struct{}),
	}

	// Start cleanup goroutine
	go ci.cleanupLoop()

	return ci
}

// AddContext adds a new context block
func (ci *ContextInjector) AddContext(ctx context.Context, key, content string, priority Priority, opts ...ContextOption) error {
	ci.mutex.Lock()
	defer ci.mutex.Unlock()

	// Check if context already exists
	if _, exists := ci.contextBlocks[key]; exists {
		return fmt.Errorf("context block with key '%s' already exists", key)
	}

	// Apply options
	block := &ContextBlock{
		Key:       key,
		Content:   content,
		Priority:  priority,
		Timestamp: time.Now(),
		Tags:      make([]string, 0),
		Metadata:  make(map[string]string),
	}

	for _, opt := range opts {
		opt(block)
	}

	// Set default expiration if not provided
	if block.Expires == nil && ci.config.DefaultTTL > 0 {
		expires := time.Now().Add(ci.config.DefaultTTL)
		block.Expires = &expires
	}

	ci.contextBlocks[key] = block

	return nil
}

// GetContext retrieves the formatted context string
func (ci *ContextInjector) GetContext(ctx context.Context, maxTokens *int) string {
	ci.mutex.RLock()
	defer ci.mutex.RUnlock()

	// Clean up expired blocks first
	ci.cleanupExpiredBlocks()

	// Convert map to slice for sorting
	blocks := make([]*ContextBlock, 0, len(ci.contextBlocks))
	for _, block := range ci.contextBlocks {
		if !block.IsExpired() {
			blocks = append(blocks, block)
		}
	}

	// Sort by priority (descending) then by timestamp (newest first)
	sort.Slice(blocks, func(i, j int) bool {
		if blocks[i].Priority != blocks[j].Priority {
			return blocks[i].Priority > blocks[j].Priority
		}
		return blocks[i].Timestamp.After(blocks[j].Timestamp)
	})

	// Build context string
	var contextParts []string
	currentSize := 0
	maxSize := ci.config.MaxContextSize
	if maxTokens != nil {
		maxSize = *maxTokens
	}

	for _, block := range blocks {
		if currentSize+len(block.Content) > maxSize {
			break
		}
		contextParts = append(contextParts, fmt.Sprintf("## %s\n%s\n", block.Key, block.Content))
		currentSize += len(block.Content)
	}

	return strings.Join(contextParts, "\n")
}

// GetContextBlocks returns context blocks matching the given filters
func (ci *ContextInjector) GetContextBlocks(ctx context.Context, filterTags []string) []*ContextBlock {
	ci.mutex.RLock()
	defer ci.mutex.RUnlock()

	var filtered []*ContextBlock
	for _, block := range ci.contextBlocks {
		if block.IsExpired() {
			continue
		}

		if len(filterTags) > 0 {
			hasMatch := false
			for _, filterTag := range filterTags {
				if block.HasTag(filterTag) {
					hasMatch = true
					break
				}
			}
			if !hasMatch {
				continue
			}
		}

		filtered = append(filtered, block)
	}

	// Sort by priority and timestamp
	sort.Slice(filtered, func(i, j int) bool {
		if filtered[i].Priority != filtered[j].Priority {
			return filtered[i].Priority > filtered[j].Priority
		}
		return filtered[i].Timestamp.After(filtered[j].Timestamp)
	})

	return filtered
}

// RemoveContext removes a context block by key
func (ci *ContextInjector) RemoveContext(ctx context.Context, key string) bool {
	ci.mutex.Lock()
	defer ci.mutex.Unlock()

	if _, exists := ci.contextBlocks[key]; exists {
		delete(ci.contextBlocks, key)
		return true
	}
	return false
}

// ClearContext removes all context blocks
func (ci *ContextInjector) ClearContext(ctx context.Context) {
	ci.mutex.Lock()
	defer ci.mutex.Unlock()

	ci.contextBlocks = make(map[string]*ContextBlock)
}

// AddConversationMessage adds a message to the conversation history
func (ci *ContextInjector) AddConversationMessage(ctx context.Context, role, content string, metadata map[string]string) error {
	ci.mutex.Lock()
	defer ci.mutex.Unlock()

	if len(ci.conversationHistory) >= ci.config.MaxConversationLength {
		// Remove oldest message
		ci.conversationHistory = ci.conversationHistory[1:]
	}

	message := ConversationMessage{
		Role:      role,
		Content:   content,
		Timestamp: time.Now(),
		Metadata:  metadata,
	}

	ci.conversationHistory = append(ci.conversationHistory, message)

	return nil
}

// GetConversationHistory returns the conversation history
func (ci *ContextInjector) GetConversationHistory(ctx context.Context, limit *int, roleFilter *string) []ConversationMessage {
	ci.mutex.RLock()
	defer ci.mutex.RUnlock()

	history := make([]ConversationMessage, len(ci.conversationHistory))
	copy(history, ci.conversationHistory)

	// Filter by role
	if roleFilter != nil {
		var filtered []ConversationMessage
		for _, msg := range history {
			if msg.Role == *roleFilter {
				filtered = append(filtered, msg)
			}
		}
		history = filtered
	}

	// Apply limit
	if limit != nil && len(history) > *limit {
		history = history[len(history)-*limit:]
	}

	return history
}

// GetStatistics returns statistics about the context injector
func (ci *ContextInjector) GetStatistics(ctx context.Context) *ContextStatistics {
	ci.mutex.RLock()
	defer ci.mutex.RUnlock()

	stats := &ContextStatistics{
		TotalBlocks:       len(ci.contextBlocks),
		TotalMessages:     len(ci.conversationHistory),
		BlocksByPriority:  make(map[string]int),
		UniqueTags:        make([]string, 0),
		TotalContentSize:  0,
	}

	var oldestTime, newestTime time.Time
	firstBlock := true
	tagSet := make(map[string]bool)

	for _, block := range ci.contextBlocks {
		// Count by priority
		stats.BlocksByPriority[block.Priority.String()]++

		// Track oldest/newest
		if firstBlock {
			oldestTime = block.Timestamp
			newestTime = block.Timestamp
			firstBlock = false
		} else {
			if block.Timestamp.Before(oldestTime) {
				oldestTime = block.Timestamp
			}
			if block.Timestamp.After(newestTime) {
				newestTime = block.Timestamp
			}
		}

		// Count content size
		stats.TotalContentSize += int64(len(block.Content))

		// Collect unique tags
		for _, tag := range block.Tags {
			if !tagSet[tag] {
				tagSet[tag] = true
				stats.UniqueTags = append(stats.UniqueTags, tag)
			}
		}
	}

	if !firstBlock {
		stats.OldestBlockAgeSeconds = int64(time.Since(oldestTime).Seconds())
		stats.NewestBlockAgeSeconds = int64(time.Since(newestTime).Seconds())
	}

	stats.TotalTags = len(stats.UniqueTags)

	// Sort unique tags
	sort.Strings(stats.UniqueTags)

	return stats
}

// Close stops the context injector and cleans up resources
func (ci *ContextInjector) Close() error {
	close(ci.done)
	ci.cleanupTicker.Stop()
	return nil
}

// cleanupExpiredBlocks removes expired context blocks
func (ci *ContextInjector) cleanupExpiredBlocks() {
	for key, block := range ci.contextBlocks {
		if block.IsExpired() {
			delete(ci.contextBlocks, key)
		}
	}
}

// cleanupLoop runs the periodic cleanup
func (ci *ContextInjector) cleanupLoop() {
	for {
		select {
		case <-ci.cleanupTicker.C:
			ci.mutex.Lock()
			ci.cleanupExpiredBlocks()
			ci.mutex.Unlock()
		case <-ci.done:
			return
		}
	}
}

// ContextOption represents an option for configuring context blocks
type ContextOption func(*ContextBlock)

// WithExpires sets the expiration time for the context block
func WithExpires(expires time.Time) ContextOption {
	return func(cb *ContextBlock) {
		cb.Expires = &expires
	}
}

// WithTTL sets the time-to-live for the context block
func WithTTL(ttl time.Duration) ContextOption {
	return func(cb *ContextBlock) {
		expires := time.Now().Add(ttl)
		cb.Expires = &expires
	}
}

// WithTags sets the tags for the context block
func WithTags(tags []string) ContextOption {
	return func(cb *ContextBlock) {
		cb.Tags = make([]string, len(tags))
		copy(cb.Tags, tags)
	}
}

// WithMetadata sets the metadata for the context block
func WithMetadata(metadata map[string]string) ContextOption {
	return func(cb *ContextBlock) {
		cb.Metadata = make(map[string]string)
		for k, v := range metadata {
			cb.Metadata[k] = v
		}
	}
}