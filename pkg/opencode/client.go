package opencode

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/go-git/go-git/v5"
	"github.com/go-git/go-git/v5/plumbing"
	"github.com/go-git/go-git/v5/plumbing/object"
	"github.com/sirupsen/logrus"
	"gopkg.in/yaml.v3"
)

// ResearchStage represents the stage of research
type ResearchStage string

const (
	StageIdeas        ResearchStage = "ideas"
	StageSynthesis     ResearchStage = "synthesis"
	StageImplementation ResearchStage = "implementation"
)

// DocumentType represents the type of research document
type DocumentType string

const (
	TypeTheory       DocumentType = "theory"
	TypeFramework    DocumentType = "framework"
	TypeExperiment   DocumentType = "experiment"
	TypeProof        DocumentType = "proof"
	TypeCode         DocumentType = "code"
	TypeAnalysis     DocumentType = "analysis"
)

// ResearchDocument represents a research document in the opencode system
type ResearchDocument struct {
	ID          string                 `json:"id" yaml:"id"`
	Title       string                 `json:"title" yaml:"title"`
	Content     string                 `json:"content" yaml:"content"`
	Stage       ResearchStage          `json:"stage" yaml:"stage"`
	DocType     DocumentType          `json:"doc_type" yaml:"doc_type"`
	Tags        []string               `json:"tags" yaml:"tags"`
	Dependencies []string               `json:"dependencies" yaml:"dependencies"`
	Metadata    map[string]interface{} `json:"metadata" yaml:"metadata"`
	CreatedAt   time.Time              `json:"created_at" yaml:"created_at"`
	UpdatedAt   time.Time              `json:"updated_at" yaml:"updated_at"`
	Author      string                 `json:"author" yaml:"author"`
}

// GetSemanticTags returns semantic tags based on stage and type
func (rd *ResearchDocument) GetSemanticTags() []string {
	tags := []string{
		fmt.Sprintf("#%s", rd.Stage),
		fmt.Sprintf("#%s", rd.DocType),
	}
	
	// Add custom tags
	for _, tag := range rd.Tags {
		if !strings.HasPrefix(tag, "#") {
			tag = "#" + tag
		}
		tags = append(tags, tag)
	}
	
	// Remove duplicates
	uniqueTags := make([]string, 0)
	tagSet := make(map[string]bool)
	for _, tag := range tags {
		if !tagSet[tag] {
			tagSet[tag] = true
			uniqueTags = append(uniqueTags, tag)
		}
	}
	
	return uniqueTags
}

// ProjectStructure represents the opencode project structure
type ProjectStructure struct {
	ResearchAreas []ResearchArea `json:"research_areas" yaml:"research_areas"`
	SemanticTags  []string       `json:"semantic_tags" yaml:"semantic_tags"`
	Workflows     []string       `json:"workflows" yaml:"workflows"`
	Version       string         `json:"version" yaml:"version"`
}

// ResearchArea represents a research area in the project
type ResearchArea struct {
	Name        string                 `json:"name" yaml:"name"`
	Description string                 `json:"description" yaml:"description"`
	Metadata    map[string]interface{} `json:"metadata" yaml:"metadata"`
}

// WorkflowRule represents a workflow transition rule
type WorkflowRule struct {
	Trigger    string                 `json:"trigger" yaml:"trigger"`
	Conditions map[string]interface{} `json:"conditions" yaml:"conditions"`
	Actions    []string               `json:"actions" yaml:"actions"`
}

// Config represents the opencode integration configuration
type Config struct {
	Enabled       bool          `yaml:"enabled" json:"enabled"`
	APIKey        string        `yaml:"api_key" json:"api_key"`
	Workspace     string        `yaml:"workspace" json:"workspace"`
	GitRepo       string        `yaml:"git_repo" json:"git_repo"`
	LocalPath     string        `yaml:"local_path" json:"local_path"`
	Timeout       time.Duration `yaml:"timeout" json:"timeout"`
	AutoCommit    bool          `yaml:"auto_commit" json:"auto_commit"`
	AuthorName    string        `yaml:"author_name" json:"author_name"`
	AuthorEmail   string        `yaml:"author_email" json:"author_email"`
}

// DefaultConfig returns the default opencode configuration
func DefaultConfig() *Config {
	return &Config{
		Enabled:     true,
		Timeout:     30 * time.Second,
		AutoCommit:  true,
		AuthorName:  "Advanced Research",
		AuthorEmail: "research@advanced-research.com",
		LocalPath:   "./research-docs",
	}
}

// Client represents the opencode integration client
type Client struct {
	config          *Config
	logger          *logrus.Logger
	projectStructure *ProjectStructure
	documents       map[string]*ResearchDocument
	gitRepo         *git.Repository
	mutex           sync.RWMutex
}

// NewClient creates a new opencode client
func NewClient(config *Config, logger *logrus.Logger) (*Client, error) {
	if config == nil {
		config = DefaultConfig()
	}

	if logger == nil {
		logger = logrus.New()
	}

	client := &Client{
		config:    config,
		logger:    logger,
		documents: make(map[string]*ResearchDocument),
	}

	if config.Enabled {
		if err := client.initialize(context.Background()); err != nil {
			return nil, fmt.Errorf("failed to initialize opencode client: %w", err)
		}
	}

	return client, nil
}

// initialize sets up the opencode integration
func (c *Client) initialize(ctx context.Context) error {
	// Create local directory if it doesn't exist
	if err := os.MkdirAll(c.config.LocalPath, 0755); err != nil {
		return fmt.Errorf("failed to create local directory: %w", err)
	}

	// Initialize or open git repository
	if err := c.initGitRepository(); err != nil {
		return fmt.Errorf("failed to initialize git repository: %w", err)
	}

	// Load project structure
	if err := c.loadProjectStructure(); err != nil {
		c.logger.WithError(err).Warn("Failed to load project structure, using defaults")
		c.projectStructure = c.getDefaultProjectStructure()
	}

	// Load existing documents
	if err := c.loadExistingDocuments(); err != nil {
		c.logger.WithError(err).Warn("Failed to load existing documents")
	}

	c.logger.Info("Opencode integration initialized successfully")
	return nil
}

// initGitRepository initializes or opens the git repository
func (c *Client) initGitRepository() error {
	repo, err := git.PlainOpen(c.config.LocalPath)
	if err == git.ErrRepositoryNotYetExists {
		// Initialize new repository
		repo, err = git.PlainInit(c.config.LocalPath, false)
		if err != nil {
			return fmt.Errorf("failed to initialize git repository: %w", err)
		}
		c.logger.Info("Initialized new git repository")
	} else if err != nil {
		return fmt.Errorf("failed to open git repository: %w", err)
	}

	c.gitRepo = repo
	return nil
}

// loadProjectStructure loads the project structure from file or creates default
func (c *Client) loadProjectStructure() error {
	structurePath := filepath.Join(c.config.LocalPath, "opencode.yaml")
	
	if _, err := os.Stat(structurePath); os.IsNotExist(err) {
		// Create default structure
		c.projectStructure = c.getDefaultProjectStructure()
		return c.saveProjectStructure()
	}

	data, err := os.ReadFile(structurePath)
	if err != nil {
		return fmt.Errorf("failed to read project structure file: %w", err)
	}

	var structure ProjectStructure
	if err := yaml.Unmarshal(data, &structure); err != nil {
		return fmt.Errorf("failed to parse project structure: %w", err)
	}

	c.projectStructure = &structure
	return nil
}

// saveProjectStructure saves the project structure to file
func (c *Client) saveProjectStructure() error {
	structurePath := filepath.Join(c.config.LocalPath, "opencode.yaml")
	
	data, err := yaml.Marshal(c.projectStructure)
	if err != nil {
		return fmt.Errorf("failed to marshal project structure: %w", err)
	}

	if err := os.WriteFile(structurePath, data, 0644); err != nil {
		return fmt.Errorf("failed to write project structure: %w", err)
	}

	return nil
}

// getDefaultProjectStructure returns the default project structure
func (c *Client) getDefaultProjectStructure() *ProjectStructure {
	return &ProjectStructure{
		ResearchAreas: []ResearchArea{
			{
				Name:        "Ideas",
				Description: "Theoretical foundation and brainstorming",
				Metadata: map[string]interface{}{
					"type":     "theory",
					"priority":  "high",
				},
			},
			{
				Name:        "Apical-Synthesis",
				Description: "Integrated frameworks",
				Metadata: map[string]interface{}{
					"type":     "synthesis",
					"priority":  "medium",
				},
			},
			{
				Name:        "Implementation",
				Description: "Experimental code and validation",
				Metadata: map[string]interface{}{
					"type":     "implementation",
					"priority":  "medium",
				},
			},
		},
		SemanticTags: []string{"#theory", "#synthesis", "#implementation", "#research"},
		Workflows:    []string{"ideas_to_synthesis", "synthesis_to_implementation"},
		Version:      "1.0.0",
	}
}

// loadExistingDocuments loads existing documents from the repository
func (c *Client) loadExistingDocuments() error {
	// Scan the local directory for document files
	return filepath.Walk(c.config.LocalPath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		if info.IsDir() {
			return nil
		}

		// Only process markdown files
		if !strings.HasSuffix(strings.ToLower(path), ".md") {
			return nil
		}

		// Skip the project structure file
		if strings.HasSuffix(path, "opencode.yaml") {
			return nil
		}

		// Try to load document
		doc, err := c.loadDocumentFromFile(path)
		if err != nil {
			c.logger.WithError(err).WithField("path", path).Warn("Failed to load document")
			return nil
		}

		c.documents[doc.ID] = doc
		return nil
	})
}

// loadDocumentFromFile loads a research document from a file
func (c *Client) loadDocumentFromFile(filePath string) (*ResearchDocument, error) {
	data, err := os.ReadFile(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to read file: %w", err)
	}

	// Try to parse front matter
	content := string(data)
	var frontMatter map[string]interface{}
	
	if strings.HasPrefix(content, "---") {
		parts := strings.SplitN(content, "---", 3)
		if len(parts) >= 3 {
			if err := yaml.Unmarshal([]byte(parts[1]), &frontMatter); err != nil {
				return nil, fmt.Errorf("failed to parse front matter: %w", err)
			}
			content = parts[2]
		}
	}

	// Create document
	doc := &ResearchDocument{
		ID:      filepath.Base(filePath[:len(filePath)-3]), // Remove .md extension
		Content: content,
	}

	// Extract metadata from front matter
	if frontMatter != nil {
		if title, ok := frontMatter["title"].(string); ok {
			doc.Title = title
		}
		if stage, ok := frontMatter["stage"].(string); ok {
			doc.Stage = ResearchStage(stage)
		}
		if docType, ok := frontMatter["doc_type"].(string); ok {
			doc.DocType = DocumentType(docType)
		}
		if tags, ok := frontMatter["tags"].([]interface{}); ok {
			for _, tag := range tags {
				if tagStr, ok := tag.(string); ok {
					doc.Tags = append(doc.Tags, tagStr)
				}
			}
		}
		if deps, ok := frontMatter["dependencies"].([]interface{}); ok {
			for _, dep := range deps {
				if depStr, ok := dep.(string); ok {
					doc.Dependencies = append(doc.Dependencies, depStr)
				}
			}
		}
		if metadata, ok := frontMatter["metadata"].(map[string]interface{}); ok {
			doc.Metadata = metadata
		}
		if author, ok := frontMatter["author"].(string); ok {
			doc.Author = author
		}
		if created, ok := frontMatter["created_at"].(string); ok {
			if parsedTime, err := time.Parse(time.RFC3339, created); err == nil {
				doc.CreatedAt = parsedTime
			}
		}
		if updated, ok := frontMatter["updated_at"].(string); ok {
			if parsedTime, err := time.Parse(time.RFC3339, updated); err == nil {
				doc.UpdatedAt = parsedTime
			}
		}
	}

	// Set defaults
	if doc.CreatedAt.IsZero() {
		doc.CreatedAt = time.Now()
	}
	if doc.UpdatedAt.IsZero() {
		doc.UpdatedAt = doc.CreatedAt
	}
	if doc.Title == "" {
		doc.Title = doc.ID
	}
	if doc.Stage == "" {
		doc.Stage = StageIdeas
	}
	if doc.DocType == "" {
		doc.DocType = TypeTheory
	}

	return doc, nil
}

// CreateResearchDocument creates a new research document
func (c *Client) CreateResearchDocument(ctx context.Context, title, content string, stage ResearchStage, docType DocumentType, tags []string, metadata map[string]interface{}) (string, error) {
	if !c.config.Enabled {
		return "mock-doc-id", nil
	}

	c.mutex.Lock()
	defer c.mutex.Unlock()

	// Generate unique ID
	docID := fmt.Sprintf("doc_%d", time.Now().Unix())

	// Create document
	doc := &ResearchDocument{
		ID:          docID,
		Title:       title,
		Content:     content,
		Stage:       stage,
		DocType:     docType,
		Tags:        tags,
		Dependencies: []string{},
		Metadata:    metadata,
		CreatedAt:   time.Now(),
		UpdatedAt:   time.Now(),
		Author:      c.config.AuthorName,
	}

	// Store document
	c.documents[docID] = doc

	// Save to file
	if err := c.saveDocumentToFile(doc); err != nil {
		delete(c.documents, docID)
		return "", fmt.Errorf("failed to save document: %w", err)
	}

	// Auto-commit if enabled
	if c.config.AutoCommit {
		if err := c.commitDocument(ctx, doc, "Create research document"); err != nil {
			c.logger.WithError(err).Warn("Failed to auto-commit document")
		}
	}

	c.logger.WithFields(logrus.Fields{
		"doc_id":   docID,
		"title":     title,
		"stage":     stage,
		"doc_type":  docType,
	}).Info("Created research document")

	return docID, nil
}

// saveDocumentToFile saves a document to a markdown file
func (c *Client) saveDocumentToFile(doc *ResearchDocument) error {
	// Create stage directory if it doesn't exist
	stageDir := filepath.Join(c.config.LocalPath, string(doc.Stage))
	if err := os.MkdirAll(stageDir, 0755); err != nil {
		return fmt.Errorf("failed to create stage directory: %w", err)
	}

	// Prepare front matter
	frontMatter := map[string]interface{}{
		"id":           doc.ID,
		"title":        doc.Title,
		"stage":        doc.Stage,
		"doc_type":     doc.DocType,
		"tags":         doc.Tags,
		"dependencies": doc.Dependencies,
		"metadata":     doc.Metadata,
		"created_at":   doc.CreatedAt.Format(time.RFC3339),
		"updated_at":   doc.UpdatedAt.Format(time.RFC3339),
		"author":       doc.Author,
	}

	// Marshal front matter
	frontMatterData, err := yaml.Marshal(frontMatter)
	if err != nil {
		return fmt.Errorf("failed to marshal front matter: %w", err)
	}

	// Combine front matter and content
	content := fmt.Sprintf("---\n%s---\n%s", string(frontMatterData), doc.Content)

	// Write to file
	filePath := filepath.Join(stageDir, doc.ID+".md")
	if err := os.WriteFile(filePath, []byte(content), 0644); err != nil {
		return fmt.Errorf("failed to write document file: %w", err)
	}

	return nil
}

// commitDocument commits a document to the git repository
func (c *Client) commitDocument(ctx context.Context, doc *ResearchDocument, message string) error {
	if c.gitRepo == nil {
		return nil
	}

	worktree, err := c.gitRepo.Worktree()
	if err != nil {
		return fmt.Errorf("failed to get worktree: %w", err)
	}

	// Add file to staging
	filePath := filepath.Join(string(doc.Stage), doc.ID+".md")
	if _, err := worktree.Add(filePath); err != nil {
		return fmt.Errorf("failed to add file to staging: %w", err)
	}

	// Commit changes
	commit, err := worktree.Commit(message, &git.CommitOptions{
		Author: &object.Signature{
			Name:  c.config.AuthorName,
			Email: c.config.AuthorEmail,
			When:  time.Now(),
		},
	})
	if err != nil {
		return fmt.Errorf("failed to commit: %w", err)
	}

	c.logger.WithFields(logrus.Fields{
		"commit": commit.String(),
		"file":   filePath,
	}).Info("Committed document to repository")

	return nil
}

// SearchDocuments searches for documents based on criteria
func (c *Client) SearchDocuments(ctx context.Context, query string, stage *ResearchStage, docType *DocumentType, tags []string) ([]*ResearchDocument, error) {
	if !c.config.Enabled {
		return []*ResearchDocument{
			{
				ID:      "doc_1",
				Title:   fmt.Sprintf("Document matching: %s", query),
				Stage:   StageIdeas,
				DocType: TypeTheory,
				Tags:    tags,
			},
		}, nil
	}

	c.mutex.RLock()
	defer c.mutex.RUnlock()

	var results []*ResearchDocument
	queryLower := strings.ToLower(query)

	for _, doc := range c.documents {
		// Text search
		if !strings.Contains(strings.ToLower(doc.Title), queryLower) &&
			!strings.Contains(strings.ToLower(doc.Content), queryLower) {
			continue
		}

		// Stage filter
		if stage != nil && doc.Stage != *stage {
			continue
		}

		// Type filter
		if docType != nil && doc.DocType != *docType {
			continue
		}

		// Tags filter
		if len(tags) > 0 {
			hasMatch := false
			for _, filterTag := range tags {
				for _, docTag := range doc.Tags {
					if docTag == filterTag {
						hasMatch = true
						break
					}
				}
				if hasMatch {
					break
				}
			}
			if !hasMatch {
				continue
			}
		}

		results = append(results, doc)
	}

	return results, nil
}

// GetDocument retrieves a document by ID
func (c *Client) GetDocument(ctx context.Context, docID string) (*ResearchDocument, error) {
	c.mutex.RLock()
	defer c.mutex.RUnlock()

	doc, exists := c.documents[docID]
	if !exists {
		return nil, fmt.Errorf("document not found: %s", docID)
	}

	return doc, nil
}

// UpdateDocument updates an existing document
func (c *Client) UpdateDocument(ctx context.Context, docID string, updates map[string]interface{}) error {
	if !c.config.Enabled {
		return nil
	}

	c.mutex.Lock()
	defer c.mutex.Unlock()

	doc, exists := c.documents[docID]
	if !exists {
		return fmt.Errorf("document not found: %s", docID)
	}

	// Apply updates
	if title, ok := updates["title"].(string); ok {
		doc.Title = title
	}
	if content, ok := updates["content"].(string); ok {
		doc.Content = content
	}
	if stage, ok := updates["stage"].(string); ok {
		doc.Stage = ResearchStage(stage)
	}
	if docType, ok := updates["doc_type"].(string); ok {
		doc.DocType = DocumentType(docType)
	}
	if tags, ok := updates["tags"].([]string); ok {
		doc.Tags = tags
	}
	if metadata, ok := updates["metadata"].(map[string]interface{}); ok {
		doc.Metadata = metadata
	}

	doc.UpdatedAt = time.Now()

	// Save to file
	if err := c.saveDocumentToFile(doc); err != nil {
		return fmt.Errorf("failed to save updated document: %w", err)
	}

	// Auto-commit if enabled
	if c.config.AutoCommit {
		if err := c.commitDocument(ctx, doc, "Update research document"); err != nil {
			c.logger.WithError(err).Warn("Failed to auto-commit document update")
		}
	}

	c.logger.WithField("doc_id", docID).Info("Updated research document")
	return nil
}

// GetProjectStructure returns the current project structure
func (c *Client) GetProjectStructure(ctx context.Context) *ProjectStructure {
	c.mutex.RLock()
	defer c.mutex.RUnlock()

	if c.projectStructure == nil {
		return c.getDefaultProjectStructure()
	}

	// Return a copy to avoid external modification
	structure := *c.projectStructure
	return &structure
}

// Close closes the opencode client
func (c *Client) Close(ctx context.Context) error {
	c.logger.Info("Closing opencode client")
	return nil
}