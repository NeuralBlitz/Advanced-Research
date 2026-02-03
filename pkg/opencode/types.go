package opencode

import (
	"time"
)

// WorkflowReport represents a report about the research workflow
type WorkflowReport struct {
	TotalDocuments      int    `json:"total_documents"`
	TotalAreas         int    `json:"total_areas"`
	TotalWorkflows     int    `json:"total_workflows"`
	TotalSemanticTags  int    `json:"total_semantic_tags"`
	LastUpdated        time.Time `json:"last_updated"`
}