package pipeline

import (
    "strings"
    "testing"
)

func TestGenerateFinalReportIncludesEveryStage(t *testing.T) {
    p := NewAutomatedPipeline(
        &PipelineConfig{
            Name: "test-pipeline",
            Stages: []PipelineStage{StageIdeation, StageSynthesis},
        },
        nil, nil,
    )
    p.metrics.TotalTasks = 2
    p.metrics.CompletedTasks = 1
    p.metrics.FailedTasks = 1
    p.tasks["idea"] = &PipelineTask{ID: "idea", Stage: StageIdeation, Status: StatusCompleted}
    p.tasks["synthesis"] = &PipelineTask{ID: "synthesis", Stage: StageSynthesis, Status: StatusFailed}

    report := p.generateFinalReport()
    for _, expected := range []string{
        "test-pipeline",
        "Success Rate: 50.0%",
        "ideation: 1/1 tasks completed",
        "synthesis: 0/1 tasks completed",
        "**Generated**:",
    } {
        if !strings.Contains(report, expected) {
            t.Errorf("expected report to include %q, got: %s", expected, report)
        }
    }
    if got := p.generateExecutionSummary(); !strings.Contains(got, "synthesis: 0/1 tasks completed") {
        t.Fatalf("execution summary missing final stage: %s", got)
    }
}

func TestGenerateFinalReportWithNoTasks(t *testing.T) {
    p := NewAutomatedPipeline(
        &PipelineConfig{Name: "empty", Stages: []PipelineStage{StageIdeation}},
        nil, nil,
    )
    result := p.generateFinalReport()
    if !strings.Contains(result, "Success Rate: 0.0%") {
        t.Fatalf("empty pipeline should report zero success rate: %s", result)
    }
    if strings.Contains(result, "NaN") || strings.Contains(result, "+Inf") {
        t.Fatalf("empty pipeline report contains invalid number: %s", result)
    }
}
