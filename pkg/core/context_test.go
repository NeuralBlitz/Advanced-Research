package context

import (
    stdcontext "context"
    "strings"
    "sync"
    "testing"
    "time"
)

func TestGetContextOrdersByPriority(t *testing.T) {
    cfg := DefaultConfig()
    injector := NewContextInjector(cfg)
    defer injector.Close()

    ctx := stdcontext.Background()
    if err := injector.AddContext(ctx, "low", "low content", PriorityLow); err != nil {
        t.Fatal(err)
    }
    if err := injector.AddContext(ctx, "high", "high content", PriorityHigh); err != nil {
        t.Fatal(err)
    }

    result := injector.GetContext(ctx, nil)
    high := strings.Index(result, "## high")
    low := strings.Index(result, "## low")
    if high < 0 || low < 0 || high > low {
        t.Fatalf("expected high-priority context first, got: %q", result)
    }
}

func TestGetContextExcludesExpired(t *testing.T) {
    cfg := DefaultConfig()
    cfg.DefaultTTL = 0
    cfg.CleanupInterval = time.Hour
    injector := NewContextInjector(cfg)
    defer injector.Close()

    ctx := stdcontext.Background()
    if err := injector.AddContext(
        ctx, "expired", "not visible", PriorityHigh,
        WithExpires(time.Now().Add(-time.Minute)),
    ); err != nil {
        t.Fatal(err)
    }

    if got := injector.GetContext(ctx, nil); got != "" {
        t.Fatalf("expected no expired context, got: %q", got)
    }
}

func TestConcurrentContextReadersWithExpiredBlock(t *testing.T) {
    cfg := DefaultConfig()
    cfg.DefaultTTL = 0
    cfg.CleanupInterval = time.Hour
    injector := NewContextInjector(cfg)
    defer injector.Close()

    ctx := stdcontext.Background()
    if err := injector.AddContext(
        ctx, "expired", "not visible", PriorityHigh,
        WithExpires(time.Now().Add(-time.Minute)),
    ); err != nil {
        t.Fatal(err)
    }

    var wg sync.WaitGroup
    for i := 0; i < 12; i++ {
        wg.Add(1)
        go func() {
            defer wg.Done()
            for j := 0; j < 30; j++ {
                if got := injector.GetContext(ctx, nil); got != "" {
                    t.Errorf("expired block appeared in context: %q", got)
                    return
                }
            }
        }()
    }
    wg.Wait()
}
