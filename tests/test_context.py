"""Regression tests for context ordering, expiration and size accounting."""

from datetime import datetime, timedelta

import pytest

from advanced_research.core.context import ContextInjector, Priority


def test_context_uses_priority_then_newest_first():
    injector = ContextInjector(max_context_size=1000)
    injector.add_context("old", "old content", Priority.HIGH)
    injector.add_context("new", "new content", Priority.HIGH)
    injector.add_context("lower", "lower content", Priority.MEDIUM)

    now = datetime.now()
    injector.context_blocks["old"].timestamp = now - timedelta(days=2)
    injector.context_blocks["new"].timestamp = now - timedelta(minutes=1)
    injector.context_blocks["lower"].timestamp = now

    rendered = injector.get_context()
    assert rendered.index("## new") < rendered.index("## old")
    assert rendered.index("## old") < rendered.index("## lower")
    assert [b.key for b in injector.get_context_blocks()] == [
        "new", "old", "lower"
    ]


def test_context_limit_counts_headers_and_separators():
    injector = ContextInjector(max_context_size=1000)
    injector.add_context("first", "A", Priority.HIGH)
    injector.add_context("second", "B", Priority.MEDIUM)

    exact = "## first\nA\n\n## second\nB\n"
    assert injector.get_context(max_tokens=len(exact)) == exact
    truncated = injector.get_context(max_tokens=len(exact) - 1)
    assert truncated == "## first\nA\n"
    assert len(truncated) <= len(exact) - 1
    assert injector.get_context(max_tokens=0) == ""


def test_context_can_skip_oversized_high_priority_block():
    injector = ContextInjector(max_context_size=1000)
    injector.add_context("huge", "X" * 100, Priority.HIGH)
    injector.add_context("tiny", "z", Priority.LOW)

    allowed = len("## tiny\nz\n")
    assert injector.get_context(max_tokens=allowed) == "## tiny\nz\n"


def test_context_rejects_negative_limit():
    injector = ContextInjector()
    with pytest.raises(ValueError, match="non-negative"):
        injector.get_context(max_tokens=-1)


def test_expired_blocks_filtered_from_context_and_statistics():
    injector = ContextInjector()
    injector.add_context("expired", "old", Priority.HIGH, tags=["history"])
    injector.context_blocks["expired"].expires = datetime.now() - timedelta(seconds=1)

    assert injector.get_context() == ""
    assert injector.get_context_blocks(filter_tags=["history"]) == []
    assert injector.get_statistics()["total_blocks"] == 0


def test_statistics_report_oldest_and_newest_ages_correctly():
    injector = ContextInjector()
    injector.add_context("old", "a")
    injector.add_context("new", "b")
    now = datetime.now()
    injector.context_blocks["old"].timestamp = now - timedelta(days=2)
    injector.context_blocks["new"].timestamp = now - timedelta(minutes=1)

    stats = injector.get_statistics()
    assert stats["total_blocks"] == 2
    assert stats["oldest_block_age"] > stats["newest_block_age"]
    assert stats["oldest_block_age"] >= 60 * 60
    assert stats["newest_block_age"] < 60 * 60
