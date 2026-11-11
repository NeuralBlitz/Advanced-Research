"""Issue #70: simulated integrations must never impersonate real side effects."""

import asyncio

import pytest

from advanced_research.core.integrations import (
    IntegrationNotImplementedError,
    IntegrationsManager,
    LRSIntegration,
    NeuralBlitzIntegration,
    OpencodeIntegration,
)


def run(awaitable):
    return asyncio.run(awaitable)


def test_integrations_default_to_disabled_mock_mode():
    manager = IntegrationsManager({})
    for name in ("lrs", "opencode", "neuralblitz"):
        service = manager.get_integration(name)
        assert service.enabled is False
        assert service.mode == "mock"
    run(manager.initialize_all())
    run(manager.shutdown_all())


def test_mock_lrs_flush_does_not_claim_or_perform_delivery(capsys):
    lrs = LRSIntegration({"enabled": True, "mode": "mock"})
    run(lrs.initialize())
    run(lrs.record_learning_event({"account": {"name": "local"}}, "tested", {"id": "mock:test"}))
    assert len(lrs.learning_records) == 1
    run(lrs.flush_records())
    assert len(lrs.learning_records) == 1
    assert "0 sent" in capsys.readouterr().out


def test_real_lrs_fails_before_record_creation():
    lrs = LRSIntegration({"enabled": True, "mode": "real", "endpoint": "https://example.invalid/xapi"})
    with pytest.raises(IntegrationNotImplementedError, match="initialize"):
        run(lrs.initialize())
    with pytest.raises(IntegrationNotImplementedError, match="record_learning_event"):
        run(lrs.record_learning_event({"actor": "local"}, "tested", {"id": "test"}))
    assert lrs.learning_records == []


def test_mock_research_documents_not_presented_as_persisted(capsys):
    service = OpencodeIntegration({"enabled": True, "mode": "mock", "workspace": "research"})
    run(service.initialize())
    ident = run(service.create_research_document("A", "text", "Ideas", ["research"]))
    assert ident.startswith("mock:")
    assert "not persisted" in capsys.readouterr().out
    hits = run(service.search_documents("test"))
    assert hits and all(hit.get("mock") is True for hit in hits)
    assert all(hit["id"].startswith("mock:") for hit in hits)


def test_real_opencode_fails_closed():
    service = OpencodeIntegration({"enabled": True, "mode": "real"})
    with pytest.raises(IntegrationNotImplementedError, match="initialize"):
        run(service.initialize())
    with pytest.raises(IntegrationNotImplementedError, match="create_research_document"):
        run(service.create_research_document("A", "text", "Ideas", []))
    with pytest.raises(IntegrationNotImplementedError, match="search_documents"):
        run(service.search_documents("foo"))


def test_geometric_results_label_constants_as_simulated():
    service = NeuralBlitzIntegration({"enabled": True, "mode": "mock"})
    result = run(service.compute_geometric_features("arbitrary input"))
    assert result["metadata"]["mock"] is True
    assert result["metadata"]["status"] == "simulated"
    assert result["metadata"]["computation_time"] is None
    optimizer = run(service.optimize_on_manifold(None, [1.0], {}))
    assert optimizer["mock"] is True
    assert optimizer["status"] == "simulated"


def test_real_geometric_computation_fails_closed():
    service = NeuralBlitzIntegration({"enabled": True, "mode": "real"})
    with pytest.raises(IntegrationNotImplementedError, match="initialize"):
        run(service.initialize())
    with pytest.raises(IntegrationNotImplementedError, match="compute_geometric_features"):
        run(service.compute_geometric_features([[1.0, 0.0]]))
    with pytest.raises(IntegrationNotImplementedError, match="optimize_on_manifold"):
        run(service.optimize_on_manifold(None, [1.0], {}))


def test_unknown_integration_mode_rejected():
    with pytest.raises(ValueError, match="mode"):
        LRSIntegration({"enabled": True, "mode": "unknown"})
