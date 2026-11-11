"""Python CLI must report mock integrations without false real-world success."""
from pathlib import Path

import yaml
from typer.testing import CliRunner

from advanced_research.cli import app

runner = CliRunner()


def test_initialize_writes_disabled_mock_integrations(tmp_path: Path):
    path = tmp_path / "config.yaml"
    result = runner.invoke(app, ["init", "--config", str(path)])
    assert result.exit_code == 0, result.output
    cfg = yaml.safe_load(path.read_text())
    for name in ("lrs", "opencode", "neuralblitz"):
        assert cfg[name]["enabled"] is False
        assert cfg[name]["mode"] == "mock"
    assert "initialized" in result.output.lower()
    assert "simulated" in result.output.lower()


def test_test_integrations_labels_mock_behavior(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({
        "lrs": {"enabled": True, "mode": "mock"},
        "opencode": {"enabled": True, "mode": "mock"},
        "neuralblitz": {"enabled": True, "mode": "mock"},
    }))
    result = runner.invoke(app, ["test-integrations", "--config", str(path)])
    assert result.exit_code == 0, result.output
    assert "MOCK" in result.output
    assert "not tested" in result.output.lower()
    assert "integration test passed" not in result.output.lower()


def test_test_integrations_real_mode_fails_closed(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({
        "lrs": {"enabled": True, "mode": "real"},
        "opencode": {"enabled": False, "mode": "mock"},
        "neuralblitz": {"enabled": False, "mode": "mock"},
    }))
    result = runner.invoke(app, ["test-integrations", "--config", str(path)])
    assert result.exit_code != 0
    assert "not implemented" in result.output.lower()
