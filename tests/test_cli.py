import sys

import pytest
from typer.testing import CliRunner

from model_radar.cli import app, main
from model_radar.config import load_config
from model_radar.pipeline import run_pipeline

FIXTURE = "tests/fixtures/models.json"
runner = CliRunner()


def config_for(tmp_path):
    path = tmp_path / "app.yaml"
    path.write_text(
        f"""output: {tmp_path / "out"}
fetch:
  max_pages: 10
  max_records: 100
sources:
  - name: fixture
    kind: fixture
    required: true
    path: {FIXTURE}
""",
        encoding="utf-8",
    )
    return load_config(path)


def test_inspect_dumps_full_snapshot(tmp_path):
    run_pipeline(config_for(tmp_path), source_override="fixture", publish=True)
    result = runner.invoke(app, ["inspect", "--output", str(tmp_path / "out")])
    assert result.exit_code == 0
    assert '"snapshot_id"' in result.output


def test_inspect_prints_single_model_with_provenance(tmp_path):
    run_pipeline(config_for(tmp_path), source_override="fixture", publish=True)
    result = runner.invoke(app, ["inspect", "atlas-7b", "--output", str(tmp_path / "out")])
    assert result.exit_code == 0
    assert '"atlas-7b"' in result.output
    assert '"provenance"' in result.output
    assert '"field_provenance"' in result.output


def test_inspect_matches_by_name(tmp_path):
    run_pipeline(config_for(tmp_path), source_override="fixture", publish=True)
    result = runner.invoke(app, ["inspect", "Atlas 7B", "--output", str(tmp_path / "out")])
    assert result.exit_code == 0
    assert '"atlas-7b"' in result.output


def test_inspect_unknown_model_exits_four(tmp_path):
    run_pipeline(config_for(tmp_path), source_override="fixture", publish=True)
    result = runner.invoke(app, ["inspect", "nope", "--output", str(tmp_path / "out")])
    assert result.exit_code == 4
    assert "model not found" in result.output


def test_inspect_missing_snapshot_exits_four(tmp_path):
    result = runner.invoke(app, ["inspect", "--output", str(tmp_path / "missing")])
    assert result.exit_code == 4
    assert "Traceback" not in result.output


def test_sources_lists_sources(tmp_path):
    config_for(tmp_path)
    result = runner.invoke(app, ["sources", "--config", str(tmp_path / "app.yaml")])
    assert result.exit_code == 0
    assert "fixture" in result.output


def test_sources_bad_config_exits_78(tmp_path):
    result = runner.invoke(app, ["sources", "--config", str(tmp_path / "missing.yaml")])
    assert result.exit_code == 78
    assert "Traceback" not in result.output


def test_main_usage_error_exits_64(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["model-radar", "run", "--bad-flag"])
    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 64


def test_main_propagates_documented_exit_code(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["model-radar", "inspect", "nope", "--output", str(tmp_path)])
    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 4


def test_main_returns_cleanly_on_success(monkeypatch, tmp_path):
    run_pipeline(config_for(tmp_path), source_override="fixture", publish=True)
    monkeypatch.setattr(
        sys,
        "argv",
        ["model-radar", "inspect", "--output", str(tmp_path / "out")],
    )
    main()
