import json
from datetime import UTC, datetime

from distill.codec import parse_snapshot
from distill.config import load_config
from distill.pipeline import run_pipeline
from distill.tui import top_rows

FIXTURE = "tests/fixtures/models.json"


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


def test_fixture_end_to_end_publishes_all_artifacts(tmp_path, monkeypatch):
    snapshot = run_pipeline(config_for(tmp_path), source_override="fixture", publish=True)
    current = tmp_path / "out" / "current"
    assert snapshot.status == "complete"
    assert {path.name for path in current.iterdir()} == {
        "snapshot.json",
        "distill.html",
        "manifest.json",
    }
    published = parse_snapshot((current / "snapshot.json").read_bytes())
    manifest = json.loads((current / "manifest.json").read_text(encoding="utf-8"))
    assert published.snapshot_id == snapshot.snapshot_id
    assert manifest["files"]["snapshot.json"]["size"] > 0
    html = (current / "distill.html").read_text(encoding="utf-8")
    assert "Artificial Analysis performance views are unavailable" not in html
    assert "<h2>Models</h2>" not in html


def test_fixed_generated_at_is_deterministic(tmp_path, monkeypatch):
    config = config_for(tmp_path)
    generated_at = datetime(2026, 9, 18, tzinfo=UTC)
    first = run_pipeline(config, generated_at=generated_at, publish=False)
    second = run_pipeline(config, generated_at=generated_at, publish=False)
    assert first.model_dump(mode="json") == second.model_dump(mode="json")


def test_models_json_controls_model_type_tabs(tmp_path):
    app_path = tmp_path / "app.yaml"
    app_path.write_text(
        """output: out
sources: []
""",
        encoding="utf-8",
    )
    (tmp_path / "models.json").write_text(
        '{"model_types": {"llm": ["performance-top5"], "image-to-video": ["performance-per-token-top5"]}}',
        encoding="utf-8",
    )

    config = load_config(app_path)

    assert config.model_type_tabs == {
        "llm": ["performance-top5"],
        "image-to-video": ["performance-per-token-top5"],
    }


def test_org_json_controls_copilot_catalog(tmp_path):
    app_path = tmp_path / "app.yaml"
    app_path.write_text(
        """output: out
copilot:
  enabled: true
sources: []
""",
        encoding="utf-8",
    )
    (tmp_path / "org.json").write_text(
        '{"source": "test-org", "models": [{"name": "GPT Sol", "family": "GPT", "level": "sol", "input_credits_per_million": 10, "output_credits_per_million": 20}]}',
        encoding="utf-8",
    )

    config = load_config(app_path)

    assert config.copilot.source == "test-org"
    assert config.copilot.models[0].family == "GPT"
    assert config.copilot.models[0].level == "sol"


def test_fixed_generated_at_publishes_byte_identical_artifacts(tmp_path, monkeypatch):
    config = config_for(tmp_path)
    generated_at = datetime(2026, 9, 18, tzinfo=UTC)
    run_pipeline(config, generated_at=generated_at, publish=True)
    first = run_pipeline(config, generated_at=generated_at, publish=False)
    second = run_pipeline(config, generated_at=generated_at, publish=False)

    assert first.model_dump(mode="json") == second.model_dump(mode="json")


def test_snapshot_and_tui_use_same_view_order(tmp_path, monkeypatch):
    snapshot = run_pipeline(config_for(tmp_path), publish=False)
    view = next(item for item in snapshot.views if item.view_id == "performance-top5")
    rows = top_rows(snapshot)
    assert [row["model_id"] for row in rows] == view.model_ids
    assert rows == []
