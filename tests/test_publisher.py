from datetime import UTC, datetime

import pytest

from distill.analysis import build_views, normalize
from distill.models import RawRecord, Snapshot
from distill.publisher import PublicationError, Publisher


def make_snapshot(name: str = "Stable", day: int = 18) -> Snapshot:
    models = normalize(
        [RawRecord(source_id=name, name=name, benchmarks={"score": 80}, provenance=["test"])]
    )
    return Snapshot(
        snapshot_id=name.lower(),
        generated_at=datetime(2026, 9, day, tzinfo=UTC),
        status="complete",
        source_status=[],
        models=models,
        views=build_views(models),
    )


def test_failed_publication_preserves_previous_release(tmp_path, monkeypatch):
    publisher = Publisher(tmp_path / "out")
    publisher.publish(make_snapshot("Stable"))
    old_snapshot = (tmp_path / "out" / "current" / "snapshot.json").read_bytes()

    def fail_render(_snapshot):
        raise RuntimeError("render exploded")

    monkeypatch.setattr("distill.publisher.render_html", fail_render)
    with pytest.raises(PublicationError, match="render exploded"):
        publisher.publish(make_snapshot("Replacement"))

    assert (tmp_path / "out" / "current" / "snapshot.json").read_bytes() == old_snapshot
    assert not (tmp_path / "out" / ".publish.lock").exists()


def test_successful_publication_replaces_previous_release(tmp_path):
    publisher = Publisher(tmp_path / "out")
    publisher.publish(make_snapshot("First"))
    publisher.publish(make_snapshot("Second"))
    assert b"second" in (tmp_path / "out" / "current" / "snapshot.json").read_bytes()


def test_existing_lock_has_distinct_publication_failure(tmp_path):
    publisher = Publisher(tmp_path / "out")
    publisher.output.mkdir()
    publisher.lock.write_text("held", encoding="utf-8")
    with pytest.raises(Exception, match="publication lock"):
        publisher.publish(make_snapshot())


def test_artifact_validation_failure_preserves_previous_release(tmp_path, monkeypatch):
    publisher = Publisher(tmp_path / "out")
    publisher.publish(make_snapshot("Stable"))
    old_snapshot = (tmp_path / "out" / "current" / "snapshot.json").read_bytes()

    def fail_validation(_stage, _snapshot):
        raise PublicationError("manifest validation failed")

    monkeypatch.setattr(publisher, "_validate_artifacts", fail_validation)
    with pytest.raises(PublicationError, match="manifest validation failed"):
        publisher.publish(make_snapshot("Replacement"))

    assert (tmp_path / "out" / "current" / "snapshot.json").read_bytes() == old_snapshot


def test_configured_html_limit_is_enforced(tmp_path):
    publisher = Publisher(tmp_path / "out", max_html_bytes=10)

    with pytest.raises(PublicationError, match="HTML exceeds"):
        publisher.publish(make_snapshot())


def test_publication_writes_permanent_root_and_history(tmp_path):
    publisher = Publisher(tmp_path / "out")
    publisher.publish(make_snapshot("First", day=18))
    publisher.publish(make_snapshot("Second", day=19))

    assert (tmp_path / "out" / "distill.html").is_file()
    assert (tmp_path / "out" / "snapshot.json").is_file()
    assert (tmp_path / "out" / "manifest.json").is_file()
    assert (tmp_path / "out" / "current").is_dir()
    assert not (tmp_path / "out" / "current").is_symlink()
    assert len(list((tmp_path / "out" / "history").glob("*.json"))) == 2
    html = (tmp_path / "out" / "distill.html").read_text(encoding="utf-8")
    assert "What's new" in html


def test_history_replaces_same_day_snapshot(tmp_path):
    publisher = Publisher(tmp_path / "out")
    publisher.publish(make_snapshot("First", day=18))
    publisher.publish(make_snapshot("Second", day=18))

    history = list((tmp_path / "out" / "history").glob("*.json"))
    assert [path.name for path in history] == ["2026-09-18.json"]
    assert b"second" in history[0].read_bytes()
