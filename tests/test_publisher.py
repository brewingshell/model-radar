from datetime import UTC, datetime

import pytest

from model_radar.analysis import build_views, normalize
from model_radar.models import RawRecord, Snapshot
from model_radar.publisher import PublicationError, Publisher, _summarize_changes


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


def make_leader_snapshot(models: list[RawRecord], day: int = 18) -> Snapshot:
    normalized = normalize(models)
    return Snapshot(
        snapshot_id=f"day-{day}",
        generated_at=datetime(2026, 9, day, tzinfo=UTC),
        status="complete",
        source_status=[],
        models=normalized,
        views=build_views(normalized),
    )


def test_leaderboard_update_reports_new_leader_and_delta():
    previous = make_leader_snapshot(
        [
            RawRecord(
                source_id="aa/opus-5",
                name="Claude Opus 5",
                intelligence_index=51.0,
                provenance=["artificial-analysis"],
            )
        ]
    )
    current = make_leader_snapshot(
        [
            RawRecord(
                source_id="aa/opus-5",
                name="Claude Opus 5",
                intelligence_index=51.0,
                provenance=["artificial-analysis"],
            ),
            RawRecord(
                source_id="aa/opus-5-5",
                name="Claude Opus 5.5 (max)",
                intelligence_index=58.0,
                provenance=["artificial-analysis"],
            ),
        ]
    )

    changes = _summarize_changes(current, previous)

    leaders = [item for item in changes["items"] if item["kind"] == "leaderboard"]
    assert len(leaders) == 1
    assert leaders[0]["count"] == 1
    assert any(
        "Claude Opus 5.5" in example["text"]
        and "Claude Opus 5" in example["text"]
        and "+7" in example["text"]
        for example in leaders[0]["examples"]
    )


def test_window_and_daily_new_model_cards():
    oldest = make_leader_snapshot(
        [RawRecord(source_id="aa/a", name="Model A", provenance=["artificial-analysis"])],
        day=17,
    )
    previous = make_leader_snapshot(
        [
            RawRecord(source_id="aa/a", name="Model A", provenance=["artificial-analysis"]),
            RawRecord(source_id="aa/b", name="Model B", provenance=["artificial-analysis"]),
        ],
        day=18,
    )
    current = make_leader_snapshot(
        [
            RawRecord(source_id="aa/a", name="Model A", provenance=["artificial-analysis"]),
            RawRecord(source_id="aa/b", name="Model B", provenance=["artificial-analysis"]),
            RawRecord(source_id="aa/c", name="Model C", provenance=["artificial-analysis"]),
        ],
        day=19,
    )

    changes = _summarize_changes(current, previous, oldest)
    items = {item["kind"]: item for item in changes["items"]}

    assert items["new-window"]["count"] == 2
    assert [example["text"] for example in items["new-window"]["examples"]] == ["Model B"]
    assert items["new-window"]["detail"].endswith("since 2026-09-17.")
    assert items["new"]["count"] == 1
    assert [example["text"] for example in items["new"]["examples"]] == ["Model C"]
    assert items["new"]["detail"].endswith("since 2026-09-18.")


def test_window_card_excludes_bubbles_already_in_daily_card():
    oldest = make_leader_snapshot(
        [RawRecord(source_id="aa/a", name="Model A", provenance=["artificial-analysis"])],
        day=17,
    )
    previous = make_leader_snapshot(
        [
            RawRecord(source_id="aa/a", name="Model A", provenance=["artificial-analysis"]),
            RawRecord(source_id="aa/b", name="Model B", provenance=["artificial-analysis"]),
        ],
        day=18,
    )
    current = make_leader_snapshot(
        [
            RawRecord(source_id="aa/a", name="Model A", provenance=["artificial-analysis"]),
            RawRecord(source_id="aa/b", name="Model B", provenance=["artificial-analysis"]),
            RawRecord(source_id="aa/c", name="Model C", provenance=["artificial-analysis"]),
        ],
        day=19,
    )

    items = {item["kind"]: item for item in _summarize_changes(current, previous, oldest)["items"]}
    window_names = {example["text"] for example in items["new-window"]["examples"]}
    daily_names = {example["text"] for example in items["new"]["examples"]}

    assert window_names.isdisjoint(daily_names)


def test_window_card_omitted_when_no_unique_bubbles():
    oldest = make_leader_snapshot(
        [RawRecord(source_id="aa/a", name="Model A", provenance=["artificial-analysis"])],
        day=17,
    )
    previous = make_leader_snapshot(
        [RawRecord(source_id="aa/a", name="Model A", provenance=["artificial-analysis"])],
        day=18,
    )
    current = make_leader_snapshot(
        [
            RawRecord(source_id="aa/a", name="Model A", provenance=["artificial-analysis"]),
            RawRecord(source_id="aa/b", name="Model B", provenance=["artificial-analysis"]),
        ],
        day=19,
    )

    items = {item["kind"]: item for item in _summarize_changes(current, previous, oldest)["items"]}

    assert "new-window" not in items
    assert [example["text"] for example in items["new"]["examples"]] == ["Model B"]


def test_new_model_examples_carry_source_url():
    previous = make_leader_snapshot([], day=18)
    current = make_leader_snapshot(
        [
            RawRecord(
                source_id="openai/gpt-6",
                name="openai/gpt-6",
                source_url="https://huggingface.co/openai/gpt-6",
                provenance=["huggingface"],
            )
        ],
        day=19,
    )

    changes = _summarize_changes(current, previous)

    new_item = next(item for item in changes["items"] if item["kind"] == "new")
    assert new_item["examples"] == [
        {"text": "openai/gpt-6", "url": "https://huggingface.co/openai/gpt-6"}
    ]


def test_window_card_omitted_when_only_one_prior_day():
    previous = make_leader_snapshot(
        [RawRecord(source_id="aa/a", name="Model A", provenance=["artificial-analysis"])],
        day=18,
    )
    current = make_leader_snapshot(
        [
            RawRecord(source_id="aa/a", name="Model A", provenance=["artificial-analysis"]),
            RawRecord(source_id="aa/b", name="Model B", provenance=["artificial-analysis"]),
        ],
        day=19,
    )

    changes = _summarize_changes(current, previous, previous)

    kinds = [item["kind"] for item in changes["items"]]
    assert "new-window" not in kinds
    assert "new" in kinds


def test_failed_publication_preserves_previous_release(tmp_path, monkeypatch):
    publisher = Publisher(tmp_path / "out")
    publisher.publish(make_snapshot("Stable"))
    old_snapshot = (tmp_path / "out" / "current" / "snapshot.json").read_bytes()

    def fail_render(_snapshot):
        raise RuntimeError("render exploded")

    monkeypatch.setattr("model_radar.publisher.render_html", fail_render)
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

    assert (tmp_path / "out" / "model-radar.html").is_file()
    assert (tmp_path / "out" / "snapshot.json").is_file()
    assert (tmp_path / "out" / "manifest.json").is_file()
    assert (tmp_path / "out" / "current").is_dir()
    assert not (tmp_path / "out" / "current").is_symlink()
    assert len(list((tmp_path / "out" / "history").glob("*.json"))) == 2
    html = (tmp_path / "out" / "model-radar.html").read_text(encoding="utf-8")
    assert "What's new" in html


def test_changes_include_structured_items(tmp_path):
    from model_radar.codec import parse_snapshot

    publisher = Publisher(tmp_path / "out")
    publisher.publish(make_snapshot("First", day=18))
    publisher.publish(make_snapshot("Second", day=19))

    published = parse_snapshot((tmp_path / "out" / "current" / "snapshot.json").read_bytes())
    kinds = {item["kind"] for item in published.changes["items"]}
    assert "new" in kinds
    assert "source" not in kinds
    assert all("label" in item and "detail" in item for item in published.changes["items"])


def test_history_replaces_same_day_snapshot(tmp_path):
    publisher = Publisher(tmp_path / "out")
    publisher.publish(make_snapshot("First", day=18))
    publisher.publish(make_snapshot("Second", day=18))

    history = list((tmp_path / "out" / "history").glob("*.json"))
    assert [path.name for path in history] == ["2026-09-18.json"]
    assert b"second" in history[0].read_bytes()
