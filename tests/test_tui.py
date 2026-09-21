from datetime import UTC, datetime

from model_radar.analysis import build_views, normalize
from model_radar.models import RawRecord, Snapshot
from model_radar.tui import plain_top, top_rows


def test_tui_reads_snapshot_data_without_recomputing_scores():
    models = normalize(
        [
            RawRecord(
                source_id="orbit",
                name="Orbit",
                intelligence_index=90,
                provenance=["artificial_analysis"],
            ),
            RawRecord(
                source_id="atlas",
                name="Atlas",
                intelligence_index=80,
                provenance=["artificial_analysis"],
            ),
        ]
    )
    snapshot = Snapshot(
        snapshot_id="test",
        generated_at=datetime(2026, 9, 18, tzinfo=UTC),
        status="complete",
        source_status=[],
        models=models,
        views=build_views(models),
    )

    rows = top_rows(snapshot)

    performance_view = next(view for view in snapshot.views if view.view_id == "performance-top5")
    assert [row["model_id"] for row in rows] == performance_view.model_ids
    assert plain_top(snapshot).splitlines()[0].endswith("Orbit (90.0)")
