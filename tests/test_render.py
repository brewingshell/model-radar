from datetime import UTC, datetime

from model_radar.analysis import attach_benchmarks, build_views, normalize
from model_radar.models import RawRecord, Snapshot
from model_radar.render import (
    model_type,
    model_type_group_categories,
    model_type_groups,
    normalise_model_type_tabs,
    render_html,
    view_model_types,
)


def test_html_escapes_untrusted_model_fields():
    models = normalize(
        [
            RawRecord(
                source_id="x",
                name="<script>alert(1)</script>",
                organization="<b>bad</b>",
                intelligence_index=91.0,
                artificial_analysis_model_url="https://artificialanalysis.ai/models/x",
                provenance=["test"],
            )
        ]
    )
    snapshot = Snapshot(
        snapshot_id="x",
        generated_at=datetime(2026, 9, 18, tzinfo=UTC),
        status="complete",
        source_status=[],
        models=models,
        views=build_views(models),
    )
    html = render_html(snapshot).decode("utf-8")
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "<b>bad</b>" not in html
    assert "Content-Security-Policy" in html
    assert "default-src &#39;none&#39;" in html


def test_html_leads_with_primary_views_and_token_price_label():
    model = normalize(
        [
            RawRecord(
                source_id="aa/atlas",
                name="Atlas",
                intelligence_index=91.5,
                cost_per_task_usd=0.12,
                artificial_analysis_input_price_per_million=0.2,
                artificial_analysis_output_price_per_million=1.2,
                artificial_analysis_model_url="https://artificialanalysis.ai/models/atlas",
                provenance=["artificial_analysis"],
            )
        ]
    )
    snapshot = Snapshot(
        snapshot_id="aa",
        generated_at=datetime(2026, 9, 18, tzinfo=UTC),
        status="complete",
        source_status=[],
        models=model,
        views=build_views(model),
    )

    html = render_html(snapshot).decode("utf-8")

    assert "Performance top 10" in html
    assert "AA Intelligence / weighted USD per 1M tokens" in html
    assert "Cost per benchmark task (USD)" in html
    assert 'href="https://artificialanalysis.ai/models/atlas"' in html
    assert "<h2>Models</h2>" not in html
    assert 'class="stats"' not in html
    assert "Source coverage" not in html
    assert '<body data-theme="dark">' in html
    assert 'id="theme-toggle"' in html
    assert "applyTheme" in html


def test_html_formats_dates_as_day_month_year():
    model = normalize(
        [
            RawRecord(
                source_id="hf/date-model",
                name="Date Model",
                created_at="2026-09-21T12:34:56Z",
                provenance=["huggingface"],
            )
        ]
    )
    snapshot = Snapshot(
        snapshot_id="date",
        generated_at=datetime(2026, 9, 21, 12, 34, 56, tzinfo=UTC),
        status="complete",
        source_status=[],
        models=model,
        views=build_views(model),
    )

    html = render_html(snapshot).decode("utf-8")

    assert "Generated 21 Sep 2026" in html
    assert "21 Sep 2026" in html
    assert "2026-09-21T12:34:56" not in html


def test_html_integrates_benchmark_columns_into_existing_tabs():
    models = normalize(
        [
            RawRecord(
                source_id="aa/model",
                name="Model (max)",
                intelligence_index=50,
                provenance=["artificial_analysis"],
            )
        ]
    )
    models = attach_benchmarks(
        models,
        [
            RawRecord(
                source_id="live/model",
                name="Model (max effort)",
                benchmark_source="livebench",
                benchmark_index=80,
                provenance=["livebench"],
            ),
            RawRecord(
                source_id="deep/model",
                name="Model (max)",
                benchmark_source="deepswe",
                benchmark_index=70,
                provenance=["deepswe"],
            ),
        ],
    )
    snapshot = Snapshot(
        snapshot_id="benchmark-columns",
        generated_at=datetime(2026, 9, 21, tzinfo=UTC),
        status="complete",
        source_status=[],
        models=models,
        views=build_views(models),
    )

    html = render_html(snapshot).decode("utf-8")

    assert "LiveBench</th>" in html
    assert "EvalPlus</th>" not in html
    assert "DeepSWE</th>" not in html
    assert "Benchmark synthesis top 10" not in html


def test_decision_filters_render_model_types_and_open_weight_metadata():
    models = normalize(
        [
            RawRecord(
                source_id="acme/atlas",
                name="Atlas",
                modalities=["text"],
                open_weights=True,
                intelligence_index=80,
                provenance=["test"],
            ),
            RawRecord(
                source_id="acme/pixel",
                name="Pixel",
                capabilities=["text-to-image"],
                open_weights=False,
                intelligence_index=75,
                provenance=["test"],
            ),
            RawRecord(
                source_id="acme/motion",
                name="Motion",
                capabilities=["image-to-video"],
                open_weights=True,
                provenance=["test"],
            ),
        ]
    )
    snapshot = Snapshot(
        snapshot_id="filters",
        generated_at=datetime(2026, 9, 21, tzinfo=UTC),
        status="complete",
        source_status=[],
        models=models,
        views=build_views(models),
        changes={"summary": ["A model changed."]},
    )

    html = render_html(snapshot).decode("utf-8")

    assert '<select class="filter-select" id="decision-modality">' in html
    assert '<option value="llm" data-categories="llm" selected>LLM</option>' in html
    assert (
        '<option value="image" data-categories="text-to-image,image-to-image">Image</option>'
        in html
    )
    assert (
        '<option value="video" data-categories="text-to-video,image-to-video">Video</option>'
        in html
    )
    assert 'data-model-table="text-to-image"' in html
    assert 'data-model-table="image-to-image"' in html
    assert 'data-model-table="text-to-video"' in html
    assert 'data-model-table="image-to-video"' in html
    assert 'id="decision-open-weight"' in html
    assert 'data-model-type="llm" data-open-weight="true"' in html
    assert 'data-model-type="text-to-image" data-open-weight="false"' in html
    assert "open-weight-mark" in html
    assert "--open-row" in html
    assert "Pixel" in html
    assert "Motion" in html
    assert "applyDecisionFilters" in html
    assert "model-type-catalog" not in html
    assert "What's new" in html
    assert 'aria-label="Warnings"' not in html
    by_name = {model.name: model for model in models}
    assert model_type(by_name["Atlas"]) == "llm"
    assert model_type(by_name["Pixel"]) == "text-to-image"
    assert model_type(by_name["Motion"]) == "image-to-video"


def test_image_and_video_modalities_share_one_select_option_each():
    groups = {
        "llm": ["org-copilot-best-top10", "performance-top5"],
        "image": ["performance-top5"],
        "video": ["performance-top5"],
    }

    assert model_type_groups(groups) == ["llm", "image", "video"]
    assert model_type_group_categories("image") == ["text-to-image", "image-to-image"]
    assert model_type_group_categories("video") == ["text-to-video", "image-to-video"]
    assert view_model_types("performance-top5", groups) == ["llm", "image", "video"]
    assert view_model_types("org-copilot-best-top10", groups) == ["llm"]


def test_legacy_per_modality_keys_are_normalised_into_groups():
    legacy = {
        "llm": ["performance-top5"],
        "text-to-image": ["performance-top5"],
        "image-to-image": ["meaningful-new-hf-top5"],
        "text-to-video": ["performance-top5"],
        "image-to-video": ["performance-top5"],
    }

    normalised = normalise_model_type_tabs(legacy)

    assert list(normalised) == ["llm", "image", "video"]
    assert normalised["image"] == ["performance-top5", "meaningful-new-hf-top5"]
    assert normalised["video"] == ["performance-top5"]


def test_decision_views_retain_top_open_weight_candidates():
    models = normalize(
        [
            RawRecord(
                source_id=f"closed/model-{index}",
                name=f"Closed Model {index}",
                intelligence_index=100 - index,
                open_weights=False,
                provenance=["test"],
            )
            for index in range(10)
        ]
        + [
            RawRecord(
                source_id=f"open/model-{index}",
                name=f"Open Model {index}",
                intelligence_index=50 - index,
                open_weights=True,
                provenance=["test"],
            )
            for index in range(10)
        ]
    )

    performance = next(view for view in build_views(models) if view.view_id == "performance-top5")

    assert len(performance.model_ids) == 20


def test_decision_tables_are_wrapped_for_horizontal_scroll():
    models = normalize(
        [
            RawRecord(
                source_id=f"aa/model-{index}",
                name=f"Model {index} (max)",
                intelligence_index=90 - index,
                provenance=["artificial-analysis"],
                artificial_analysis_model_url=f"https://artificialanalysis.ai/models/{index}",
            )
            for index in range(12)
        ]
    )
    snapshot = Snapshot(
        snapshot_id="scroll",
        generated_at=datetime(2026, 9, 21, tzinfo=UTC),
        status="complete",
        source_status=[],
        models=models,
        views=build_views(models),
    )

    html = render_html(snapshot).decode("utf-8")

    assert '<div class="table-scroll"><table class="decision-table"' in html
    assert html.count('<div class="table-scroll">') == html.count("</table></div>")
    assert ".table-scroll{max-width:100%;overflow-x:auto" in html
    assert "nth-child(n+4)" not in html
    assert ".ranking-card{overflow:hidden}" not in html
