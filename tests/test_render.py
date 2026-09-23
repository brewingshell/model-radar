from datetime import UTC, datetime

from model_radar.analysis import attach_benchmarks, build_views, normalize
from model_radar.models import RawRecord, Snapshot
from model_radar.render import (
    format_size_gb,
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
    assert 'data-model-table="image-to-video"' in html
    assert 'data-model-table="image-to-image"' not in html
    assert 'data-model-table="text-to-video"' not in html
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
    assert 'class="change-list"' in html
    by_name = {model.name: model for model in models}
    assert model_type(by_name["Atlas"]) == "llm"
    assert model_type(by_name["Pixel"]) == "text-to-image"
    assert model_type(by_name["Motion"]) == "image-to-video"


def test_whats_new_renders_structured_cards():
    models = normalize(
        [
            RawRecord(
                source_id="aa/best",
                name="Best Power",
                intelligence_index=60.0,
                provenance=["artificial-analysis"],
            )
        ]
    )
    snapshot = Snapshot(
        snapshot_id="changes",
        generated_at=datetime(2026, 9, 21, tzinfo=UTC),
        status="complete",
        source_status=[],
        models=models,
        views=build_views(models),
        changes={
            "summary": ["2 model records entered the catalog."],
            "items": [
                {
                    "kind": "new",
                    "label": "New models",
                    "count": 2,
                    "detail": "2 model records entered the catalog.",
                    "examples": ["Alpha", "Beta"],
                },
                {
                    "kind": "leaderboard",
                    "label": "Leaderboard update",
                    "detail": "Claude Opus 5.5 overtook Claude Opus 5 on Performance, 58 vs 51 (+7).",
                    "examples": ["Claude Opus 5", "Claude Opus 5.5"],
                },
                {
                    "kind": "quiet",
                    "label": "No changes",
                    "detail": "No material model or shortlist changes were detected.",
                    "examples": [],
                },
            ],
        },
    )

    html = render_html(snapshot).decode("utf-8")

    assert 'class="change-grid"' in html
    assert 'class="change-card change-new"' in html
    assert 'class="change-card change-leaderboard"' in html
    assert 'class="change-card change-quiet"' in html
    assert 'class="change-badge">New models' in html
    assert 'class="change-badge">Leaderboard update' in html
    assert 'class="change-count">2' in html
    assert ">Alpha<" in html
    assert 'class="change-list"' not in html


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


def test_modality_tabs_split_per_modality():
    models = normalize(
        [
            RawRecord(
                source_id=f"aa/{category}-{index}",
                name=f"{category} Model {index}",
                capabilities=[category],
                artificial_analysis_modality=category,
                artificial_analysis_modality_elo=1200 - index,
                intelligence_index=90 - index,
                provenance=["artificial-analysis"],
            )
            for category in ("text-to-image", "image-to-image", "text-to-video", "image-to-video")
            for index in range(3)
        ]
    )
    snapshot = Snapshot(
        snapshot_id="modality",
        generated_at=datetime(2026, 9, 21, tzinfo=UTC),
        status="complete",
        source_status=[],
        models=models,
        views=build_views(models),
    )

    html = render_html(snapshot).decode("utf-8")

    assert "Performance t2i top 10" in html
    assert "Performance i2i top 10" in html
    assert "Performance t2v top 10" in html
    assert "Performance i2v top 10" in html
    assert 'data-model-table="text-to-image"' in html
    assert 'data-model-table="image-to-image"' in html
    assert 'data-model-table="text-to-video"' in html
    assert 'data-model-table="image-to-video"' in html
    assert html.count('data-model-types="image"') >= 4
    assert html.count('data-model-types="video"') >= 4
    assert "modality-tabs" not in html
    assert "modality-tab" not in html
    assert "tab-panel is-active" in html


def test_llm_tabs_have_no_modality_suffix():
    models = normalize(
        [
            RawRecord(
                source_id=f"aa/model-{index}",
                name=f"Model {index}",
                intelligence_index=90 - index,
                provenance=["artificial-analysis"],
            )
            for index in range(5)
        ]
    )
    snapshot = Snapshot(
        snapshot_id="llm-tabs",
        generated_at=datetime(2026, 9, 21, tzinfo=UTC),
        status="complete",
        source_status=[],
        models=models,
        views=build_views(models),
    )

    html = render_html(snapshot).decode("utf-8")

    assert "Performance top 10" in html
    assert "Performance t2i top 10" not in html
    assert 'data-model-types="llm"' in html


def test_tiny_and_on_device_tabs_render_for_llm():
    models = normalize(
        [
            RawRecord(
                source_id="aa/k2-7b",
                name="K2 Horizon 7B",
                intelligence_index=21.0,
                provenance=["artificial-analysis"],
            ),
            RawRecord(
                source_id="Cactus-Compute/needle2",
                name="Cactus-Compute/needle2",
                capabilities=["on-device", "tool-calling"],
                downloads=32939,
                provenance=["huggingface"],
            ),
        ]
    )
    snapshot = Snapshot(
        snapshot_id="tiny",
        generated_at=datetime(2026, 9, 21, tzinfo=UTC),
        status="complete",
        source_status=[],
        models=models,
        views=build_views(models),
    )

    html = render_html(snapshot).decode("utf-8")

    assert ">Tiny LLM top 10<" in html
    assert ">On-device models top 10<" in html
    assert "Cactus-Compute/needle2" in html
    assert "K2 Horizon 7B" in html
    assert 'data-model-table="llm"' in html


def test_highlight_cards_show_top_llm_picks():
    models = normalize(
        [
            RawRecord(
                source_id="aa/best",
                name="Best Power",
                intelligence_index=60.0,
                provenance=["artificial-analysis"],
            ),
            RawRecord(
                source_id="aa/tiny",
                name="Tiny Star 4B",
                intelligence_index=20.0,
                provenance=["artificial-analysis"],
            ),
            RawRecord(
                source_id="aa/value",
                name="Value Star",
                intelligence_index=40.0,
                provenance=["artificial-analysis"],
            ),
        ]
    )
    snapshot = Snapshot(
        snapshot_id="highlights",
        generated_at=datetime(2026, 9, 21, tzinfo=UTC),
        status="complete",
        source_status=[],
        models=models,
        views=build_views(models),
    )

    html = render_html(snapshot).decode("utf-8")

    assert 'class="highlights"' in html
    assert "Best power LLM" in html
    assert "Best value per token" in html
    assert "Best tiny LLM" in html
    assert "Best Power" in html
    assert "Tiny Star 4B" in html
    assert 'data-goto-tab="' in html
    assert ".highlight:hover{border-color:var(--teal)}" in html


def test_format_size_gb_estimates_fp16_weights():
    assert format_size_gb(7.0) == "14.0"
    assert format_size_gb(2.6) == "5.2"
    assert format_size_gb(0.5) == "1.0"
    assert format_size_gb(None) == "unknown"


def test_tiny_llm_table_shows_params_size_and_benchmark():
    models = normalize(
        [
            RawRecord(
                source_id="aa/k2-7b",
                name="K2 Horizon 7B",
                intelligence_index=21.0,
                provenance=["artificial-analysis"],
            ),
            RawRecord(
                source_id="aa/lfm-2-6b",
                name="LFM2.5-2.6B",
                intelligence_index=8.0,
                provenance=["artificial-analysis"],
            ),
        ]
    )
    snapshot = Snapshot(
        snapshot_id="tiny-cols",
        generated_at=datetime(2026, 9, 21, tzinfo=UTC),
        status="complete",
        source_status=[],
        models=models,
        views=build_views(models),
    )

    html = render_html(snapshot).decode("utf-8")
    panel = html.split("<h3>Tiny LLM top 10</h3>", 1)[1]
    tiny = panel.split("</table>", 1)[0]

    assert "<th>Params (B)</th>" in tiny
    assert "<th>Size (GB)</th>" in tiny
    assert "<th>Benchmark (AA Index)</th>" in tiny
    assert "<td>7.0</td>" in tiny
    assert "<td>14.0</td>" in tiny
    assert "<td>5.2</td>" in tiny


def test_mini_llm_table_shows_size_and_unknown_benchmark():
    models = normalize(
        [
            RawRecord(
                source_id="aa/qwen-0-6b",
                name="Qwen3-0.6B",
                intelligence_index=4.0,
                provenance=["artificial-analysis"],
            ),
            RawRecord(
                source_id="Cactus-Compute/needle3",
                name="Cactus-Compute/needle3",
                capabilities=["on-device", "edge"],
                downloads=54528,
                provenance=["huggingface"],
            ),
        ]
    )
    snapshot = Snapshot(
        snapshot_id="mini-cols",
        generated_at=datetime(2026, 9, 21, tzinfo=UTC),
        status="complete",
        source_status=[],
        models=models,
        views=build_views(models),
    )

    html = render_html(snapshot).decode("utf-8")
    panel = html.split("<h3>Mini LLM top 10</h3>", 1)[1].split("</table>", 1)[0]

    assert "Best mini LLM" in html
    assert "<th>Params (B)</th>" in panel
    assert "<th>Size (GB)</th>" in panel
    assert "<th>Benchmark (AA Index)</th>" in panel
    assert "<td>0.6</td>" in panel
    assert "<td>1.2</td>" in panel
    assert "Cactus-Compute/needle3" in panel


def test_on_device_table_shows_params_size_and_matched_benchmark():
    models = normalize(
        [
            RawRecord(
                source_id="aa/minicpm-2b",
                name="MiniCPM5-2B",
                intelligence_index=12.0,
                provenance=["artificial-analysis"],
            ),
            RawRecord(
                source_id="openbmb/MiniCPM5-2B",
                name="openbmb/MiniCPM5-2B",
                capabilities=["on-device", "tool-calling"],
                downloads=460533,
                provenance=["huggingface"],
            ),
            RawRecord(
                source_id="Cactus-Compute/needle2",
                name="Cactus-Compute/needle2",
                capabilities=["on-device", "edge"],
                downloads=32939,
                provenance=["huggingface"],
            ),
        ]
    )
    snapshot = Snapshot(
        snapshot_id="edge-cols",
        generated_at=datetime(2026, 9, 21, tzinfo=UTC),
        status="complete",
        source_status=[],
        models=models,
        views=build_views(models),
    )

    html = render_html(snapshot).decode("utf-8")
    panel = html.split("<h3>On-device models top 10</h3>", 1)[1].split("</table>", 1)[0]

    assert "<th>Params (B)</th>" in panel
    assert "<th>Size (GB)</th>" in panel
    assert "<th>Benchmark (AA Index)</th>" in panel
    assert "<th>LiveBench</th>" in panel
    assert "<th>Downloads</th>" in panel
    assert "<td>12.0</td>" in panel
    assert "<td>4.0</td>" in panel
    assert "Cactus-Compute/needle2" in panel


def test_benchmark_enrichment_matches_by_family_name():
    from model_radar.render import _benchmark_index, benchmark_enrichment

    models = normalize(
        [
            RawRecord(
                source_id="aa/minicpm-2b",
                name="MiniCPM5-2B",
                intelligence_index=12.0,
                provenance=["artificial-analysis"],
            ),
            RawRecord(
                source_id="openbmb/MiniCPM5-2B",
                name="openbmb/MiniCPM5-2B",
                provenance=["huggingface"],
            ),
            RawRecord(
                source_id="Cactus-Compute/needle2",
                name="Cactus-Compute/needle2",
                provenance=["huggingface"],
            ),
        ]
    )
    models = attach_benchmarks(
        models,
        [
            RawRecord(
                source_id="livebench:minicpm5-2b",
                name="minicpm5-2b",
                benchmark_source="livebench",
                benchmark_index=55.0,
                provenance=["livebench"],
            )
        ],
    )

    enriched = benchmark_enrichment(models, _benchmark_index(models))

    hf = next(model for model in models if model.slug == "openbmb-minicpm5-2b")
    needle = next(model for model in models if model.slug == "cactus-compute-needle2")
    assert enriched[hf.model_id] == (12.0, 55.0)
    assert needle.model_id not in enriched
