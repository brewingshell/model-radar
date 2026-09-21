from datetime import UTC, datetime

from model_radar.analysis import (
    _benchmark_family_key,
    _edge_model_ids,
    _parameter_size_b,
    apply_copilot_catalog,
    attach_benchmarks,
    build_views,
    normalize,
)
from model_radar.models import CopilotConfig, CopilotModelConfig, RawRecord


def test_live_metadata_normalizes_and_builds_honest_views():
    records = [
        RawRecord(
            source_id="acme/expensive",
            name="Expensive",
            source_url="https://openrouter.ai/acme/expensive",
            updated_at="2026-09-17T00:00:00Z",
            context_length=32768,
            price_per_million=1.2,
            output_price_per_million=4.0,
            downloads=100,
            likes=10,
            tool_calling=True,
            supported_parameters=["tools"],
            provenance=["openrouter"],
        ),
        RawRecord(
            source_id="acme/long",
            name="Long",
            source_url="https://huggingface.co/acme/long",
            updated_at="2026-09-16T00:00:00Z",
            context_length=131072,
            price_per_million=0.3,
            downloads=1000,
            likes=5,
            provenance=["huggingface"],
        ),
    ]

    models = normalize(records)
    views = {view.view_id: view for view in build_views(models)}

    assert views["performance-top5"].model_ids == []
    assert views["performance-top5"].degraded is True
    assert views["performance-top5"].annotations["status"] == "unavailable"
    assert views["performance-per-token-top5"].model_ids == []
    assert views["performance-per-token-top5"].degraded is True
    assert views["popular"].model_ids == [models[1].model_id, models[0].model_id]
    assert views["lowest-price"].model_ids == [models[1].model_id, models[0].model_id]
    assert views["long-context"].model_ids == [models[1].model_id, models[0].model_id]
    assert views["tool-calling"].model_ids == [models[0].model_id]
    assert models[0].source_urls == ["https://openrouter.ai/acme/expensive"]
    assert models[0].output_price_per_million == 4.0
    assert models[0].recent_signals["downloads"] == 100.0


def test_primary_performance_views_rank_by_aa_and_task_cost_efficiency():
    records = [
        RawRecord(
            source_id=f"aa/{name.lower()}",
            name=name,
            intelligence_index=score,
            cost_per_task_usd=cost,
            artificial_analysis_input_price_per_million=cost,
            artificial_analysis_output_price_per_million=cost * 2 if cost is not None else None,
            artificial_analysis_model_url=f"https://artificialanalysis.ai/models/{name.lower()}",
            provenance=["artificial_analysis"],
        )
        for name, score, cost in [
            ("Alpha", 100.0, 2.0),
            ("Bravo", 90.0, 1.0),
            ("Charlie", 80.0, 0.5),
            ("Delta", 70.0, 3.0),
            ("Echo", 60.0, 0.0),
            ("Foxtrot", 50.0, None),
        ]
    ]
    models = normalize(records)
    views = {view.view_id: view for view in build_views(models)}
    names = {model.model_id: model.name for model in models}

    assert [names[model_id] for model_id in views["performance-top5"].model_ids] == [
        "Alpha",
        "Bravo",
        "Charlie",
        "Delta",
        "Echo",
        "Foxtrot",
    ]
    assert [names[model_id] for model_id in views["performance-per-token-top5"].model_ids] == [
        "Charlie",
        "Bravo",
        "Alpha",
        "Delta",
    ]
    assert views["performance-top5"].degraded is False


def test_meaningful_new_hf_view_excludes_noise_and_requires_adoption():
    records = [
        RawRecord(
            source_id="acme/newest",
            name="acme/newest",
            created_at="2026-09-18T00:00:00Z",
            updated_at="2026-09-18T00:00:00Z",
            downloads=1000,
            provenance=["huggingface"],
        ),
        RawRecord(
            source_id="acme/liked",
            name="acme/liked",
            created_at="2026-09-17T00:00:00Z",
            likes=25,
            provenance=["huggingface"],
        ),
        RawRecord(
            source_id="acme/parameterized",
            name="acme/parameterized",
            created_at="2026-09-16T00:00:00Z",
            parameters_b=1.0,
            provenance=["huggingface"],
        ),
        RawRecord(
            source_id="acme/old",
            name="acme/old",
            created_at="2026-09-15T00:00:00Z",
            downloads=999,
            likes=24,
            provenance=["huggingface"],
        ),
        RawRecord(
            source_id="acme/quantized",
            name="acme/quantized-gguf",
            created_at="2026-09-19T00:00:00Z",
            downloads=100000,
            provenance=["huggingface"],
        ),
    ]
    models = normalize(records)
    views = {view.view_id: view for view in build_views(models)}
    names = {model.model_id: model.name for model in models}

    assert [names[model_id] for model_id in views["meaningful-new-hf-top5"].model_ids] == [
        "acme/newest",
        "acme/liked",
        "acme/parameterized",
    ]
    assert all(
        model.meaningful_new_hf is True
        for model in models
        if model.name in {"acme/newest", "acme/liked", "acme/parameterized"}
    )
    assert next(model for model in models if model.name == "acme/old").meaningful_new_hf is False
    assert (
        next(model for model in models if model.name == "acme/quantized-gguf").meaningful_new_hf
        is False
    )


def test_meaningful_new_view_excludes_future_source_dates():
    models = normalize(
        [
            RawRecord(
                source_id="acme/future",
                name="acme/future",
                created_at="2026-09-19T00:00:00Z",
                downloads=10000,
                provenance=["huggingface"],
            )
        ]
    )

    views = {
        view.view_id: view for view in build_views(models, as_of=datetime(2026, 9, 18, tzinfo=UTC))
    }

    assert views["meaningful-new-hf-top5"].model_ids == []


def test_primary_rankings_diversify_creators_and_families():
    records = [
        RawRecord(
            source_id=f"aa/{name.lower().replace(' ', '-')}",
            name=name,
            organization=creator,
            intelligence_index=score,
            artificial_analysis_input_price_per_million=0.2,
            artificial_analysis_output_price_per_million=1.0,
            provenance=["artificial_analysis"],
        )
        for name, creator, score in [
            ("Fable (max)", "Anthropic", 53),
            ("Fable (xhigh)", "Anthropic", 52),
            ("Astra (max)", "OpenAI", 51),
            ("GLM Flash", "Z AI", 45),
            ("Qwen Flash", "Alibaba", 44),
            ("DeepSeek V4 Flash", "DeepSeek", 43),
        ]
    ]

    models = normalize(records)
    views = {view.view_id: view for view in build_views(models)}
    names = {model.model_id: model.name for model in models}

    assert [names[model_id] for model_id in views["performance-top5"].model_ids] == [
        "Fable (max)",
        "Astra (max)",
        "GLM Flash",
        "Qwen Flash",
        "DeepSeek V4 Flash",
    ]


def test_copilot_view_is_unavailable_without_explicit_source():
    model = normalize(
        [
            RawRecord(
                source_id="aa/model",
                name="Model",
                intelligence_index=40,
                tool_calling=True,
                license="apache-2.0",
                provenance=["artificial_analysis"],
            )
        ]
    )

    views = {view.view_id: view for view in build_views(model)}

    assert model[0].copilot_ready is None
    assert views["org-copilot-per-token-top10"].degraded is True
    assert views["org-copilot-best-top10"].degraded is True
    assert "organization catalog" in views["org-copilot-best-top10"].annotations["reason"]


def test_meaningful_new_view_deduplicates_hf_family_variants():
    models = normalize(
        [
            RawRecord(
                source_id="Qwen/Qwen-Image-2.1",
                name="Qwen/Qwen-Image-2.1",
                created_at="2026-09-15T00:00:00Z",
                likes=350,
                provenance=["huggingface"],
            ),
            RawRecord(
                source_id="Qwen/Qwen-Image-2.1-PE-I2I",
                name="Qwen/Qwen-Image-2.1-PE-I2I",
                created_at="2026-09-20T00:00:00Z",
                likes=30,
                provenance=["huggingface"],
            ),
        ]
    )

    views = {
        view.view_id: view for view in build_views(models, as_of=datetime(2026, 9, 21, tzinfo=UTC))
    }
    names = {model.model_id: model.name for model in models}

    assert [names[model_id] for model_id in views["meaningful-new-hf-top5"].model_ids] == [
        "Qwen/Qwen-Image-2.1"
    ]


def test_meaningful_new_view_prioritizes_first_party_launches():
    models = normalize(
        [
            RawRecord(
                source_id="Qwen/Qwen-Image-2.1",
                name="Qwen/Qwen-Image-2.1",
                created_at="2026-09-14T00:00:00Z",
                likes=100,
                provenance=["huggingface"],
            ),
            RawRecord(
                source_id="third-party/popular",
                name="third-party/popular",
                created_at="2026-09-16T00:00:00Z",
                likes=250,
                provenance=["huggingface"],
            ),
        ]
    )

    views = {
        view.view_id: view for view in build_views(models, as_of=datetime(2026, 9, 21, tzinfo=UTC))
    }
    names = {model.model_id: model.name for model in models}

    assert names[views["meaningful-new-hf-top5"].model_ids[0]] == "Qwen/Qwen-Image-2.1"


def test_meaningful_new_view_keeps_distinct_non_text_modalities():
    models = normalize(
        [
            RawRecord(
                source_id="text/alpha",
                name="text/alpha",
                created_at="2026-09-20T00:00:00Z",
                modalities=["text"],
                likes=100,
                provenance=["huggingface"],
            ),
            RawRecord(
                source_id="text/bravo",
                name="text/bravo",
                created_at="2026-09-19T00:00:00Z",
                modalities=["text"],
                likes=90,
                provenance=["huggingface"],
            ),
            RawRecord(
                source_id="Qwen/Qwen-Image-2.1",
                name="Qwen/Qwen-Image-2.1",
                created_at="2026-09-15T00:00:00Z",
                modalities=["image"],
                capabilities=["text-to-image"],
                likes=30,
                provenance=["huggingface"],
            ),
        ]
    )

    views = {
        view.view_id: view for view in build_views(models, as_of=datetime(2026, 9, 21, tzinfo=UTC))
    }
    names = {model.model_id: model.name for model in models}

    assert "Qwen/Qwen-Image-2.1" in [
        names[model_id] for model_id in views["meaningful-new-hf-top5"].model_ids
    ]


def test_copilot_catalog_matches_family_and_preserves_thinking_level():
    models = normalize(
        [
            RawRecord(
                source_id="aa/alpha-model",
                name="Alpha Model (xhigh)",
                intelligence_index=42,
                provenance=["artificial_analysis"],
            )
        ]
    )
    enriched = apply_copilot_catalog(
        models,
        CopilotConfig(
            enabled=True,
            source="test-org",
            models=[
                CopilotModelConfig(
                    name="Alpha Model",
                    input_credits_per_million=20,
                    output_credits_per_million=120,
                )
            ],
        ),
    )

    assert enriched[0].model_family == "Alpha Model"
    assert enriched[0].effort_level == "xhigh"
    assert enriched[0].copilot_ready is True
    assert enriched[0].copilot_model_name == "Alpha Model"
    assert enriched[0].scores["copilot_token_efficiency"].value == 0.933333


def test_copilot_per_token_requires_intelligence_index_of_25():
    models = normalize(
        [
            RawRecord(
                source_id="aa/low",
                name="Low Model (max)",
                intelligence_index=24,
                provenance=["artificial_analysis"],
            ),
            RawRecord(
                source_id="aa/eligible",
                name="Eligible Model (max)",
                intelligence_index=25,
                provenance=["artificial_analysis"],
            ),
        ]
    )
    enriched = apply_copilot_catalog(
        models,
        CopilotConfig(
            enabled=True,
            models=[
                CopilotModelConfig(
                    name="Low Model",
                    input_credits_per_million=20,
                    output_credits_per_million=120,
                ),
                CopilotModelConfig(
                    name="Eligible Model",
                    input_credits_per_million=20,
                    output_credits_per_million=120,
                ),
            ],
        ),
    )
    views = {view.view_id: view for view in build_views(enriched)}

    eligible = next(model for model in enriched if model.name == "Eligible Model (max)")
    assert views["org-copilot-per-token-top10"].model_ids == [eligible.model_id]


def test_copilot_coverage_matches_future_versions_by_family_and_level():
    models = normalize(
        [
            RawRecord(
                source_id="aa/gpt-6-luna",
                name="GPT-6 Luna",
                intelligence_index=60,
                provenance=["artificial_analysis"],
            ),
            RawRecord(
                source_id="aa/gpt-7-sol",
                name="GPT-7 Sol",
                intelligence_index=65,
                provenance=["artificial_analysis"],
            ),
            RawRecord(
                source_id="aa/gpt-6-random",
                name="GPT-6 Random",
                intelligence_index=70,
                provenance=["artificial_analysis"],
            ),
        ]
    )
    enriched = apply_copilot_catalog(
        models,
        CopilotConfig(
            enabled=True,
            models=[
                CopilotModelConfig(
                    name="GPT Luna",
                    family="GPT",
                    level="luna",
                    input_credits_per_million=20,
                    output_credits_per_million=120,
                ),
                CopilotModelConfig(
                    name="GPT Sol",
                    family="GPT",
                    level="sol",
                    input_credits_per_million=400,
                    output_credits_per_million=2000,
                ),
            ],
        ),
    )

    by_name = {model.name: model for model in enriched}
    assert by_name["GPT-6 Luna"].copilot_ready is True
    assert by_name["GPT-7 Sol"].copilot_ready is True
    assert by_name["GPT-6 Random"].copilot_ready is None


def test_benchmark_sources_merge_without_touching_aa_score():
    models = normalize(
        [
            RawRecord(
                source_id="aa/gpt-6-astra-max",
                name="GPT-6 Astra (max)",
                intelligence_index=53,
                provenance=["artificial_analysis"],
            )
        ]
    )
    benchmarks = [
        RawRecord(
            source_id="livebench:gpt-6-astra-max",
            name="GPT-6 Astra (max effort)",
            benchmark_source="livebench",
            benchmark_index=80,
            provenance=["livebench"],
        ),
        RawRecord(
            source_id="evalplus:gpt-6-astra-max",
            name="GPT-6 Astra (max)",
            benchmark_source="evalplus",
            benchmark_index=90,
            provenance=["evalplus"],
        ),
    ]

    enriched = attach_benchmarks(models, benchmarks)

    assert enriched[0].livebench_index == 80
    assert enriched[0].evalplus_index == 90
    assert enriched[0].merged_benchmark_index == 85
    assert enriched[0].benchmark_coverage == 2
    assert enriched[0].intelligence_index == 53
    assert enriched[0].scores["merged_benchmark_index"].provisional is True


def test_benchmark_family_key_aligns_slugs_with_display_names():
    assert _benchmark_family_key("claude-opus-4-6-thinking-auto-high-effort") == (
        _benchmark_family_key("Anthropic: Claude Opus 4.6")
    )
    assert _benchmark_family_key("gpt-5.2-2025-12-11-high") == _benchmark_family_key(
        "OpenAI: GPT-5.2"
    )
    assert _benchmark_family_key("deepseek-v4-flash-0731") == _benchmark_family_key(
        "DeepSeek V4 Flash 0731 (max)"
    )
    assert _benchmark_family_key("glm-5.2") == _benchmark_family_key("zai-org/GLM-5.2")
    assert _benchmark_family_key("claude-fable-5-1-max-effort") == _benchmark_family_key(
        "Claude Fable 5.1 (max with fallback)"
    )


def test_benchmark_family_key_keeps_distinct_models_apart():
    assert _benchmark_family_key("OpenAI: GPT-5.2") != _benchmark_family_key("OpenAI: GPT-5.2 Pro")
    assert _benchmark_family_key("Claude Opus 5") != _benchmark_family_key("Claude Opus 4.5")
    assert _benchmark_family_key("Kimi K3 (max)") != _benchmark_family_key("Kimi K2.7 Code")


def test_livebench_prefers_artificial_analysis_and_exact_effort():
    models = normalize(
        [
            RawRecord(
                source_id="aa/claude-opus-5-max",
                name="Claude Opus 5 (max)",
                intelligence_index=80.0,
                provenance=["artificial-analysis"],
            ),
            RawRecord(
                source_id="openrouter/anthropic/claude-opus-5",
                name="Anthropic: Claude Opus 5",
                provenance=["openrouter"],
            ),
        ]
    )
    benchmarks = [
        RawRecord(
            source_id="livebench:claude-opus-5-max-effort",
            name="claude-opus-5-max-effort",
            benchmark_source="livebench",
            benchmark_release="2026-06-25",
            benchmark_index=81.5,
            benchmark_effort="max",
            provenance=["livebench"],
        )
    ]

    enriched = {model.name: model for model in attach_benchmarks(models, benchmarks)}

    assert enriched["Claude Opus 5 (max)"].livebench_index == 81.5
    assert enriched["Anthropic: Claude Opus 5"].livebench_index is None


def test_livebench_without_effort_targets_flagship_variant():
    models = normalize(
        [
            RawRecord(
                source_id="aa/kimi-k3-max",
                name="Kimi K3 (max)",
                intelligence_index=79.0,
                provenance=["artificial-analysis"],
            ),
            RawRecord(
                source_id="aa/kimi-k3-low",
                name="Kimi K3 (low)",
                intelligence_index=55.0,
                provenance=["artificial-analysis"],
            ),
        ]
    )
    benchmarks = [
        RawRecord(
            source_id="livebench:kimi-k3",
            name="kimi-k3",
            benchmark_source="livebench",
            benchmark_index=70.0,
            provenance=["livebench"],
        )
    ]

    enriched = {model.name: model for model in attach_benchmarks(models, benchmarks)}

    assert enriched["Kimi K3 (max)"].livebench_index == 70.0
    assert enriched["Kimi K3 (low)"].livebench_index is None


def test_livebench_absent_model_is_not_force_matched():
    models = normalize(
        [
            RawRecord(
                source_id="aa/glm-5-3",
                name="GLM-5.3",
                intelligence_index=70.0,
                provenance=["artificial-analysis"],
            )
        ]
    )
    benchmarks = [
        RawRecord(
            source_id="livebench:smaug-agentic",
            name="smaug-agentic",
            benchmark_source="livebench",
            benchmark_index=79.0,
            provenance=["livebench"],
        )
    ]

    enriched = attach_benchmarks(models, benchmarks)

    assert enriched[0].livebench_index is None


def test_livebench_slug_with_date_matches_catalog_display_name():
    models = normalize(
        [
            RawRecord(
                source_id="openrouter/openai/gpt-5-2",
                name="OpenAI: GPT-5.2",
                provenance=["openrouter"],
            )
        ]
    )
    benchmarks = [
        RawRecord(
            source_id="livebench:gpt-5.2-2025-12-11-high",
            name="gpt-5.2-2025-12-11-high",
            benchmark_source="livebench",
            benchmark_index=75.19,
            benchmark_effort="high",
            provenance=["livebench"],
        )
    ]

    enriched = attach_benchmarks(models, benchmarks)

    assert enriched[0].livebench_index == 75.19
    assert enriched[0].effort_level == "high"


def test_parameter_size_parsed_from_model_names():
    assert _parameter_size_b("Qwen3.5 4B") == 4.0
    assert _parameter_size_b("LFM2.5-2.6B") == 2.6
    assert _parameter_size_b("Gemma 4 12B") == 12.0
    assert _parameter_size_b("K2 Horizon 26B A4B") == 26.0
    assert _parameter_size_b("Exaone 4.0 1.2B") == 1.2
    assert _parameter_size_b("Bonsai-2-27B-1bit") == 27.0
    assert _parameter_size_b("Cactus-Compute/needle2") is None


def test_normalize_fills_parameters_b_from_name():
    models = normalize([RawRecord(source_id="Qwen/Qwen3.5-4B", name="Qwen3.5 4B")])

    assert models[0].parameters_b == 4.0


def test_tiny_llm_view_keeps_only_small_models():
    models = normalize(
        [
            RawRecord(
                source_id="aa/qwen-small",
                name="Qwen3.5 4B",
                intelligence_index=13.0,
                provenance=["artificial-analysis"],
            ),
            RawRecord(
                source_id="aa/k2-small",
                name="K2 Horizon 7B",
                intelligence_index=21.0,
                provenance=["artificial-analysis"],
            ),
            RawRecord(
                source_id="aa/big",
                name="Qwen3.8 27B",
                intelligence_index=34.0,
                provenance=["artificial-analysis"],
            ),
        ]
    )

    tiny = next(view for view in build_views(models) if view.view_id == "tiny-llm-top10")
    names = {model.name for model in models if model.model_id in tiny.model_ids}

    assert names == {"Qwen3.5 4B", "K2 Horizon 7B"}
    assert "total parameters <= 8B" in tiny.annotations["metric"]


def test_edge_view_includes_tagged_models_and_collapses_quants():
    models = normalize(
        [
            RawRecord(
                source_id="Cactus-Compute/needle2",
                name="Cactus-Compute/needle2",
                capabilities=["on-device", "tool-calling"],
                downloads=32939,
                provenance=["huggingface"],
            ),
            RawRecord(
                source_id="prism-ml/Ternary-Bonsai-2-27B-gguf",
                name="prism-ml/Ternary-Bonsai-2-27B-gguf",
                capabilities=["on-device", "gguf"],
                downloads=2227879,
                provenance=["huggingface"],
            ),
            RawRecord(
                source_id="dealignai/Bonsai-2-27B-Ternary-CRACK-GGUF",
                name="dealignai/Bonsai-2-27B-Ternary-CRACK-GGUF",
                capabilities=["on-device", "gguf"],
                downloads=33051,
                provenance=["huggingface"],
            ),
            RawRecord(
                source_id="openbmb/MiniCPM5-2B-GGUF",
                name="openbmb/MiniCPM5-2B-GGUF",
                capabilities=["edge", "gguf"],
                downloads=188174,
                provenance=["huggingface"],
            ),
            RawRecord(
                source_id="meta-llama/Llama-3.1-8B",
                name="meta-llama/Llama-3.1-8B",
                capabilities=["text-generation"],
                downloads=537978,
                provenance=["huggingface"],
            ),
        ]
    )

    ids = _edge_model_ids(models)
    selected = {model.name: model for model in models if model.model_id in ids}

    assert "Cactus-Compute/needle2" in selected
    assert "meta-llama/Llama-3.1-8B" not in selected
    assert sum(name.startswith(("prism-ml/", "dealignai/")) for name in selected) == 1
