from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict
from collections.abc import Callable, Iterable
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from model_radar.models import (
    CopilotConfig,
    CopilotModelConfig,
    ModelRecord,
    RawRecord,
    Score,
    View,
)

_PARAMETER_SIZE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*([bBmM])(?![a-zA-Z])")
_EDGE_CAPABILITIES = frozenset({"on-device", "edge", "tiny", "tinyllm", "mobile"})
MINI_PARAMETER_LIMIT_B = 1.0
TINY_PARAMETER_LIMIT_B = 8.0


def _parameter_size_b(name: str) -> float | None:
    """Best-effort parameter count in billions parsed from a model name.

    Sources do not expose a reliable size field, so names like ``Qwen3.5 4B``,
    ``LFM2.5-2.6B``, ``LFM2.5-350M``, or ``K2 Horizon 26B A4B`` (total, not
    active) are parsed. Million-scale sizes are converted to billions. Models
    whose names carry no size return ``None``.
    """
    sizes = [
        float(match.group(1)) / (1000.0 if match.group(2).casefold() == "m" else 1.0)
        for match in _PARAMETER_SIZE.finditer(name)
    ]
    return max(sizes) if sizes else None


def _base_model_size_b(capabilities: Iterable[str]) -> float | None:
    """Parameter count of the declaring base model, from Hugging Face tags.

    Tags such as ``base_model:Qwen/Qwen3-0.6B`` or
    ``base_model:finetune:Qwen/Qwen3-0.6B`` name the upstream checkpoint, which
    is the best available size hint when the model name itself has no size.
    """
    sizes: list[float] = []
    for capability in capabilities:
        value = capability.casefold()
        if not value.startswith("base_model:"):
            continue
        target = value.split(":", 1)[1]
        for prefix in ("finetune:", "quantized:", "adapter:"):
            target = target.removeprefix(prefix)
        size = _parameter_size_b(target)
        if size is not None:
            sizes.append(size)
    return max(sizes) if sizes else None


def _is_edge_model(model: ModelRecord) -> bool:
    capabilities = {value.casefold() for value in model.capabilities}
    return bool(capabilities & _EDGE_CAPABILITIES)


_EDGE_VARIANT_TOKENS = (
    r"(?:ternary|gguf|mlx|1bit|2bit|3bit|4bit|8bit|16bit|awq|gptq|fp8|bf16|int4|int8|"
    r"q4|q8|mtp|dflash|dspark)"
)
_EDGE_VARIANT = re.compile(rf"(?:^|[-_]){_EDGE_VARIANT_TOKENS}(?=[-_]|$)")
_EDGE_DERIVATIVE = re.compile(r"abliterated|uncensored|heretic|crack|derisked")
_EDGE_NON_LLM = re.compile(r"tts|asr|whisper|embedding|rerank")


def _edge_family_key(name: str) -> str:
    """Collapse quantisation and vendor suffixes so re-uploads share one key."""
    value = name.casefold().split("/", 1)[-1]
    previous = None
    while previous != value:
        previous = value
        value = _EDGE_VARIANT.sub("-", value)
    return re.sub(r"[^a-z0-9]+", "-", value).strip("-")


def _edge_model_ids(models: list[ModelRecord], limit: int = 10) -> list[str]:
    """Hugging Face models tagged for on-device or edge use, by downloads.

    Community re-uploads (derivative names) and non-language models are
    excluded so the list surfaces original models rather than quantised copies.
    """
    candidates = [
        model
        for model in models
        if "huggingface" in model.provenance
        and _is_edge_model(model)
        and not _EDGE_DERIVATIVE.search(model.name.casefold())
        and not _EDGE_NON_LLM.search(model.name.casefold())
    ]
    representatives: dict[str, ModelRecord] = {}
    for model in candidates:
        key = _edge_family_key(model.name)
        current = representatives.get(key)
        if current is None or _popularity(model) > _popularity(current):
            representatives[key] = model
    ranked = sorted(representatives.values(), key=_popularity, reverse=True)
    return [model.model_id for model in ranked[:limit]]


def notable_new_model_names(models: list[ModelRecord], limit: int = 6) -> list[str]:
    """Human-facing names of the most significant new models, one per family.

    Raw catalog diffs are dominated by community re-uploads, so this ranks by
    authoritative coverage (Artificial Analysis), then first-party releases,
    open weights, and adoption, and collapses thinking-level variants so a
    release like ``Claude Opus 5.5`` appears once.
    """

    def importance(model: ModelRecord) -> tuple[bool, float, bool, bool, bool, int, int, str]:
        value = model.name.casefold()
        noisy = bool(_EDGE_VARIANT.search(value) or _EDGE_DERIVATIVE.search(value))
        return (
            model.intelligence_index is not None,
            model.intelligence_index or 0.0,
            not noisy,
            _is_first_party_hf_name(model.name),
            model.open_weights is True,
            model.likes or 0,
            model.downloads or 0,
            model.name.casefold(),
        )

    selected: list[str] = []
    families: set[str] = set()
    for model in sorted(models, key=importance, reverse=True):
        family = _benchmark_family_key(model.name)
        if family in families:
            continue
        families.add(family)
        selected.append(_model_family_name(model.name))
        if len(selected) == limit:
            break
    return selected


def _mini_model_ids(models: list[ModelRecord], limit: int = 10) -> list[str]:
    """Toaster-class models at or below the mini size ceiling.

    Any language model with a parsed size at or below
    ``MINI_PARAMETER_LIMIT_B`` qualifies. Tagged on-device or edge models with
    no parsed size are also included so tag-only models such as
    ``Cactus-Compute/needle3`` surface, keeping their size and benchmark
    unknown. Quantised and derivative re-uploads are collapsed. Ranked by
    Artificial Analysis Intelligence Index where present, then by adoption, so
    unscored models follow scored ones deterministically.
    """
    sized = [
        model
        for model in models
        if model.parameters_b is not None
        and model.parameters_b <= MINI_PARAMETER_LIMIT_B
        and model_type(model) == "llm"
        and not _EDGE_NON_LLM.search(model.name.casefold())
    ]
    tagged = [
        model
        for model in models
        if model.parameters_b is None
        and _is_edge_model(model)
        and model_type(model) == "llm"
        and not _EDGE_DERIVATIVE.search(model.name.casefold())
        and not _EDGE_NON_LLM.search(model.name.casefold())
    ]
    representatives: dict[str, ModelRecord] = {}
    for model in tagged:
        key = _edge_family_key(model.name)
        current = representatives.get(key)
        if current is None or _popularity(model) > _popularity(current):
            representatives[key] = model
    ranked = sorted(
        [*sized, *representatives.values()],
        key=lambda item: (
            item.intelligence_index is None,
            -(item.intelligence_index or 0.0),
            -(item.downloads or 0),
            -(item.likes or 0),
            item.name.casefold(),
            item.model_id,
        ),
    )
    return [model.model_id for model in ranked[:limit]]


def normalize(records: Iterable[RawRecord]) -> list[ModelRecord]:
    grouped: dict[str, list[RawRecord]] = defaultdict(list)
    for record in records:
        grouped[_canonical_key(record)].append(record)
    models: list[ModelRecord] = []
    for key, assertions in sorted(grouped.items()):
        ordered = sorted(
            assertions, key=lambda item: (item.source_id, item.name, item.source_url or "")
        )
        first = ordered[0]
        model_id = _stable_id(key)
        benchmarks = _merge_numbers(ordered, "benchmarks")
        pricing = _merge_numbers(ordered, "pricing")
        price_values = [
            item.price_per_million for item in ordered if item.price_per_million is not None
        ]
        meaningful_new_hf, meaningful_new_hf_reason = _hf_meaningfulness(ordered)
        capabilities = sorted({value for item in ordered for value in item.capabilities})
        model = ModelRecord(
            model_id=model_id,
            slug=_slug(first.name),
            organization=first.organization,
            family=first.family,
            name=first.name,
            model_family=_model_family_name(first.name),
            effort_level=_effort_level(first.name),
            variant=first.variant,
            source_urls=sorted({item.source_url for item in ordered if item.source_url}),
            created_at=_date_extreme(ordered, "created_at", minimum=True),
            updated_at=_date_extreme(ordered, "updated_at"),
            modalities=sorted({value for item in ordered for value in item.modalities}),
            capabilities=capabilities,
            supported_parameters=sorted(
                {value for item in ordered for value in item.supported_parameters}
            ),
            tool_calling=_merge_bool(ordered, "tool_calling"),
            architecture=_merge_mapping(ordered, "architecture"),
            provider_details=_merge_mapping(ordered, "provider_details"),
            parameters_b=first.parameters_b
            or _parameter_size_b(first.name)
            or _base_model_size_b(capabilities),
            context_length=max((item.context_length or 0 for item in ordered), default=0) or None,
            quantization=first.quantization,
            release_date=_date_extreme(ordered, "release_date"),
            license=first.license,
            open_weights=_merge_bool(ordered, "open_weights"),
            commercial_use=_merge_bool(ordered, "commercial_use"),
            origin=_merge_origin(ordered),
            providers=sorted({value for item in ordered for value in item.providers}),
            price_per_million=sum(price_values) / len(price_values) if price_values else None,
            output_price_per_million=_average_field(ordered, "output_price_per_million"),
            pricing=pricing,
            downloads=_max_field(ordered, "downloads"),
            likes=_max_field(ordered, "likes"),
            benchmarks=benchmarks,
            recent_signals=_recent_signals(ordered),
            artificial_analysis_source_url=_first_field(ordered, "artificial_analysis_source_url"),
            artificial_analysis_model_url=_first_field(ordered, "artificial_analysis_model_url"),
            artificial_analysis_creator=_first_field(ordered, "artificial_analysis_creator"),
            artificial_analysis_context_window=_max_field(
                ordered, "artificial_analysis_context_window"
            ),
            intelligence_index=_average_field(ordered, "intelligence_index"),
            cost_per_task_usd=_average_field(ordered, "cost_per_task_usd"),
            artificial_analysis_input_price_per_million=_average_field(
                ordered, "artificial_analysis_input_price_per_million"
            ),
            artificial_analysis_output_price_per_million=_average_field(
                ordered, "artificial_analysis_output_price_per_million"
            ),
            artificial_analysis_modality=_first_field(ordered, "artificial_analysis_modality"),
            artificial_analysis_modality_elo=_average_field(
                ordered, "artificial_analysis_modality_elo"
            ),
            artificial_analysis_modality_samples=_max_field(
                ordered, "artificial_analysis_modality_samples"
            ),
            artificial_analysis_modality_release=_first_field(
                ordered, "artificial_analysis_modality_release"
            ),
            artificial_analysis_modality_cost=_average_field(
                ordered, "artificial_analysis_modality_cost"
            ),
            artificial_analysis_modality_cost_unit=_first_field(
                ordered, "artificial_analysis_modality_cost_unit"
            ),
            median_output_tokens_per_second=_average_field(
                ordered, "median_output_tokens_per_second"
            ),
            latency_first_chunk_seconds=_average_field(ordered, "latency_first_chunk_seconds"),
            total_response_seconds=_average_field(ordered, "total_response_seconds"),
            meaningful_new_hf=meaningful_new_hf,
            meaningful_new_hf_reason=meaningful_new_hf_reason,
            provenance=sorted({value for item in ordered for value in item.provenance}),
            field_provenance=_merge_provenance(ordered),
            confidence=min(item.confidence for item in ordered),
            scores=_scores(
                benchmarks,
                price_values,
                _average_field(ordered, "intelligence_index"),
                _average_field(ordered, "cost_per_task_usd"),
                _average_field(ordered, "artificial_analysis_input_price_per_million"),
                _average_field(ordered, "artificial_analysis_output_price_per_million"),
            ),
            copilot_ready=_copilot_status(ordered),
        )
        models.append(model)
    return models


def apply_copilot_catalog(models: list[ModelRecord], config: CopilotConfig) -> list[ModelRecord]:
    if not config.enabled:
        return models
    catalog = {_model_family_key(item.name): item for item in config.models if item.enabled}
    enriched: list[ModelRecord] = []
    for model in models:
        entry = catalog.get(_model_family_key(model.model_family or model.name))
        if entry is None:
            entry = _copilot_coverage_match(model, config.models)
        if entry is None:
            enriched.append(model)
            continue
        weighted_price = blended_token_price(
            entry.input_credits_per_million,
            entry.output_credits_per_million,
        )
        efficiency = (
            model.intelligence_index / weighted_price
            if model.intelligence_index is not None and weighted_price and weighted_price > 0
            else None
        )
        copilot_score = Score(
            value=_round(efficiency),
            components={
                "intelligence_index": _round(model.intelligence_index),
                "input_credits_per_million": entry.input_credits_per_million,
                "output_credits_per_million": entry.output_credits_per_million,
                "weighted_credits_per_million": _round(weighted_price),
            },
            cohort=f"Copilot organization catalog: {config.source}",
            algorithm_version="copilot-v1",
            configuration_hash=hashlib.sha256(config.source.encode()).hexdigest()[:16],
            provisional=efficiency is None,
        )
        enriched.append(
            model.model_copy(
                update={
                    "copilot_ready": True,
                    "copilot_model_name": entry.name,
                    "copilot_input_credits_per_million": entry.input_credits_per_million,
                    "copilot_output_credits_per_million": entry.output_credits_per_million,
                    "scores": {**model.scores, "copilot_token_efficiency": copilot_score},
                }
            )
        )
    return enriched


def _copilot_coverage_match(
    model: ModelRecord, entries: list[CopilotModelConfig]
) -> CopilotModelConfig | None:
    model_text = " ".join(
        value for value in (model.name, model.organization or "", model.model_family or "")
    ).casefold()
    for entry in entries:
        if not entry.enabled or not entry.family:
            continue
        if entry.family.casefold() not in model_text:
            continue
        level = (entry.level or "default").casefold()
        if level != "default" and level not in model_text and level != (model.effort_level or ""):
            continue
        return entry
    return None


def _select_benchmark_target(
    candidates: list[ModelRecord], benchmark_effort: str | None
) -> ModelRecord | None:
    """Pick the single model a benchmark row belongs to, deterministically.

    Artificial Analysis records carry the performance view and split models by
    effort level, so they are preferred over catalog duplicates from other
    sources. Within a family the exact effort wins; a variant with no declared
    effort is next; otherwise the flagship (highest Intelligence Index) is used.
    """
    if not candidates:
        return None
    analysis = [model for model in candidates if "artificial-analysis" in model.provenance]
    pool = analysis or candidates
    if benchmark_effort:
        exact = [model for model in pool if model.effort_level == benchmark_effort]
        if exact:
            pool = exact
    if len(pool) == 1:
        return pool[0]
    no_effort = [model for model in pool if model.effort_level is None]
    if no_effort:
        pool = no_effort
    return max(
        pool,
        key=lambda model: (
            model.intelligence_index is not None,
            model.intelligence_index or 0.0,
            model.name.casefold(),
            model.model_id,
        ),
    )


def attach_benchmarks(
    models: list[ModelRecord], benchmark_records: Iterable[RawRecord]
) -> list[ModelRecord]:
    for record in benchmark_records:
        key = _benchmark_family_key(record.name)
        if not key:
            continue
        candidates = [model for model in models if _benchmark_family_key(model.name) == key]
        model = _select_benchmark_target(candidates, record.benchmark_effort)
        if model is None:
            continue
        field = {
            "livebench": "livebench_index",
            "evalplus": "evalplus_index",
            "deepswe": "deepswe_index",
        }.get(record.benchmark_source or "")
        if field is None:
            continue
        values = {field: record.benchmark_index, "benchmark_release": record.benchmark_release}
        if record.benchmark_effort and model.effort_level is None:
            values["effort_level"] = record.benchmark_effort
        if record.benchmark_cost_per_task is not None:
            values["benchmark_cost_per_task"] = record.benchmark_cost_per_task
        models[models.index(model)] = model.model_copy(update=values)

    enriched: list[ModelRecord] = []
    for model in models:
        source_indexes = {
            "livebench": model.livebench_index,
            "evalplus": model.evalplus_index,
            "deepswe": model.deepswe_index,
        }
        available = {key: value for key, value in source_indexes.items() if value is not None}
        merged = sum(available.values()) / len(available) if len(available) >= 2 else None
        scores = dict(model.scores)
        for source, value in available.items():
            scores[f"{source}_index"] = Score(
                value=value,
                cohort=source,
                algorithm_version="benchmark-v1",
                configuration_hash="benchmark-v1",
                provisional=False,
            )
        scores["merged_benchmark_index"] = Score(
            value=merged,
            components={key: value for key, value in available.items()},
            weights={key: 1 / len(available) for key in available},
            cohort=f"{len(available)}/3 benchmark sources" if merged is not None else None,
            algorithm_version="benchmark-merged-v1",
            configuration_hash="benchmark-merged-v1",
            provisional=merged is not None and len(available) < 3,
        )
        enriched.append(
            model.model_copy(
                update={
                    "merged_benchmark_index": merged,
                    "benchmark_coverage": len(available),
                    "scores": scores,
                }
            )
        )
    return enriched


def build_views(
    models: list[ModelRecord], degraded: bool = False, as_of: datetime | None = None
) -> list[View]:
    performance_models = [item for item in models if item.intelligence_index is not None]
    performance_floor = max(
        20.0,
        max((item.intelligence_index or 0.0 for item in performance_models), default=0.0) * 0.5,
    )
    token_efficiency_models = [
        item
        for item in performance_models
        if (item.intelligence_index or 0.0) >= performance_floor
        and _aa_token_efficiency(item) is not None
    ]
    meaningful_hf_models = [
        item
        for item in models
        if item.meaningful_new_hf is True and _hf_is_recent_release(item, as_of)
    ]
    meaningful_hf_models = _meaningful_representatives(meaningful_hf_models)
    performance_view = View(
        view_id="performance-top5",
        title="Performance top 10",
        model_ids=_decision_ranked_ids(performance_models, lambda item: item.intelligence_index),
        annotations={"metric": "Artificial Analysis Intelligence Index"},
    )
    efficiency_view = View(
        view_id="performance-per-token-top5",
        title="AA Intelligence per token-cost dollar top 10",
        model_ids=_decision_ranked_ids(token_efficiency_models, _aa_token_efficiency),
        annotations={
            "metric": "intelligence_index / weighted token price per 1M tokens",
            "price_basis": "(3 * input price + output price) / 4",
            "minimum_intelligence_index": str(round(performance_floor, 2)),
        },
    )
    meaningful_view = View(
        view_id="meaningful-new-hf-top5",
        title="Meaningful new Hugging Face models top 10",
        model_ids=_decision_release_ids(meaningful_hf_models, as_of),
        annotations={
            "heuristic": (
                "HF createdAt release date within 14 days; family-deduplicated; prioritizes first-party releases, adoption, and recency; "
                "requires downloads >= 1000 OR likes >= 25 OR parameters_b >= 1"
            )
        },
    )
    if not performance_view.model_ids:
        performance_view = _unavailable_view(
            "performance-top5",
            "Performance top 5",
            "Artificial Analysis Intelligence Index data is unavailable",
        )
    if not efficiency_view.model_ids:
        efficiency_view = _unavailable_view(
            "performance-per-token-top5",
            "AA Intelligence per token-cost dollar",
            "Artificial Analysis Intelligence Index must meet the performance floor and token prices must be positive",
        )
    if not meaningful_view.model_ids:
        meaningful_view = _unavailable_view(
            "meaningful-new-hf-top5",
            "Meaningful new Hugging Face models",
            "No Hugging Face models meet the date, noise, and adoption heuristic",
        )

    copilot_models = [model for model in models if model.copilot_ready is True]
    copilot_per_token_models = [
        model
        for model in copilot_models
        if (model.intelligence_index or 0.0) >= 25
        and _score_value(model, "copilot_token_efficiency") is not None
    ]
    copilot_best_models = [
        model for model in copilot_models if model.intelligence_index is not None
    ]
    copilot_per_token_view = View(
        view_id="org-copilot-per-token-top10",
        title="Org Copilot per token top 10",
        model_ids=_decision_ranked_ids(
            copilot_per_token_models,
            lambda item: _score_value(item, "copilot_token_efficiency"),
            diverse=False,
        ),
        annotations={
            "metric": "AA Intelligence per weighted Copilot credits per 1M tokens",
            "minimum_intelligence_index": "25",
            "source": "organization Copilot catalog",
        },
    )
    copilot_best_view = View(
        view_id="org-copilot-best-top10",
        title="Org Copilot best top 10",
        model_ids=_decision_ranked_ids(
            copilot_best_models,
            lambda item: item.intelligence_index,
            diverse=False,
        ),
        annotations={
            "metric": "Artificial Analysis Intelligence Index; each thinking level is ranked separately",
            "source": "organization Copilot catalog",
        },
    )
    if not copilot_per_token_view.model_ids:
        copilot_per_token_view = _unavailable_view(
            "org-copilot-per-token-top10",
            "Org Copilot per token top 10",
            "No organization Copilot combination has AA Intelligence Index >= 25 with token pricing",
        )
    if not copilot_best_view.model_ids:
        copilot_best_view = _unavailable_view(
            "org-copilot-best-top10",
            "Org Copilot best top 10",
            "No matching Copilot organization catalog entries have AA performance",
        )

    tiny_models = [
        model
        for model in performance_models
        if model.parameters_b is not None and model.parameters_b <= TINY_PARAMETER_LIMIT_B
    ]
    tiny_view = View(
        view_id="tiny-llm-top10",
        title="Tiny LLM top 10",
        model_ids=_decision_ranked_ids(tiny_models, lambda item: item.intelligence_index),
        annotations={
            "metric": (
                "Artificial Analysis Intelligence Index; "
                f"total parameters <= {int(TINY_PARAMETER_LIMIT_B)}B; "
                "size is estimated weights at FP16"
            )
        },
    )
    if not tiny_view.model_ids:
        tiny_view = _unavailable_view(
            "tiny-llm-top10",
            "Tiny LLM top 10",
            f"No model at or below {int(TINY_PARAMETER_LIMIT_B)}B parameters has an "
            "Artificial Analysis Intelligence Index",
        )
    mini_view = View(
        view_id="mini-llm-top10",
        title="Mini LLM top 10",
        model_ids=_mini_model_ids(models),
        annotations={
            "metric": (
                "Artificial Analysis Intelligence Index; "
                f"total parameters <= {int(MINI_PARAMETER_LIMIT_B)}B or tagged on-device/edge "
                "with an unknown size; unscored models rank after scored ones and show unknown; "
                "size is estimated weights at FP16"
            )
        },
    )
    if not mini_view.model_ids:
        mini_view = _unavailable_view(
            "mini-llm-top10",
            "Mini LLM top 10",
            f"No model at or below {int(MINI_PARAMETER_LIMIT_B)}B parameters or tagged "
            "on-device/edge is available",
        )
    edge_ids = _edge_model_ids(models)
    edge_view = View(
        view_id="edge-models-top10",
        title="On-device models top 10",
        model_ids=edge_ids,
        annotations={
            "metric": (
                "Hugging Face downloads; models tagged on-device or edge, quantised re-uploads "
                "collapsed; size is estimated weights at FP16; benchmark is matched by model "
                "name where Artificial Analysis covers it"
            )
        },
    )
    if not edge_view.model_ids:
        edge_view = _unavailable_view(
            "edge-models-top10",
            "On-device models top 10",
            "No Hugging Face model is tagged on-device or edge",
        )

    views = [
        copilot_per_token_view,
        copilot_best_view,
        performance_view,
        efficiency_view,
        tiny_view,
        mini_view,
        edge_view,
        meaningful_view,
        View(
            view_id="popular",
            title="Popular",
            model_ids=_ordered_ids(
                [
                    model
                    for model in models
                    if model.downloads is not None or model.likes is not None
                ],
                key=_popularity,
                reverse=True,
            )[:10],
        ),
        View(
            view_id="lowest-price",
            title="Lowest input price",
            model_ids=_ordered_ids(
                [model for model in models if model.price_per_million is not None],
                key=lambda item: (item.price_per_million or 0, item.model_id),
            )[:10],
        ),
        View(
            view_id="long-context",
            title="Longest context",
            model_ids=_ordered_ids(
                [model for model in models if model.context_length is not None],
                key=lambda item: (item.context_length or 0, item.model_id),
                reverse=True,
            )[:10],
        ),
        View(
            view_id="tool-calling",
            title="Tool calling",
            model_ids=_ordered_ids(
                [model for model in models if model.tool_calling is True],
                key=lambda item: (item.context_length or 0, item.model_id),
                reverse=True,
            )[:10],
        ),
        View(
            view_id="open-weights",
            title="Open weights",
            model_ids=[m.model_id for m in models if m.open_weights is True],
        ),
        View(
            view_id="commercial-use",
            title="Commercial use",
            model_ids=[m.model_id for m in models if m.commercial_use is True],
        ),
        View(
            view_id="local-24gb",
            title="Fits local 24GB",
            model_ids=[
                m.model_id for m in models if m.parameters_b is not None and m.parameters_b <= 14
            ],
        ),
    ]
    if degraded:
        views = [
            view.model_copy(
                update={
                    "degraded": True,
                    "annotations": {
                        **view.annotations,
                        "source_status": "optional source data unavailable",
                    },
                }
            )
            for view in views
        ]
    return views


def _scores(
    benchmarks: dict[str, float],
    prices: list[float],
    intelligence_index: float | None = None,
    cost_per_task_usd: float | None = None,
    input_price_per_million: float | None = None,
    output_price_per_million: float | None = None,
) -> dict[str, Score]:
    config_hash = hashlib.sha256(b"quality=v1;value=v1;aa=v1").hexdigest()[:16]
    quality = sum(benchmarks.values()) / len(benchmarks) if benchmarks else None
    price = sum(prices) / len(prices) if prices else None
    value = quality / max(price, 0.001) if quality is not None and price is not None else None
    efficiency = (
        intelligence_index / cost_per_task_usd
        if intelligence_index is not None
        and intelligence_index > 0
        and cost_per_task_usd is not None
        and cost_per_task_usd > 0
        else None
    )
    token_price = blended_token_price(input_price_per_million, output_price_per_million)
    token_efficiency = (
        intelligence_index / token_price
        if intelligence_index is not None
        and intelligence_index > 0
        and token_price is not None
        and token_price > 0
        else None
    )
    return {
        "quality": Score(
            value=_round(quality),
            components={key: _round(item) for key, item in benchmarks.items()},
            cohort="provided benchmarks" if benchmarks else None,
            weights={key: 1 / len(benchmarks) for key in benchmarks} if benchmarks else {},
            configuration_hash=config_hash,
            provisional=not bool(benchmarks),
        ),
        "value": Score(
            value=_round(value),
            components={"quality": _round(quality), "price": _round(price)},
            configuration_hash=config_hash,
            provisional=value is None,
        ),
        "intelligence_index": Score(
            value=_round(intelligence_index),
            cohort="Artificial Analysis leaderboard" if intelligence_index is not None else None,
            configuration_hash=config_hash,
            provisional=intelligence_index is None,
        ),
        "aa_task_dollar_efficiency": Score(
            value=_round(efficiency),
            components={
                "intelligence_index": _round(intelligence_index),
                "cost_per_task_usd": _round(cost_per_task_usd),
            },
            cohort="Artificial Analysis leaderboard" if efficiency is not None else None,
            configuration_hash=config_hash,
            provisional=efficiency is None,
        ),
        "aa_token_dollar_efficiency": Score(
            value=_round(token_efficiency),
            components={
                "intelligence_index": _round(intelligence_index),
                "input_price_per_million": _round(input_price_per_million),
                "output_price_per_million": _round(output_price_per_million),
                "weighted_token_price_per_million": _round(token_price),
            },
            cohort="Artificial Analysis token pricing" if token_efficiency is not None else None,
            configuration_hash=config_hash,
            provisional=token_efficiency is None,
        ),
    }


def _canonical_key(record: RawRecord) -> str:
    identity = record.source_id.lower().strip()
    identity = re.sub(r"[^a-z0-9./_-]+", "-", identity)
    return identity


def _stable_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _merge_numbers(records: list[RawRecord], field: str) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for record in records:
        values.update({})
        for key, value in getattr(record, field).items():
            values[key].append(value)
    return {key: round(sum(items) / len(items), 6) for key, items in sorted(values.items())}


def _score_value(model: ModelRecord, score_name: str) -> float | None:
    score = model.scores.get(score_name)
    return score.value if score and score.value is not None else None


def _merge_bool(records: list[RawRecord], field: str) -> bool | None:
    values = [getattr(item, field) for item in records if getattr(item, field) is not None]
    return values[0] if values and all(value == values[0] for value in values) else None


def _merge_origin(records: list[RawRecord]) -> Literal["open", "proprietary", "unknown"]:
    values = {item.origin for item in records}
    return next(iter(values)) if len(values) == 1 else "unknown"


def _merge_mapping(records: list[RawRecord], field: str) -> dict[str, str]:
    merged: dict[str, set[str]] = defaultdict(set)
    for record in records:
        for key, value in getattr(record, field).items():
            merged[key].add(value)
    return {key: min(values) for key, values in sorted(merged.items()) if values}


def _merge_provenance(records: list[RawRecord]) -> dict[str, list[str]]:
    merged: dict[str, set[str]] = defaultdict(set)
    for record in records:
        for key, values in record.field_provenance.items():
            merged[key].update(values)
    return {key: sorted(values) for key, values in sorted(merged.items())}


def _max_field(records: list[RawRecord], field: str) -> int | None:
    values = [getattr(item, field) for item in records if getattr(item, field) is not None]
    return max(values) if values else None


def _average_field(records: list[RawRecord], field: str) -> float | None:
    values = [getattr(item, field) for item in records if getattr(item, field) is not None]
    return sum(values) / len(values) if values else None


def _first_field(records: list[RawRecord], field: str) -> str | None:
    values = [getattr(item, field) for item in records if getattr(item, field) is not None]
    return values[0] if values else None


def _max_numbers(records: list[RawRecord], field: str) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for record in records:
        for key, value in getattr(record, field).items():
            values[key].append(value)
    return {key: max(items) for key, items in sorted(values.items())}


def _recent_signals(records: list[RawRecord]) -> dict[str, float]:
    signals = _max_numbers(records, "recent_signals")
    for field in ("downloads", "likes"):
        value = _max_field(records, field)
        if value is not None:
            signals[field] = float(value)
    return dict(sorted(signals.items()))


def _ranked_ids(
    models: list[ModelRecord],
    value_getter: Callable[[ModelRecord], float | None],
    limit: int = 10,
) -> list[str]:
    ranked = [(model, value) for model in models if (value := value_getter(model)) is not None]
    return [
        model.model_id
        for model, _value in sorted(
            ranked,
            key=lambda item: (-float(item[1] or 0), item[0].name.casefold(), item[0].model_id),
        )[:limit]
    ]


def _ranked_diverse_ids(
    models: list[ModelRecord],
    value_getter: Callable[[ModelRecord], float | None],
    limit: int = 10,
) -> list[str]:
    ranked = sorted(
        (model for model in models if value_getter(model) is not None),
        key=lambda item: (-float(value_getter(item) or 0), item.name.casefold(), item.model_id),
    )
    selected: list[str] = []
    creators: set[str] = set()
    families: set[str] = set()
    for model in ranked:
        creator = (model.organization or model.model_id).casefold()
        family = _model_family_key(model.name)
        if creator in creators or family in families:
            continue
        selected.append(model.model_id)
        creators.add(creator)
        families.add(family)
        if len(selected) == limit:
            break
    return selected


_MODEL_TYPES = ("llm", "text-to-image", "image-to-image", "text-to-video", "image-to-video")


def model_type(model: ModelRecord) -> str:
    values: list[str] = [model.name, model.family or "", model.model_family or ""]
    values.extend(model.modalities)
    values.extend(model.capabilities)
    values.extend(value for value in model.architecture.values() if isinstance(value, str))
    text = " ".join(values).casefold().replace("_", "-")
    patterns = (
        (
            "image-to-video",
            ("image-to-video", "image to video", "image2video", "img2vid", "i2v"),
        ),
        ("text-to-video", ("text-to-video", "text to video", "text2video", "txt2vid", "t2v")),
        (
            "image-to-image",
            ("image-to-image", "image to image", "image2image", "img2img", "i2i", "image edit"),
        ),
        ("text-to-image", ("text-to-image", "text to image", "text2image", "txt2img", "t2i")),
    )
    for category, needles in patterns:
        if any(needle in text for needle in needles):
            return category
    return "llm"


def _decision_ranked_ids(
    models: list[ModelRecord],
    value_getter: Callable[[ModelRecord], float | None],
    diverse: bool = True,
) -> list[str]:
    groups = [models]
    groups.extend(
        [model for model in models if model_type(model) == category] for category in _MODEL_TYPES
    )
    groups.append([model for model in models if model.open_weights is True])
    groups.extend(
        [model for model in models if model.open_weights is True and model_type(model) == category]
        for category in _MODEL_TYPES
    )
    candidate_ids: set[str] = set()
    for group in groups:
        ranked = (
            _ranked_diverse_ids(group, value_getter)
            if diverse
            else _ranked_ids(group, value_getter)
        )
        candidate_ids.update(ranked)
    candidates = [model for model in models if model.model_id in candidate_ids]
    return [
        model.model_id
        for model in sorted(
            candidates,
            key=lambda item: (
                -float(value_getter(item) or 0),
                item.name.casefold(),
                item.model_id,
            ),
        )
    ]


def _decision_release_ids(models: list[ModelRecord], as_of: datetime | None) -> list[str]:
    groups = [models]
    groups.extend(
        [model for model in models if model_type(model) == category] for category in _MODEL_TYPES
    )
    groups.append([model for model in models if model.open_weights is True])
    groups.extend(
        [model for model in models if model.open_weights is True and model_type(model) == category]
        for category in _MODEL_TYPES
    )
    candidate_ids: set[str] = set()
    for group in groups:
        candidate_ids.update(_new_release_ids(group, as_of))
    candidates = [model for model in models if model.model_id in candidate_ids]
    return [
        model.model_id
        for model in sorted(
            candidates,
            key=lambda item: _hf_meaningful_score(item, as_of),
            reverse=True,
        )
    ]


def _model_family_key(name: str) -> str:
    value = name.casefold().split("/", 1)[-1]
    value = re.sub(r"\s*\([^)]*\)", "", value)
    value = re.sub(r"[-_](?:max|xhigh|high|medium|low|non[- ]reasoning|batch)$", "", value)
    return re.sub(r"[^a-z0-9]+", "-", value).strip("-")


_BENCHMARK_NOISE = re.compile(
    r"\b(?:max|xhigh|high|medium|low|non[- ]reasoning|reasoning|open|thinking|effort|auto)\b"
)
_BENCHMARK_DATE_PATTERNS = (
    re.compile(r"\b(?:19|20)\d{2}[-/.]\d{1,2}[-/.]\d{1,2}\b"),
    re.compile(r"\b(?:19|20)\d{6}\b"),
    re.compile(r"\b(?:sept|oct|nov|dec|jan|feb|mar|apr|may|jun|jul|aug)\w*\s+\d{4}\b"),
    re.compile(r"(?<=[- ])0\d{3}(?=$|[- ])"),
    re.compile(r"(?<=[- ])\d{1,3}k(?=$|[- ])"),
)


def _benchmark_family_key(name: str) -> str:
    """Normalise a display name or slug so benchmark rows match model rows.

    LiveBench publishes slugs such as ``claude-opus-4-6-thinking-auto-high-effort``
    while Artificial Analysis publishes display names such as
    ``Claude Opus 4.6 (max)``. Strip organisation prefixes, effort/thinking
    markers, context-window sizes, and date stamps so both sides collapse to the
    same family key.
    """
    value = name.casefold()
    value = value.split("/", 1)[-1]
    value = value.split(":", 1)[-1]
    value = re.sub(r"[\[(][^\])]*[\])]", " ", value)
    for pattern in _BENCHMARK_DATE_PATTERNS:
        value = pattern.sub(" ", value)
    value = _BENCHMARK_NOISE.sub(" ", value)
    return re.sub(r"[^a-z0-9]+", "-", value).strip("-")


def _model_family_name(name: str) -> str:
    return re.sub(r"\s*\([^)]*\)\s*$", "", name).strip()


def _effort_level(name: str) -> str | None:
    match = re.search(r"\(([^)]*)\)", name)
    if match is None:
        return None
    value = match.group(1).casefold()
    for level in ("non-reasoning", "xhigh", "max", "high", "medium", "low"):
        if level in value:
            return level
    return value.strip() or None


def _meaningful_representatives(models: list[ModelRecord]) -> list[ModelRecord]:
    grouped: dict[str, list[ModelRecord]] = defaultdict(list)
    for model in models:
        grouped[_hf_family_key(model.name)].append(model)
    return [max(group, key=_hf_representative_key) for group in grouped.values()]


def _new_release_ids(models: list[ModelRecord], as_of: datetime | None) -> list[str]:
    ranked = sorted(models, key=lambda item: _hf_meaningful_score(item, as_of), reverse=True)
    selected: list[str] = []
    non_text_modalities: set[str] = set()
    text_count = 0
    for model in ranked:
        non_text = {_release_modality(model)}
        non_text.discard("text")
        if non_text and not non_text_modalities and text_count >= 8:
            selected.append(model.model_id)
            non_text_modalities.update(non_text)
            if len(selected) == 5:
                break
            continue
        if non_text and non_text & non_text_modalities:
            continue
        if not non_text and text_count >= 8:
            continue
        if non_text and non_text & non_text_modalities:
            continue
        selected.append(model.model_id)
        if non_text:
            non_text_modalities.update(non_text)
        else:
            text_count += 1
        if len(selected) == 10:
            break
    return selected


def _release_modality(model: ModelRecord) -> str:
    capabilities = {value.casefold() for value in model.capabilities}
    if any("text-to-image" in value or "image-generation" in value for value in capabilities):
        return "image-generation"
    if any("text-to-video" in value or "video-generation" in value for value in capabilities):
        return "video-generation"
    return "text"


def _hf_family_key(name: str) -> str:
    value = name.casefold().split("/", 1)[-1]
    value = re.sub(r"[-_](?:pe|t2i|i2i|text[- ]encoder|vision|edit)(?:[-_].*)?$", "", value)
    value = re.sub(r"[-_](?:gguf|gptq|awq|lora|adapter|merge|checkpoint).*$", "", value)
    return re.sub(r"[^a-z0-9]+", "-", value).strip("-")


def _hf_representative_key(model: ModelRecord) -> tuple[int, int, int, str]:
    name = model.name.casefold()
    derivative = any(token in name for token in ("-pe", "-t2i", "-i2i", "encoder", "vision"))
    return (not derivative, model.likes or 0, model.downloads or 0, _hf_activity_date(model))


def _hf_meaningful_score(model: ModelRecord, as_of: datetime | None) -> tuple[float, str]:
    date = _hf_release_date(model)
    try:
        timestamp = datetime.fromisoformat(date)
        reference = as_of or datetime.now(UTC)
        age_days = max(0.0, (reference - timestamp).total_seconds() / 86400)
    except (TypeError, ValueError):
        age_days = 30.0
    score = (
        4 * math.log1p(model.likes or 0)
        + 2 * math.log1p(model.downloads or 0)
        + max(0.0, 14.0 - age_days)
        + (10.0 if _is_first_party_hf_name(model.name) else 0.0)
    )
    return score, date


def _is_first_party_hf_name(name: str) -> bool:
    if "/" not in name:
        return False
    owner, model_name = name.split("/", 1)
    owner_tokens = set(re.findall(r"[a-z0-9]+", owner.casefold()))
    model_tokens = set(re.findall(r"[a-z0-9]+", model_name.casefold()))
    return bool(owner_tokens & model_tokens)


def _aa_efficiency(model: ModelRecord) -> float | None:
    if (
        model.intelligence_index is None
        or model.intelligence_index <= 0
        or model.cost_per_task_usd is None
        or model.cost_per_task_usd <= 0
    ):
        return None
    return model.intelligence_index / model.cost_per_task_usd


def _aa_token_efficiency(model: ModelRecord) -> float | None:
    price = blended_token_price(
        model.artificial_analysis_input_price_per_million,
        model.artificial_analysis_output_price_per_million,
    )
    if (
        model.intelligence_index is None
        or model.intelligence_index <= 0
        or price is None
        or price <= 0
    ):
        return None
    return model.intelligence_index / price


def blended_token_price(input_price: float | None, output_price: float | None) -> float | None:
    if input_price is None and output_price is None:
        return None
    if input_price is None:
        return output_price
    if output_price is None:
        return input_price
    return (3 * input_price + output_price) / 4


def _hf_activity_date(model: ModelRecord) -> str:
    dates = [date for date in (model.created_at, model.updated_at) if date]
    return max(dates) if dates else ""


def _hf_release_date(model: ModelRecord) -> str:
    return model.created_at or model.release_date or model.updated_at or ""


def _hf_is_recent_release(model: ModelRecord, as_of: datetime | None) -> bool:
    date = _hf_release_date(model)
    if not date:
        return False
    try:
        released = datetime.fromisoformat(date)
        reference = as_of or datetime.now(UTC)
        return timedelta(0) <= reference - released <= timedelta(days=14)
    except (TypeError, ValueError):
        return False


def _hf_meaningfulness(records: list[RawRecord]) -> tuple[bool | None, str | None]:
    hf_records = [record for record in records if _is_hf_record(record)]
    if not hf_records:
        return None, None
    activity_dates = [
        date for record in hf_records for date in (record.created_at, record.updated_at) if date
    ]
    if not activity_dates:
        return False, "excluded: Hugging Face createdAt/lastModified is missing"
    identity = " ".join(record.name for record in hf_records).lower()
    noise_tokens = (
        "quant",
        "gguf",
        "gptq",
        "awq",
        "lora",
        "adapter",
        "merge",
        "checkpoint",
        "dflash",
        "mtp",
        "finetune",
        "test",
        "demo",
        "upload",
        "converted",
        "benchmark",
        "eval",
        "judge",
        "classifier",
        "embedding",
        "encoder",
        "text-encoder",
    )
    if any(token in identity for token in noise_tokens):
        return False, "excluded: model name contains a derivative or noise marker"
    downloads = _max_field(hf_records, "downloads") or 0
    likes = _max_field(hf_records, "likes") or 0
    parameters = max(
        (record.parameters_b or 0 for record in hf_records),
        default=0,
    )
    if downloads >= 1000:
        return True, "included: downloads >= 1000"
    if likes >= 25:
        return True, "included: likes >= 25"
    if parameters >= 1:
        return True, "included: parameters_b >= 1"
    return False, "excluded: downloads < 1000, likes < 25, and parameters_b < 1"


def _copilot_status(records: list[RawRecord]) -> bool | None:
    explicit = [record for record in records if "copilot" in record.provenance]
    if not explicit:
        return None
    return any(record.tool_calling is True and record.commercial_use is True for record in explicit)


def _is_hf_record(record: RawRecord) -> bool:
    return any(value.lower() in {"huggingface", "hf"} for value in record.provenance) or (
        record.source_url is not None and "huggingface.co/" in record.source_url
    )


def _date_extreme(records: list[RawRecord], field: str, minimum: bool = False) -> str | None:
    values = [getattr(item, field) for item in records if getattr(item, field)]
    return (min if minimum else max)(values) if values else None


def _unavailable_view(view_id: str, title: str, reason: str) -> View:
    return View(
        view_id=view_id,
        title=title,
        annotations={"status": "unavailable", "reason": reason},
        degraded=True,
    )


def _ordered_ids(
    models: list[ModelRecord], key: Callable[[ModelRecord], Any], reverse: bool = False
) -> list[str]:
    return [item.model_id for item in sorted(models, key=key, reverse=reverse)]


def _model_date(model: ModelRecord) -> str | None:
    return model.release_date or model.updated_at or model.created_at


def _popularity(model: ModelRecord) -> tuple[int, int, str]:
    return (model.downloads or 0, model.likes or 0, model.model_id)


def _round(value: float | None) -> float | None:
    return round(value, 6) if value is not None else None
