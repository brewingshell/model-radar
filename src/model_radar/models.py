from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class SourceConfig(StrictModel):
    name: str
    kind: Literal[
        "fixture",
        "huggingface",
        "openrouter",
        "artificial_analysis",
        "artificial_analysis_modality",
        "livebench",
        "evalplus",
        "deepswe",
    ]
    enabled: bool = True
    required: bool = False
    path: str | None = None
    base_url: str | None = None
    endpoint: str | None = None
    token_env: str | None = None
    page_size: int = Field(default=50, ge=1, le=1000)


class FetchConfig(StrictModel):
    concurrency: int = Field(default=4, ge=1, le=32)
    max_pages: int = Field(default=100, ge=1, le=10000)
    max_records: int = Field(default=10000, ge=1, le=1_000_000)
    timeout_seconds: float = Field(default=10.0, gt=0, le=300)


class RenderConfig(StrictModel):
    max_html_bytes: int = Field(default=2_000_000, ge=1000)
    max_models: int = Field(default=1000, ge=1)


class CopilotModelConfig(StrictModel):
    name: str
    family: str | None = None
    level: str | None = None
    input_credits_per_million: float = Field(ge=0)
    output_credits_per_million: float = Field(ge=0)
    enabled: bool = True


class CopilotConfig(StrictModel):
    enabled: bool = False
    source: str = "organization-catalog"
    models: list[CopilotModelConfig] = Field(default_factory=list)


class AppConfig(StrictModel):
    schema_version: str = "1"
    output: str = "out"
    fetch: FetchConfig = FetchConfig()
    render: RenderConfig = RenderConfig()
    copilot: CopilotConfig = CopilotConfig()
    model_type_tabs: dict[str, list[str]] = Field(default_factory=dict)
    sources: list[SourceConfig] = Field(default_factory=list)


class SourceFailure(StrictModel):
    code: str
    message: str
    required: bool


class SourceStatus(StrictModel):
    name: str
    kind: str
    required: bool
    enabled: bool
    status: Literal["ok", "failed", "disabled"]
    pages: int = 0
    records: int = 0
    total_records: int | None = None
    coverage: str | None = None
    warning: str | None = None
    duration_ms: int = 0
    failure: SourceFailure | None = None


class RawRecord(StrictModel):
    source_id: str
    name: str
    organization: str | None = None
    family: str | None = None
    variant: str | None = None
    source_url: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    modalities: list[str] = Field(default_factory=list)
    capabilities: list[str] = Field(default_factory=list)
    supported_parameters: list[str] = Field(default_factory=list)
    tool_calling: bool | None = None
    architecture: dict[str, str] = Field(default_factory=dict)
    provider_details: dict[str, str] = Field(default_factory=dict)
    parameters_b: float | None = None
    context_length: int | None = None
    quantization: str | None = None
    release_date: str | None = None
    license: str | None = None
    open_weights: bool | None = None
    commercial_use: bool | None = None
    origin: Literal["open", "proprietary", "unknown"] = "unknown"
    providers: list[str] = Field(default_factory=list)
    price_per_million: float | None = None
    output_price_per_million: float | None = None
    pricing: dict[str, float] = Field(default_factory=dict)
    downloads: int | None = Field(default=None, ge=0)
    likes: int | None = Field(default=None, ge=0)
    benchmarks: dict[str, float] = Field(default_factory=dict)
    recent_signals: dict[str, float] = Field(default_factory=dict)
    artificial_analysis_source_url: str | None = None
    artificial_analysis_model_url: str | None = None
    artificial_analysis_creator: str | None = None
    artificial_analysis_context_window: int | None = None
    intelligence_index: float | None = None
    cost_per_task_usd: float | None = None
    artificial_analysis_input_price_per_million: float | None = None
    artificial_analysis_output_price_per_million: float | None = None
    artificial_analysis_modality: str | None = None
    artificial_analysis_modality_elo: float | None = None
    artificial_analysis_modality_samples: int | None = None
    artificial_analysis_modality_release: str | None = None
    artificial_analysis_modality_cost: float | None = None
    artificial_analysis_modality_cost_unit: str | None = None
    median_output_tokens_per_second: float | None = None
    latency_first_chunk_seconds: float | None = None
    total_response_seconds: float | None = None
    meaningful_new_hf: bool | None = None
    meaningful_new_hf_reason: str | None = None
    benchmark_source: str | None = None
    benchmark_release: str | None = None
    benchmark_index: float | None = Field(default=None, ge=0, le=100)
    benchmark_components: dict[str, float] = Field(default_factory=dict)
    benchmark_cost_per_task: float | None = Field(default=None, ge=0)
    benchmark_effort: str | None = None
    benchmark_harness: str | None = None
    benchmark_confidence_interval: float | None = Field(default=None, ge=0)
    provenance: list[str] = Field(default_factory=list)
    field_provenance: dict[str, list[str]] = Field(default_factory=dict)
    confidence: float = Field(default=0.5, ge=0, le=1)


class Score(StrictModel):
    value: float | None = None
    components: dict[str, float | None] = Field(default_factory=dict)
    cohort: str | None = None
    weights: dict[str, float] = Field(default_factory=dict)
    algorithm_version: str = "1"
    configuration_hash: str
    provisional: bool = False


class ModelRecord(StrictModel):
    model_id: str
    slug: str
    organization: str | None = None
    family: str | None = None
    name: str
    model_family: str | None = None
    effort_level: str | None = None
    variant: str | None = None
    source_urls: list[str] = Field(default_factory=list)
    created_at: str | None = None
    updated_at: str | None = None
    modalities: list[str] = Field(default_factory=list)
    capabilities: list[str] = Field(default_factory=list)
    supported_parameters: list[str] = Field(default_factory=list)
    tool_calling: bool | None = None
    architecture: dict[str, str] = Field(default_factory=dict)
    provider_details: dict[str, str] = Field(default_factory=dict)
    parameters_b: float | None = None
    context_length: int | None = None
    quantization: str | None = None
    release_date: str | None = None
    license: str | None = None
    open_weights: bool | None = None
    commercial_use: bool | None = None
    origin: Literal["open", "proprietary", "unknown"] = "unknown"
    providers: list[str] = Field(default_factory=list)
    price_per_million: float | None = None
    output_price_per_million: float | None = None
    pricing: dict[str, float] = Field(default_factory=dict)
    downloads: int | None = Field(default=None, ge=0)
    likes: int | None = Field(default=None, ge=0)
    benchmarks: dict[str, float] = Field(default_factory=dict)
    recent_signals: dict[str, float] = Field(default_factory=dict)
    artificial_analysis_source_url: str | None = None
    artificial_analysis_model_url: str | None = None
    artificial_analysis_creator: str | None = None
    artificial_analysis_context_window: int | None = None
    intelligence_index: float | None = None
    cost_per_task_usd: float | None = None
    artificial_analysis_input_price_per_million: float | None = None
    artificial_analysis_output_price_per_million: float | None = None
    artificial_analysis_modality: str | None = None
    artificial_analysis_modality_elo: float | None = None
    artificial_analysis_modality_samples: int | None = None
    artificial_analysis_modality_release: str | None = None
    artificial_analysis_modality_cost: float | None = None
    artificial_analysis_modality_cost_unit: str | None = None
    median_output_tokens_per_second: float | None = None
    latency_first_chunk_seconds: float | None = None
    total_response_seconds: float | None = None
    meaningful_new_hf: bool | None = None
    meaningful_new_hf_reason: str | None = None
    livebench_index: float | None = Field(default=None, ge=0, le=100)
    evalplus_index: float | None = Field(default=None, ge=0, le=100)
    deepswe_index: float | None = Field(default=None, ge=0, le=100)
    merged_benchmark_index: float | None = Field(default=None, ge=0, le=100)
    benchmark_coverage: int = 0
    benchmark_release: str | None = None
    benchmark_effort: str | None = None
    benchmark_cost_per_task: float | None = Field(default=None, ge=0)
    benchmark_harness: str | None = None
    provenance: list[str] = Field(default_factory=list)
    field_provenance: dict[str, list[str]] = Field(default_factory=dict)
    confidence: float = Field(default=0.5, ge=0, le=1)
    scores: dict[str, Score] = Field(default_factory=dict)
    copilot_ready: bool | None = None
    copilot_model_name: str | None = None
    copilot_input_credits_per_million: float | None = None
    copilot_output_credits_per_million: float | None = None


class View(StrictModel):
    view_id: str
    title: str
    model_ids: list[str] = Field(default_factory=list)
    annotations: dict[str, str] = Field(default_factory=dict)
    degraded: bool = False


class Snapshot(StrictModel):
    schema_version: str = "1"
    snapshot_id: str
    generated_at: datetime
    status: Literal["complete", "degraded"]
    source_status: list[SourceStatus]
    warnings: list[str] = Field(default_factory=list)
    models: list[ModelRecord] = Field(default_factory=list)
    views: list[View] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)
    truncation: dict[str, Any] = Field(default_factory=dict)
    changes: dict[str, Any] = Field(default_factory=dict)
