from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from model_radar.analysis import apply_copilot_catalog, attach_benchmarks, build_views, normalize
from model_radar.connectors import (
    ArtificialAnalysisConnector,
    ArtificialAnalysisModalityConnector,
    BoundedHttpClient,
    DeepSweConnector,
    EvalPlusConnector,
    FixtureConnector,
    HuggingFaceConnector,
    LiveBenchConnector,
    OpenRouterConnector,
)
from model_radar.models import (
    AppConfig,
    FetchConfig,
    RawRecord,
    Snapshot,
    SourceConfig,
    SourceStatus,
)
from model_radar.publisher import Publisher
from model_radar.source import Connector, fetch_sources


def make_connector(config: SourceConfig, client: BoundedHttpClient | None = None) -> Connector:
    if config.kind == "fixture":
        return FixtureConnector(config)
    if config.kind == "huggingface":
        return HuggingFaceConnector(config, client=client)
    if config.kind == "openrouter":
        return OpenRouterConnector(config, client=client)
    if config.kind == "artificial_analysis":
        return ArtificialAnalysisConnector(config, client=client)
    if config.kind == "artificial_analysis_modality":
        return ArtificialAnalysisModalityConnector(config, client=client)
    if config.kind == "livebench":
        return LiveBenchConnector(config, client=client)
    if config.kind == "evalplus":
        return EvalPlusConnector(config, client=client)
    if config.kind == "deepswe":
        return DeepSweConnector(config, client=client)
    raise ValueError(f"unsupported source kind {config.kind}")


async def _fetch_configured_sources(
    configs: list[SourceConfig], fetch_config: FetchConfig
) -> tuple[list[RawRecord], list[SourceStatus], list[str], bool]:
    async with BoundedHttpClient(timeout_seconds=fetch_config.timeout_seconds) as client:
        connectors: list[Connector] = [make_connector(item, client=client) for item in configs]
        return await fetch_sources(connectors, fetch_config)


def run_pipeline(
    config: AppConfig,
    source_override: str | None = None,
    generated_at: datetime | None = None,
    publish: bool = True,
) -> Snapshot:
    configs = [
        item
        for item in config.sources
        if item.enabled
        and (
            source_override is None or item.kind == source_override or item.name == source_override
        )
    ]
    records, statuses, warnings, degraded = asyncio.run(
        _fetch_configured_sources(configs, config.fetch)
    )
    benchmark_records = [record for record in records if record.benchmark_source is not None]
    models = normalize([record for record in records if record.benchmark_source is None])
    models = attach_benchmarks(models, benchmark_records)
    models = apply_copilot_catalog(models, config.copilot)
    timestamp = generated_at or datetime.now(UTC)
    views = build_views(models, degraded, as_of=timestamp)
    benchmark_model_count = sum(bool(model.benchmarks) for model in models)
    aa_model_count = sum(model.intelligence_index is not None for model in models)
    meaningful_new_hf_count = sum(model.meaningful_new_hf is True for model in models)
    if not aa_model_count:
        warnings.append(
            "Artificial Analysis performance views are unavailable: no Intelligence Index data"
        )
    if not meaningful_new_hf_count:
        warnings.append(
            "meaningful-new-hf view is unavailable: no Hugging Face model passed the heuristic"
        )
    digest = hashlib.sha256(
        json.dumps(
            [model.model_dump(mode="json") for model in models],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()[:16]
    snapshot = Snapshot(
        snapshot_id=f"{timestamp.strftime('%Y%m%dT%H%M%SZ')}-{digest}",
        generated_at=timestamp,
        status="degraded" if degraded else "complete",
        source_status=statuses,
        warnings=warnings,
        models=models,
        views=views,
        summary={
            "model_count": len(models),
            "source_count": len(statuses),
            "source_record_count": sum(status.records for status in statuses),
            "source_total_count": sum(
                status.total_records for status in statuses if status.total_records is not None
            ),
            "benchmark_model_count": benchmark_model_count,
            "benchmark_coverage": benchmark_model_count / len(models) if models else 0.0,
            "aa_performance_model_count": aa_model_count,
            "meaningful_new_hf_count": meaningful_new_hf_count,
            "primary_view_ids": [
                "performance-top5",
                "performance-per-token-top5",
                "meaningful-new-hf-top5",
                "org-copilot-per-token-top10",
                "org-copilot-best-top10",
            ],
            "model_type_tabs": config.model_type_tabs,
            "source_coverage": {
                status.name: {
                    "records": status.records,
                    "total_records": status.total_records,
                    "coverage": status.coverage,
                }
                for status in statuses
            },
        },
        truncation={
            "records": any(
                status.coverage in {"partial_catalog", "bounded_catalog", "bounded_recent"}
                for status in statuses
            )
        },
    )
    if publish:
        Publisher(config.output, max_html_bytes=config.render.max_html_bytes).publish(snapshot)
    return snapshot


def load_published_snapshot(output: str | Path) -> Snapshot:
    root = Path(output)
    path = root / "snapshot.json"
    if not path.exists():
        path = root / "current" / "snapshot.json"
    return Snapshot.model_validate(json.loads(path.read_text(encoding="utf-8")))
