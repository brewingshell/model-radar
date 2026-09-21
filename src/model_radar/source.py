from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Protocol

from model_radar.models import FetchConfig, RawRecord, SourceConfig, SourceFailure, SourceStatus


class ConnectorError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class PaginationError(ConnectorError):
    pass


class RequiredSourceError(ConnectorError):
    pass


@dataclass(frozen=True)
class Page:
    records: list[RawRecord]
    next_cursor: str | None = None
    total_records: int | None = None
    coverage: str | None = None
    warning: str | None = None


class Connector(Protocol):
    config: SourceConfig

    async def page(self, cursor: str | None) -> Page: ...


async def fetch_connector(
    connector: Connector, max_pages: int, max_records: int
) -> tuple[list[RawRecord], SourceStatus]:
    started = time.monotonic()
    records: list[RawRecord] = []
    cursor: str | None = None
    seen: set[str | None] = set()
    pages = 0
    total_records: int | None = None
    coverage: str | None = None
    source_warning: str | None = None
    while True:
        if pages >= max_pages:
            raise PaginationError("max_pages", f"source {connector.config.name} exceeded max_pages")
        if cursor in seen:
            raise PaginationError("cursor_loop", f"source {connector.config.name} repeated cursor")
        seen.add(cursor)
        page = await connector.page(cursor)
        pages += 1
        records.extend(page.records)
        if page.total_records is not None:
            total_records = max(total_records or 0, page.total_records)
        coverage = page.coverage or coverage
        source_warning = page.warning or source_warning
        if len(records) > max_records:
            raise PaginationError(
                "max_records", f"source {connector.config.name} exceeded max_records"
            )
        if page.next_cursor is None:
            break
        cursor = page.next_cursor
    status = SourceStatus(
        name=connector.config.name,
        kind=connector.config.kind,
        required=connector.config.required,
        enabled=True,
        status="ok",
        pages=pages,
        records=len(records),
        total_records=total_records,
        coverage=coverage or "complete",
        warning=source_warning,
        duration_ms=int((time.monotonic() - started) * 1000),
    )
    return records, status


async def fetch_sources(
    connectors: list[Connector], fetch_config: FetchConfig
) -> tuple[list[RawRecord], list[SourceStatus], list[str], bool]:
    max_pages = fetch_config.max_pages
    max_records = fetch_config.max_records
    results = await asyncio.gather(
        *(fetch_connector(connector, max_pages, max_records) for connector in connectors),
        return_exceptions=True,
    )
    records: list[RawRecord] = []
    statuses: list[SourceStatus] = []
    warnings: list[str] = []
    degraded = False
    for connector, result in zip(connectors, results, strict=True):
        if isinstance(result, BaseException):
            failure = SourceFailure(
                code=getattr(result, "code", "source_error"),
                message=str(result),
                required=connector.config.required,
            )
            status = SourceStatus(
                name=connector.config.name,
                kind=connector.config.kind,
                required=connector.config.required,
                enabled=True,
                status="failed",
                failure=failure,
            )
            statuses.append(status)
            if connector.config.required:
                raise RequiredSourceError(failure.code, failure.message)
            degraded = True
            warnings.append(f"{connector.config.name}: {failure.message}")
        else:
            source_records, status = result
            records.extend(source_records)
            statuses.append(status)
            if status.warning:
                warnings.append(f"{status.name}: {status.warning}")
    return records, statuses, warnings, degraded
