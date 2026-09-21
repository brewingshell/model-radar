from __future__ import annotations

import csv
import io
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, ClassVar, Self
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from tenacity import AsyncRetrying, RetryCallState, retry_if_exception_type, stop_after_attempt
from tenacity.wait import wait_base

from model_radar.models import RawRecord, SourceConfig
from model_radar.source import ConnectorError, Page


class FixtureConnector:
    def __init__(self, config: SourceConfig) -> None:
        if not config.path:
            raise ConnectorError("fixture_config", f"fixture source {config.name} needs path")
        self.config = config
        try:
            with Path(config.path).open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.pages = [
                Page(
                    records=[RawRecord.model_validate(item) for item in page.get("records", [])],
                    next_cursor=page.get("next_cursor"),
                )
                for page in payload["pages"]
            ]
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise ConnectorError(
                "fixture_load", f"cannot load fixture {config.path}: {exc}"
            ) from exc
        self._cursor_index: dict[str | None, int] = {None: 0}
        for index, page in enumerate(self.pages):
            if page.next_cursor is not None:
                self._cursor_index[page.next_cursor] = index + 1

    async def page(self, cursor: str | None) -> Page:
        index = self._cursor_index.get(cursor)
        if index is None or index >= len(self.pages):
            raise ConnectorError("fixture_cursor", f"unknown fixture cursor {cursor!r}")
        return self.pages[index]


class BoundedHttpClient:
    def __init__(
        self,
        timeout_seconds: float = 10.0,
        max_bytes: int = 5_000_000,
        transport: httpx.AsyncBaseTransport | None = None,
        max_attempts: int = 3,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_bytes = max_bytes
        self.transport = transport
        self.max_attempts = max_attempts
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> Self:
        timeout = httpx.Timeout(
            self.timeout_seconds,
            connect=self.timeout_seconds,
            read=self.timeout_seconds,
            write=self.timeout_seconds,
            pool=self.timeout_seconds,
        )
        self._client = httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=False,
            transport=self.transport,
            headers={"User-Agent": "model-radar/0.1"},
        )
        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def get_json(
        self, url: str, headers: dict[str, str] | None = None, params: dict[str, Any] | None = None
    ) -> Any:
        if self._client is None:
            async with self:
                return await self.get_json(url, headers=headers, params=params)

        retrying = AsyncRetrying(
            stop=stop_after_attempt(self.max_attempts),
            wait=_RetryAfterWait(),
            retry=retry_if_exception_type(RetryableHttpError),
            reraise=True,
        )
        try:
            async for attempt in retrying:
                with attempt:
                    return await self._get_json_once(url, headers, params)
        except ConnectorError:
            raise
        except (httpx.TimeoutException, httpx.RequestError) as exc:
            raise ConnectorError("http", f"request failed for {url}: {exc}") from exc
        except ValueError as exc:
            raise ConnectorError("json", f"source response was not valid JSON: {exc}") from exc

        raise ConnectorError("http", f"request failed for {url}")

    async def get_html(self, url: str, headers: dict[str, str] | None = None) -> str:
        if self._client is None:
            async with self:
                return await self.get_html(url, headers=headers)

        retrying = AsyncRetrying(
            stop=stop_after_attempt(self.max_attempts),
            wait=_RetryAfterWait(),
            retry=retry_if_exception_type(RetryableHttpError),
            reraise=True,
        )
        try:
            async for attempt in retrying:
                with attempt:
                    return await self._get_html_once(url, headers)
        except ConnectorError:
            raise
        except (httpx.TimeoutException, httpx.RequestError) as exc:
            raise ConnectorError("http", f"request failed for {url}: {exc}") from exc
        raise ConnectorError("http", f"request failed for {url}")

    async def get_text(self, url: str, headers: dict[str, str] | None = None) -> str:
        if self._client is None:
            async with self:
                return await self.get_text(url, headers=headers)
        try:
            response = await self._client.get(url, headers=headers)
            response.raise_for_status()
        except (httpx.TimeoutException, httpx.RequestError, httpx.HTTPStatusError) as exc:
            raise ConnectorError("http", f"request failed for {url}: {exc}") from exc
        if len(response.content) > self.max_bytes:
            raise ConnectorError("response_size", "response exceeded configured size limit")
        return response.text

    async def _get_json_once(
        self, url: str, headers: dict[str, str] | None, params: dict[str, Any] | None
    ) -> Any:
        if self._client is None:
            raise ConnectorError("client_state", "HTTP client is not open")
        try:
            response = await self._client.get(url, headers=headers, params=params)
        except httpx.TimeoutException as exc:
            raise RetryableHttpError("timeout", f"request timed out for {url}") from exc
        except httpx.RequestError as exc:
            raise RetryableHttpError("connection", f"request failed for {url}: {exc}") from exc
        if response.status_code in {408, 425, 429} or response.status_code >= 500:
            raise RetryableHttpError(
                "http_retry",
                f"source returned HTTP {response.status_code} for {url}",
                retry_after=_retry_after(response),
            )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ConnectorError(
                "http_status", f"source returned HTTP {response.status_code} for {url}"
            ) from exc
        if len(response.content) > self.max_bytes:
            raise ConnectorError("response_size", "response exceeded configured size limit")
        content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if content_type != "application/json" and not content_type.endswith("+json"):
            raise ConnectorError("content_type", "source response was not JSON")
        return response.json()

    async def _get_html_once(self, url: str, headers: dict[str, str] | None) -> str:
        if self._client is None:
            raise ConnectorError("client_state", "HTTP client is not open")
        try:
            response = await self._client.get(url, headers=headers)
        except httpx.TimeoutException as exc:
            raise RetryableHttpError("timeout", f"request timed out for {url}") from exc
        except httpx.RequestError as exc:
            raise RetryableHttpError("connection", f"request failed for {url}") from exc
        if response.status_code in {408, 425, 429} or response.status_code >= 500:
            raise RetryableHttpError(
                "http_retry",
                f"source returned HTTP {response.status_code} for {url}",
                retry_after=_retry_after(response),
            )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ConnectorError(
                "http_status", f"source returned HTTP {response.status_code} for {url}"
            ) from exc
        if len(response.content) > self.max_bytes:
            raise ConnectorError("response_size", "response exceeded configured size limit")
        content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if content_type not in {"text/html", "application/xhtml+xml"}:
            raise ConnectorError("content_type", "source response was not HTML")
        return response.text


class RetryableHttpError(ConnectorError):
    def __init__(self, code: str, message: str, retry_after: float | None = None) -> None:
        super().__init__(code, message)
        self.retry_after = retry_after


class _RetryAfterWait(wait_base):
    def __call__(self, retry_state: RetryCallState) -> float:
        if retry_state.outcome is not None:
            error = retry_state.outcome.exception()
            if isinstance(error, RetryableHttpError) and error.retry_after is not None:
                return min(max(error.retry_after, 0.0), 30.0)
        return float(min(2 ** max(retry_state.attempt_number - 1, 0), 8.0))


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("Retry-After")
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


class HuggingFaceConnector:
    def __init__(self, config: SourceConfig, client: BoundedHttpClient | None = None) -> None:
        self.config = config
        self.client = client or BoundedHttpClient()
        self.base_url = config.base_url or "https://huggingface.co/api/models"

    async def page(self, cursor: str | None) -> Page:
        payload = await self.client.get_json(
            self.base_url,
            params={"limit": self.config.page_size, "cursor": cursor}
            if cursor
            else {"limit": self.config.page_size},
        )
        items: object
        next_cursor: str | None = None
        warning: str | None = None
        if isinstance(payload, list):
            items = payload
            total_records = None
            coverage = "bounded_recent"
            warning = (
                "Hugging Face returned a bounded recent list without total-count or cursor metadata"
            )
        elif isinstance(payload, dict):
            items = payload.get("models", payload.get("data", []))
            next_cursor = _optional_str(payload.get("next_cursor"))
            total_records = _optional_int(payload.get("total_count"))
            coverage = "complete" if total_records is not None else "bounded_recent"
            warning = (
                None
                if total_records is not None
                else "Hugging Face did not expose total-count metadata for this response"
            )
        else:
            raise ConnectorError("payload", "Hugging Face response must be a list or object")
        if not isinstance(items, list):
            raise ConnectorError("payload", "Hugging Face models must be a list")
        records = [
            RawRecord.model_validate(_hf_record(item, self.config.name))
            for item in items
            if isinstance(item, dict)
        ]
        return Page(
            records=records,
            next_cursor=next_cursor,
            total_records=total_records,
            coverage=coverage,
            warning=warning,
        )


class OpenRouterConnector:
    def __init__(
        self,
        config: SourceConfig,
        client: BoundedHttpClient | None = None,
        environ: dict[str, str] | None = None,
    ) -> None:
        self.config = config
        self.client = client or BoundedHttpClient()
        env = os.environ if environ is None else environ
        token = env.get(config.token_env or "OPENROUTER_API_KEY")
        self.headers = {"Authorization": f"Bearer {token}"} if token else {}
        self.base_url = config.base_url or "https://openrouter.ai/api/v1/models"

    async def page(self, cursor: str | None) -> Page:
        payload = await self.client.get_json(
            self.base_url,
            headers=self.headers,
            params={"limit": self.config.page_size, "cursor": cursor}
            if cursor
            else {"limit": self.config.page_size},
        )
        if not isinstance(payload, dict):
            raise ConnectorError("payload", "OpenRouter response must be an object")
        items = payload.get("data", [])
        if not isinstance(items, list):
            raise ConnectorError("payload", "OpenRouter data must be a list")
        total_records = _optional_int(payload.get("total_count"))
        returned_records = len(items)
        coverage = (
            "complete_catalog"
            if total_records is not None and returned_records >= total_records
            else "partial_catalog"
            if total_records is not None
            else "bounded_catalog"
        )
        warning = None
        if coverage == "partial_catalog":
            warning = (
                f"OpenRouter reported {total_records} catalog records but returned "
                f"{returned_records}; no cursor was exposed for completion"
            )
        if coverage == "bounded_catalog":
            warning = "OpenRouter did not expose total-count metadata for this response"
        records = [
            RawRecord.model_validate(_or_record(item, self.config.name))
            for item in items
            if isinstance(item, dict)
        ]
        return Page(
            records=records,
            total_records=total_records,
            coverage=coverage,
            warning=warning,
        )


class ArtificialAnalysisConnector:
    _HEADERS = (
        "Model",
        "Context Window",
        "Creator",
        "Artificial Analysis Intelligence Index",
        "Cost per Task USD",
        "Median Tokens/s",
        "Latency First Chunk (s)",
        "Total Response (s)",
        "Further Analysis",
    )

    def __init__(self, config: SourceConfig, client: BoundedHttpClient | None = None) -> None:
        self.config = config
        self.client = client or BoundedHttpClient()
        self.base_url = config.base_url or "https://artificialanalysis.ai/leaderboards/models"

    async def page(self, cursor: str | None) -> Page:
        if cursor is not None:
            raise ConnectorError("pagination", "Artificial Analysis leaderboard is not paginated")
        html = await self.client.get_html(self.base_url)
        soup = BeautifulSoup(html, "html.parser")
        embedded_metrics = _aa_embedded_metrics(html)
        table = next(
            (
                candidate
                for candidate in soup.find_all("table")
                if _table_headers(candidate) == list(self._HEADERS)
            ),
            None,
        )
        if table is None:
            raise ConnectorError("schema", "Artificial Analysis leaderboard table was not found")
        records: list[RawRecord] = []
        for row in table.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) != len(self._HEADERS):
                continue
            link = next(
                (
                    candidate
                    for candidate in row.find_all("a", href=True)
                    if str(candidate.get("href", "")).startswith("/models/")
                    and "/providers" not in str(candidate.get("href", ""))
                ),
                None,
            )
            if link is None:
                continue
            model_name = " ".join(cells[0].get_text(" ", strip=True).split())
            model_url = urljoin(self.base_url, str(link.get("href")))
            records.append(
                RawRecord.model_validate(
                    _aa_record(
                        source=self.config.name,
                        source_url=self.base_url,
                        model_url=model_url,
                        name=model_name,
                        cells=[" ".join(cell.get_text(" ", strip=True).split()) for cell in cells],
                        embedded_metrics=embedded_metrics.get(
                            urlparse(model_url).path.rstrip("/").rsplit("/", 1)[-1]
                        ),
                    )
                )
            )
        return Page(
            records=records,
            total_records=len(records),
            coverage="complete_leaderboard",
            warning="Artificial Analysis cost is cost per benchmark task, not token pricing",
        )


class ArtificialAnalysisModalityConnector:
    _REQUIRED_HEADERS: ClassVar[frozenset[str]] = frozenset({"model", "elo", "samples", "released"})

    def __init__(self, config: SourceConfig, client: BoundedHttpClient | None = None) -> None:
        self.config = config
        self.client = client or BoundedHttpClient()
        self.base_url = config.base_url or "https://artificialanalysis.ai/leaderboards/models"
        self.modality = config.endpoint or ""
        if not self.base_url or self.modality not in {
            "text-to-image",
            "image-to-image",
            "text-to-video",
            "image-to-video",
        }:
            raise ConnectorError(
                "modality_config",
                f"{config.name} needs a leaderboard URL and a supported modality endpoint",
            )

    async def page(self, cursor: str | None) -> Page:
        if cursor is not None:
            raise ConnectorError(
                "pagination", "Artificial Analysis modality leaderboard is not paginated"
            )
        html = await self.client.get_html(self.base_url)
        soup = BeautifulSoup(html, "html.parser")
        table = next(
            (
                candidate
                for candidate in soup.find_all("table")
                if self._REQUIRED_HEADERS.issubset(
                    {_normalize_header(header) for header in _table_headers(candidate)}
                )
            ),
            None,
        )
        if table is None:
            raise ConnectorError(
                "schema", f"Artificial Analysis {self.modality} leaderboard table was not found"
            )
        headers = [_normalize_header(header) for header in _table_headers(table)]
        model_index = headers.index("model")
        creator_index = headers.index("creator") if "creator" in headers else None
        elo_index = headers.index("elo")
        samples_index = headers.index("samples")
        released_index = headers.index("released")
        price_index = next(
            (index for index, header in enumerate(headers) if header.startswith("api pricing")),
            None,
        )
        records: list[RawRecord] = []
        for row in table.find_all("tr"):
            cells = [
                " ".join(cell.get_text(" ", strip=True).split())
                for cell in row.find_all("td", recursive=False)
            ]
            if len(cells) <= max(model_index, elo_index, samples_index, released_index):
                continue
            name = cells[model_index]
            if not name or name.casefold() == "model":
                continue
            creator = cells[creator_index] if creator_index is not None else None
            release = cells[released_index] or None
            cost, cost_unit = _modality_cost(cells[price_index] if price_index is not None else "")
            source_id = f"artificial-analysis:{self.modality}:{_slug_text(name)}"
            source_url = self.base_url
            open_weights = "open weight" in name.casefold()
            records.append(
                RawRecord(
                    source_id=source_id,
                    name=name,
                    organization=creator or None,
                    source_url=source_url,
                    created_at=_aa_release_iso(release),
                    release_date=_aa_release_iso(release),
                    modalities=["video" if "video" in self.modality else "image"],
                    capabilities=[self.modality],
                    open_weights=True if open_weights else None,
                    origin="open" if open_weights else "proprietary",
                    artificial_analysis_source_url=source_url,
                    artificial_analysis_model_url=source_url,
                    artificial_analysis_modality=self.modality,
                    artificial_analysis_modality_elo=_number(cells[elo_index]),
                    artificial_analysis_modality_samples=(
                        int(samples)
                        if (samples := _number(cells[samples_index])) is not None
                        else None
                    ),
                    artificial_analysis_modality_release=release,
                    artificial_analysis_modality_cost=cost,
                    artificial_analysis_modality_cost_unit=cost_unit,
                    provenance=[self.config.name],
                    confidence=0.9,
                )
            )
        return Page(
            records=records,
            total_records=len(records),
            coverage="complete_leaderboard",
            warning=f"Artificial Analysis {self.modality} API cost is reported per {cost_unit or 'source unit'}",
        )


class LiveBenchConnector:
    def __init__(self, config: SourceConfig, client: BoundedHttpClient | None = None) -> None:
        self.config = config
        self.client = client or BoundedHttpClient()
        self.base_url = config.base_url or "https://livebench.ai"
        self.release = "2026-06-25"

    async def page(self, cursor: str | None) -> Page:
        if cursor is not None:
            raise ConnectorError("pagination", "LiveBench release is a single static dataset")
        release = self.config.endpoint or self.release
        suffix = release.replace("-", "_")
        table_text = await self.client.get_text(f"{self.base_url}/table_{suffix}.csv")
        category_payload = json.loads(
            await self.client.get_text(f"{self.base_url}/categories_{suffix}.json")
        )
        cost_text = await self.client.get_text(f"{self.base_url}/cost_{suffix}.csv")
        rows = list(csv.DictReader(io.StringIO(table_text)))
        cost_rows = {row.get("model"): row for row in csv.DictReader(io.StringIO(cost_text))}
        category_columns = [column for values in category_payload.values() for column in values]
        records = []
        for row in rows:
            scores: list[float] = []
            for column in category_columns:
                score = _number(str(row.get(column, "")))
                if score is not None:
                    scores.append(score)
            if not scores:
                continue
            cost_row = cost_rows.get(row.get("model"), {})
            records.append(
                RawRecord(
                    source_id=f"livebench:{row['model']}",
                    name=row["model"],
                    source_url=self.base_url,
                    benchmark_source=self.config.name,
                    benchmark_release=release,
                    benchmark_index=round(sum(scores) / len(scores), 4),
                    benchmark_effort=_benchmark_effort(row["model"]),
                    benchmark_components={
                        key: score
                        for key in category_columns
                        if (score := _number(str(row.get(key, "")))) is not None
                    },
                    benchmark_cost_per_task=_number(
                        str(cost_row.get("cost_per_successful_task", ""))
                    ),
                    provenance=[self.config.name],
                    confidence=0.85,
                )
            )
        return Page(records=records, total_records=len(records), coverage="complete_leaderboard")


class EvalPlusConnector:
    def __init__(self, config: SourceConfig, client: BoundedHttpClient | None = None) -> None:
        self.config = config
        self.client = client or BoundedHttpClient()
        self.base_url = config.base_url or "https://evalplus.github.io/results.json"

    async def page(self, cursor: str | None) -> Page:
        if cursor is not None:
            raise ConnectorError("pagination", "EvalPlus results are a single static dataset")
        payload = await self.client.get_json(self.base_url)
        if not isinstance(payload, dict):
            raise ConnectorError("payload", "EvalPlus results must be an object")
        records = []
        for name, item in payload.items():
            pass_at_1 = item.get("pass@1", {}) if isinstance(item, dict) else {}
            values: list[float] = []
            for key in ("humaneval+", "mbpp+"):
                value = pass_at_1.get(key)
                if isinstance(value, (int, float)):
                    values.append(float(value))
            if not values:
                continue
            records.append(
                RawRecord(
                    source_id=f"evalplus:{name}",
                    name=name,
                    source_url=str(item.get("link", self.base_url)),
                    benchmark_source=self.config.name,
                    benchmark_release="evalplus-results",
                    benchmark_index=round(sum(values) / len(values), 4),
                    benchmark_components={
                        key: float(value)
                        for key, value in pass_at_1.items()
                        if isinstance(value, (int, float))
                    },
                    provenance=[self.config.name],
                    confidence=0.75,
                )
            )
        return Page(records=records, total_records=len(records), coverage="static_leaderboard")


class DeepSweConnector:
    def __init__(self, config: SourceConfig, client: BoundedHttpClient | None = None) -> None:
        self.config = config
        self.client = client or BoundedHttpClient()
        self.base_url = config.base_url or "https://deepswe.datacurve.ai/"

    async def page(self, cursor: str | None) -> Page:
        if cursor is not None:
            raise ConnectorError("pagination", "DeepSWE is a single public leaderboard")
        soup = BeautifulSoup(await self.client.get_html(self.base_url), "html.parser")
        rows = soup.select("[data-chart-pin-source]")
        if not rows:
            raise ConnectorError("schema", "DeepSWE leaderboard rows were not found")
        records = []
        for row in rows:
            text = " ".join(row.get_text(" ", strip=True).split())
            model_node = row.select_one("div.flex.min-w-0.items-center")
            name = model_node.get_text(" ", strip=True).split("[")[0].strip() if model_node else ""
            match = re.search(
                r"(\d+(?:\.\d+)?)\s*%.*?Avg cost\s*\$([\d.]+).*?Out tok\s*([^ ]+).*?Steps\s*(\d+)",
                text,
            )
            if not name or match is None:
                continue
            effort_match = re.search(r"\[\s*([^]]+)\s*\]", text)
            records.append(
                RawRecord(
                    source_id=f"deepswe:{name}:{effort_match.group(1) if effort_match else 'default'}",
                    name=name,
                    source_url=self.base_url,
                    benchmark_source=self.config.name,
                    benchmark_release="current",
                    benchmark_index=float(match.group(1)),
                    benchmark_cost_per_task=float(match.group(2)),
                    benchmark_effort=effort_match.group(1).strip() if effort_match else None,
                    benchmark_harness="DeepSWE",
                    benchmark_components={
                        "cost": float(match.group(2)),
                        "steps": float(match.group(4)),
                    },
                    provenance=[self.config.name],
                    confidence=0.8,
                )
            )
        return Page(records=records, total_records=len(records), coverage="complete_leaderboard")


def _aa_record(
    source: str,
    source_url: str,
    model_url: str,
    name: str,
    cells: list[str],
    embedded_metrics: dict[str, float | None] | None = None,
) -> dict[str, Any]:
    slug = urlparse(model_url).path.rstrip("/").rsplit("/", 1)[-1] or name
    creator = cells[2] or None
    values: dict[str, Any] = {
        "source_id": f"artificial-analysis:{slug}",
        "name": name,
        "organization": creator,
        "source_url": model_url,
        "artificial_analysis_source_url": source_url,
        "artificial_analysis_model_url": model_url,
        "artificial_analysis_creator": creator,
        "artificial_analysis_context_window": _context_window(cells[1]),
        "context_length": _context_window(cells[1]),
        "intelligence_index": _number(cells[3]),
        "cost_per_task_usd": _number(cells[4]),
        "artificial_analysis_input_price_per_million": (embedded_metrics or {}).get(
            "input_price_per_million"
        ),
        "artificial_analysis_output_price_per_million": (embedded_metrics or {}).get(
            "output_price_per_million"
        ),
        "median_output_tokens_per_second": _number(cells[5]),
        "latency_first_chunk_seconds": _number(cells[6]),
        "total_response_seconds": _number(cells[7]),
        "provenance": [source],
        "confidence": 0.9,
    }
    values["field_provenance"] = _field_provenance(source_url, values)
    return values


def _aa_embedded_metrics(html: str) -> dict[str, dict[str, float | None]]:
    starts = [match.start() - 2 for match in re.finditer(r'\\"slug\\":\\"', html)]
    if not starts:
        return {}
    metrics: dict[str, dict[str, float | None]] = {}
    for start, end in zip(starts, starts[1:] + [len(html)], strict=True):
        chunk = html[start:end]
        if '\\"modelCreatorName\\":' not in chunk:
            continue
        slug = _escaped_string(chunk, "slug")
        if slug is None or '\\"price1mInputTokens\\":' not in chunk:
            continue
        metrics[slug] = {
            "input_price_per_million": _escaped_number(chunk, "price1mInputTokens"),
            "output_price_per_million": _escaped_number(chunk, "price1mOutputTokens"),
        }
    return metrics


def _escaped_string(value: str, key: str) -> str | None:
    match = re.search(rf'\\"{key}\\":\\"([^"\\]*)\\"', value)
    return match.group(1) if match else None


def _escaped_number(value: str, key: str) -> float | None:
    match = re.search(rf'\\"{key}\\":(null|-?\d+(?:\.\d+)?)', value)
    if match is None or match.group(1) == "null":
        return None
    return float(match.group(1))


def _table_headers(table: Any) -> list[str]:
    for row in table.find_all("tr"):
        headers = [" ".join(cell.get_text(" ", strip=True).split()) for cell in row.find_all("th")]
        if "Model" in headers:
            return headers
    return []


def _normalize_header(value: str) -> str:
    return " ".join(value.casefold().split())


def _modality_cost(value: str) -> tuple[float | None, str | None]:
    lowered = value.casefold()
    unit = "minute" if "/min" in lowered else "1,000 images" if "/1k imgs" in lowered else None
    match = re.search(r"\$\s*([0-9]+(?:\.[0-9]+)?)", value)
    return (float(match.group(1)), unit) if match and unit else (None, unit)


def _aa_release_iso(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%b %Y").replace(tzinfo=UTC).date().isoformat()
    except ValueError:
        return None


def _slug_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")


def _context_window(value: str) -> int | None:
    number = _number(value)
    if number is None:
        return None
    normalized = value.replace(",", "").upper()
    multiplier = 1
    if "K" in normalized:
        multiplier = 1024
    elif "M" in normalized:
        multiplier = 1024**2
    elif "G" in normalized:
        multiplier = 1024**3
    return int(number * multiplier)


def _number(value: str) -> float | None:
    cleaned = value.replace(",", "").replace("$", "").replace("*", "").strip()
    if not cleaned or cleaned.lower() in {"--", "n/a", "na", "null", "unknown"}:
        return None
    match = re.search(r"[-+]?\d+(?:\.\d+)?", cleaned)
    if match is None:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _benchmark_effort(value: str) -> str | None:
    lowered = value.casefold()
    for effort in ("non-reasoning", "xhigh", "max", "high", "medium", "low"):
        if effort in lowered:
            return effort
    return None


def _hf_record(item: dict[str, Any], source: str) -> dict[str, Any]:
    source_id = str(item.get("id", item.get("modelId", "unknown")))
    source_url = f"https://huggingface.co/{source_id}"
    license_name = _optional_str(item.get("license") or _nested_value(item, "cardData", "license"))
    tags = _string_list(item.get("tags"))
    pipeline_tag = _optional_str(item.get("pipeline_tag"))
    modalities = _modalities_from_text(pipeline_tag)
    capabilities = sorted(set(tags + ([str(pipeline_tag)] if pipeline_tag else [])))
    tool_calling = _has_tool_calling(tags)
    values = {
        "source_id": source_id,
        "name": source_id,
        "source_url": source_url,
        "created_at": item.get("createdAt"),
        "updated_at": item.get("lastModified"),
        "organization": source_id.split("/", 1)[0] or None,
        "license": license_name,
        "release_date": item.get("createdAt"),
        "open_weights": None if item.get("private") is None else not bool(item.get("private")),
        "commercial_use": _commercial_use(license_name),
        "origin": "open" if item.get("private") is not True else "unknown",
        "modalities": modalities,
        "capabilities": capabilities,
        "supported_parameters": [],
        "tool_calling": tool_calling,
        "downloads": _optional_int(item.get("downloads")),
        "likes": _optional_int(item.get("likes")),
        "recent_signals": _signals(item),
    }
    values["field_provenance"] = _field_provenance(source_url, values)
    values["provenance"] = [source]
    return {
        **values,
        "confidence": 0.7,
    }


def _or_record(item: dict[str, Any], source: str) -> dict[str, Any]:
    model_id = str(item.get("id", "unknown"))
    source_url = f"https://openrouter.ai/{model_id}"
    pricing_payload = item.get("pricing", {})
    pricing = (
        {
            str(key): price
            for key, value in pricing_payload.items()
            if (price := _price(value)) is not None
        }
        if isinstance(pricing_payload, dict)
        else {}
    )
    architecture_payload = item.get("architecture", {})
    architecture = _string_mapping(architecture_payload)
    supported_parameters = _string_list(item.get("supported_parameters"))
    open_weights = item.get("open_weights") if isinstance(item.get("open_weights"), bool) else None
    top_provider = _string_mapping(item.get("top_provider"))
    values = {
        "source_id": model_id,
        "name": _optional_str(item.get("name")) or model_id,
        "source_url": source_url,
        "created_at": _date_value(item.get("created")),
        "updated_at": _date_value(item.get("last_updated")),
        "organization": model_id.split("/", 1)[0] or None,
        "context_length": _optional_int(item.get("context_length")),
        "modalities": _modalities_from_text(architecture.get("modality")),
        "capabilities": sorted(set(supported_parameters)),
        "supported_parameters": supported_parameters,
        "tool_calling": _has_tool_calling(supported_parameters),
        "architecture": architecture,
        "provider_details": top_provider,
        "price_per_million": pricing.get("prompt"),
        "output_price_per_million": pricing.get("completion"),
        "pricing": pricing,
        "open_weights": open_weights,
        "origin": "open"
        if open_weights is True
        else "proprietary"
        if open_weights is False
        else "unknown",
        "providers": _string_list(item.get("providers")),
        "release_date": _date_value(item.get("created")),
    }
    values["field_provenance"] = _field_provenance(source_url, values)
    values["provenance"] = [source]
    return {
        **values,
        "confidence": 0.7,
    }


def _price(value: Any) -> float | None:
    try:
        return float(value) * 1_000_000 if value is not None else None
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return sorted({str(item) for item in value if item is not None})


def _string_mapping(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {str(key): str(item) for key, item in value.items() if item is not None}


def _nested_value(value: dict[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _modalities_from_text(value: Any) -> list[str]:
    if not isinstance(value, str):
        return []
    lowered = value.lower()
    modalities = [
        modality for modality in ("text", "image", "audio", "video", "3d") if modality in lowered
    ]
    return modalities or ["text"]


def _has_tool_calling(values: list[str]) -> bool:
    tokens = {value.lower().replace("_", "-") for value in values}
    return bool(tokens & {"tools", "tool-use", "tool-calling", "tool-choice", "function-calling"})


def _commercial_use(license_name: Any) -> bool | None:
    if not isinstance(license_name, str):
        return None
    normalized = license_name.lower()
    if any(token in normalized for token in ("non-commercial", "-nc", "cc-by-nc")):
        return False
    if normalized in {"apache-2.0", "mit", "bsd-2-clause", "bsd-3-clause", "isc"}:
        return True
    return None


def _signals(item: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(value)
        for key in ("downloads", "likes")
        if (value := _optional_int(item.get(key))) is not None
    }


def _field_provenance(source_url: str, values: dict[str, Any]) -> dict[str, list[str]]:
    return {key: [source_url] for key, value in values.items() if value not in (None, [], {}, "")}


def _date_value(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=UTC).isoformat().replace("+00:00", "Z")
    return None
