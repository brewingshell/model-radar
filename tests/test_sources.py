import asyncio

import httpx
import pytest

from model_radar.connectors import (
    ArtificialAnalysisConnector,
    ArtificialAnalysisModalityConnector,
    BoundedHttpClient,
    DeepSweConnector,
    EvalPlusConnector,
    HuggingFaceConnector,
    LiveBenchConnector,
    OpenRouterConnector,
)
from model_radar.models import SourceConfig
from model_radar.source import (
    ConnectorError,
    Page,
    PaginationError,
    RequiredSourceError,
    fetch_connector,
    fetch_sources,
)


class JsonClient:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    async def get_json(self, url, headers=None, params=None):
        self.calls.append((url, headers, params))
        return self.payload


class HtmlClient:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    async def get_html(self, url, headers=None):
        self.calls.append((url, headers))
        return self.payload

    async def get_text(self, url, headers=None):
        self.calls.append((url, headers))
        return self.payload


class MultiTextClient:
    def __init__(self, payloads):
        self.payloads = payloads

    async def get_text(self, url, headers=None):
        return self.payloads[url]


class LoopConnector:
    def __init__(self, required: bool = False):
        self.config = SourceConfig(name="loop", kind="fixture", required=required)

    async def page(self, cursor):
        return Page(records=[], next_cursor="same")


class BrokenConnector:
    def __init__(self, required: bool):
        self.config = SourceConfig(name="broken", kind="fixture", required=required)

    async def page(self, cursor):
        raise ConnectorError("unavailable", "fixture source unavailable")


class FetchSettings:
    max_pages = 10
    max_records = 10


def test_repeated_cursor_is_a_pagination_failure():
    with pytest.raises(PaginationError, match="repeated cursor"):
        asyncio.run(fetch_connector(LoopConnector(), 10, 10))


def test_livebench_and_evalplus_parse_normalized_indexes():
    live_client = MultiTextClient(
        {
            "https://live/table_2026_06_25.csv": "model,reasoning,coding\nGPT-6 Astra (max),80,90\n",
            "https://live/categories_2026_06_25.json": '{"Reasoning":["reasoning"],"Coding":["coding"]}',
            "https://live/cost_2026_06_25.csv": "model,cost_per_successful_task\nGPT-6 Astra (max),0.5\n",
        }
    )
    live = LiveBenchConnector(
        SourceConfig(
            name="livebench", kind="livebench", base_url="https://live", endpoint="2026-06-25"
        ),
        client=live_client,
    )
    live_record = asyncio.run(live.page(None)).records[0]
    assert live_record.benchmark_index == 85.0
    assert live_record.benchmark_cost_per_task == 0.5

    eval_client = JsonClient({"GPT-6 Astra (max)": {"pass@1": {"humaneval+": 80, "mbpp+": 90}}})
    evalplus = EvalPlusConnector(SourceConfig(name="evalplus", kind="evalplus"), client=eval_client)
    eval_record = asyncio.run(evalplus.page(None)).records[0]
    assert eval_record.benchmark_index == 85.0


def test_livebench_overall_is_mean_of_category_averages():
    client = MultiTextClient(
        {
            "https://live/table_2026_06_25.csv": (
                "model,r1,r2,r3,c1\ngpt-6-astra-max,90,90,90,30\n"
            ),
            "https://live/categories_2026_06_25.json": (
                '{"Reasoning":["r1","r2","r3"],"Coding":["c1"]}'
            ),
            "https://live/cost_2026_06_25.csv": "model,cost_per_successful_task\n",
        }
    )
    connector = LiveBenchConnector(
        SourceConfig(
            name="livebench", kind="livebench", base_url="https://live", endpoint="2026-06-25"
        ),
        client=client,
    )

    record = asyncio.run(connector.page(None)).records[0]

    assert record.benchmark_index == 60.0
    assert record.benchmark_components == {"Reasoning": 90.0, "Coding": 30.0}
    assert record.benchmark_effort == "max"


def test_deepswe_parses_effort_pass_rate_and_cost():
    client = HtmlClient(
        """
        <div data-chart-pin-source="">
          <div class="flex min-w-0 items-center"><span>gpt-6-astra</span> <span>[ xhigh ]</span></div>
          <div>74 % +/- 3 % Avg cost $6.52 Out tok 30k Steps 29</div>
        </div>
        """
    )
    connector = DeepSweConnector(
        SourceConfig(name="deepswe", kind="deepswe", base_url="https://deepswe.example"),
        client=client,
    )

    record = asyncio.run(connector.page(None)).records[0]

    assert record.name == "gpt-6-astra"
    assert record.benchmark_index == 74
    assert record.benchmark_cost_per_task == 6.52
    assert record.benchmark_effort == "xhigh"


def test_optional_failure_returns_degraded_status():
    records, statuses, warnings, degraded = asyncio.run(
        fetch_sources([BrokenConnector(False)], FetchSettings())
    )
    assert records == []
    assert statuses[0].status == "failed"
    assert degraded is True
    assert warnings == ["broken: fixture source unavailable"]


def test_required_failure_aborts_with_required_error():
    with pytest.raises(RequiredSourceError, match="fixture source unavailable"):
        asyncio.run(fetch_sources([BrokenConnector(True)], FetchSettings()))


def test_huggingface_accepts_top_level_list_and_preserves_live_metadata():
    client = JsonClient(
        [
            {
                "id": "acme/atlas-7b",
                "modelId": "acme/atlas-7b",
                "createdAt": "2026-01-10T12:00:00.000Z",
                "lastModified": "2026-09-17T12:00:00.000Z",
                "pipeline_tag": "text-generation",
                "tags": ["tool-use", "transformers"],
                "downloads": 1234,
                "likes": 42,
                "private": False,
                "license": "apache-2.0",
            }
        ]
    )
    connector = HuggingFaceConnector(
        SourceConfig(name="huggingface", kind="huggingface", page_size=100), client=client
    )

    page = asyncio.run(connector.page(None))

    assert page.next_cursor is None
    assert page.coverage == "bounded_recent"
    assert page.records[0].downloads == 1234
    assert page.records[0].likes == 42
    assert page.records[0].created_at == "2026-01-10T12:00:00.000Z"
    assert page.records[0].updated_at == "2026-09-17T12:00:00.000Z"
    assert page.records[0].source_url == "https://huggingface.co/acme/atlas-7b"
    assert page.records[0].tool_calling is True
    assert client.calls[0][2] == {"limit": 100}


def test_openrouter_parses_catalog_object_and_pricing_metadata():
    client = JsonClient(
        {
            "data": [
                {
                    "id": "acme/atlas-7b",
                    "name": "Atlas 7B",
                    "context_length": 131072,
                    "pricing": {"prompt": "0.00000025", "completion": "0.000001"},
                    "architecture": {
                        "modality": "text->text",
                        "tokenizer": "Atlas",
                        "instruct_type": "chatml",
                    },
                    "supported_parameters": ["temperature", "tools"],
                    "top_provider": {"context_length": 131072, "is_moderated": False},
                }
            ],
            "links": {"first": "https://openrouter.ai/api/v1/models"},
            "total_count": 1,
        }
    )
    connector = OpenRouterConnector(
        SourceConfig(name="openrouter", kind="openrouter", page_size=50), client=client
    )

    page = asyncio.run(connector.page(None))

    record = page.records[0]
    assert page.next_cursor is None
    assert page.total_records == 1
    assert page.coverage == "complete_catalog"
    assert record.context_length == 131072
    assert record.architecture["tokenizer"] == "Atlas"
    assert record.modalities == ["text"]
    assert record.price_per_million == 0.25
    assert record.output_price_per_million == 1.0
    assert record.tool_calling is True
    assert record.providers == []
    assert record.provider_details["is_moderated"] == "False"
    assert client.calls[0][2] == {"limit": 50}


def test_artificial_analysis_parses_public_leaderboard_and_missing_values():
    client = HtmlClient(
        """
                <table>
                    <thead><tr>
                        <th>Model</th><th>Context Window</th><th>Creator</th>
                        <th>Artificial Analysis Intelligence Index</th><th>Cost per Task USD</th>
                        <th>Median Tokens/s</th><th>Latency First Chunk (s)</th>
                        <th>Total Response (s)</th><th>Further Analysis</th>
                    </tr></thead>
                    <tbody><tr>
                        <td><a href="/models/acme-atlas">Atlas</a></td><td>128K</td><td>Acme</td>
                        <td>91.5</td><td>$0.12</td><td>1,234</td><td>0.45</td><td>2.10</td><td>link</td>
                    </tr><tr>
                        <td><a href="/models/acme-missing">Missing</a></td><td>--</td><td>Acme</td>
                        <td>*</td><td>--</td><td>--</td><td></td><td>*</td><td>link</td>
                    </tr></tbody>
                </table>
                """
    )
    connector = ArtificialAnalysisConnector(
        SourceConfig(
            name="artificial-analysis",
            kind="artificial_analysis",
            base_url="https://artificialanalysis.ai/leaderboards/models",
        ),
        client=client,
    )

    page = asyncio.run(connector.page(None))

    first, second = page.records
    assert page.coverage == "complete_leaderboard"
    assert page.total_records == 2
    assert first.source_url == "https://artificialanalysis.ai/models/acme-atlas"
    assert first.artificial_analysis_source_url == (
        "https://artificialanalysis.ai/leaderboards/models"
    )
    assert first.artificial_analysis_model_url == first.source_url
    assert first.artificial_analysis_creator == "Acme"
    assert first.artificial_analysis_context_window == 131072
    assert first.context_length == 131072
    assert first.intelligence_index == 91.5
    assert first.cost_per_task_usd == 0.12
    assert first.median_output_tokens_per_second == 1234.0
    assert first.latency_first_chunk_seconds == 0.45
    assert first.total_response_seconds == 2.10
    assert second.intelligence_index is None
    assert second.cost_per_task_usd is None
    assert client.calls == [("https://artificialanalysis.ai/leaderboards/models", None)]


def test_artificial_analysis_modality_parses_elo_and_cost():
    client = HtmlClient(
        """
        <table><thead><tr>
            <th></th><th>Range</th><th>Creator</th><th>Model</th><th>Elo</th>
            <th>95% CI</th><th>Samples</th><th>Released</th><th>API Pricing 1</th>
        </tr></thead><tbody><tr>
            <td>1</td><td>1-2</td><td>OpenAI</td><td>GPT Image 2.5 Flare (max)</td>
            <td>1188</td><td>-11/11</td><td>5,941</td><td>Sep 2026</td><td>$210.7 /1k imgs</td>
        </tr><tr>
            <td>2</td><td>2</td><td>MiniMax</td><td>MiniMax H3 Hugging Face Open Weights</td>
            <td>1220</td><td>-8/8</td><td>8,602</td><td>Jul 2026</td><td>$7.80 /min</td>
        </tr></tbody></table>
        """
    )
    connector = ArtificialAnalysisModalityConnector(
        SourceConfig(
            name="artificial-analysis-image",
            kind="artificial_analysis_modality",
            base_url="https://artificialanalysis.ai/image/leaderboard/text-to-image",
            endpoint="text-to-image",
        ),
        client=client,
    )

    page = asyncio.run(connector.page(None))

    first, second = page.records
    assert page.total_records == 2
    assert first.artificial_analysis_modality == "text-to-image"
    assert first.artificial_analysis_modality_elo == 1188
    assert first.artificial_analysis_modality_samples == 5941
    assert first.artificial_analysis_modality_release == "Sep 2026"
    assert first.artificial_analysis_modality_cost == 210.7
    assert first.artificial_analysis_modality_cost_unit == "1,000 images"
    assert second.open_weights is True
    assert client.calls == [("https://artificialanalysis.ai/image/leaderboard/text-to-image", None)]


@pytest.mark.parametrize(
    ("endpoint", "expected_modality", "expected_media"),
    [
        ("image-to-image", "image-to-image", "image"),
        ("image-to-video", "image-to-video", "video"),
    ],
)
def test_artificial_analysis_modality_accepts_image_endpoints(
    endpoint, expected_modality, expected_media
):
    client = HtmlClient(
        """
        <table><tr><th></th><th>Range</th><th>Creator</th><th>Model</th><th>Elo</th>
        <th>95% CI</th><th>Samples</th><th>Released</th><th>API Pricing 1</th></tr>
        <tr><td>1</td><td>1</td><td>Acme</td><td>Media Model</td><td>1000</td>
        <td>-5/5</td><td>100</td><td>Sep 2026</td><td>$10.0 /min</td></tr></table>
        """
    )
    connector = ArtificialAnalysisModalityConnector(
        SourceConfig(
            name=f"artificial-analysis-{endpoint}",
            kind="artificial_analysis_modality",
            base_url=f"https://example.test/{endpoint}",
            endpoint=endpoint,
        ),
        client=client,
    )

    record = asyncio.run(connector.page(None)).records[0]

    assert record.artificial_analysis_modality == expected_modality
    assert record.modalities == [expected_media]
    assert record.capabilities == [expected_modality]


def test_artificial_analysis_extracts_embedded_token_prices():
    client = HtmlClient(
        """
        <script>self.__next_f.push([1,"d:[{\\"slug\\":\\"acme-atlas\\",\\"modelCreatorName\\":\\"Acme\\",\\"price1mInputTokens\\":0.2,\\"price1mOutputTokens\\":1.2}]"])</script>
        <table><tr>
            <th>Model</th><th>Context Window</th><th>Creator</th>
            <th>Artificial Analysis Intelligence Index</th><th>Cost per Task USD</th>
            <th>Median Tokens/s</th><th>Latency First Chunk (s)</th>
            <th>Total Response (s)</th><th>Further Analysis</th>
        </tr><tr>
            <td><a href="/models/acme-atlas">Atlas</a></td><td>128K</td><td>Acme</td>
            <td>91.5</td><td>$0.12</td><td>1,234</td><td>0.45</td><td>2.10</td><td>link</td>
        </tr></table>
        """
    )
    connector = ArtificialAnalysisConnector(
        SourceConfig(name="artificial-analysis", kind="artificial_analysis"),
        client=client,
    )

    record = asyncio.run(connector.page(None)).records[0]

    assert record.artificial_analysis_input_price_per_million == 0.2
    assert record.artificial_analysis_output_price_per_million == 1.2


@pytest.mark.parametrize("status_code", [429, 500])
def test_bounded_http_client_retries_transient_responses(status_code):
    responses = iter(
        [
            httpx.Response(status_code, headers={"Retry-After": "0"}),
            httpx.Response(200, headers={"content-type": "application/json"}, json={"ok": True}),
        ]
    )

    async def run():
        transport = httpx.MockTransport(lambda _request: next(responses))
        async with BoundedHttpClient(transport=transport) as client:
            return await client.get_json("https://example.test/models")

    assert asyncio.run(run()) == {"ok": True}
