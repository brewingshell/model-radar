# Model Radar

Model Radar is a snapshot-only model landscape tool. `solution.md` is authoritative for the implementation boundary and `model-radar.md` is the product contract.

A `model-radar run` invocation fetches the enabled current-state sources, validates and normalizes records in memory, resolves identities, computes deterministic current scores, renders a self-contained HTML report, and atomically publishes:

- `out/snapshot.json`
- `out/model-radar.html`
- `out/manifest.json`
- `out/history/YYYY-MM-DD.json`, exactly one JSON result per calendar day for the last 14 days

The HTML is a permanent file generated from the latest JSON result. A regular `out/current/`
compatibility directory is also refreshed, but it is not a symlink. The report includes a
What's New section comparing the latest run with the previous retained day. Re-running on the same
day replaces that day's history file.

There is no database, ORM, migration, persistent cache, checkpoint, resume flow, or built-in
scheduler. `model-radar top` reads the root snapshot JSON. Missing source fields stay `null` or
`unknown` and are surfaced as warnings where applicable; the tool does not invent benchmark data.

## Install

With `uv`:

```bash
uv sync --extra dev
```

With `pip`:

```bash
python3 -m pip install -e '.[dev]'
```

The tests are offline and do not need credentials or network access.

## Run the live snapshot

```bash
python3 -m model_radar run --config config/app.yaml --output ./model-radar
python3 -m model_radar validate ./model-radar/snapshot.json
python3 -m model_radar top --output ./model-radar --plain
```

The default config calls public Hugging Face, OpenRouter, and Artificial Analysis endpoints,
including the Artificial Analysis text-to-image, image-to-image, text-to-video, and image-to-video
leaderboards. Those modality
leaderboards provide Elo rankings, evaluation sample counts, release months, open-weight markers,
and API cost per 1,000 images or per minute of video.
`HF_TOKEN` and `OPENROUTER_API_KEY` are optional; set them only when the APIs require
authenticated rate limits. Artificial Analysis is public HTML and requires no login or API key.
The live fetch is bounded to 1,000 records per source. Hugging Face currently returns a top-level
recent-model list without a total count or cursor, so its snapshot coverage is labelled
`bounded_recent`. OpenRouter returns `data` and `total_count`, but the observed response exposes no
usable cursor; a catalog larger than the returned bound is labelled partial and reported as a
warning. The report never claims the full model universe.

Live source failures are explicit and include the source name, HTTP/schema error category, and
required-source exit code. A successful live snapshot can still contain degraded or unavailable
views when upstream metadata is incomplete.

## Run the offline fixture demo

```bash
python3 -m model_radar run --config config/fixture.yaml --output ./out --source fixture

python3 -m model_radar top --output ./out --plain
```

To make output reproducible, pass an explicit timestamp:

```bash
python3 -m model_radar run --config config/fixture.yaml --source fixture --generated-at 2026-09-18T00:00:00+00:00
```

Useful commands include `model-radar validate`, `model-radar inspect`, and `model-radar sources`. Exit codes are `0` complete, `3` degraded optional source, `4` analysis/render/publication failure, `64` usage, `69` required source failure, `75` publication lock, and `78` configuration failure.

Artificial Analysis Intelligence Index is the authoritative performance metric. Its cost field is
cost per benchmark task and is reported separately. The second primary view uses Artificial
Analysis input/output USD per 1M tokens, weighted as `(3 * input + output) / 4`, and ranks
`intelligence_index / weighted_token_price`; it must not be confused with benchmark-task cost.
The image and video tabs use the modality-specific Artificial Analysis Elo rankings and display
their API costs separately from LLM token pricing.
The meaningful-new Hugging Face view uses
source `createdAt`/`lastModified` plus a transparent adoption heuristic; OpenRouter remains an
optional secondary metadata source. A failed required AA source leaves the primary performance
views unavailable and prevents a production publication rather than substituting catalog metadata.

The report includes two organization Copilot tabs. Its catalog is configured in
`config/app.yaml` from the organization model list. Model families and thinking levels are
extracted generically from names such as `Luna (max)`, `Sol (high)`, and `Opus (xhigh)`, rather
than hardcoded to a model version. **Org Copilot per token** ranks combinations by Artificial
Analysis Intelligence per weighted Copilot credits per million tokens and excludes combinations
with AA Intelligence Index below `25`. **Org Copilot best** ranks the top ten by raw Artificial Analysis Intelligence, retaining
each thinking level as a separate combination. All primary decision views are top ten.

`config/models.json` controls which decision tabs are available for each model type. The selected
model type automatically selects its first configured tab, and the mapping is embedded in each
snapshot so the permanent HTML remains self-contained. `config/org.json` controls the organization
Copilot catalog, including the covered family and level for each model.

The report enriches the existing decision tabs with a separate LiveBench column. LiveBench is
matched to model+thinking variants without replacing Artificial Analysis. EvalPlus and DeepSWE
are not active report sources.

## Test

```bash
pytest -q
```

## GitHub Pages

This repository includes two GitHub Actions workflows:

- `.github/workflows/ci.yml` runs tests, lint, formatting, type checks, and an offline fixture smoke test on pushes and pull requests.
- `.github/workflows/pages.yml` runs the live analysis daily at 02:17 UTC and supports manual dispatch. It validates the snapshot, creates `model-radar/index.html`, and deploys the complete `model-radar/` artifact to GitHub Pages.

One-time setup:

1. Add these files to a GitHub repository without committing generated output.
2. In **Settings → Pages**, select **GitHub Actions** as the source.
3. Optionally add `HF_TOKEN` and `OPENROUTER_API_KEY` as repository secrets for authenticated rate limits.
4. Run **Publish model dashboard → Run workflow** once to test the deployment.

The scheduled workflow does not create commits. To keep the What's New comparison and the 14-day
history working across runs, the workflow restores the previous `snapshot.json` and `history/`
directory from the Actions cache before generating and saves the updated copy afterward. A failed
required source or validation step stops deployment before the cache is saved, leaving both the
previous Pages artifact and the last good history available. GitHub schedules are approximate and
may be delayed; the manual dispatch is available for an immediate refresh.
