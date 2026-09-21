# model-radar: Product Plan

> An on-demand tool that fetches the current AI model landscape, resolves and ranks it, then
> publishes one point-in-time HTML report and one machine-readable snapshot for a terminal viewer.

`solution.md` is the authoritative implementation design. This document defines product scope,
behavior, source policy, ranking intent, outputs, and acceptance criteria.

## 1. Problem

AI model information is fragmented across model hubs, provider catalogs, benchmark sites, vendor
announcements, and feeds. Engineers must reconcile several problems before making a choice:

- Model hubs contain many quantizations, adapters, merges, and re-uploads of the same base model.
- Capability, price, licence, provenance, context length, and hardware needs live in different
  sources.
- “Open source,” “open weights,” and “free API” are often conflated.
- Benchmark values are difficult to compare across modalities and cohorts.
- The landscape changes quickly enough that a report must state exactly when it was generated.

The product condenses those current sources into a portable answer to practical model-selection
questions. It is not a continuously maintained catalog.

## 2. Product Contract

One `model-radar run` invocation:

1. Fetches every page from all enabled current-state sources.
2. Validates and normalizes the fetched records.
3. Resolves canonical models, families, and variants.
4. Enriches current metadata and computes deterministic scores.
5. Audits source completeness, identity quality, and score integrity.
6. Produces one immutable point-in-time snapshot.
7. Renders a self-contained HTML report from that snapshot.
8. Publishes the HTML, JSON snapshot, and manifest atomically.
9. Retains exactly one JSON snapshot per calendar day for 14 days and adds a short What's New
  summary against the prior retained day.
10. Exits.

The same command may be invoked manually, by CI, or by an external scheduler. Scheduling is
optional and adds no product state or behavior.

## 3. Goals

| ID | Goal |
|---|---|
| G1 | One canonical record per real model, with derivatives represented as variants. |
| G2 | Fetch the complete configured current-state scope, with explicit source outcomes. |
| G3 | Answer recent-release questions from source-provided timestamps. |
| G4 | Rank current models by quality, value, and practical suitability. |
| G5 | Filter by modality, openness, origin, licence, provider, and hardware footprint. |
| G6 | Publish a defensible Copilot suitability rubric. |
| G7 | Preserve field-level provenance and confidence in the generated snapshot. |
| G8 | Produce deterministic output for fixed inputs, configuration, and generation time. |
| G9 | Run on a laptop without a database, daemon, server, or frontend build system. |
| G10 | Give HTML and TUI users exactly the same values and ordering. |

## 4. Non-goals

- Maintaining a persistent model catalog or source-assertion store.
- Maintaining more than one history JSON result for a calendar day.
- Maintaining history beyond the rolling 14-day report window.
- Comparing against a previous run or deriving local trends, movers, or rank deltas.
- Locally tracking first-seen dates, download velocity, or historical heat.
- Incremental synchronization, caches, watermarks, backfills, checkpoints, or resume.
- Running an embedded scheduler or requiring daily execution.
- Running original model benchmarks or downloading model weights.
- Routing inference traffic or acting as a model gateway.
- Hosting a web application or API service in v1.
- Treating LLM-generated research as authoritative data.
- Social features, personalized feeds, comments, or votes.

## 5. Users and Questions

| Persona | Question | View |
|---|---|---|
| Engineer choosing a coding model | Best coding models usable from VS Code with commercial terms? | `copilot-ready` |
| Cost owner | Best quality per dollar above a quality floor? | `value-top5` |
| Researcher | Which recently released models appear significant? | `recent-significant` |
| Self-hosting team | Which open-weight models fit 24 GB VRAM at 4-bit? | `local-24gb` |
| Creative team | Which current image models lead their cohorts? | `image-leaders` |
| Strategy team | How does the current model landscape break down by origin? | `origin-breakdown` |

Every view operates over the same point-in-time model set.

## 6. Source Policy

Prefer sources in this order:

1. Official APIs.
2. Published feeds or machine-readable datasets.
3. Licensed third-party APIs.
4. HTML extraction only when terms and robots policy permit it.

### 6.1 Initial Sources

| Source | Role | Initial status |
|---|---|---|
| Hugging Face Hub API | Model identity, metadata, lineage, downloads, likes | Required |
| Hugging Face model cards | Licence, language, base model, current claims | Required where available |
| OpenRouter models API | Hosted price, context, modality, provider availability | Required |
| Open LLM benchmark datasets | Current benchmark observations | Required for quality views |
| Vendor announcement feeds | Release timestamps and first-party context | Optional |
| Artificial Analysis LLM leaderboard | Quality, latency, throughput, token price | Required |
| Artificial Analysis image/video leaderboards | Text-to-image, image-to-image, text-to-video, and image-to-video Elo, samples, release, API cost | Required |
| Civitai | Image-model metadata and derivatives | Later enrichment phase |
| MTEB | Embedding and reranking quality | Later modality phase |

Each connector declares whether it is required, its attribution, redistribution constraints,
request limits, maximum pages, and supported fields.

### 6.2 “Fetch Everything”

“Everything” means every page exposed by each enabled connector for its configured current-state
scope. It does not mean private data, arbitrary web search, model weights, or unlimited event
history.

A connector must prove pagination completion. Repeated cursors, rejected schemas, page-limit
exhaustion, or partial responses count as source failure rather than a successful partial fetch.

## 7. Current Model Record

The generated snapshot contains current canonical records with:

- Stable snapshot-local model ID and public slug.
- Organization, family, model name, and variant relationships.
- Modality and capabilities.
- Parameter count, active parameters, architecture, context length, and quantization.
- Source-provided release and announcement timestamps.
- Current hosted providers, pricing, latency, and throughput where available.
- Current benchmark observations and cohort labels.
- Licence classification, open-weight status, commercial-use status, and origin.
- Current popularity or recent-window signals directly supplied by sources.
- Quality, value, significance, and Copilot suitability scores.
- Field-level source URL, observed time, confidence, and competing assertions.

Do not create `first_seen_at` or any locally historical field.

## 8. Identity and Derivatives

Apply identity rules from strongest to weakest:

1. Curated cross-source alias.
2. Exact normalized source identifier.
3. Declared base-model lineage.
4. Structural fingerprint using architecture and tokenizer identity.
5. Fuzzy name match only when parsed parameter and variant tuples are identical.
6. Otherwise retain a separate unresolved identity.

Fuzzy names alone never merge records. Quantizations, adapters, LoRAs, and merges are variants by
default. Declared lineage outranks heuristics. Cycles and conflicting parentage fail the identity
audit or remain visibly unresolved.

Resolve canonical fields independently with field-specific source precedence. Preserve competing
current assertions and the rule used to select the displayed value.

## 9. Openness and Licence

Keep these concepts separate:

- **Open weights:** weights can be downloaded.
- **Open source:** code, weights, and licence meet the configured definition.
- **Commercial use:** current licence metadata permits the intended commercial use.
- **Free hosted API:** a provider currently offers a zero-price tier.

Map exact SPDX identifiers or reviewed licence-file hashes. Unknown or conflicting licences remain
`unknown` and fail closed for commercial-use filters. The report must state that this classification
is metadata, not legal advice.

Organization origin comes from curated organization data or reviewed source metadata. Unknown
origin is `XX`; do not infer it from model names.

## 10. Scoring

All scoring is deterministic arithmetic over the current snapshot inputs. Every score stores its
components, cohort, weights, configuration hash, and algorithm version.

### 10.1 Quality

Normalize benchmark observations to percentiles only within declared modality and capability
cohorts:

$$
\operatorname{pct}_b(m)=
\frac{|\{m' \in M_b:v_b(m')<v_b(m)\}|}{|M_b|-1}\times100
$$

$$
Q(m)=\frac{\sum_{b\in B(m)}w_b\operatorname{pct}_b(m)}
{\sum_{b\in B(m)}w_b}
$$

Missing observations are absent, not zero. Scores below the configured benchmark-coverage
threshold are provisional and do not enter authoritative leader views.

### 10.2 Value

Use the median current provider quote and retain provider count and price spread:

$$
P_{blend}=\frac{3p_{in}+p_{out}}{4}
$$

$$
V=\frac{Q}{\max(P_{blend},P_{min})}
$$

Apply a configurable quality floor to value leaderboards and identify the quality/price Pareto
frontier.

### 10.3 Significance and Current Signals

Significance may combine organization weight, benchmark availability, announcement evidence, new
family status, and recent-window download/like counts supplied directly by current sources.

Do not calculate velocity from previous local runs. Omit heat when sources do not provide comparable
current-window signals.

### 10.4 Copilot Suitability

Eligibility requires a reachable supported endpoint or local runtime, adequate context, tool
calling, and acceptable current licence/API terms. The score may combine coding quality,
tool-calling reliability, latency/throughput, price, context length, caching, and structured output.
Keep FIM capability as a separate fact.

## 11. Views

Saved views support allow-listed filters, grouping, stable sorting, limits, and facets. Important
v1 views are:

- `quality-top5`
- `value-top5`
- `recent-significant`
- `open-weights`
- `commercial-use`
- `local-24gb`
- `copilot-ready`
- `origin-breakdown`
- modality-specific leaders

A view records ordered model IDs and display annotations in the snapshot. If required fields are
unavailable because an optional source failed, mark the view `degraded` or `unavailable`; never
present an authoritative empty list.

## 12. Output Contract

One successful release contains:

| Artifact | Purpose |
|---|---|
| `snapshot.json` | Complete versioned machine-readable current-state result |
| `model-radar.html` | Self-contained report rendered from the same snapshot object |
| `manifest.json` | Schema version, filenames, sizes, and SHA-256 hashes |

The snapshot includes generation time, generator/configuration/algorithm versions, source outcomes,
warnings, audit results, canonical models, provenance, scores, saved views, and summary counts.

The HTML report must:

- Work from `file://` and remain useful with JavaScript disabled.
- Use semantic server-rendered tables.
- Provide search, facets, sorting, details, comparison, and export when JavaScript is enabled.
- Make no network requests and use no CDN, analytics, or remote fonts.
- Show generation time, source status, degraded views, score coverage, and provenance.
- Escape hostile source text and enforce a strict CSP.

`model-radar top` reads `out/snapshot.json` or a path supplied with `--snapshot`. It performs no
network access and does not recompute scores. It supports filtering, sorting, comparison,
provenance, responsive columns, ASCII, no-color, and one-frame plain output.

## 13. Execution and Publication

The in-process stages are:

```text
preflight -> fetch -> normalize -> resolve -> optional research -> enrich -> score
          -> audit -> build snapshot -> render HTML -> publish -> report
```

- Bound network concurrency, response bytes, pages, records, memory, and total wall time.
- Retry only transient idempotent requests and honor `Retry-After`.
- Required-source failure or incomplete pagination exits `69` without publication.
- Optional-source failure may publish a degraded snapshot with exit `3` only when coverage gates
  pass.
- Analysis, audit, render, or artifact validation failure exits `4`.
- Configuration failure exits `78`; unexpected failure exits `1`.
- Missing source data remains unknown; never copy facts from the previous release.

Write a same-filesystem staging release, serialize JSON once, render HTML from the same immutable
snapshot, write the manifest last, validate all artifacts, then atomically replace `out/current`.
A failed invocation leaves the prior completed release untouched but never reads it as analysis
input.

Use a short publication lock only around the final replacement. There are no stage checkpoints,
resume semantics, or persistent run records.

## 14. Optional Research

Research is disabled by default. Current-run research may produce cited summaries and low-confidence
proposals, but it cannot determine identities, licences, canonical facts, filters, or scores.

Every task has a versioned prompt, bounded input documents, output schema, timeout, token/cost limit,
and citation requirements. Validate citations against supplied text. Run web-capable backends in a
credential-free sandbox with enforced resource and egress limits. Agent failure is a warning and
cannot block deterministic analysis.

## 15. Security and Compliance

- Follow source terms, attribution, and redistribution restrictions.
- Prefer official APIs and use a descriptive User-Agent.
- Validate redirect targets and block private, loopback, link-local, and metadata addresses.
- Treat model cards, feeds, names, URLs, and research text as hostile input.
- Strip terminal controls and bidi overrides.
- Escape HTML and Markdown; allow-list outbound HTTPS links.
- Never log tokens, authorization headers, secrets, or unbounded source payloads.
- Store no model weights or user PII.
- Pin dependencies and run dependency and secret scans in CI.

## 16. Testing and Acceptance

Tests use recorded complete-pagination fixtures; normal CI performs no live network access.

Required suites:

- Connector contracts for success, malformed pages, timeout, repeated cursor, and partial fetch.
- Strict source validation and explicit null/type behavior.
- Reviewed identity corpus with at least 99% merge precision; report recall separately.
- Scoring properties for bounds, monotonicity, missing data, stable ties, and finite values.
- Deterministic snapshot and HTML goldens with pinned generation time.
- JSON Schema compatibility and hostile-input cases.
- Required/optional source failure matrix.
- Failure injection at every atomic publication step.
- Playwright tests for `file://`, zero network, CSP, XSS, keyboard use, and accessibility.
- TUI snapshots at 80x24, 120x40, and 200x60, including ASCII and no-color modes.
- HTML/TUI parity over the same snapshot fixture.
- Peak-memory and full-invocation performance tests.

Optional live connector canaries detect upstream schema drift but do not create product history.

## 17. Roadmap

| Phase | Scope | Exit criterion |
|---|---|---|
| 0: Contract | Package, configuration, source outcomes, snapshot schema, manifest | Hand-built snapshot validates and opens in minimal HTML/TUI |
| 1: Fetch | Bounded HTTP client, Hugging Face, OpenRouter, complete pagination | Recorded and live smoke sources fetch completely |
| 2: Analyze | Normalization, identity, variants, enrichment, scores, views | Deterministic ranked snapshot passes identity and score gates |
| 3: HTML MVP | Audit, snapshot codec, Jinja2 report, atomic publication | `model-radar run` produces a useful offline release |
| 4: TUI | Snapshot loader, filters, comparison, provenance, responsive layouts | HTML and TUI values and ordering match |
| 5: Breadth | Approved benchmarks, additional modalities, optional research | Each addition meets the same source and provenance contracts |

## 18. Decisions

- The product is an on-demand current-state snapshot, not a daily historical catalog.
- There is no database because no cross-run analytical state is required.
- Polars is the sole in-process analysis engine unless measurement proves plain Python is simpler.
- HTML and TUI share one versioned `snapshot.json` contract.
- Previous releases exist only to make publication atomic and provide manual rollback.
- Source completeness is explicit; partial required-source results never masquerade as complete.
- Identity and scoring are deterministic and independent of optional agent output.
- Scheduling remains an external user choice.

## 19. v1 Definition of Done

- `model-radar run` fetches every page from all enabled current-state sources and exits.
- No database, cache, migration, checkpoint, or built-in scheduler is required; history is limited
  to one JSON result per day for the rolling 14-day report window.
- Required-source incompleteness prevents publication; optional failure is visible and bounded.
- Identity resolution meets its precision gate and retains unresolved records separately.
- Scores are deterministic, cohort-aware, finite, versioned, and explainable.
- One validated snapshot drives both HTML and TUI.
- The HTML report is self-contained, offline, accessible, and safe for hostile text.
- Publication is atomic and failure leaves the previous completed release untouched.
- Fixed fixtures and generation time produce byte-identical artifacts.
- Security, browser, terminal, memory, and performance gates pass.
