<p align="center">
  <img src="https://raw.githubusercontent.com/ShriniwasAhirrao/ParseIQ/master/assets/banner.svg" alt="ParseIQ Banner" width="100%"/>
</p>

<p align="center">
  <a href="https://pypi.org/project/parseiq/"><img src="https://img.shields.io/pypi/v/parseiq?color=blue&label=PyPI" alt="PyPI version"/></a>
  <a href="https://pypi.org/project/parseiq/"><img src="https://img.shields.io/pypi/pyversions/parseiq" alt="Python versions"/></a>
  <a href="https://github.com/ShriniwasAhirrao/ParseIQ/actions/workflows/test.yml"><img src="https://github.com/ShriniwasAhirrao/ParseIQ/actions/workflows/test.yml/badge.svg" alt="Tests"/></a>
  <a href="https://pypi.org/project/parseiq/"><img src="https://img.shields.io/pypi/dm/parseiq?color=orange" alt="PyPI downloads"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT License"/></a>
  <a href="https://github.com/ShriniwasAhirrao/ParseIQ/releases"><img src="https://img.shields.io/github/v/release/ShriniwasAhirrao/ParseIQ" alt="Latest release"/></a>
  <a href="https://github.com/ShriniwasAhirrao/ParseIQ/stargazers"><img src="https://img.shields.io/github/stars/ShriniwasAhirrao/ParseIQ?style=social" alt="GitHub stars"/></a>
</p>

<h3 align="center">AI-Powered Data Quality Agent</h3>
<p align="center">Profile · Score · Fix any JSON / CSV / XML / Excel file in one command</p>

<p align="center">
  <a href="https://shriniwasahirrao.github.io/ParseIQ/">🌐 <b>Web UI</b></a> ·
  <a href="https://pypi.org/project/parseiq/0.0.6/">📦 <b>PyPI v0.0.6</b></a> ·
  <a href="#quickstart"><b>Quickstart</b></a> ·
  <a href="commands.md"><b>CLI Reference</b></a> ·
  <a href="CHANGELOG.md"><b>Changelog</b></a> ·
  <a href="https://github.com/ShriniwasAhirrao/ParseIQ/discussions"><b>Discussions</b></a>
</p>

---

ParseIQ analyses any data file and produces a full quality report — statistical profiling, anomaly detection, per-table quality scores, and optional AI-generated recommendations — all in a structured Excel workbook and CSV summaries.

Built for the **data onboarding and discovery phase**: when you receive a data dump and need to know what's in it, whether it's trustworthy, and what to fix before loading it into production.

---

## Why ParseIQ?

| Feature | ParseIQ | ydata-profiling | great-expectations |
|---|:---:|:---:|:---:|
| Zero config — works out of the box | ✅ | ✅ | ❌ needs YAML |
| Nested JSON → multiple tables auto | ✅ | ❌ | ❌ |
| Multi-provider LLM enrichment | ✅ | ❌ | ❌ |
| Local mode (no API key ever needed) | ✅ | ✅ | ✅ |
| Excel report with 30-col Meta sheet | ✅ | ❌ | ❌ |
| CLI + Python API in one package | ✅ | ✅ | ✅ |
| Incremental processing (hash cache) | ✅ | ❌ | ✅ |
| pip install, one command, done | ✅ | ✅ | ❌ |

---

## Quickstart <a name="quickstart"></a>

```bash
pip install parseiq

parseiq init                              # first-time setup (API key, model)
parseiq analyze data.json --no-llm       # local mode — no API key needed
parseiq analyze data.json                # with AI enrichment (needs API key)
```

Results appear in `output/` as an Excel workbook + CSV summaries.

For a more detailed report, refer to the path shown at the end of each run:
```
For a more detailed report, refer to:
  D:\your\path\output\complete_data_analysis.xlsx
```

---

## How It Works

```
  ┌─────────────────────────────────────────────┐
  │         YOUR DATA FILE                      │
  │   JSON  /  CSV  /  XML  /  Excel            │
  └──────────────────┬──────────────────────────┘
                     │
                     ▼
  ┌─────────────────────────────────────────────┐
  │  STEP 1 — Metadata Extractor                │
  │  (always runs · no API key needed)          │
  │                                             │
  │  • Flatten nested JSON → multiple tables    │
  │  • Detect data types per column             │
  │  • Compute stats (min/max/mean/percentiles) │
  │  • Flag 8 anomaly types per column          │
  │  • Score every table 0–100                  │
  └──────────────────┬──────────────────────────┘
                     │
          ┌──────────┴──────────┐
          │ llm=True?           │ llm=False → skip
          ▼                     ▼
  ┌───────────────────┐   ┌─────────────────────────┐
  │ STEP 2 — LLM      │   │  Local / Fallback mode  │
  │ Enricher (BYOK)   │   │                         │
  │                   │   │  • Instant · 100% free  │
  │ • Sends metadata  │   │  • Zero data leaves     │
  │   + samples to    │   │    your machine         │
  │   YOUR chosen API │   │  • Structured report    │
  │ • Business-level  │   │    with quality grade   │
  │   interpretation  │   │    and recommendations  │
  │ • Prioritised     │   │  • Auto-activates if    │
  │   recommendations │   │    LLM call fails       │
  │ • Quality grade   │   └─────────────────────────┘
  │   A–F             │
  └──────────┬────────┘
             │
             ▼
  ┌─────────────────────────────────────────────┐
  │  STEP 3 — Output                            │
  │                                             │
  │  complete_data_analysis.xlsx                │
  │    ├── 00_Summary (per-table overview)      │
  │    ├── 01_LLM_Assessment                    │
  │    ├── 02_LLM_Recommendations               │
  │    ├── Data_<table>   (raw rows)            │
  │    ├── Meta_<table>   (30-col profile)      │
  │    ├── Quality_<table>(metric per attr)     │
  │    └── 99_Issues_Recommendations            │
  │  overall_dataset_summary.csv                │
  │  combined_issues_and_recommendations.csv    │
  └─────────────────────────────────────────────┘
```

### Built-in Fallback Mode

If no API key is set, the network is unavailable, or the LLM call fails for any reason, ParseIQ **automatically falls back to local mode** — you always get a complete structured Excel report with quality scores and recommendations, no matter what. No crash, no empty output, no manual retry needed.

---

## Enterprise & Data Privacy

ParseIQ is designed to fit inside any company's data stack without compromising data governance:

| | What happens |
|---|---|
| **Step 1 (always)** | Runs 100% locally. No data leaves your machine. |
| **Step 2 — LLM mode** | Sends metadata summaries and data samples **directly to your chosen LLM provider** (OpenAI, Anthropic, Google, etc.) using **your own API key**. ParseIQ has no server, no proxy, no telemetry — traffic goes straight from your machine to your provider. |
| **Step 2 — Local mode** | `--no-llm` keeps everything fully offline. Zero external calls. |
| **Fallback mode** | If the LLM call fails, ParseIQ generates a full report locally. You always get output. |
| **Bring Your Own Key** | Use your company's existing LLM contract — OpenAI enterprise, Azure OpenAI, Anthropic Claude, Gemini, or a self-hosted Ollama instance. |
| **On-prem / air-gapped** | Run with `--no-llm` or point it at a local Ollama endpoint. No internet required. |
| **Cloud-native** | Runs anywhere Python runs — AWS Lambda, GCP Cloud Run, Azure Functions, Docker, CI/CD pipelines. |

> **No ParseIQ server ever sees your data.** The tool is a local Python process — BYOK means your API key and your data stay under your own infrastructure contract.

## Demo

```bash
$ parseiq analyze input/enterprises.json --no-llm

=======================================================
  ParseIQ - AI-Powered Data Quality Agent  v0.0.6
=======================================================

File     : input/enterprises.json
Output   : output/
LLM mode : disabled (local only)

STEP 1: Extracting metadata...
  employees: analysing (960 records)...  Quality Score: 37.6/100  Anomalies: 9
  departments: analysing (96 records)... Quality Score: 93.3/100  Anomalies: 1
  ...

STEP 3: Writing outputs...
  Excel (46 sheets) → output/complete_data_analysis.xlsx

=======================================================
ANALYSIS COMPLETE
=======================================================
  Tables analysed : 14
  Total records   : 53,981
  Avg quality     : 72.4/100
  Total anomalies : 48
  LLM grade       : N/A (local mode)

Per-table quality scores:
  continents     ██████████ 100.0/100  ✓
  countries      █████████░  94.9/100  ✓
  employees      ███░░░░░░░  37.6/100  ⚠
  tasks          ████░░░░░░  48.8/100  ⚠

Anomalies detected:
  [employees.salary]    NUMERIC_OUTLIERS_DETECTED, NEGATIVE_VALUES_DETECTED
  [employees.hire_date] FUTURE_DATE_DETECTED
  [employees.email]     PATTERN_INCONSISTENCY
  ...

For a more detailed report, refer to:
  /your/path/output/complete_data_analysis.xlsx
```

---

## Installation

```bash
pip install parseiq
```

**With optional extras:**

```bash
pip install parseiq[anthropic]   # Claude models (pip install anthropic)
pip install parseiq[gemini]      # Google Gemini (pip install google-generativeai)
pip install parseiq[rules]       # YAML rules sidecar (pip install pyyaml)
pip install parseiq[all]         # all extras: dotenv, anthropic, gemini, boto3, psycopg2, pymongo, pyyaml
pip install parseiq[s3]          # S3 connector only
pip install parseiq[postgres]    # PostgreSQL connector only
pip install parseiq[mongodb]     # MongoDB connector only
```

**From source:**

```bash
git clone https://github.com/ShriniwasAhirrao/ParseIQ.git
cd ParseIQ-V0.0.1
pip install -e .
```

Requires Python 3.9+.

---

## CLI Usage

### First-time setup

```bash
parseiq init
```

Interactive wizard — choose your LLM provider, paste your API key, pick a model, test the connection, set output directory. Run once.

### Analyse a file

```bash
# Local mode — no API key, always works, instant
parseiq analyze data.json --no-llm

# With AI enrichment (free OpenRouter account works)
parseiq analyze data.json

# CSV, XML, or Excel — same command
parseiq analyze export.csv --no-llm
parseiq analyze report.xlsx --no-llm

# Custom output folder
parseiq analyze data.json --no-llm --output reports/june/

# Force reprocess (ignore incremental cache)
parseiq analyze data.json --no-llm --force

# Quiet mode for scripts / CI
parseiq analyze data.json --no-llm --quiet

# CI quality gate — exit code 1 if avg quality below 80
parseiq analyze data.json --no-llm --fail-under 80
```

### Other commands

```bash
parseiq validate data.json     # quick file check — tables, columns, record count
parseiq models                 # list available LLM models (free, paid, local)
parseiq config                 # show current settings and all detected API keys
parseiq version                # print version
```

### LLM providers

ParseIQ supports any major LLM provider — bring your own key:

```bash
# OpenRouter (default) — free models available, one account covers 100+ models
parseiq analyze data.json --llm-provider openrouter \
  --llm-model nvidia/nemotron-3-super-120b-a12b:free

# OpenAI
parseiq analyze data.json --llm-provider openai \
  --llm-model gpt-4o --llm-api-key sk-...

# Anthropic / Claude (requires: pip install anthropic)
parseiq analyze data.json --llm-provider anthropic \
  --llm-model claude-sonnet-4-5 --llm-api-key sk-ant-...

# Google Gemini (requires: pip install google-generativeai)
parseiq analyze data.json --llm-provider gemini \
  --llm-model gemini-1.5-pro --llm-api-key AIza...

# Perplexity
parseiq analyze data.json --llm-provider perplexity \
  --llm-model llama-3.1-sonar-large-128k-online

# Local Ollama — no API key, no cost, data never leaves your machine
parseiq analyze data.json --llm-provider ollama --llm-model llama3

# Pass key inline without env var
parseiq analyze data.json --llm-api-key sk-or-v1-your-key-here
```

Run `parseiq models` to see the full list with install instructions per provider.

---

## Python API

```python
from parseiq import Pipeline

# Local mode — no API key needed
result = Pipeline("data.json").run(llm=False)

# With LLM — any provider
result = Pipeline("data.json").run(
    llm=True,
    llm_provider="openrouter",           # or openai / anthropic / gemini / perplexity / ollama
    llm_api_key="sk-or-v1-...",
    llm_model="nvidia/nemotron-3-super-120b-a12b:free",
)

# Works with CSV, XML, Excel too
result = Pipeline("export.csv").run(llm=False)
result = Pipeline("report.xlsx").run(llm=False)

# Check results
print(result.tables)                # ["employees", "departments", ...]
print(result.quality_scores)        # {"employees": 37.6, "departments": 93.3}
print(result.overall_quality_score) # 72.4
print(result.total_anomalies)       # 48
print(result.llm_grade)             # "B" or None (local mode)
print(result.output_files)          # list of file paths written
```

### Class-method constructors

```python
Pipeline.from_file("data.json")
Pipeline.from_url("https://api.example.com/data.json")
Pipeline.from_s3("s3://my-bucket/data.json")
Pipeline.from_postgres("postgresql://user:pass@host/db", "SELECT * FROM orders")
Pipeline.from_mongodb("mongodb://localhost:27017", "customers")
```

### Alert rules

```python
from parseiq.alerts import slack_webhook

result = Pipeline("data.json").run(
    llm=False,
    alert_rules={
        "employees.salary": {"negative_values": True},
        "employees.email":  {"null_rate_gt": 0.05},
        "orders":           {"quality_score_lt": 70},
    },
    on_alert=slack_webhook("https://hooks.slack.com/services/..."),
)

print(result.alerts_fired)  # list of matched rules
```

### Incremental processing

```python
# First run — analyses all tables
result = Pipeline("data.json").run(llm=False)

# Second run — skips unchanged tables automatically (hash cache)
result = Pipeline("data.json").run(llm=False)

# Force full reprocess
result = Pipeline("data.json").run(llm=False, force=True)
```

---

## Output Files

Every run produces these files in the output directory:

| File | Contents |
|---|---|
| `complete_data_analysis.xlsx` | Master workbook — all sheets (see structure below) |
| `overall_dataset_summary.csv` | One row per table: records, quality score, anomaly count |
| `combined_issues_and_recommendations.csv` | All flagged issues with priority, fix, effort |
| `raw_metadata.json` | Full Step 1 technical metadata |
| `enriched_metadata.json` | Step 1 + LLM insights merged |

### Excel workbook structure

```
complete_data_analysis.xlsx
├── 00_Summary                  <- dataset-wide overview: one row per table
├── 01_LLM_Assessment           <- LLM quality grade, production readiness, concerns
├── 02_LLM_Recommendations      <- prioritised recommendations from LLM
│
├── Data_employees              <- raw data rows
├── Meta_employees              <- 30-column attribute profile:
│                                  type, nulls, unique ratio, length stats,
│                                  most common values, char distribution,
│                                  anomaly types, outlier count, numeric stats,
│                                  boolean true/false counts
├── Quality_employees           <- long-format quality breakdown:
│                                  Table_Name | Quality_Category | Metric_Name |
│                                  Metric_Value | Status | Description
│
├── Data_departments
├── Meta_departments
├── Quality_departments
├── ... (3 sheets per table, grouped by table)
│
└── 99_Issues_Recommendations   <- all issues sorted by priority (CRITICAL→LOW):
                                   Priority | Table | Column | Issue_Type |
                                   Description | Business_Impact |
                                   Recommended_Fix | Effort
```

---

## Anomaly Detection

ParseIQ flags 11 types of data quality issues — 8 column-level detectors that run automatically, plus 3 cross-table detectors available via the rules sidecar or built-in heuristics:

### Automatic (always on)

| Flag | Triggered when |
|---|---|
| `HIGH_NULL_RATE` | More than 30% of values are null |
| `LOW_UNIQUENESS` | Unique ratio below 10% with more than 10 rows (booleans exempt) |
| `MIXED_DATA_TYPES` | Column contains incompatible types (e.g. integers mixed with strings) |
| `FUTURE_DATE_DETECTED` | ISO date string is beyond today's date |
| `NUMERIC_OUTLIERS_DETECTED` | Z-score or IQR outlier detected in a numeric column |
| `NEGATIVE_VALUES_DETECTED` | Numeric column contains negative values (suppressed for financial convention columns — `drawdown`, `var_*`, `shock`, `cfi_*`, `capex`, `pnl`, etc.) |
| `PATTERN_INCONSISTENCY` | Dominant pattern (e.g. email) but 10–50% of values don't match |
| `DUPLICATE_ROWS_DETECTED` | Exact duplicate rows found at the table level |
| `RANGE_VIOLATION_DETECTED` | A numeric value in a child table falls outside the `[lo, hi]` range defined by a `*_range*` column in a parent table (auto-detected via name-match and FK-child heuristics) |
| `TYPE_CONDITIONAL_FIELD` | Column is absent for some entity types within a polymorphic table (e.g. banking-specific metrics missing for energy stocks). Informational only — 2pt penalty vs 15pt, suppressed from top issues. Discriminator column and group breakdown are included in `schema_groups` metadata. |

### Via `parseiq_rules.yaml` sidecar

| Flag | Rule type | Example |
|---|---|---|
| `SCALE_VIOLATION_DETECTED` | `max_value` / `min_value` | `marks.total > 100` on a 100-point scale |
| `CONSTRAINT_VIOLATION_DETECTED` | `cross_table_compare` | `claimed_amount > sum_assured` across two FK-linked tables |

Each flagged column incurs a quality score penalty. Table score (0–100) reflects overall severity.

---

## LLM Models

Run `parseiq models` to see the full list. Highlights:

**Free (OpenRouter — one account, no cost):**
- `nvidia/nemotron-3-super-120b-a12b:free` — default, strong reasoning
- `mistralai/mistral-small-3.1-24b-instruct:free` — faster responses
- `meta-llama/llama-3.3-70b-instruct:free` — well-rounded

**OpenAI:**
- `gpt-4o` — best overall quality
- `gpt-4o-mini` — fast and cost-efficient

**Anthropic / Claude** (`pip install anthropic`):
- `claude-sonnet-4-5` — recommended, balanced speed/quality
- `claude-opus-4-5` — most capable

**Google Gemini** (`pip install google-generativeai`):
- `gemini-1.5-pro` — large context (1M tokens), free API key available
- `gemini-2.0-flash` — latest, fast

**Local / no cost (Ollama):**
- `llama3`, `mistral`, `phi3`, `gemma2`

**Credit exhaustion:** ParseIQ detects 402 errors and automatically prints free model alternatives.

---

## Configuration

Priority order (highest to lowest):

1. Parameters passed directly to `run()` — `llm_api_key`, `llm_model`, etc.
2. Environment variables (provider-specific)
3. `.env` file in project root (auto-loaded)
4. Built-in defaults

```bash
# Provider-specific environment variables
OPENROUTER_API_KEY=sk-or-v1-...
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=AIza...
PERPLEXITY_API_KEY=pplx-...

# Model and output directory
PARSEIQ_MODEL=nvidia/nemotron-3-super-120b-a12b:free
PARSEIQ_OUTPUT_DIR=output/
```

Save to `.env` file — auto-loaded if `python-dotenv` is installed:
```bash
pip install parseiq[llm]
```

Check current config:
```bash
parseiq config
```

---

## Project Structure

```
parseiq/
├── parseiq/                          # Main package
│   ├── __init__.py                   # Public API: Pipeline, PipelineResult, Config
│   ├── pipeline.py                   # Pipeline class — orchestrates all 3 steps
│   ├── result.py                     # PipelineResult frozen dataclass
│   ├── config.py                     # Centralised configuration
│   ├── alerts.py                     # Alert rules engine + Slack/email helpers
│   ├── _cli.py                       # CLI entry point (parseiq command)
│   ├── connectors/                   # Data source connectors
│   │   ├── file.py                   # Local files (JSON, CSV, XML, Excel)
│   │   ├── url.py                    # HTTP/HTTPS URLs
│   │   ├── s3.py                     # Amazon S3
│   │   ├── postgres.py               # PostgreSQL
│   │   └── mongodb.py                # MongoDB
│   ├── file_loader/
│   │   └── loader.py                 # Multi-format loader + nested JSON flattener
│   ├── step1_metadata_extractor/
│   │   ├── extractor.py              # Metadata extraction, anomaly detection, scoring
│   │   └── utils.py                  # Statistical helpers (zscore, IQR, outliers)
│   └── step2_llm_enricher/
│       ├── llm_agent.py              # Multi-provider LLM client (BYOK)
│       └── prompt_template.txt       # LLM system prompt
│
├── web/                              # Web UI (React + FastAPI)
│   ├── run.py                        # Dev/prod launcher (--dev flag)
│   ├── api/                          # FastAPI backend
│   │   ├── main.py                   # App entry, CORS, exception handlers
│   │   ├── routes/                   # upload, results, settings endpoints
│   │   └── services/                 # Pipeline runner (thread-safe, job queue)
│   └── frontend/                     # React 19 + Vite 8 + TailwindCSS v4
│       ├── src/pages/                # Upload, Processing, Dashboard, TableDetail, Settings
│       ├── src/components/           # Reusable UI (ScoreGauge, FileDropzone, etc.)
│       ├── src/hooks/                # useJobPoller (SSE/polling)
│       └── src/lib/                  # API client, types
│
├── research/                         # Research paper materials
│   ├── technical_overview.md         # Technical details for research paper
│   ├── architecture.md               # System architecture and workflow diagrams
│   └── performance_analysis.ipynb    # Performance benchmarks and comparisons
│
├── tests/                            # 590+ pytest tests
├── input/                            # Sample input files
├── examples/                         # Runnable example scripts
├── pyproject.toml
├── CHANGELOG.md                      # Version history
├── WORKLOG.md                        # Development session log
└── commands.md                       # Full CLI reference
```

---

## Running Tests

```bash
pip install -e ".[dev]"
pytest
pytest --cov=parseiq --cov-report=term-missing
```

Current status: **590/592 passing** (v0.0.6)

---

## Supported Input Formats

| Format | Extension | Notes |
|---|---|---|
| JSON | `.json` | Any nesting depth — arrays of objects become separate tables automatically |
| CSV | `.csv` | Auto-detects delimiter (comma, semicolon, tab) and file encoding |
| XML | `.xml` | Converted via `xmltodict`, then processed as JSON |
| Excel | `.xlsx` `.xls` | Each sheet becomes a separate table |

---

## Web UI (Coming Soon)

ParseIQ now includes a **full-featured web interface** for interactive data quality analysis — drag-and-drop file upload, real-time processing with a live thought-process feed, and a rich results dashboard with per-table drill-down.

**Stack:** React 19 + Vite 8 + TailwindCSS v4 (frontend) · FastAPI + background threading (backend)

**Features:**
- Drag-and-drop file upload (JSON, CSV, XML, Excel — up to 100 MB)
- Toggle LLM enrichment on/off with any configured provider
- Real-time processing page with timestamped event feed
- Interactive results dashboard: quality scores, anomaly badges, score gauges
- Per-table detail view with column profiles, data preview, and nested table navigation
- Configurable settings (API key, model, provider) saved in browser
- Dark theme, fully responsive, accessible (ARIA roles, keyboard navigation)
- Thread-safe backend with concurrent job support (up to 4 simultaneous analyses)

> **Deployment is in progress.** The web UI source is included under `web/` and can be run locally with `python web/run.py --dev`. A hosted version will be available shortly.

---

## What's New in v0.0.6

### LLM Mode Output Fixes

- **Overall Score** and **Model Used** no longer show `N/A` in the Excel `01_LLM_Assessment` sheet
- **Quality score** now correctly reflects actual per-table scores (was always 0 due to wrong metadata lookup)
- Fallback enrichment now populates all expected fields — no blank cells in Excel output
- 6 regression tests added; **590/592 tests passing**

---

## What's New in v0.0.5

### Schema Polymorphism Detection (Issue K)

ParseIQ now handles **heterogeneous record types** within the same JSON array — a pattern common in financial datasets where Energy stocks and Banking stocks share an array but have completely different field sets.

**Before v0.0.5:** ParseIQ treated all records as a single schema. Fields absent for some entity types were flagged as `HIGH_NULL_RATE` (15pt penalty each). A 2-company `holdings` array with 39 schema-conditional fields produced 39 false-positive anomalies.

**After v0.0.5:** ParseIQ detects a *discriminator column* (e.g. `sector`) and groups records by entity type. Fields absent for some types are reclassified as `TYPE_CONDITIONAL_FIELD` — informational (2pt penalty, suppressed from top issues). Same dataset: 0 false-positive `HIGH_NULL_RATE` flags.

**What gets detected:**
- Discriminator column: a low-cardinality categorical field (2–10 unique values, ≥70% present, not an ID/key column) whose values explain which other fields are present or absent
- Type-conditional columns: fields present for some entity types but absent for others
- Schema groups metadata: written to `table_metadata.schema_groups` in raw output

**Also fixed:** `_NEGATIVE_ALLOWED_PATTERNS` extended with `_return` and `_yield` tokens — `predicted_return_1m_pct`, `dividend_yield_pct`, and similar columns no longer produce spurious `NEGATIVE_VALUES_DETECTED` flags.

---

## What's New in v0.0.4

### Bug Fixes

| Issue | Fixed |
|---|---|
| **False positives on financial columns** | `NEGATIVE_VALUES_DETECTED` is now suppressed for columns whose name contains a financial-convention token (`drawdown`, `var_*`, `shock`, `cfi_*`, `cff_*`, `capex`, `pnl`, `loss`, `deficit`, `outflow`, `return_pct`, `alpha`, …). VaR, drawdown, and capex columns no longer produce noise. |
| **Dependency version conflicts** | All 6 core version pins relaxed (`pandas>=1.5`, `numpy>=1.21`, `scipy>=1.7`, `openpyxl>=3.0`, `chardet>=4.0`, `requests>=2.28`) — ParseIQ now installs cleanly alongside projects that pin older versions. |

### New Features

#### Cross-Level Range Violation Detection (automatic, no config needed)

ParseIQ now detects when a value in a child table breaches a numeric range defined in a parent table — without any user configuration.

**How it works:** Any column named `*_range*` (e.g. `temp_range_c`, `normal_range`) whose value is a `"lo, hi"` string is treated as a bound. ParseIQ then:
1. **Name-match** — strips `_range` from the column name and searches all other tables for that measurement column (`temp_range_c` → `temp_c`).
2. **FK-child fallback** — if no name match, finds the direct FK child table and checks its 1–3 numeric columns.

```
Cold-chain example (TC-05):
  inventory_zones.temp_range_c = "2, 8"
  tracking_events.temp_c       = 10.8   ← RANGE_VIOLATION_DETECTED ✅

IoT example (TC-08):
  sensors.normal_range = "0, 10"
  readings.value       = 18.9   ← RANGE_VIOLATION_DETECTED ✅
```

#### Rules Sidecar — Domain Constraint Validation

Place a `parseiq_rules.yaml` (or `.json`) file next to your input file. ParseIQ detects it automatically and applies the rules after Step 1 — no CLI flag, no code change.

```yaml
# parseiq_rules.yaml
rules:
  # Issue I: scale/domain violation — marks total must be <= 100
  - type: max_value
    table: students_enrolled
    column: marks__total
    max: 100
    message: "Marks total={val} exceeds 100-point scale"

  # Issue H: cross-table constraint — claim cannot exceed policy limit
  - type: cross_table_compare
    left_table: claims
    left_col: claimed_amount
    right_table: policies
    right_col: sum_assured
    join_key: policy_id
    fk_key: _ref_policies_id
    rule: "left <= right"
    message: "claimed_amount exceeds sum_assured — possible fraud"
```

Supported rule types:

| Rule type | What it checks | Anomaly raised |
|---|---|---|
| `max_value` | `column > max` | `SCALE_VIOLATION_DETECTED` |
| `min_value` | `column < min` | `SCALE_VIOLATION_DETECTED` |
| `cross_table_compare` | `left_col OP right_col` across FK-linked tables | `CONSTRAINT_VIOLATION_DETECTED` |

YAML parsing requires `pyyaml` (`pip install parseiq[rules]`). JSON rules files work with no extra dependency.

Violations are injected into each affected table's `top_issues` and `anomaly_summary`, and stored in `raw_metadata["rule_violations"]`.

---

## Key Features

| Feature | Detail |
|---|---|
| **Deep nested JSON flattening** | Recursively discovers all tables to arbitrary depth; injects `_ref_<parent>_id` FK columns |
| **Multi-format input** | JSON (nested), CSV (auto-delimiter), XML, Excel `.xlsx` |
| **11 anomaly detectors** | 8 column-level (always on) + cross-level range (auto) + 2 via rules sidecar |
| **Domain-aware negative suppression** | Financial convention columns (`drawdown`, `var_*`, `capex`, `pnl`, …) never trigger false positive NEGATIVE_VALUES flags |
| **YAML/JSON rules sidecar** | `parseiq_rules.yaml` next to your file — max/min bounds and cross-table constraints, zero config |
| **Multi-provider LLM** | OpenRouter, OpenAI, Anthropic/Claude, Gemini, Perplexity, Ollama |
| **BYOK architecture** | Bring your own key — data goes to your LLM account, not ours |
| **Local mode** | `llm=False` — full Step 1 analysis, no API key, data never leaves machine |
| **Credit exhaustion detection** | 402 errors trigger a free-model suggestion automatically |
| **Graceful degradation** | LLM failure falls back to local Step 1 report, no crash |
| **Incremental processing** | Hash-based cache — unchanged tables reuse previous results |
| **Rich Meta sheet** | 30-column attribute profile per table |
| **Long-format Quality sheet** | One metric row per attribute: category, value, status, description |
| **Alert rules** | Post-analysis rule evaluation with Slack/email callback helpers |
| **Structured result object** | `PipelineResult` dataclass with quality scores, anomalies, grades |
| **Web UI** | Drag-and-drop upload, real-time processing, interactive dashboard (coming soon — run locally now) |
| **CLI + Python API** | Command-line tool or importable library |
| **592 tests** | Full pytest suite across all components |

---

## Limitations (v0.0.6)

- Free-tier OpenRouter: ~10 RPM — one LLM call per run, not per table
- LLM response time: 2–3 min for large datasets on free tier
- Max file size: 100 MB
- Web UI not yet deployed — run locally with `python web/run.py --dev`
- YAML rules sidecar requires `pip install parseiq[rules]` for `.yaml` files; `.json` rules work without it

---

## Roadmap

**v0.1.0**
- PDF report export
- Batch processing — `parseiq analyze-all data/` (folder of files in one command)
- Cross-table FK orphan detection — flag `_ref_*` values that don't exist in the parent table
- `conftest.py` — shared test fixtures
- Web UI deployment (hosted version)

**v0.2.0**
- Parquet and Google Sheets support
- Multi-tenancy / job queue (Celery + Redis)

---

## Dependencies

| Library | Purpose |
|---|---|
| `pandas` | Data manipulation and DataFrame operations |
| `numpy` / `scipy` | Statistical calculations and outlier detection |
| `openpyxl` | Excel report generation |
| `requests` | LLM API calls (OpenAI-compatible providers) |
| `xmltodict` | XML parsing |
| `chardet` | Character encoding detection |
| `python-dateutil` | Date parsing |

Optional:

| Package | Extra | Purpose |
|---|---|---|
| `python-dotenv` | `[llm]` | `.env` file loading |
| `anthropic` | `[anthropic]` | Anthropic/Claude API |
| `google-generativeai` | `[gemini]` | Google Gemini API |
| `pyyaml` | `[rules]` | YAML rules sidecar (`parseiq_rules.yaml`) |
| `boto3` | `[s3]` | S3 connector |
| `psycopg2-binary` | `[postgres]` | PostgreSQL connector |
| `pymongo` | `[mongodb]` | MongoDB connector |

---

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

- Found a bug? [Open an issue](https://github.com/ShriniwasAhirrao/ParseIQ/issues/new?template=bug_report.md)
- Have an idea? [Request a feature](https://github.com/ShriniwasAhirrao/ParseIQ/issues/new?template=feature_request.md)
- Questions? [Start a discussion](https://github.com/ShriniwasAhirrao/ParseIQ/discussions)

---

## Star History

<p align="center">
  <a href="https://star-history.com/#ShriniwasAhirrao/ParseIQ&Date">
    <img src="https://api.star-history.com/svg?repos=ShriniwasAhirrao/ParseIQ&type=Date" alt="Star History Chart" width="600"/>
  </a>
</p>

If ParseIQ saves you time, consider giving it a ⭐ — it helps others discover it.

---

## License

MIT — see [LICENSE](LICENSE)

---

## Author

Built by [Shriniwas Ahirrao](https://github.com/ShriniwasAhirrao).
