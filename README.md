# ParseIQ — AI-Powered Data Quality Agent

> Understand your data before you trust it.

ParseIQ analyses any data file (JSON, CSV, XML, Excel) and produces a full data quality report — statistical profiling, anomaly detection, per-table quality scores, and optional AI-generated recommendations — all in a structured Excel workbook and CSV summaries.

Built for the **data onboarding and discovery phase**: when you receive a data dump and need to know what's in it, whether it's trustworthy, and what needs fixing before loading into production.

---

## Quickstart

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

## What It Does

```
Input file  (JSON / CSV / XML / Excel)
         |
         v
 Step 1 — Metadata Extractor  (always runs, no API key needed)
   * Flatten deeply nested JSON into multiple tables automatically
   * Detect data types, compute statistics (min/max/mean/percentiles)
   * Flag 8 anomaly types per column
   * Score every table 0-100
         |
         v
 Step 2 — LLM Enricher  (optional, BYOK)
   * Business-level interpretation of quality issues
   * Cross-table relationship insights
   * Prioritised recommendations with effort estimates
         |
         v
 Output — Excel workbook  +  CSV summaries  +  JSON metadata files
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
pip install parseiq[all]         # all extras: dotenv, anthropic, gemini, boto3, psycopg2, pymongo
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

ParseIQ flags 8 types of data quality issues at the column level:

| Flag | Triggered when |
|---|---|
| `HIGH_NULL_RATE` | More than 30% of values are null |
| `LOW_UNIQUENESS` | Unique ratio below 10% with more than 10 rows (booleans exempt) |
| `MIXED_DATA_TYPES` | Column contains incompatible types (e.g. integers mixed with strings) |
| `FUTURE_DATE_DETECTED` | ISO date string is beyond today's date |
| `NUMERIC_OUTLIERS_DETECTED` | Z-score or IQR outlier detected in a numeric column |
| `NEGATIVE_VALUES_DETECTED` | Numeric column contains negative values |
| `PATTERN_INCONSISTENCY` | Dominant pattern (e.g. email) but 10–50% of values don't match |
| `DUPLICATE_ROWS_DETECTED` | Exact duplicate rows found at the table level |

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
├── tests/                            # 159 pytest tests
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

Current status: **159/159 passing**

---

## Supported Input Formats

| Format | Extension | Notes |
|---|---|---|
| JSON | `.json` | Any nesting depth — arrays of objects become separate tables automatically |
| CSV | `.csv` | Auto-detects delimiter (comma, semicolon, tab) and file encoding |
| XML | `.xml` | Converted via `xmltodict`, then processed as JSON |
| Excel | `.xlsx` `.xls` | Each sheet becomes a separate table |

---

## Key Features

| Feature | Detail |
|---|---|
| **Deep nested JSON flattening** | Recursively discovers all tables; injects `_ref_<parent>_id` FK columns |
| **Multi-format input** | JSON (nested), CSV (auto-delimiter), XML, Excel `.xlsx` |
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
| **CLI + Python API** | Command-line tool or importable library |
| **159 tests** | Full pytest suite across all components |

---

## Limitations (V.0.0.2)

- Free-tier OpenRouter: ~10 RPM — one LLM call per run, not per table
- LLM response time: 2–3 min for large datasets on free tier
- Max file size: 100 MB
- Output is files only — no live dashboard

---

## Roadmap

**V.0.1.0**
- PDF report export
- Batch processing (folder of files in one command)
- Cross-table FK violation detection (orphaned records)

**V.0.2.0**
- Web UI — drag-and-drop file upload, results in browser
- Custom YAML rule definitions (`salary > 0`, `email matches pattern`)
- Parquet and Google Sheets support

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
| Package | Purpose |
|---|---|
| `python-dotenv` | `.env` file loading |
| `anthropic` | Anthropic/Claude API |
| `google-generativeai` | Google Gemini API |
| `boto3` | S3 connector |
| `psycopg2-binary` | PostgreSQL connector |
| `pymongo` | MongoDB connector |

---

## License

MIT — see [LICENSE](LICENSE)

---

## Author

Built by [Shriniwas Ahirrao](https://github.com/ShriniwasAhirrao).
