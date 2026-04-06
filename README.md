# ParseIQ — AI-Powered Data Quality Agent

> Understand your data before you trust it.

ParseIQ is a Python library and CLI tool that analyses any data file (JSON, CSV, XML, Excel) and produces a full data quality report — statistical profiling, anomaly detection, per-table quality scores, and optional AI-generated recommendations — all in a structured Excel workbook and a set of CSVs.

Built for the **data onboarding and discovery phase**: when you receive a data dump and need to know what's in it, whether it's usable, and what to fix before loading it into production.

---

## Quickstart

```bash
pip install parseiq

parseiq init                              # first-time setup (API key, model)
parseiq analyze data.json --no-llm       # local mode — no API key needed
parseiq analyze data.json                # with AI enrichment (needs API key)
```

That's it. Results appear in `output/` as an Excel workbook + CSV summaries.

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
pip install parseiq[all]       # includes dotenv, boto3, psycopg2, pymongo
pip install parseiq[s3]        # S3 connector only
pip install parseiq[postgres]  # PostgreSQL connector only
pip install parseiq[mongodb]   # MongoDB connector only
```

**From source:**

```bash
git clone https://github.com/ShriniwasAhirrao/ParseIQ-V0.0.1.git
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

Interactive wizard that asks for your API key, lets you pick a model, tests the connection, and saves everything to `.env`. Run this once.

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
parseiq config                 # show current settings and API key status
parseiq version                # print version
```

### LLM providers

```bash
# OpenRouter (default) — free models available
parseiq analyze data.json --llm-provider openrouter --llm-model nvidia/nemotron-3-super-120b-a12b:free

# OpenAI
parseiq analyze data.json --llm-provider openai --llm-model gpt-4o --llm-api-key sk-...

# Local Ollama — no API key, no cost, no data leaves machine
parseiq analyze data.json --llm-provider ollama --llm-model llama3

# Pass key directly without setting env var
parseiq analyze data.json --llm-api-key sk-or-v1-your-key-here
```

---

## Python API

```python
from parseiq import Pipeline

# Local mode — no API key needed
result = Pipeline("data.json").run(llm=False)

# With LLM
result = Pipeline("data.json").run(
    llm=True,
    llm_provider="openrouter",
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
# Load from different sources
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
# First run — analyses all 14 tables
result = Pipeline("data.json").run(llm=False)

# Second run — skips unchanged tables (uses hash cache)
result = Pipeline("data.json").run(llm=False)

# Force full reprocess
result = Pipeline("data.json").run(llm=False, force=True)
```

---

## Output Files

Every run produces these files in the output directory:

| File | Contents |
|---|---|
| `complete_data_analysis.xlsx` | Master workbook — Data / Metadata / Quality sheets per table + 2 summary tabs |
| `overall_dataset_summary.csv` | One row per table: records, quality score, anomaly count |
| `combined_issues_and_recommendations.csv` | All flagged issues with recommended fixes |
| `raw_metadata.json` | Full Step 1 technical metadata |
| `enriched_metadata.json` | Step 1 + LLM insights merged |
| `llm_insights.json` | Raw LLM response (only when LLM is enabled) |

### Excel workbook structure

```
complete_data_analysis.xlsx
├── 00_Overall_Summary          <- dataset-wide quality metrics
├── 99_Issues_Recommendations   <- all issues with recommended actions
├── Data_employees              <- raw data rows for the employees table
├── Meta_employees              <- column metadata: type, nulls, stats, anomaly flags
├── Quality_employees           <- quality score breakdown per column
├── Data_departments
├── Meta_departments
├── Quality_departments
└── ... (3 sheets per table discovered)
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
| `NUMERIC_OUTLIERS_DETECTED` | Z-score or IQR outlier found in a numeric column |
| `NEGATIVE_VALUES_DETECTED` | Numeric column contains negative values |
| `PATTERN_INCONSISTENCY` | Dominant pattern exists (e.g. email format) but 10–50% of values don't match |
| `DUPLICATE_ROWS_DETECTED` | Exact duplicate rows found at the table level |

Each flagged column incurs a score penalty. The table quality score (0–100) reflects the overall severity.

---

## Key Features

| Feature | Detail |
|---|---|
| **Deep nested JSON flattening** | Recursively discovers all tables in any JSON hierarchy; injects `_ref_<parent>_id` FK columns |
| **Multi-format input** | JSON (nested), CSV (auto-delimiter), XML, Excel `.xlsx` |
| **BYOK LLM** | Bring your own key — OpenRouter, OpenAI, Azure, or local Ollama |
| **Local mode** | `llm=False` — full Step 1 analysis, no API key, data never leaves your machine |
| **Graceful degradation** | LLM call fails or times out → Step 1 report is still saved, no crash |
| **Incremental processing** | Hash-based cache — unchanged tables reuse previous results on re-runs |
| **Alert rules** | Post-analysis rule evaluation with Slack/email callback helpers |
| **Structured result object** | `PipelineResult` dataclass with quality scores, anomalies, grades |
| **CLI + Python API** | Use as a command-line tool or import directly into any Python project |
| **159 tests** | Full pytest suite across all components |

---

## LLM Models

Run `parseiq models` to see the full list. Highlights:

**Free (OpenRouter account, no cost):**
- `nvidia/nemotron-3-super-120b-a12b:free` — default, strong reasoning
- `mistralai/mistral-small-3.1-24b-instruct:free` — faster
- `meta-llama/llama-3.3-70b-instruct:free` — well-rounded

**Paid:**
- `openai/gpt-4o` — best overall quality
- `openai/gpt-4o-mini` — fast and cheap

**Local (no API key, no cost):**
- `llama3`, `mistral`, `phi3` via Ollama

---

## Configuration

Priority order (highest to lowest):

1. Parameters passed directly to `run()` — `llm_api_key`, `llm_model`, etc.
2. Environment variables — `OPENROUTER_API_KEY`, `PARSEIQ_MODEL`
3. `.env` file in project root (auto-loaded)
4. Built-in defaults

```bash
# Set in environment
export OPENROUTER_API_KEY=sk-or-v1-your-key-here
export PARSEIQ_MODEL=mistralai/mistral-small-3.1-24b-instruct:free

# Or save to .env file
echo "OPENROUTER_API_KEY=sk-or-v1-..." >> .env
```

Check current config:
```bash
parseiq config
```

---

## Project Structure

```
parseiq/
├── parseiq/                         # Main package
│   ├── __init__.py                  # Public API: Pipeline, PipelineResult, Config
│   ├── pipeline.py                  # Pipeline class + MetadataEnrichmentAgent shim
│   ├── result.py                    # PipelineResult frozen dataclass
│   ├── config.py                    # Centralised configuration
│   ├── alerts.py                    # Alert rules engine + Slack/email helpers
│   ├── _cli.py                      # CLI entry point (parseiq command)
│   ├── connectors/                  # Data source connectors
│   │   ├── file.py                  # Local files (JSON, CSV, XML, Excel)
│   │   ├── url.py                   # HTTP/HTTPS URLs
│   │   ├── s3.py                    # Amazon S3
│   │   ├── postgres.py              # PostgreSQL
│   │   └── mongodb.py               # MongoDB
│   ├── file_loader/
│   │   └── loader.py                # Multi-format loader + nested JSON flattener
│   ├── step1_metadata_extractor/
│   │   ├── extractor.py             # Metadata extraction, anomaly detection, scoring
│   │   └── utils.py                 # Statistical helpers
│   └── step2_llm_enricher/
│       ├── llm_agent.py             # LLM API client (multi-provider, BYOK)
│       └── prompt_template.txt      # LLM system prompt
│
├── examples/                        # Runnable example scripts
│   ├── from_json_file.py
│   ├── from_postgres.py
│   ├── from_s3.py
│   ├── with_alert_rules.py
│   └── with_local_llm_ollama.py
│
├── tests/                           # 159 pytest tests
├── pyproject.toml
├── commands.md                      # Full CLI command reference
└── CHANGELOG.md
```

---

## Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run full test suite
pytest

# With coverage report
pytest --cov=parseiq --cov-report=term-missing
```

Current status: **159/159 passing**

---

## Supported Input Formats

| Format | Extension | Notes |
|---|---|---|
| JSON | `.json` | Any nesting depth — all arrays of objects become separate tables automatically |
| CSV | `.csv` | Auto-detects delimiter (comma, semicolon, tab) and file encoding |
| XML | `.xml` | Converted via `xmltodict`, then processed as JSON |
| Excel | `.xlsx` `.xls` | Each sheet becomes a separate table |

---

## Limitations (V.0.0.1)

- Free-tier OpenRouter: ~10 RPM — one LLM call per run, not per table
- LLM response time: 2–3 minutes for large datasets on free tier
- Max file size: 100 MB
- No live dashboard — output is files only

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
| `requests` | LLM API calls |
| `xmltodict` | XML parsing |
| `chardet` | Character encoding detection |
| `python-dateutil` | Date parsing |

Optional:
| `python-dotenv` | `.env` file loading |
| `boto3` | S3 connector |
| `psycopg2-binary` | PostgreSQL connector |
| `pymongo` | MongoDB connector |

---

## License

MIT

---

## Author

Built by [Shriniwas Ahirrao](https://github.com/ShriniwasAhirrao).
