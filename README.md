# ParseIQ — AI-Powered Data Quality & Metadata Agent

> Understand your data before you trust it.

ParseIQ is an AI data agent that takes raw files (JSON, CSV, XML, Excel) and produces a full data quality report — anomaly flags, statistical profiling, per-table metadata, and LLM-generated recommendations — all in a structured Excel workbook and a set of CSVs.

It is designed for the **data onboarding / data discovery phase**: when you receive a data dump and need to know what's in it, whether it's usable, and what to fix before loading it into production.

---

## What It Does

```
Input file (JSON / CSV / XML / Excel)
         ↓
 Step 1 — Metadata Extractor
   • Flatten deeply nested JSON (any depth)
   • Detect table structure, data types, statistics
   • Flag 8 anomaly types per column
   • Score every table 0–100
         ↓
 Step 2 — LLM Enricher  (BYOK via OpenRouter)
   • Business-level interpretation of quality issues
   • Cross-table relationship insights
   • Prioritised action recommendations
         ↓
 Output — Excel workbook + CSV summary files
```

---

## Key Features

| Feature | Detail |
|---|---|
| **Deep nested JSON flattening** | Recursively discovers all tables in any JSON hierarchy; injects FK columns linking children back to parents |
| **8 anomaly detectors** | `HIGH_NULL_RATE`, `LOW_UNIQUENESS`, `MIXED_DATA_TYPES`, `FUTURE_DATE_DETECTED`, `NUMERIC_OUTLIERS_DETECTED`, `NEGATIVE_VALUES_DETECTED`, `PATTERN_INCONSISTENCY`, `DUPLICATE_ROWS_DETECTED` |
| **Per-table quality scores** | Every table and every column is scored 0–100 based on anomaly severity |
| **Multi-format input** | JSON (including deeply nested), CSV (auto-delimiter detection), XML, Excel `.xlsx` |
| **BYOK LLM** | Bring your own OpenRouter API key — your data goes to your LLM account, not a third party |
| **Structured Excel output** | One workbook, separate sheets per table: Data / Metadata / Quality |
| **159 tests** | Full test suite across all components |

---

## Output

Running ParseIQ on any input file produces **6 files** in the `output/` directory:

| File | Contents |
|---|---|
| `complete_data_analysis.xlsx` | Master workbook — Data, Metadata, and Quality sheets per table + two summary tabs |
| `overall_dataset_summary.csv` | One row per table: record count, quality score, anomaly count |
| `combined_issues_and_recommendations.csv` | Prioritised issue list with affected table, category, and recommended action |
| `raw_metadata.json` | Full technical metadata from Step 1 |
| `enriched_metadata.json` | Step 1 metadata merged with LLM insights |
| `llm_insights.json` | Raw LLM response |

### Excel workbook structure

```
complete_data_analysis.xlsx
├── 00_Overall_Summary          ← dataset-wide quality overview
├── 01_Issues_Recommendations   ← prioritised issues with affected tables
├── Data_employees              ← raw data for the employees table
├── Meta_employees              ← column-level metadata (type, nulls, stats, anomalies)
├── Quality_employees           ← quality score breakdown per column
├── Data_departments
├── Meta_departments
├── Quality_departments
└── ... (one set of 3 sheets per table discovered)
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- An [OpenRouter](https://openrouter.io/) API key (free tier works — the default model is `nvidia/nemotron-3-super-120b-a12b:free`)

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/your-username/ParseIQ.git
cd ParseIQ

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your API key
# Option A — environment variable (recommended)
export OPENROUTER_API_KEY="sk-or-v1-..."   # Windows: set OPENROUTER_API_KEY=sk-or-v1-...

# Option B — .env file in the project root
echo OPENROUTER_API_KEY="sk-or-v1-..." > .env
```

### Run

```bash
# Place your input file in the input/ directory, then:
python main.py
```

Output files appear in `output/`. The terminal prints a live step-by-step progress log.

---

## Project Structure

```
ParseIQ/
├── main.py                          # Pipeline entry point
├── config.py                        # All configuration (thresholds, LLM settings, file settings)
├── requirements.txt
│
├── file_loader/
│   └── loader.py                    # Multi-format loader + nested JSON flattener
│
├── step1_metadata_extractor/
│   ├── extractor.py                 # Metadata extraction, anomaly detection, quality scoring
│   └── utils.py                     # Statistical helpers
│
├── step2_llm_enricher/
│   ├── llm_agent.py                 # OpenRouter API client + metadata enrichment
│   └── prompt_template.txt          # LLM system prompt
│
├── input/                           # Place your input files here
│   └── input_data.json              # Example: 4-level nested org dataset (13 tables, 50 records)
│
├── output/                          # Generated reports land here (git-ignored)
│
├── tests/
│   ├── test_comprehensive.py        # 109 tests — all components, all branches
│   ├── test_bug_fixes.py            # 10 targeted bug-fix verification tests
│   ├── test_config.py
│   ├── test_file_loader.py
│   ├── test_metadata_extractor.py
│   ├── test_llm_enricher.py
│   ├── test_main.py
│   ├── test_integration.py
│   └── test_statistical_utils.py
│
└── scripts/
    └── generate_stress_test.py      # Generates a 14-level, 53k-record stress test dataset
```

---

## Anomaly Detection

ParseIQ flags 8 types of data quality anomalies at the column level:

| Flag | Triggered when |
|---|---|
| `HIGH_NULL_RATE` | > 30% of values are null |
| `LOW_UNIQUENESS` | Unique ratio < 10% and > 10 rows (boolean columns exempt) |
| `MIXED_DATA_TYPES` | Same column contains values of incompatible types (e.g. int + string) |
| `FUTURE_DATE_DETECTED` | ISO date string is beyond today's date |
| `NUMERIC_OUTLIERS_DETECTED` | Z-score or IQR outlier found in numeric column |
| `NEGATIVE_VALUES_DETECTED` | Numeric column contains negative values |
| `PATTERN_INCONSISTENCY` | Column has a dominant regex pattern but 10–50% of values don't match it |
| `DUPLICATE_ROWS_DETECTED` | Table-level: exact duplicate rows found |

Each flagged column incurs a quality score penalty. Tables with many issues score lower.

---

## Configuration

Edit `config.py` to adjust behaviour without touching pipeline code:

```python
# Change the LLM model (any OpenRouter model works)
MODEL_NAME = "nvidia/nemotron-3-super-120b-a12b:free"

# Adjust anomaly sensitivity
ANOMALY_THRESHOLDS = {
    'high_null_rate': 30.0,       # % nulls to trigger HIGH_NULL_RATE
    'min_unique_ratio': 0.1,      # unique ratio to trigger LOW_UNIQUENESS
    'z_score_threshold': 3.0,     # std deviations for outlier detection
    'iqr_multiplier': 1.5,        # IQR multiplier for outlier detection
}

# LLM call settings
LLM_SETTINGS = {
    'max_tokens': 4096,
    'temperature': 0.1,
    'timeout': 240,
    'retry_attempts': 3,
}
```

---

## Supported Input Formats

| Format | Notes |
|---|---|
| **JSON** | Any depth of nesting. Sibling arrays with the same name are merged into one table. Child records get a `_ref_<parent>_id` FK column injected automatically. |
| **CSV** | Auto-detects delimiter (comma, semicolon, tab). Handles encoding detection. Null cells correctly preserved as `None`. |
| **XML** | Converted via `xmltodict` then processed as JSON. |
| **Excel** | `.xlsx` files loaded via `openpyxl`. |

---

## Running Tests

```bash
# Run the full test suite
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=. --cov-report=term-missing

# Run only the bug-fix verification tests
pytest tests/test_bug_fixes.py -v
```

Current status: **159/159 tests passing**

---

## Limitations (V.0.0.1)

- **File-based input only** — no direct database or cloud storage connectors yet
- **Single-run, no incremental processing** — re-runs the full analysis each time
- **Static reports** — no live dashboards or alerting hooks yet
- **Single-user design** — no concurrency or multi-tenant isolation

These are planned for V.0.1 (Python library + CLI release). See [TODO.md](TODO.md) for the full roadmap.

---

## Roadmap

**V.0.1 — Python Library + CLI**
- `pip install parseiq`
- `parseiq run mydata.json --output ./reports/ --api-key sk-...`
- `from parseiq import Pipeline; Pipeline(api_key=...).run("mydata.json")`
- Configurable LLM provider (OpenAI, Azure, Ollama, OpenRouter)

**V.0.2 — Connectors**
- `Pipeline.from_postgres(conn_string, table)`
- `Pipeline.from_s3(bucket, key)`
- `Pipeline.from_mongodb(uri, collection)`

**V.0.3 — Incremental + Alerting**
- State file to skip unchanged tables between runs
- `alert_rules` + `on_alert` callback for integration into existing pipelines

---

## Dependencies

| Library | Purpose |
|---|---|
| `pandas` | Data manipulation, DataFrame operations |
| `numpy` / `scipy` | Statistical calculations, outlier detection |
| `openpyxl` | Excel report generation |
| `requests` | OpenRouter API calls |
| `xmltodict` | XML parsing |
| `python-dotenv` | `.env` file support for API key |
| `chardet` | Character encoding detection for CSV files |
| `python-dateutil` | Date parsing |
| `pytest` / `pytest-cov` | Testing and coverage |

---

## License

MIT

---

## Author

Built by [your name] as part of an internship project at Ilink Digital.
