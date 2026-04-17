# ParseIQ — Technical Overview for Research Paper

> Author: Shriniwas Ahirrao
> Project: ParseIQ — AI-Powered Data Quality Agent
> Version: 0.0.6 (library) + Web UI Platform
> Date: April 2026

---

## 1. Abstract Summary

ParseIQ is an open-source, AI-powered data quality assessment agent that automates the profiling, anomaly detection, and quality scoring of structured data files. It implements a three-step pipeline — metadata extraction, optional LLM enrichment, and structured output generation — that transforms raw JSON, CSV, XML, or Excel files into comprehensive quality reports. The system operates in two modes: a fully offline local mode requiring no external services, and an AI-augmented mode that leverages large language models (LLMs) via a Bring Your Own Key (BYOK) architecture for business-level interpretation and recommendations.

Key contributions:
1. A recursive nested-JSON flattener that extracts arbitrarily deep hierarchical structures into relational tables with auto-injected foreign key relationships
2. Eleven statistical anomaly detectors including a novel schema polymorphism detector that eliminates false positives in heterogeneous datasets
3. A multi-provider LLM integration layer supporting 7+ providers with automatic fallback to local mode
4. A rate-based quality scoring algorithm that produces meaningful 0-100 scores even for wide tables with hundreds of attributes
5. A full-stack web platform for interactive data quality analysis

---

## 2. Problem Statement

The data onboarding and discovery phase presents a critical challenge in modern data engineering: when organisations receive external data dumps, they must quickly assess data quality, identify anomalies, understand schema structure, and determine trustworthiness before loading data into production systems. Existing tools either:

- **Require extensive configuration** (Great Expectations — YAML expectation suites)
- **Cannot handle nested JSON** (ydata-profiling — flat DataFrames only)
- **Lack automated anomaly detection** (pandas-profiling — descriptive statistics only)
- **Have no LLM integration** for business-level interpretation
- **Require specific infrastructure** (Great Expectations — database backends, checkpoint configs)

ParseIQ addresses these gaps with a zero-configuration, multi-format tool that works immediately on any structured data file.

---

## 3. System Architecture

### 3.1 Pipeline Overview

```
Input File (JSON/CSV/XML/Excel)
        |
        v
[Step 1: Metadata Extractor]     <-- Always runs, fully local
  |-- File loading + format detection
  |-- Nested JSON recursive flattening
  |-- Per-table statistical profiling
  |-- 11 anomaly detectors
  |-- Quality scoring (0-100)
  |-- Cross-table relationship analysis
  |-- Schema polymorphism detection
        |
        v
[Step 2: LLM Enricher]           <-- Optional, BYOK
  |-- Multi-provider routing
  |-- Metadata + sample compression
  |-- Business-level interpretation
  |-- Quality grade (A-F)
  |-- Prioritised recommendations
  |-- Automatic fallback to local
        |
        v
[Step 3: Output Generator]
  |-- Excel workbook (grouped sheets)
  |-- CSV summaries
  |-- JSON metadata files
```

### 3.2 Data Flow

1. **Input Normalisation**: `FileLoader` detects format, encoding (via `chardet`), and delimiter (CSV). All formats are normalised to `Dict[str, List[Dict]]` — a mapping of table names to lists of record dictionaries.

2. **Nested JSON Flattening**: `_flatten_nested_json()` recursively walks the JSON tree:
   - Array-of-dicts → extracted as child tables with `_ref_{parent}_id` FK injection
   - Nested dicts → flattened to scalar leaves with `__`-joined key paths (e.g., `financials__fy2025__revenue`)
   - Primitive arrays → joined as comma-separated strings
   - Handles arbitrary nesting depth (tested up to 14 levels)

3. **Per-Attribute Profiling**: For each attribute in each table, 30+ statistics are computed:
   - Core: data type, present/missing count, missing percentage, unique values, unique ratio
   - String: min/max/avg/median length, most common values, character distribution, recognised patterns
   - Numeric: min/max/mean/median, standard deviation, outlier count (Z-score + IQR)
   - Boolean: true/false count and percentage
   - Anomaly: flags, types, severity

4. **Quality Scoring**: Composite score per table (0-100) combining:
   - Per-attribute penalties (15pt per anomaly flag)
   - Rate-based table penalty: `min(anomalous_attrs / total_attrs * 20, 20)` — caps contribution so wide tables aren't penalised disproportionately
   - Missing value penalties
   - Duplicate row penalties: `min(duplicate_rate * 20, 20)`

5. **LLM Enrichment** (optional): Compressed metadata + data samples sent to the user's chosen LLM provider for:
   - Business-level data quality assessment
   - Quality grade (A-F) with justification
   - Prioritised recommendations with effort estimates
   - Production readiness evaluation

---

## 4. Anomaly Detection Algorithms

### 4.1 Column-Level Detectors (Always Active)

| Detector | Algorithm | Threshold | Penalty |
|---|---|---|---|
| `HIGH_NULL_RATE` | `missing_count / total_count` | > 30% | 15pt |
| `LOW_UNIQUENESS` | `unique_values / total_count` (booleans exempt) | < 10% with > 10 rows | 15pt |
| `MIXED_DATA_TYPES` | Type counting (int+float grouped as `numeric`) | > 1 type present | 15pt |
| `FUTURE_DATE_DETECTED` | ISO-8601 date parsing + comparison to `datetime.today()` | Any date > today | 15pt |
| `NUMERIC_OUTLIERS_DETECTED` | Dual method: Z-score (|z| > 3) OR IQR (< Q1-1.5*IQR or > Q3+1.5*IQR) | Either triggers | 15pt |
| `NEGATIVE_VALUES_DETECTED` | `any(value < 0)` with domain-aware suppression | Any negative value | 15pt |
| `PATTERN_INCONSISTENCY` | Dominant pattern detection (email, phone, UUID) + mismatch rate | 10-50% non-conforming | 15pt |
| `DUPLICATE_ROWS_DETECTED` | Full-row hash comparison | Any exact duplicates | 2pt/dup, max 20pt |

### 4.2 Cross-Table Detectors (Automatic)

| Detector | Algorithm | Trigger |
|---|---|---|
| `RANGE_VIOLATION_DETECTED` | Parent `*_range*` column parsed as `[lo, hi]`; child measurement column checked. Two-tier: (1) name-match strip `_range`, (2) FK-child fallback | Value outside parent-defined range |
| `TYPE_CONDITIONAL_FIELD` | Schema polymorphism detector: finds discriminator column (low-cardinality categorical, 2-10 unique, >=70% present), groups records, classifies fields absent for some types | Column present in some entity types but absent in others |

### 4.3 Rule-Based Detectors (Via YAML/JSON Sidecar)

| Detector | Rule Type | Example |
|---|---|---|
| `SCALE_VIOLATION_DETECTED` | `max_value` / `min_value` | `marks.total > 100` |
| `CONSTRAINT_VIOLATION_DETECTED` | `cross_table_compare` | `claimed_amount > sum_assured` across FK-linked tables |

### 4.4 Domain-Aware Negative Suppression

The `NEGATIVE_VALUES_DETECTED` detector includes a suppression mechanism for financial convention columns. A module-level tuple of 15 token patterns (`drawdown`, `var_`, `shock`, `cfi_`, `cff_`, `capex`, `pnl`, `loss`, `deficit`, `outflow`, `return_pct`, `change_pct`, `alpha`, `_return`, `_yield`) is matched against the lowercased column name. If any pattern appears, the negative flag is suppressed. This eliminates false positives on columns like `max_drawdown_pct`, `var_1d_99_pct`, `predicted_return_1m_pct`.

### 4.5 Schema Polymorphism Detection Algorithm

**Problem**: JSON arrays often contain records of fundamentally different entity types (e.g., Energy stocks and Banking stocks in a `holdings` array). Fields specific to one type produce mass `HIGH_NULL_RATE` false positives.

**Algorithm** (`_detect_schema_groups`):
1. **Candidate discriminator selection**: For each column, check:
   - Presence rate >= 70%
   - 2-10 unique values (categorical)
   - Non-numeric data type
   - Not an identifier column (filtered by `_id` suffix, `isin`, `name`, `ticker`, etc.)
2. **Record grouping**: Group records by discriminator value
3. **Purity scoring**: For each column with 15-85% overall presence rate, compute per-group presence purity (>= 80% within each group = pure)
4. **Acceptance**: If `pure_columns / variable_columns > 0.25`, accept discriminator
5. **Reclassification**: Type-conditional columns get `TYPE_CONDITIONAL_FIELD` (2pt penalty) instead of `HIGH_NULL_RATE` (15pt penalty)

**Result**: On a 2-company holdings array with 39 schema-conditional fields, this eliminates 39 false-positive anomalies and raises the quality score from ~50 to 95.66.

---

## 5. Quality Scoring Methodology

### 5.1 Per-Attribute Score (0-100)

```
base = 100
base -= missing_percentage_penalty  (scaled by severity)
base -= anomaly_flags * 15          (per flag)
base -= outlier_penalty             (if applicable)
score = max(0, base)
```

### 5.2 Per-Table Score (0-100)

```
base = avg(attribute_scores)
anomaly_rate = anomalous_attrs / total_attrs
base -= min(anomaly_rate * 20, 20)              # rate-based, capped
base -= min(duplicate_rate * 20, 20)            # if duplicates exist
score = max(0, min(100, base))
```

**Design rationale**: The rate-based penalty (capped at 20) replaced the original `total_anomalies * 3` formula which drove wide tables (62+ columns) to score 0. Per-attribute scores already penalise individual flags; the table penalty only adds a proportional context signal.

### 5.3 Overall Dataset Score

```
overall = avg(table_scores, weighted by record_count)
```

---

## 6. LLM Integration Architecture

### 6.1 Bring Your Own Key (BYOK) Design

ParseIQ implements a **zero-telemetry, zero-proxy BYOK architecture**:
- No ParseIQ server is involved — API calls go directly from the user's machine to their chosen LLM provider
- The user's API key is used for authentication; ParseIQ never stores, proxies, or logs keys
- 7 providers supported: OpenRouter, OpenAI, Anthropic, Google Gemini, Perplexity, Azure OpenAI, Ollama (local)

### 6.2 Provider Routing

```python
_PROVIDER_BASE_URLS = {
    'openrouter': 'https://openrouter.ai/api/v1',
    'openai':     'https://api.openai.com/v1',
    'perplexity': 'https://api.perplexity.ai',
}
```

Provider detection from model name: `claude-*` -> anthropic, `gemini-*` -> gemini, etc.
Native SDK integration for Anthropic (`anthropic` SDK) and Gemini (`google-generativeai` SDK); OpenAI-compatible REST for all others.

### 6.3 Graceful Degradation

```
LLM call attempt
    |
    +-- Success -> Merge LLM insights into metadata
    |
    +-- Failure (network, rate limit, auth, 402)
         |
         +-- Generate local fallback enrichment
         +-- 402 specifically -> Print free model alternatives
         +-- Continue with full Step 1 report
```

The user always receives a complete structured report regardless of LLM availability.

### 6.4 Credit Exhaustion Detection

402 HTTP errors are intercepted and the user is shown free model alternatives:
```
Credits exhausted on your current plan.
  Free alternatives:
    nvidia/nemotron-3-super-120b-a12b:free  (via openrouter.ai)
    mistralai/mistral-small-3.1-24b-instruct:free
    meta-llama/llama-3.3-70b-instruct:free
```

---

## 7. Nested JSON Flattening Algorithm

### 7.1 Problem

Real-world JSON data contains arbitrarily deep nesting: `org -> departments -> team -> head -> performance -> fy2025 -> margins`. Existing tools either reject nested JSON or require pre-processing.

### 7.2 Algorithm

```
_flatten_nested_json(data, path, tables):
    for each key-value pair:
        if value is list of dicts:
            -> Extract as child table
            -> Inject _ref_{parent}_id FK column
            -> Recurse into each child record
        elif value is dict:
            -> Recurse into dict for child table extraction
            -> _deep_flatten_scalars(value, key) -> add scalar leaves to parent record
        elif value is primitive list:
            -> Join as comma-separated string
        else:
            -> Add as column to parent record

_deep_flatten_scalars(obj, prefix):
    Walk dict tree collecting every scalar leaf
    Key path: prefix__key1__key2__...
    Return flat dict of {path: value}
```

### 7.3 Performance

Tested on 37 MB stress test dataset (`stress_test_data.json`):
- 14-level deep nesting
- 53,981 records across 14 tables
- Flattening time: ~1.1 seconds
- All scalar leaves preserved with `__`-joined paths

---

## 8. Incremental Processing

ParseIQ implements hash-based incremental processing:

1. **Hash computation**: SHA-256 hash of each table's record content
2. **State file**: `.parseiq_cache.json` stores `{table_name: hash}` from last run
3. **Skip logic**: On subsequent runs, unchanged tables (same hash) reuse previous results
4. **Force override**: `--force` flag or `run(force=True)` ignores cache

This is particularly valuable for large datasets where only a subset of tables changes between runs.

---

## 9. Output Generation

### 9.1 Excel Workbook Structure

Sheets are grouped per table (not per type) for easier navigation:
```
00_Summary              <- One row per table: records, quality, anomalies
01_LLM_Assessment       <- Grade, production readiness, concerns (if LLM used)
02_LLM_Recommendations  <- Prioritised recommendations (if LLM used)
Data_{table}            <- Raw data rows
Meta_{table}            <- 30-column attribute profile
Quality_{table}         <- Long-format quality breakdown
99_Issues               <- All issues sorted CRITICAL -> HIGH -> MEDIUM -> LOW
```

### 9.2 Issues Prioritisation

Each issue includes:
- **Priority**: CRITICAL / HIGH / MEDIUM / LOW
- **Table** and **Column**: affected location
- **Issue Type**: anomaly flag name
- **Description**: human-readable explanation (auto-generated by `_describe_issue()`)
- **Business Impact**: potential downstream effect
- **Recommended Fix**: actionable guidance
- **Effort**: LOW / MEDIUM / HIGH

---

## 10. Web UI Platform

### 10.1 Architecture

```
Browser (React SPA)  <-->  FastAPI Backend  <-->  ParseIQ Pipeline
                                |
                           Thread Pool (max 4 concurrent jobs)
                                |
                           Job Store (in-memory dict with locks)
```

### 10.2 Technology Stack

| Layer | Technology | Version |
|---|---|---|
| Frontend Framework | React | 19 |
| Build Tool | Vite | 8 |
| CSS Framework | TailwindCSS | v4 |
| Backend Framework | FastAPI | latest |
| ASGI Server | Uvicorn | latest |
| HTTP Client | Axios | latest |
| Charts | Recharts | 3.x |

### 10.3 Real-Time Processing

The backend captures the ParseIQ pipeline's stdout output in real-time:
1. Worker thread replaces `sys.stdout` with a `StringIO` buffer (protected by `_stdout_lock`)
2. Events are parsed from stdout lines and stored as `JobEvent` objects
3. Frontend polls `/api/job/{id}` for status + events (with `since` parameter for incremental updates)
4. Events displayed in a timestamped, auto-scrolling feed

### 10.4 Security Measures

| Measure | Implementation |
|---|---|
| XSS Prevention | Regex validation of analytics IDs, no innerHTML interpolation |
| Path Traversal | `os.path.realpath` + `startswith` on file download |
| API Key Redaction | Regex replacement of key patterns in event stream |
| CORS | Configurable via `PARSEIQ_CORS_ORIGINS` env var |
| Upload Safety | 1MB chunked streaming with early size-check abort |
| Error Consistency | Global exception handlers for `{success, data, error}` shape |

---

## 11. Comparison with Existing Tools

### 11.1 Feature Comparison

| Capability | ParseIQ | ydata-profiling | Great Expectations | pandas-profiling |
|---|:---:|:---:|:---:|:---:|
| Zero config | Yes | Yes | No (YAML) | Yes |
| Nested JSON support | Arbitrary depth | No | No | No |
| Multi-format (JSON/CSV/XML/Excel) | Yes | CSV/DataFrame | Any via connectors | CSV/DataFrame |
| Automated anomaly detection | 11 types | Basic warnings | User-defined only | Warnings |
| Schema polymorphism detection | Yes | No | No | No |
| LLM integration | Multi-provider BYOK | No | No | No |
| Quality scoring (0-100) | Per-attribute + per-table | Yes (correlations) | Pass/Fail | Limited |
| Cross-table analysis | Range + constraint | No | Yes (via suites) | No |
| Incremental processing | Hash-based cache | No | Checkpoint-based | No |
| Web UI | Yes (React + FastAPI) | HTML report | Data Docs (static) | HTML report |
| BYOK (data privacy) | Yes | N/A | N/A | N/A |
| Local/offline mode | Yes | Yes | Yes | Yes |
| Excel report output | Yes (30-col Meta) | No | No | No |

### 11.2 Use Case Positioning

- **ParseIQ**: Data onboarding/discovery — "what's in this file, is it trustworthy, what to fix"
- **ydata-profiling**: Exploratory data analysis — descriptive statistics and correlations
- **Great Expectations**: Production pipeline validation — "does this batch meet our expectations"
- **pandas-profiling**: Quick EDA — summary statistics and basic alerts

### 11.3 Performance Characteristics

| Metric | ParseIQ | ydata-profiling | Great Expectations |
|---|---|---|---|
| Setup time | 0 (pip install + run) | 0 | Hours (YAML config) |
| 1K records | ~2s (local) | ~5s | ~3s |
| 50K records | ~8s (local) | ~45s | ~15s |
| Nested JSON (14 levels) | ~1.1s flatten + ~8s profile | N/A | N/A |
| LLM enrichment overhead | 10-60s (provider dependent) | N/A | N/A |
| Memory footprint | ~2x dataset size | ~5-10x | ~1.5x |

*Note: Performance numbers are approximate and vary by hardware, dataset characteristics, and LLM provider. Formal benchmarking methodology is provided in `performance_analysis.ipynb`.*

---

## 12. Testing

### 12.1 Test Suite

- **590+ tests** across all components
- Test categories:
  - Unit tests: individual functions and methods
  - Integration tests: end-to-end pipeline runs
  - Regression tests: bug-specific tests for each fixed issue
  - Comprehensive tests: 109 tests across 21 test classes

### 12.2 Test Coverage Areas

| Component | Tests | Coverage |
|---|---|---|
| File Loader | ~40 | JSON, CSV, XML, Excel, encoding, delimiter detection |
| Metadata Extractor | ~80 | All 11 anomaly types, scoring, edge cases |
| LLM Enricher | ~30 | Provider routing, fallback, error handling |
| Pipeline (E2E) | ~50 | Full runs with all file formats and options |
| Quality Scoring | ~20 | Rate-based penalty, attribute penalties, edge cases |
| CLI | ~15 | Command parsing, flag handling, output |
| Output Generation | ~25 | Excel structure, CSV format, JSON metadata |
| Schema Polymorphism | ~15 | Discriminator detection, reclassification |
| Rules Engine | ~10 | max_value, min_value, cross_table_compare |

### 12.3 Multi-Domain Test Suite

10 domain-specific test cases with intentionally injected anomalies:

| Test Case | Domain | Anomalies Tested |
|---|---|---|
| TC-01 | E-Commerce | Negative values, future dates, mixed types |
| TC-02 | HR / Payroll | Null salary, negative bonus, duplicates |
| TC-03 | Hospital / EMR | Negative dosage, future admission, missing fields |
| TC-04 | University | Scale violation (marks > 100) |
| TC-05 | Supply Chain | Cross-level range violation (temperature) |
| TC-06 | Banking / KYC | Negative balance, future date, pattern inconsistency |
| TC-07 | Social Media | Negative metrics, null engagement |
| TC-08 | IoT Manufacturing | Cross-level range violation (sensor reading) |
| TC-09 | Insurance | Cross-table constraint (claim > sum_assured) |
| TC-10 | Conglomerate | Missing sibling dict key (fiscal year) |

---

## 13. Data Privacy Model

### 13.1 Privacy Guarantees

| Mode | Data Exposure | External Calls |
|---|---|---|
| Local (`--no-llm`) | Zero | None |
| LLM (BYOK) | Metadata + samples to user's chosen provider | Direct to provider API |
| Web UI | Server-local only | Same as above for LLM |

### 13.2 What Is Sent to LLM

- Table names, column names, data types
- Statistical summaries (not raw data)
- Sample values (configurable count)
- Anomaly flags and quality scores

### 13.3 What Is NOT Sent

- Full dataset
- API keys of other services
- File system paths
- User identity information

---

## 14. Deployment Options

| Environment | How |
|---|---|
| Local CLI | `pip install parseiq && parseiq analyze data.json` |
| Python API | `from parseiq import Pipeline; Pipeline("data.json").run()` |
| Web UI (local) | `python web/run.py --dev` |
| Docker | Standard Python Docker image + pip install |
| CI/CD Pipeline | `parseiq analyze data.json --no-llm --fail-under 80` (exit code 1 if quality below threshold) |
| Air-gapped | `--no-llm` or Ollama (local LLM, no internet) |
| Cloud Functions | AWS Lambda / GCP Cloud Run / Azure Functions |

---

## 15. Limitations and Future Work

### Current Limitations
- Free-tier LLM rate limits (~10 RPM on OpenRouter)
- 100 MB file size limit
- No streaming/real-time data source support
- Web UI not yet deployed as hosted service
- No PDF report export (planned for v0.1.0)
- Schema polymorphism detection requires >= 2 entity types

### Future Directions
- PDF report generation
- Batch processing (folder of files)
- Cross-table FK orphan detection
- Parquet and Google Sheets input support
- Multi-tenancy with Celery + Redis job queue
- Hosted web UI deployment
- Real-time data stream monitoring
- Custom anomaly detector plugins

---

## 16. Technical Stack Summary

| Component | Technology |
|---|---|
| Language | Python 3.9+ |
| Data Processing | pandas, numpy, scipy |
| Statistical Analysis | Z-score, IQR, frequency analysis |
| Excel Output | openpyxl |
| HTTP | requests (LLM API calls) |
| XML Parsing | xmltodict |
| Encoding Detection | chardet |
| Date Parsing | python-dateutil |
| LLM SDKs | anthropic (optional), google-generativeai (optional) |
| Web Frontend | React 19, Vite 8, TailwindCSS v4, Recharts |
| Web Backend | FastAPI, Uvicorn |
| Package Management | setuptools, pip |
| Testing | pytest |

---

## References

- [ParseIQ GitHub Repository](https://github.com/ShriniwasAhirrao/ParseIQ)
- [ParseIQ on PyPI](https://pypi.org/project/parseiq/)
- [ydata-profiling](https://github.com/ydataai/ydata-profiling)
- [Great Expectations](https://github.com/great-expectations/great_expectations)
- [pandas-profiling](https://github.com/pandas-profiling/pandas-profiling)
