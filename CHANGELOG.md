# Changelog

All notable changes to ParseIQ are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [0.0.2] — 2026-04-06

### Added
- **Multi-provider LLM support** — any provider, bring your own key:
  - `anthropic` — Claude models via Anthropic SDK (`pip install anthropic`)
  - `gemini` — Google Gemini via `google-generativeai` SDK (`pip install google-generativeai`)
  - `perplexity` — Perplexity AI via OpenAI-compatible REST
  - `openai`, `openrouter`, `azure`, `ollama` — existing providers retained
  - Auto base-URL routing per provider (no manual `--llm-base-url` needed for standard providers)
  - Provider detection from model name (`claude-*` → anthropic, `gemini-*` → gemini)
- **Credit exhaustion detection** — 402 API errors now print a free-model fallback list and re-run command instead of silently failing
- **Rich Meta sheet** (30 columns) — replaces the old 7-column version:
  - Core: Table_Name, Attribute_Name, Data_Type, Total_Records, Present_Count, Missing_Count, Missing_Percentage, Unique_Values, Unique_Ratio, Quality_Score
  - String-specific: Min_Length, Max_Length, Avg_Length, Median_Length, Most_Common_Values, Character_Distribution
  - Anomaly: Anomaly_Count, Anomaly_Types, Has_Outliers, Recognized_Patterns
  - Numeric-specific: Min_Value, Max_Value, Mean_Value, Median_Value, Std_Deviation, Outliers_Count
  - Boolean-specific: True_Count, False_Count, True_Percentage, False_Percentage
- **Long-format Quality sheet** — replaces old 6-column wide format:
  - Columns: Table_Name, Quality_Category, Metric_Name, Metric_Value, Status, Description
  - Categories: Overall, Structure, Volume, Attribute Quality, Uniqueness, Completeness, Anomalies, Outliers
  - One metric row per attribute — human-readable, filterable in Excel
- **Expanded `parseiq init`** — provider selection wizard covering all 6 providers with model menus per provider
- **Expanded `parseiq models`** — full breakdown by provider with install instructions, API key env vars, and free tier links
- **Expanded `parseiq config`** — shows all 5 provider API keys detected (OpenRouter, OpenAI, Anthropic, Gemini, Perplexity)
- **Expanded `--llm-provider` choices** — now accepts: `openrouter openai anthropic claude gemini perplexity azure ollama`
- **Provider-aware key lookup** — `_get_api_key_from_env(provider)` picks the right env var per provider
- **Ollama auto-key** — Ollama sets a placeholder key automatically so no `--llm-api-key` is needed
- **Detailed report footer** — CLI now prints absolute Excel path at end of every run: `For a more detailed report, refer to: ...`
- **Optional dependencies in pyproject.toml** — `parseiq[anthropic]`, `parseiq[gemini]` extras
- **UTF-8 stdout reconfiguration** — CLI reconfigures `sys.stdout` to UTF-8 at startup to prevent encoding errors on Windows consoles

### Changed
- `_print_banner()` now reads `__version__` dynamically (no hardcoded version string)
- Interactive model list in `parseiq analyze` adapts to selected provider
- `--llm-provider` default suggestions in no-key prompt updated to include all providers
- `parseiq analyze` LLM mode display includes provider name

### Fixed
- `parseiq models` crashing with `UnicodeEncodeError` on Windows cp1252 console (replaced `──` with `--`)

---

## [0.0.1] — 2026-04-03  *(Initial release)*

### Added
- 3-step pipeline: Extract → LLM Enrich → Output
- Multi-format input: JSON (any nesting depth), CSV, XML, Excel
- Recursive nested-JSON flattener with FK injection and sibling-table merging
- 8 anomaly types: HIGH_NULL_RATE, LOW_UNIQUENESS, MIXED_DATA_TYPES, FUTURE_DATE_DETECTED, NUMERIC_OUTLIERS_DETECTED, NEGATIVE_VALUES_DETECTED, PATTERN_INCONSISTENCY, DUPLICATE_ROWS_DETECTED
- Quality scoring per table (0–100) with per-attribute penalties
- Step 2 LLM enrichment via OpenRouter (BYOK)
- Rate limiting: token-bucket (10 RPM), 429 Retry-After respect, adaptive backoff
- Output: Excel workbook (Data/Meta/Quality sheets per table) + 2 CSV + 3 JSON files
- Excel sheet ordering: grouped per table (Data→Meta→Quality), not type-first
- `99_Issues_Recommendations` sheet: Priority, Table, Column, Issue_Type, Description, Business_Impact, Recommended_Fix, Effort — sorted CRITICAL→HIGH→MEDIUM→LOW
- `00_Summary` + `01_LLM_Assessment` + `02_LLM_Recommendations` tabs
- CLI: `parseiq analyze`, `validate`, `init`, `models`, `config`, `version`
- CLI flags: `--no-llm`, `--output`, `--force`, `--quiet`, `--fail-under`, `--llm-api-key`, `--llm-model`, `--llm-provider`, `--llm-base-url`
- Python API: `Pipeline`, `PipelineResult`, `Config`, `alerts`
- Class-method constructors: `from_file()`, `from_url()`, `from_s3()`, `from_postgres()`, `from_mongodb()`
- Alert rules engine: 7 rule types + `on_alert` callback
- Incremental processing: SHA-256 hash state file, `run(force=True)` override
- Connectors: file, URL, S3, PostgreSQL, MongoDB
- `MetadataEnrichmentAgent` shim for backward compatibility
- 159 passing tests (109 comprehensive + 10 regression + 40 integration)

### Fixed (patch releases during 0.0.1 development)
- Data sheets coercing all cell values to strings — fixed with `astype(object).where()`
- FALSE `LOW_UNIQUENESS` anomaly on boolean columns — added `data_type != "boolean"` guard
- CSV NaN values not converting to `None` — added `astype(object)` before `.where(df.notna())`
- `main_table` showing instead of filename stem — renamed in loader after flatten
- zscore `RuntimeWarning` on constant columns — suppressed + `nan_to_num` fallback
- `enrich_metadata()` rejecting BYOK kwargs — extended method signature
- `LLMEnricher.__init__` raising immediately on missing key — deferred to call time
- Total records showing table count instead of row count — fixed to use `pipeline_info`
- `_ref_*` FK columns being profiled as regular attributes — excluded in extractor
- Artifact null columns from flattener — pre-scan dominant type per field, skip null dict/list rows

---

## [0.1.0] — *upcoming*

### Planned
- PDF report export
- Batch processing (folder of files in one command)
- Cross-table FK violation detection (orphaned records)
- Custom YAML rule definitions
- Parquet + Google Sheets input support
