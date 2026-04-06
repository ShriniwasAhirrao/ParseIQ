# Changelog

All notable changes to ParseIQ are documented here.

---

## [0.0.1] — 2026-04-03  *(Baseline)*

### Added
- 3-step pipeline: Extract → LLM Enrich → Output
- Multi-format input: JSON (any nesting depth), CSV, XML, Excel
- Recursive nested-JSON flattener with FK injection and sibling-table merging
- Step 1 metadata extraction: 8 anomaly types, quality scoring (0–100), cross-table FK detection
- Step 2 LLM enrichment via OpenRouter (nvidia/nemotron-3-super-120b-a12b:free)
- Step 3 output: Excel workbook (Data/Meta/Quality sheets per table) + 3 JSON + 2 CSV files
- 159 passing tests (109 branch-coverage + 10 regression + 40 integration)

### Fixed (2026-04-04)
- Data sheets coercing all cell values to strings (fixed with `astype(object).where()`)
- FALSE LOW_UNIQUENESS anomaly on boolean columns (added `data_type != "boolean"` guard)
- `Affected_Tables` column showing `"Multiple"` for all rows (now parses `[table_name]` prefix)

---

## [0.1.0] — *upcoming*

### Added
- `parseiq` Python package (`pip install parseiq`)
- `Pipeline` public API with class-method constructors:
  - `Pipeline.from_file()`, `.from_url()`, `.from_s3()`, `.from_postgres()`, `.from_mongodb()`
- `PipelineResult` structured return object (`tables`, `quality_scores`, `anomalies`, `output_files`, `llm_insights`)
- **BYOK LLM**: `run(llm=True, llm_provider=..., llm_api_key=..., llm_model=..., llm_base_url=...)`
  - Supports: OpenRouter, OpenAI, Azure OpenAI, Ollama (local)
  - `llm=False` for pure local mode — no API call, data never leaves machine
  - Graceful degradation: LLM failure falls back to local analysis automatically
- **Incremental processing**: hash-based state file (`output/.parseiq_state.json`), `run(force=True)` override
- **Alert rules**: `run(alert_rules={...}, on_alert=callback)` with 7 rule types
  - Built-in helpers: `parseiq.alerts.slack_webhook()`, `parseiq.alerts.email()`
- `parseiq analyze <file>` CLI command
- `examples/` folder with 5 runnable scripts
- `Config.validate()` with actionable missing-field messages
- Concurrency-safe: each `Pipeline` instance uses its own output directory, no global state
- Backward-compatible shims: `from main import MetadataEnrichmentAgent` still works
