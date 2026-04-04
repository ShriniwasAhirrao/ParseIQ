# ParseIQ V.0.0.1 — TODO & Backlog

> Last updated: 2026-04-04
> Status legend:  ✅ Done  |  🔄 In Progress  |  ⏳ Next  |  💡 Future

---

## ✅ Completed (This Session — 2026-04-03)

- ✅ Fix 400 Bad Request from OpenRouter API (wrong content format — list vs string)
- ✅ Fix model ID errors (removed hardcoded invalid model IDs, use `config.MODEL_NAME`)
- ✅ Fix 429 lockout too aggressive (threshold was `attempt >= 2`, now `retry_after >= 300`)
- ✅ Fix duplicate log lines (5× output) — root logger handler deduplication
- ✅ Fix debug_output files accumulating every run
- ✅ Fix Excel sheet names over 31 chars — `_safe_sheet()` helper with per-prefix budget
- ✅ Shorten `prompt_template.txt` from ~17k chars to ~4k (removed verbose JSON example)
- ✅ Fix quality score always 100/100 — removed misleading +10 uniqueness reward, added per-anomaly deduction
- ✅ Fix output showing only 1 table — `convert_enriched_json_to_csv` now handles any root key
- ✅ Fix LLM receiving 0 records — added translation layer from extractor format to LLM compressor format
- ✅ Fix table names all showing as `main_table` — patch actual name after extraction
- ✅ Fix LLM Assessment showing N/A — read correct fields (`overall_score`, `quality_grade`)
- ✅ Project cleanup — cleared 80+ debug files, 27 old logs, stale CSVs, moved tests to `tests/`
- ✅ Created complex nested test dataset (13 tables, 4-level nesting, 22+ quality issues)
- ✅ Rewrote `_flatten_nested_json` — FK injection, embedded object flattening, merge same-name tables
- ✅ Fix FK column picking up parent ID instead of record's own ID (`_ref_*` guard)
- ✅ Fix CSV NaN values not counted as null (`df.where(df.notna(), other=None)`)
- ✅ Fix artifact columns inflating null rates (pre-scan field types, skip null/empty for dict/list fields)
- ✅ Fix `Config.validate_config()` missing — added method
- ✅ Fix test mocks using outdated `load_file` return type (list → dict)
- ✅ Fix `test_load_file_json` wrong assertion after flatten step added
- ✅ Fix `summary['total_tables']` always None
- ✅ **Test suite: 40/40 passing**

## ✅ Completed (Session — 2026-04-04)

- ✅ Flag duplicate rows as anomaly — `DUPLICATE_ROWS_DETECTED` in `anomaly_summary`, quality deduction applied
- ✅ Flag future dates as anomaly — `FUTURE_DATE_DETECTED` in `anomaly_flags` for date-like string columns
- ✅ Flag mixed data types — `MIXED_DATA_TYPES` in `anomaly_flags` when column has incompatible Python types
- ✅ Exclude `_ref_*` FK columns from quality analysis — skipped in `_analyze_table_detailed()`
- ✅ **Test suite: 40/40 passing** (all new logic smoke-tested)
- ✅ Output file reduction — per-table CSVs skipped by default; only Excel + 2 summary CSVs + JSON (6 files vs 45)
- ✅ Auto-clean stale output files at `run_pipeline()` start
- ✅ 14-level 53K-record stress test dataset (`input/stress_test_data.json`, 37 MB)
- ✅ Generator script (`scripts/generate_stress_test.py`) for reproducible stress data
- ✅ Confirmed flattener handles 14 levels in 1.1 s, extracts all 14 tables correctly

---

## ⏳ Priority Next Tasks

---

### MEDIUM — Output & Pipeline

- [ ] **Verify LLM full pipeline end-to-end after today's fixes**
  - The LLM was receiving 0 records before today's fix. Run a complete pipeline with
    LLM enabled and verify `llm_insights.json` now shows real record counts and
    catches actual quality issues from the new dataset.

- [ ] **Fix "STEP 2 banner prints many times" in terminal**
  - Appears to be a Windows PowerShell / VS Code terminal buffer artifact during the
    long LLM wait, not a real code bug. Investigate if adding `sys.stdout.flush()` or
    `print(..., flush=True)` after the Step 2 print resolves it.

---

### LOW — Code Quality & Robustness

- [ ] **Add `conftest.py` with shared test fixtures**
  - Many tests repeat the same mock setup. Centralise into `tests/conftest.py`:
    - `mock_agent` fixture
    - `sample_flat_tables` fixture
    - `sample_nested_tables` fixture

- [ ] **Rotate `app.log` in `logs/`**
  - `logs/app.log` grows unboundedly. Add `RotatingFileHandler` (max 5 MB, 3 backups).

- [ ] **Add XML test case**
  - `_load_xml()` exists but no test covers it. Add a test with a simple XML file.

- [ ] **Add Excel test case**
  - `_load_excel()` exists but no test covers it. Add a test with a minimal `.xlsx`.

- [ ] **Handle deeply nested embedded objects gracefully**
  - Currently objects with nested dicts/lists are JSON-stringified as a fallback.
  - Consider a configurable max-depth for inline flattening (e.g., 2 levels deep).

---

## 🚀 Deployment Strategy (decided 2026-04-04)

### Phase 1 — Python Library + CLI (`pip install parseiq`)
- [ ] Refactor `main.py` into a clean `Pipeline` class with a public API
      `from parseiq import Pipeline; report = Pipeline("data.json").run()`
- [ ] Add `parseiq` CLI entry point: `parseiq analyze data.json --output report/`
- [ ] Make LLM optional — skip Step 2 if no API key (pure local mode)
- [ ] Publish on PyPI

### Phase 2 — Data Source Connectors
- [ ] `Pipeline.from_s3("s3://bucket/file.json")`
- [ ] `Pipeline.from_postgres(conn_string, query)`
- [ ] `Pipeline.from_mongodb(conn_string, collection)`
- [ ] REST API response direct input

### Phase 3 — Hosted Web Option
- [ ] Simple web UI — upload file, download report (for non-technical users)
- [ ] Host on Railway / Render / AWS

**Why library-first:** Data never leaves their machine (GDPR/HIPAA safe).
Works in any cloud environment. No infra cost for you. Integrates into their ETL scripts.

---

## 🏗️ Enterprise-Ready Pre-Deployment Checklist

> Read this section fully before starting deployment work.
> Complete items in order — code changes first, packaging last.

---

### 1. Package Structure & PyPI Publishing
- [ ] Restructure project into proper Python package layout (`src/parseiq/`)
- [ ] Add `pyproject.toml` / `setup.cfg` with metadata (name, version, author, dependencies)
- [ ] Add `parseiq` CLI entry point via `console_scripts` in `pyproject.toml`
- [ ] Test `pip install -e .` locally in a clean virtualenv before publishing
- [ ] Publish to PyPI using `twine upload dist/*`

---

### 2. Clean Public API — `Pipeline` Class
- [ ] Refactor `main.py` into a `Pipeline` class with clean public methods
- [ ] `Pipeline("data.json").run()` as the single main entry point
- [ ] Add input connectors as class methods:
  - `Pipeline.from_file("data.json")`
  - `Pipeline.from_s3("s3://bucket/file.json")`  — uses `boto3`
  - `Pipeline.from_postgres(conn_string, "SELECT * FROM orders")`  — uses `psycopg2`
  - `Pipeline.from_mongodb(conn_string, "collection_name")`  — uses `pymongo`
  - `Pipeline.from_url("https://api.example.com/data", headers={...})`  — uses `requests`
  - All connectors produce a Python dict/list → existing flattener handles the rest unchanged
- [ ] Return a structured result object from `.run()`, not just write files silently
  - `result.tables` — list of table names extracted
  - `result.quality_scores` — per-table scores
  - `result.anomalies` — all anomaly flags
  - `result.output_files` — paths to generated files
  - `result.llm_insights` — LLM output if enabled, else `None`

---

### 3. LLM Architecture — Bring Your Own Key (BYOK)
- [ ] Make LLM fully opt-in: `run(llm=True)` / `run(llm=False)`
  - `llm=False` → pure local mode, instant, no API needed, data never leaves machine
  - `llm=True` → user must supply their own key
- [ ] Accept LLM config as `run()` params:
  - `llm_provider` — `"openrouter"` | `"openai"` | `"azure"` | `"ollama"`
  - `llm_api_key` — user's own key, never hardcoded
  - `llm_model` — e.g. `"gpt-4o"`, `"llama3"`, `"mistral"`
  - `llm_base_url` — for Azure OpenAI or local Ollama endpoint
- [ ] Remove all hardcoded API keys from `config.py` — replace with env var reads only
- [ ] Add graceful degradation: if LLM call fails / times out / key is wrong,
      catch the exception, log a warning, and still return the Step 1 report
      (currently the pipeline crashes and produces nothing if Step 2 fails)
- [ ] Never log or print API keys anywhere in the codebase

---

### 4. Incremental Processing (State File)
- [ ] After each run, write `output/.parseiq_state.json`:
      `{"employees": {"hash": "abc123", "last_run": "2026-04-04T10:00:00"}, ...}`
- [ ] On next run: hash current data per table, compare to stored hash
- [ ] Skip re-analysis for tables whose hash is unchanged — reuse previous result
- [ ] Add `run(force=True)` param to override and reprocess all tables regardless
- [ ] This makes re-runs on large files ~10× faster when only a few tables changed

---

### 5. Actionable Alert Rules
- [ ] Accept `alert_rules` dict in `run()`:
      `{"employees.email": {"null_rate_gt": 0.05}, "orders.amount": {"negative_values": True}}`
- [ ] After Step 1 completes, run a post-processing rule-matching pass against results
- [ ] Accept `on_alert` callback: `on_alert=lambda rule, table, metric: your_function(...)`
- [ ] Ship built-in alert helpers in `parseiq.alerts`:
  - `parseiq.alerts.slack_webhook(url)` — posts alert to Slack channel
  - `parseiq.alerts.email(smtp_config)` — sends alert email
- [ ] Supported rule types: `null_rate_gt`, `negative_values`, `duplicate_rows`,
      `future_dates`, `mixed_types`, `outliers_detected`, `quality_score_lt`

---

### 6. Concurrency — Library Design Guarantee
- [ ] Document explicitly: library model = one `Pipeline` instance per user,
      no shared global state, each run is fully isolated
- [ ] Each `Pipeline.run()` uses a unique temp working dir if no output dir specified
- [ ] No global file locks, no shared `output/` directory conflicts between parallel runs
- [ ] Users running it in Airflow, Prefect, or a thread pool will get clean isolation

---

### 7. Config & Secrets Handling
- [ ] Priority order for config: direct params → env vars → `.env` file → defaults
- [ ] Support `.env` file loading via `python-dotenv` (optional dependency)
- [ ] All sensitive fields (`API_KEY`, `DB_PASSWORD`) read from env only — never from code
- [ ] `Config` class exposes a `validate()` method that lists any missing required fields
      and tells the user exactly what to set before running
- [ ] Document every config option in README with type, default, and example

---

### 8. Documentation & Examples
- [ ] `README.md` with 5-line quickstart that works out of the box
- [ ] `examples/` folder with runnable scripts:
  - `examples/from_json_file.py`
  - `examples/from_postgres.py`
  - `examples/from_s3.py`
  - `examples/with_alert_rules.py`
  - `examples/with_local_llm_ollama.py`
- [ ] Docstrings on all public methods (`Pipeline`, connectors, alert helpers)
- [ ] `CHANGELOG.md` starting from V.0.0.1

---

### 9. Testing Before Publish
- [ ] Test `pip install -e .` from scratch in a clean virtualenv — no import errors
- [ ] Test `parseiq analyze data.json --output report/` CLI command end-to-end
- [ ] Test `run(llm=False)` — pure local mode, no API key, produces full Step 1 report
- [ ] Test graceful degradation — wrong API key → Step 1 report still saved, warning printed
- [ ] Test `from_file()`, `from_url()` connectors with real data
- [ ] Run full test suite (`pytest tests/`) — all 40 tests must still pass
- [ ] Test on a clean machine / CI environment with only `pip install parseiq`

---

## 💡 Future / V.0.0.2 Ideas

- [ ] **GUI / Web interface** — upload file, view results in browser
- [ ] **Batch processing** — run pipeline on a folder of files
- [ ] **PDF report generation** — export quality report as PDF
- [ ] **Cross-table FK violation detection** — flag `_ref_*` values that don't exist
      in the referenced parent table (orphaned records)
- [ ] **Custom business rule definitions** — YAML config to define domain rules
      (e.g., `salary must be > 0`, `email must match pattern`)
- [ ] **Incremental runs** — only re-analyse tables that changed since last run
- [ ] **Support more input formats** — Parquet, Avro, Google Sheets
- [ ] **Actionable alerts** — Slack/email webhook when quality threshold breached
      (e.g., null rate > 10% on a critical column)
- [ ] **Multi-tenancy / job queue** — support concurrent users, Celery + Redis

---

## Known Limitations (Not Bugs — Acceptable for V.0.0.1)

- Free-tier OpenRouter: 20 RPM limit — single LLM call only, no per-table enrichment
- LLM response time: 2–3 minutes for large datasets on free tier
- Max file size: 100 MB
- Excel sheet name max 31 chars (handled)
- `features` primitive array in `pricing_tiers` is joined as string — loses individual
  element queryability in output CSV
