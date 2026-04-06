# ParseIQ — TODO & Backlog

> Last updated: 2026-04-06
> Status legend:  ✅ Done  |  ⏳ Next  |  💡 Future

---

## ✅ Completed — v0.0.2 (2026-04-06)

- ✅ Multi-provider LLM: Anthropic/Claude, Gemini, Perplexity, OpenAI, OpenRouter, Azure, Ollama
- ✅ Provider auto-routing from model name (`claude-*` → anthropic, `gemini-*` → gemini)
- ✅ Credit exhaustion detection (402) — prints free-model alternatives automatically
- ✅ Meta sheet expanded to 30 columns (string / numeric / boolean stats per attribute)
- ✅ Quality sheet rewritten to long format (Category | Metric | Value | Status | Description)
- ✅ CLI `parseiq init` — 7-provider wizard with per-provider model menus
- ✅ CLI `parseiq models` — full provider breakdown with install instructions
- ✅ CLI `parseiq config` — shows all 5 provider API keys detected
- ✅ CLI `--llm-provider` expanded: openrouter / openai / anthropic / claude / gemini / perplexity / azure / ollama
- ✅ CLI footer: prints absolute Excel path at end of every run
- ✅ UTF-8 stdout fix — no more encoding errors on Windows console
- ✅ `pyproject.toml` optional extras: `parseiq[anthropic]`, `parseiq[gemini]`
- ✅ Published to PyPI — `pip install parseiq` (v0.0.2)
- ✅ Published to TestPyPI — v0.0.1 and v0.0.2
- ✅ GitHub repo renamed to `ParseIQ`, made public under ShriniwasAhirrao
- ✅ Co-authored-by lines removed from all git commits
- ✅ README, CHANGELOG, WORKLOG, commands.md updated for deployment-ready state
- ✅ 159/159 tests passing

---

## ✅ Completed — v0.0.1 (2026-04-03 to 2026-04-05)

- ✅ 3-step pipeline: Extract → LLM Enrich → Output
- ✅ Multi-format input: JSON (any nesting depth), CSV, XML, Excel
- ✅ Recursive nested-JSON flattener with FK injection and sibling-table merging
- ✅ 8 anomaly types: HIGH_NULL_RATE, LOW_UNIQUENESS, MIXED_DATA_TYPES, FUTURE_DATE_DETECTED, NUMERIC_OUTLIERS_DETECTED, NEGATIVE_VALUES_DETECTED, PATTERN_INCONSISTENCY, DUPLICATE_ROWS_DETECTED
- ✅ Quality scoring per table (0–100) with per-attribute penalties
- ✅ Excel workbook: Data/Meta/Quality sheets grouped per table + 00_Summary + 99_Issues
- ✅ CLI: `parseiq analyze`, `validate`, `init`, `models`, `config`, `version`
- ✅ Python API: `Pipeline`, `PipelineResult`, `Config`, connectors, alerts
- ✅ BYOK LLM: `run(llm=True/False, llm_provider, llm_api_key, llm_model, llm_base_url)`
- ✅ Incremental processing: SHA-256 hash state file, `run(force=True)` override
- ✅ Alert rules engine + `on_alert` callback helpers
- ✅ Connectors: file, URL, S3, PostgreSQL, MongoDB
- ✅ 159 passing tests (109 comprehensive + 10 regression + 40 integration)

---

## ⏳ Next — v0.1.0

- [ ] **PDF report export** — export full quality report as PDF alongside Excel
- [ ] **Batch processing** — `parseiq analyze-all data/` (folder of files in one command)
- [ ] **Cross-table FK violation detection** — flag `_ref_*` values that don't exist in parent table (orphaned records)
- [ ] **`conftest.py`** — shared test fixtures to reduce repetition across test files
- [ ] **XML + Excel test coverage** — `_load_xml()` and `_load_excel()` have no dedicated tests

---

## 💡 Future — v0.2.0+

- [ ] **Web UI** — drag-and-drop file upload, quality report in browser
- [ ] **Custom YAML rule definitions** — `salary > 0`, `email matches pattern`
- [ ] **Parquet + Google Sheets** input support
- [ ] **Multi-tenancy / job queue** — Celery + Redis for concurrent users
- [ ] **Rotating log handler** — prevent `logs/` growing unboundedly

---

## Known Limitations (v0.0.2)

- Free-tier OpenRouter: ~10 RPM — one LLM call per run, not per table
- LLM response time: 2–3 min for large datasets on free tier
- Max file size: 100 MB
- Output is files only — no live dashboard
