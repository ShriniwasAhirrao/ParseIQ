# ParseIQ — TODO & Backlog

> Last updated: 2026-04-10
> Status legend:  ✅ Done  |  ⏳ Next  |  💡 Future

---

## ✅ Completed — v0.0.6 (2026-04-10)

- ✅ **Overall Score N/A in Excel** — `_generate_outputs()` now reads `overall_score` then falls back to `corrected_score`; never shows N/A when a real score exists
- ✅ **Model Used N/A in Excel** — `_create_fallback_enrichment()` now includes `model_used` in `enrichment_metadata`
- ✅ **corrected_score always 0** — `_calculate_corrected_quality_score()` extended with two additional lookup paths matching the real pipeline metadata structure (`dataset_overview.table_summaries[*].quality_score` and `tables[*].table_metadata.data_quality_score`)
- ✅ **Fallback missing fields** — `overall_score`, `key_strengths`, `primary_concerns` now always present in fallback `overall_assessment`
- ✅ 6 regression tests added — 165/165 tests passing

---

## ✅ Completed — v0.0.5 (2026-04-09)

- ✅ **Issue K** — Schema polymorphism: `_detect_schema_groups()` finds a discriminator column (low-cardinality categorical, ≥70% present, not an ID/key) and classifies type-conditional columns (absent for some entity types) as `TYPE_CONDITIONAL_FIELD` instead of `HIGH_NULL_RATE`
- ✅ **TYPE_CONDITIONAL_FIELD** anomaly type added: 2pt quality penalty (vs 15pt), null-rate deduction suppressed, excluded from `top_issues` surfacing; `schema_groups` metadata block written to table output
- ✅ **NEGATIVE_ALLOWED_PATTERNS** extended: `_return` and `_yield` tokens added — `predicted_return_1m_pct`, `dividend_yield_pct`, etc. no longer produce spurious `NEGATIVE_VALUES_DETECTED` flags
- ✅ **`_describe_issue()`** updated in `pipeline.py` with actionable guidance for `TYPE_CONDITIONAL_FIELD`
- ✅ **Result on apex_capital dataset**: quality score 95.66 (was ~50s); ~58 false-positive `HIGH_NULL_RATE` flags eliminated
- ✅ 159/159 tests passing

---

## ✅ Completed — v0.0.4 (2026-04-09)

- ✅ **Issue E** — Relax version pins in `pyproject.toml` so ParseIQ installs cleanly alongside older pandas/numpy
- ✅ **Issue F** — NEGATIVE_VALUES_DETECTED suppressed for domain-conventional columns (`drawdown`, `var_*`, `shock`, `cfi_*`, `cff_*`, `capex`, `pnl`, etc.)
- ✅ **Issue G** — Cross-level range violations now detected: `*_range*` parent columns are parsed as [lo, hi]; matching measurement columns in other tables are checked and `RANGE_VIOLATION_DETECTED` is raised (TC-05, TC-08)
- ✅ **Issue H** — `cross_table_compare` rule type in `parseiq_rules.yaml` sidecar enables cross-table constraint detection (e.g. `claimed_amount <= sum_assured`) (TC-09)
- ✅ **Issue I** — `max_value` / `min_value` rule types in `parseiq_rules.yaml` enable scale/domain violation detection (e.g. marks total ≤ 100) (TC-04)
- ✅ **Issue J** — Missing sibling dict key already detected automatically: when `fy2024` is absent for one subsidiary, deep-flatten leaves all `financials__fy2024__*` columns as null → HIGH_NULL_RATE fires at 50% (> 30% threshold). No code change needed; verified in TC-10.
- ✅ Example rule files added: `test_cases/tc04_university_rules.yaml`, `test_cases/tc09_insurance_rules.yaml`
- ✅ 159/159 tests passing

---

## ✅ Completed — v0.0.3 (2026-04-09)

- ✅ Deep JSON flattening — nested dicts recurse to arbitrary depth (`_deep_flatten_scalars` helper)
- ✅ No more blob columns in Excel Data sheets (financials, ml_signals, risk_framework, head.performance)
- ✅ `stress_scenarios` and `top_signals` now extracted as proper tables (were missing)
- ✅ Quality score = 0 bug fixed — rate-based table penalty capped at 20 pts
- ✅ Duplicate table processing guard (`visited_tables` set in output loop)
- ✅ Excel blob truncation — `_truncate_blobs()` on all Data_ sheets (120 char limit)
- ✅ Context bleed between tables fixed (side-effect of deep-flatten fix)
- ✅ `'None'` string in 00_Summary Top_Issues replaced with empty cell
- ✅ Privacy fix — prompt template path logs filename only (not full absolute path)
- ✅ Community bug docs — TODO.md Known Bugs + CONTRIBUTING.md Good First Issues sections
- ✅ Test updated for new quality score formula — 159/159 passing

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

## ⏳ Next — v0.0.7

- [ ] **Bug: CSV delimiter detection fails on Unicode-heavy CSVs** — `csv.Sniffer().sniff()` raises `Could not determine delimiter` when the sample contains mostly Unicode characters. Affects `_load_csv()` in `parseiq/file_loader/loader.py`. Fix: fall back to `','` when sniffer fails. (2 tests failing: `test_02_unicode_heavy`, `test_rules_sidecar_yaml_applied`)
- [ ] **XML + Excel test coverage** — `_load_xml()` and `_load_excel()` have no dedicated tests
- [ ] **`conftest.py`** — shared test fixtures to reduce repetition across test files

---

## ⏳ Future — v0.1.0

- [ ] **PDF report export** — export full quality report as PDF alongside Excel
- [ ] **Batch processing** — `parseiq analyze-all data/` (folder of files in one command)
- [ ] **Cross-table FK violation detection** — flag `_ref_*` values that don't exist in parent table (orphaned records)

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

---

## Known Bugs — Open for Community PRs

> Each entry includes: symptoms · affected file · suggested fix · PR label.

### ✅ Bug A — Duplicate Table Analysis Loop — FIXED in v0.0.3
`visited_tables: set` guard added to `parseiq/pipeline.py` output builder loops.

---

### ✅ Bug B — Attribute Context Bleeding Between Tables — FIXED in v0.0.3
Fixed as a side-effect of the deep-flatten loader rewrite. Root cause was blob-stringification
mixing sibling-context attributes into the same table record.

---

### ✅ Bug C — Inconsistent Flattening Depth — FIXED in v0.0.3
`_deep_flatten_scalars()` helper added to `parseiq/file_loader/loader.py`.
All nested dicts now recurse to arbitrary depth with `__`-joined column names.
Previously missing tables (`stress_scenarios`, `top_signals`) now extracted correctly.

---

### ✅ Bug D — Excel Blob Columns — FIXED in v0.0.3
`_truncate_blobs()` applied to all `Data_*` sheets in `parseiq/pipeline.py`.
Strings > 120 chars starting with `{` or `[` are truncated with `…`.

---

### ✅ Issue E — pip install Environment Collision — FIXED in v0.0.4
**Symptom:** Running `pip install parseiq` into an existing project virtual environment can
downgrade or conflict with user's pinned dependencies (pandas, openpyxl, etc.).

**Recommended workaround for users (available now):**
```bash
# Option 1 — pipx (cleanest, zero pollution)
pipx install parseiq
parseiq analyze data.json

# Option 2 — dedicated venv
python -m venv .parseiq-env
source .parseiq-env/bin/activate   # Windows: .parseiq-env\Scripts\activate
pip install parseiq
parseiq analyze data.json
```

**Long-term fix:** Relax version pins in `pyproject.toml` to wide ranges
(`pandas>=1.5`, `openpyxl>=3.0`) and add a `[tool.parseiq]` isolation note to the docs.

**File:** `pyproject.toml`, `README.md`
**Label:** `enhancement` `documentation`

---

### ✅ Issue F — NEGATIVE_VALUES_DETECTED False Positives — FIXED in v0.0.4
**Symptom:** Columns like `var_1d_99_pct`, `max_drawdown_pct`, `equity_shock`, `cfi_cr` are
legitimately negative by financial convention (VaR, drawdown, capex outflows) but are flagged
as `NEGATIVE_VALUES_DETECTED` anomalies, lowering quality scores and producing noisy issues.

**Suggested fix:** Allow users to pass a domain hint or an allowlist of columns/patterns that
are expected to be negative (e.g. `--allow-negatives "var_*,drawdown*,cf*_cr"`), or add a
built-in heuristic that suppresses the flag on columns whose name contains `var`, `drawdown`,
`shock`, `cfi`, `cff`.

**File:** `parseiq/step1_metadata_extractor/extractor.py` — anomaly detection section
**Label:** `enhancement` `good first issue`

---

### ✅ Issue G — Cross-Level Range Violations — FIXED in v0.0.4
**Symptom:** When a valid-range spec lives at one nesting level (e.g. `temp_range_c: [2, 8]`
inside a zone object) and the breaching value lives 4 levels deeper inside `tracking_events`,
ParseIQ flattens them into separate tables with no link. The breach is never flagged in
`--no-llm` mode.  Same problem in IoT/manufacturing JSON: `normal_range: [0, 10]` lives in the
`sensors` table while the breaching reading `18.9` lives in the child `readings` table.

**Affected test cases:** TC-05 (supply chain cold-chain), TC-08 (IoT manufacturing)

**Suggested fix:** After FK-injected parent-child table pairs are built, walk parent columns
whose name ends in `_range` or `_limit` and compare them against the child table's numeric
columns that share the same FK prefix.  Flag `RANGE_VIOLATION_DETECTED` anomaly when any
child value falls outside the parent's `[min, max]` pair.

**File:** `parseiq/step1_metadata_extractor/extractor.py` — post-extraction cross-table check
**Label:** `enhancement` `good first issue`

---

### ✅ Issue H — Cross-Table Constraint Violations — FIXED in v0.0.4 (via rules sidecar)
**Symptom:** Constraints that span two separately extracted tables (e.g. `claimed_amount` in
`claims[]` vs `sum_assured` in `policies[]`) are undetectable in `--no-llm` mode because
ParseIQ produces flat, unjoined tables.  The FK key is injected but no join + comparison is
ever performed.

**Affected test cases:** TC-09 (insurance) — `claimed_amount: 450000 > sum_assured: 300000`

**Suggested fix:** Allow users to define cross-table constraint rules in a YAML sidecar file:
```yaml
constraints:
  - left_table: claims
    left_col: claimed_amount
    right_table: policies
    right_col: sum_assured
    join_key: policy_id
    rule: left <= right
    anomaly: CONSTRAINT_VIOLATION
```
ParseIQ reads the sidecar and applies the join + comparison after extraction.

**File:** `parseiq/step1_metadata_extractor/extractor.py`, new `parseiq/rule_engine.py`
**Label:** `enhancement`

---

### ✅ Issue I — Scale/Domain Violations — FIXED in v0.0.4 (via rules sidecar)
**Symptom:** When a numeric value is arithmetically coherent but semantically out-of-scale
(e.g. `total: 128` in a 100-point marks system), ParseIQ does not flag it.  It sees an integer
and has no concept of the domain upper bound.

**Affected test cases:** TC-04 (university) — `mid_term(38) + end_term(72) + assignment(18) = 128`

**Suggested fix:** Allow a `max_value` annotation on the parent object (e.g. `max_marks: 100`)
or a YAML rule (`marks.total <= 100`).  Without this, detection requires LLM mode.

**File:** `parseiq/step1_metadata_extractor/extractor.py`
**Label:** `enhancement`

---

### ✅ Issue J — Missing Sibling Dict Key — Already Handled (HIGH_NULL_RATE, verified v0.0.4)
**Symptom:** When a parent object has two sibling dict keys (e.g. `fy2025: {...}` and
`fy2024: {...}`) and one record is missing `fy2024` entirely, ParseIQ does not flag it.
The key absence is a structural gap (missing fiscal year), but because the parent is a dict
(not an array-of-dicts), ParseIQ never compares key presence across records.

**Affected test cases:** TC-10 (conglomerate) — `financials.fy2024` missing for one subsidiary

**Suggested fix:** After deep-flattening, collect all `__`-prefixed column names produced per
record.  For records in the same table that are missing columns present in other records,
raise `HIGH_NULL_RATE` only if the column represents a known time-series key (heuristic:
column name ends with 4 digits that look like a year).

**File:** `parseiq/file_loader/loader.py`, `parseiq/step1_metadata_extractor/extractor.py`
**Label:** `enhancement`

---

### ✅ Issue K — Schema Polymorphism False Positives — FIXED in v0.0.5
**Symptom:** When a JSON array contains records of fundamentally different types
(e.g. Energy stocks and Banking stocks in `holdings[]`, or Equity funds and Quant funds in
`portfolios[]`), ParseIQ treats them as one uniform schema. Fields that only apply to one
entity type are counted as "null" for all records, producing mass `HIGH_NULL_RATE` false
positives (39 for holdings, 19 for portfolios on the apex_capital dataset — ~58 of 63 total
anomalies were false positives).

**Root cause:** No entity-type clustering before per-attribute statistics were computed.

**Fix:** New `_detect_schema_groups(records)` method in `extractor.py`:
- Finds a *discriminator column*: low-cardinality (2–10 unique values), ≥70% present,
  categorical (non-numeric), not an ID/key column (skips `_id` suffix, `isin`, `name`, etc.)
- Scores how well the discriminator explains variable-presence columns (purity ≥ 80% per group)
- Marks columns present in some groups but absent in others as `type_conditional_cols`
- Type-conditional columns get `TYPE_CONDITIONAL_FIELD` instead of `HIGH_NULL_RATE`:
  2pt quality penalty, no null-rate deduction, suppressed from top_issues
- `schema_groups` metadata block written to table output (discriminator, group count, score,
  type_conditional_columns list)

**Also fixed:** `_NEGATIVE_ALLOWED_PATTERNS` extended with `_return` and `_yield` tokens.

**File:** `parseiq/step1_metadata_extractor/extractor.py`, `parseiq/pipeline.py`
**Label:** `enhancement`
