# ParseIQ V.0.0.1 — Work Log

---

## Session: 2026-04-04 (Bug Fixes + Comprehensive Testing)

### Overview
Three confirmed bugs fixed across `main.py` and `step1_metadata_extractor/extractor.py`.
Comprehensive bug-fix verification test suite written (`tests/test_bug_fixes.py` — 10 tests).
Full regression run confirmed 159/159 tests passing.

---

### Bug 1 — Data Sheets Coercing All Values to Strings

**File:** `main.py` — `convert_enriched_json_to_csv()` (~line 746)

**Problem:**
```python
# Old (broken):
for col in df_table.columns:
    df_table[col] = df_table[col].astype(str).replace('nan', '')
```
Every cell in every Data sheet was converted to a string before writing to Excel.
Integer `30` became `'30'`, boolean `True` became `'True'`, and `None` became `''`.
This meant numeric Excel columns had left-aligned text, no SUM/AVERAGE formulas worked,
and boolean columns couldn't be filtered as true/false.

**Fix:**
```python
# New (correct):
df_table = df_table.astype(object).where(df_table.notna(), other=None)
```
Casts the DataFrame to Python object dtype so `None` can be stored (float64 columns
cannot hold Python `None` — they silently stay `NaN` even after `.where()`).
Numbers, booleans, dates, and strings all write to Excel as their proper types.
`None` cells write as blank (empty) cells.

**Impact:** All Data sheets in the Excel workbook now contain correct cell types.
Excel formulas, sorting, and filtering work correctly on all columns.

---

### Bug 2 — LOW_UNIQUENESS False Positive on Boolean Fields

**File:** `step1_metadata_extractor/extractor.py` — `_detect_anomalies()` (~line 799)

**Problem:**
```python
# Old (buggy):
if unique_ratio < 0.1 and len(values) > 10:
    anomalies.append("LOW_UNIQUENESS")
```
Boolean columns like `is_active` and `is_hq` always have at most 2 distinct values
(`True`/`False`), giving `unique_ratio = 2/N`. For 50+ records, ratio = 0.04 — always
below the 0.1 threshold. Every boolean field was being flagged as a data quality issue,
cluttering the anomaly report with false positives.

**Fix:**
```python
# New (correct):
if unique_ratio < 0.1 and len(values) > 10 and attr_metadata.get("data_type") != "boolean":
    anomalies.append("LOW_UNIQUENESS")
```
Added a `data_type != "boolean"` guard. Low cardinality is *expected* for boolean
fields — it is not an anomaly.

**Regression guard:** Low-cardinality string columns (e.g., `status` with 3 values in
50 rows) and low-cardinality integer columns still correctly fire `LOW_UNIQUENESS`.
Only `data_type == "boolean"` is exempt.

---

### Bug 3 — `Affected_Tables` Hardcoded to `"Multiple"` in Issues CSV

**File:** `main.py` — `convert_enriched_json_to_csv()` (~line 1058)

**Problem:**
```python
# Old (hardcoded):
"Affected_Tables": "Multiple",
```
The `combined_issues_and_recommendations.csv` Issues sheet showed `Affected_Tables:
Multiple` for every row, even single-table issues. It was impossible to tell which
table each issue referred to.

**Root cause:** `top_issues` items are already formatted as `"[table_name] issue text"`
(e.g. `"[employees] HIGH_NULL_RATE in salary"`) when coming from nested JSON analysis,
but the code ignored this prefix and hardcoded `"Multiple"`.

**Fix:**
```python
# New (extracts table name from prefix):
affected_table = "Multiple"
if issue.startswith("["):
    end_bracket = issue.find("]")
    if end_bracket > 0:
        affected_table = issue[1:end_bracket]
```
Parses the `[table_name]` prefix where present and uses it as the `Affected_Tables`
value. Issues without a prefix (unbracketed global issues) still fall back to
`"Multiple"`. Edge cases (unclosed bracket, empty string) handled safely with no crash.

---

### Comprehensive Testing — 109 New Tests (test_comprehensive.py)

Written and run as part of the alpha/beta testing phase prior to the bug fixes above.
All 109 tests were written from scratch to cover branches and edge cases not reached
by the original 40 tests.

**Test classes and coverage:**

| Class | What it covers |
|---|---|
| `TestFileLoaderCSV` | Delimiter detection, encoding, NaN→None for numeric columns |
| `TestFlattenNestedJSON` | Sibling merging, FK injection, path-collision resolution, embedded objects, primitive arrays, empty arrays, null handling |
| `TestDetermineDataType` | All 7 type inference branches (boolean, integer, float, date, email, URL, string) |
| `TestDetectAnomalies` | All 8 anomaly flags: HIGH_NULL_RATE, AVG_LENGTH_TOO_SHORT/LONG, NUMERIC_OUTLIERS, NEGATIVE_VALUES, LOW_UNIQUENESS, PATTERN_INCONSISTENCY, FUTURE_DATE, MIXED_DATA_TYPES |
| `TestQualityScoring` | Score deductions per anomaly flag, penalty capping |
| `TestGenerateAnomalySummary` | Anomaly type counts, total aggregation |
| `TestIdentifyTopIssues` | Top-5 issue message format |
| `TestDataProfilingHelpers` | Completeness, duplicate detection, data freshness |
| `TestTypeSpecificAnalysis` | Numeric stats (min/max/mean/outliers), string length stats, date range stats, boolean value counts |
| `TestAnalyzePatterns` | Regex pattern matching across email, phone, URL, date patterns |
| `TestExportMetadataReport` | Excel and CSV export, sheet naming |
| `TestDuplicateRowAnomalyTableLevel` | Duplicate row deduction applied to quality score |
| `TestFKColumnsExcluded` | `_ref_*` columns not profiled as regular attributes |
| `TestAggregateAnomalySummary` | Multi-table anomaly rollup |
| `TestExtractMetadataEndToEnd` | Full pipeline on single-table and multi-table inputs |
| `TestSafeSheet` | Sheet name truncation (≤31 chars), invalid char sanitization |
| `TestCreateFallbackEnrichment` | Fallback LLM enrichment structure when LLM is unavailable |
| `TestConfigPromptTemplatePath` | Config path resolution |
| `TestStatisticalUtilsEdgeCases` | Outlier detection edge cases (< 3 values, constant arrays) |
| `TestFlattenThenExtract` | Flattener output fed directly into extractor (integration) |

**CSV null-handling bug found during testing:**
`test_load_csv_null_values_become_none` failed — numeric CSV columns with empty cells
stayed as `float('nan')` because `df.where(df.notna(), other=None)` cannot store Python
`None` in a `float64` column. Fixed by prepending `df.astype(object)` before the
`.where()` call in `file_loader/loader.py`.

**Excel output single-sheet bug found during testing:**
`convert_enriched_json_to_csv` used a brittle if/elif block to detect tables from the
input JSON. It failed for `{"organization": {...}}` structure, collapsing everything
into a single `main_data` sheet. Fixed by replacing the ~40-line detection block with
a direct call to `_fl._flatten_nested_json(input_data)` — the same flattener the
pipeline uses — guaranteeing table names match `raw_metadata['tables']` keys exactly.
Result: 41 sheets (13 tables × 3 sheets + 2 summaries) correctly generated.

---

### Bug-Fix Verification Tests (test_bug_fixes.py — 10 tests)

Written specifically to confirm each fix works and to guard against regressions:

| Test | Verifies |
|---|---|
| `test_bug1_numeric_preserved` | int/float/bool/None types survive `astype(object).where()` |
| `test_bug1_old_code_would_fail` | Old `astype(str)` converts `30`→`'30'`, `True`→`'True'` |
| `test_bug1_excel_round_trip` | Types survive full write-to-Excel → openpyxl read-back cycle |
| `test_bug2_boolean_no_low_uniqueness` | `is_active` (50 records, 2 values) — no LOW_UNIQUENESS |
| `test_bug2_status_field_low_uniqueness_still_fires` | `status` string (3 values, 50 rows) — still flagged (regression guard) |
| `test_bug2_boolean_with_all_true` | All-True column (unique_ratio=1/30) — still not flagged |
| `test_bug2_numeric_low_uniqueness_still_fires` | Integer 0/1 alternating — still flagged (regression guard) |
| `test_bug3_bracketed_table_name_extracted` | `[employees] HIGH_NULL...` → `Affected_Tables=employees` |
| `test_bug3_unbracketed_falls_back_to_multiple` | Plain issue text → `Affected_Tables=Multiple` |
| `test_bug3_partial_bracket_falls_back` | Unclosed bracket, empty string — no crash |

---

### Test Results

| Suite | Count | Result |
|---|---|---|
| `tests/test_comprehensive.py` | 109 | All pass |
| `tests/test_bug_fixes.py` | 10 | All pass |
| All other suites | 40 | All pass |
| **Total** | **159** | **159/159 PASSED** |

---

### Files Changed

| File | Change |
|---|---|
| `main.py` | Bug 1: replaced `astype(str)` loop with `astype(object).where()` |
| `main.py` | Bug 3: parse `[table_name]` prefix from top_issues for `Affected_Tables` column |
| `main.py` | Table detection fix: replaced brittle if/elif block with `_fl._flatten_nested_json()` |
| `step1_metadata_extractor/extractor.py` | Bug 2: added `data_type != "boolean"` guard to LOW_UNIQUENESS check |
| `file_loader/loader.py` | CSV null fix: added `astype(object)` before `.where(df.notna(), other=None)` |
| `tests/test_comprehensive.py` | New — 109 comprehensive tests |
| `tests/test_bug_fixes.py` | New — 10 targeted bug-fix verification tests |

---

## Session: 2026-04-04 (Strategic Planning)

### Overview
No code changes. Strategic analysis of ParseIQ's real-world positioning, deployment path,
and enterprise readiness. Decisions made here feed directly into the deployment project
(ParseIQ V.0.1 — Python Library).

### Key Decisions Made

**1. Deployment model: Python Library + CLI first**
Decided to zip V.0.0.1 as-is (web UI baseline) and duplicate the project as a
"deployment project" targeting `pip install parseiq`. Library-first because:
- Data never leaves user's machine (GDPR/HIPAA safe)
- Works in any cloud environment without infra from our side
- Integrates into existing ETL scripts and notebooks
- No server costs, no scaling headaches for V.0.1

**2. LLM stays central — BYOK architecture**
ParseIQ's identity is "AI data agent" so making LLM optional-by-design was rejected.
Instead: LLM provider is fully configurable. Users bring their own API key and choose
their own model (OpenAI, Azure, Ollama, OpenRouter). Their data goes to their LLM
account, not ours. This solves privacy/compliance without killing the AI-agent value.

**3. Real-world use cases confirmed**
The problem space is genuine. Complex nested JSON is common from:
SaaS API exports (Salesforce, Stripe, HubSpot), MongoDB/DynamoDB dumps, ERP system
exports (SAP, Oracle), IoT telemetry, data migration projects, microservice log aggregation.
Sweet spot: data onboarding / discovery phase — "what's in this dump and is it usable?"

**4. Disadvantages identified and mitigation strategy documented**
| Disadvantage | Mitigation |
|---|---|
| File-based input only | `from_*` connector class methods (S3, Postgres, MongoDB, URL) |
| LLM privacy / downtime | BYOK architecture + graceful degradation fallback to Step 1 |
| No incremental processing | `.parseiq_state.json` hash file, skip unchanged tables |
| Static reports only | `alert_rules` + `on_alert` callback post-processing layer |
| Single-user design | Library model = isolated by design; concurrent use is safe |

### Artefacts Updated
- `TODO.md` — Added `🚀 Deployment Strategy` section (Phase 1/2/3)
- `TODO.md` — Added `🏗️ Enterprise-Ready Pre-Deployment Checklist` (9 items, ordered
  code-changes-first then packaging, with full implementation detail per item)

---

## Session: 2026-04-04

### Overview
Implemented four data quality detection improvements in `step1_metadata_extractor/extractor.py`.
All 40 tests continue to pass; new logic confirmed via smoke tests.

### Changes Made

| File | Change |
|------|--------|
| `step1_metadata_extractor/extractor.py` | Skip `_ref_*` attrs in `_analyze_table_detailed` |
| `step1_metadata_extractor/extractor.py` | Post-profiling duplicate row deduction & anomaly flag |
| `step1_metadata_extractor/extractor.py` | `FUTURE_DATE_DETECTED` in `_detect_anomalies` |
| `step1_metadata_extractor/extractor.py` | `MIXED_DATA_TYPES` in `_detect_anomalies` |
| `step1_metadata_extractor/extractor.py` | New anomaly types in `_identify_top_issues` messages |
| `TODO.md` | Marked 4 items complete, removed completed MEDIUM item |

### Details

**1. `_ref_*` column exclusion**
FK columns injected by the flattener (e.g. `_ref_departments_id`) were being profiled as
real data attributes, inflating attribute counts and adding false anomalies. Fix: `continue`
for any `attr_name.startswith('_ref_')` in the attribute analysis loop.

**2. `DUPLICATE_ROWS_DETECTED`**
The `_analyze_duplicates` profiling already counted duplicates but the result was never
surfaced as an anomaly. Fix: after `_perform_data_profiling()`, check
`duplicate_analysis['total_duplicates'] > 0` and (a) inject into `anomaly_summary`, (b)
deduct 2 pts per duplicate row (max 20) from `data_quality_score`, (c) prepend message to
`top_issues`.

**3. `FUTURE_DATE_DETECTED`**
Any string column containing ISO-8601 date values beyond `datetime.today()` now raises this
flag. Implemented inside `_detect_anomalies()` — iterates non-null values, parses `YYYY-MM-DD`
prefix, compares to today.

**4. `MIXED_DATA_TYPES`**
Columns where non-null values span more than one incompatible Python category (bool / numeric /
str / list / dict) now flag `MIXED_DATA_TYPES`. int+float are grouped as `numeric` to avoid
false positives on naturally mixed numeric columns.

### Test Results
- 40/40 passing (unchanged)
- Smoke test confirmed all four flags fire correctly on targeted synthetic records

---

## Session: 2026-04-04 (continued)

### Overview
Output file reduction, auto-clean pipeline, and a 14-level 53K-record stress test dataset.

### Changes Made

| File | Change |
|------|--------|
| `main.py` | `convert_enriched_json_to_csv`: added `csv_per_table=False` param; per-table CSVs skipped by default |
| `main.py` | Auto-clean `output/*.csv` and `output/*.xlsx` at the start of `run_pipeline()` |
| `main.py` | Updated output summary print to reflect lean file set |
| `input/stress_test_data.json` | New 14-level nested stress test dataset |
| `scripts/generate_stress_test.py` | Generator script for the stress test data |

### Output File Reduction

**Before**: 13 tables × 3 CSVs each = 39 CSV files + 2 summary + 1 Excel + 3 JSON = **45 files**
**After**:  1 Excel (all sheets) + 2 summary CSVs + 3 JSON = **6 files**

The Excel workbook already contained all per-table data in separate sheets. The individual
CSVs were redundant. Set `csv_per_table=True` to restore them if needed.

### Stress Test Dataset

**File**: `input/stress_test_data.json` (37 MB)
**Generated by**: `scripts/generate_stress_test.py`

| Level | Table | Records |
|-------|-------|---------|
| 0  | enterprise      | 2      |
| 1  | continents      | 3      |
| 2  | countries       | 6      |
| 3  | regions         | 12     |
| 4  | cities          | 24     |
| 5  | offices         | 48     |
| 6  | departments     | 96     |
| 7  | teams           | 192    |
| 8  | employees       | 960    |
| 9  | assignments     | 1,698  |
| 10 | tasks           | 3,396  |
| 11 | subtasks        | 6,792  |
| 12 | activities      | 13,584 |
| 13 | activity_logs   | 27,168 |

**Total**: 53,981 records  
**Flattener performance**: 14 tables extracted in 1.1 seconds

**Anomalies injected** (per-employee, systematically varied):
- `DUPLICATE_ROWS_DETECTED` — exact duplicate employees per team; duplicate enterprise record
- `FUTURE_DATE_DETECTED` — hire_date=2029, audit_date=2099, subtask due_date=2099
- `MIXED_DATA_TYPES` — `priority` column mixes strings+int+None; `type` in activities mixes string+int+None; `is_active` mixes bool+string+int; `action` in logs mixes string+int
- `NEGATIVE_VALUES_DETECTED` — negative salary, negative budget, negative billing rate, negative lease cost
- `NUMERIC_OUTLIERS_DETECTED` — salary=9,999,999 outlier
- `HIGH_NULL_RATE` — `middle_name` ~40% null across all employees
- `PATTERN_INCONSISTENCY` — `email` column has valid + invalid emails mixed
- `LOW_UNIQUENESS` — `focus` (7 values across 192 teams), `currency`, `status`
- Impossible values — `completion_pct=150%`, `allocation_pct=120%`, `discount_pct=200%`

### Test Results
- 40/40 passing

---

## Session: 2026-04-03

---

### Overview

Full debugging, testing, algorithmic improvement, and stabilisation of the ParseIQ pipeline.
Started with a broken output (1 table, score always 100/100, LLM seeing 0 records)
and ended with 40/40 tests passing, 13 tables extracted correctly, and realistic quality scores.

---

## What Was Done

---

### 1. Fixed Quality Score Always Showing 100/100

**File:** `step1_metadata_extractor/extractor.py`

**Problem:**
Every table was scoring 100/100 even when anomalies were clearly detected.
Two flaws working together:
- `_calculate_attribute_quality_score()` had a `+10` uniqueness reward that nearly
  always applied (ratio in 0.1–0.9), which cancelled the `–10` anomaly penalty.
- `_calculate_quality_score()` (table-level) had a flat `+5` bonus for tables with
  5–20 attributes, pushing any near-100 average over the cap.

**Fix:**
- Removed the uniqueness reward entirely.
- Increased anomaly penalty from `–10` to `–15` per flag for better sensitivity.
- Replaced the `+5` attribute count bonus with a `–3 × total_anomalies` table-level
  deduction, so tables with many issues score meaningfully lower.

**Result:** Scores now range 53–100 based on actual issues found.

---

### 2. Fixed Output Showing Only 1 Table Instead of All Tables

**File:** `main.py` — `convert_enriched_json_to_csv()`

**Problem:**
Function checked `if "database" in input_data` but the new input JSON uses
`{"company": {"tables": {...}}}`. When the check failed, it fell through to treating
the entire JSON as a single row called `main_data`.

**Fix:**
Added a generic loop that checks all top-level keys for a value that is a dict
containing a `"tables"` sub-key. Works for any root name (`"company"`, `"database"`,
`"org"`, etc.).

**Result:** All 6 original tables (employees, departments, projects, timesheets,
expenses, performance_reviews) appeared in CSV and Excel output.

---

### 3. Fixed LLM Receiving 0 Records for All Tables

**Files:** `main.py`, `step2_llm_enricher/llm_agent.py`

**Problem:**
The LLM's `_compress_metadata_for_prompt()` reads from `dataset_overview.table_summaries`
and expects keys `record_count`, `field_analysis`, `completeness_rate`.
But `main.py` was feeding it the raw extractor output which uses different key names:
`dataset_info.total_records`, `attributes`, and no `completeness_rate` at top level.
Result: LLM received `record_count: 0` for every table and concluded "No records present."

**Fix (main.py):**
Added a translation layer that converts the extractor's format into what the LLM
compressor expects:
- `dataset_info.total_records` → `record_count`
- `attributes` → `field_analysis` (each attribute translated to `null_percentage`,
  `anomalies`, `outlier_count`, `unique_percentage`, etc.)
- `data_profiling.record_completeness.avg_completeness` → `completeness_rate`
- `data_profiling.duplicate_analysis.duplicate_rows` → `duplicate_count`

**Fix (llm_agent.py):** `_detect_table_structure()` now checks three possible locations
for record count: `record_count` → `summary.total_records` →
`table_metadata.dataset_info.total_records`, in order.

---

### 4. Fixed Table Names Showing as `main_table` Instead of Real Names

**File:** `main.py`

**Problem:**
The extractor was called with a bare list (e.g., `[{employee records}]`), so it
defaulted the table name to `"main_table"`. The real name (e.g., `"employees"`) was
stored only in the outer loop variable, never passed into the extractor result.

**Fix:**
After extraction, patch the `table_metadata['table_name']` field with the actual table
name from the outer loop.

---

### 5. Fixed LLM Assessment Showing N/A in Pipeline Summary

**File:** `main.py` — `_print_pipeline_summary()`

**Problem:**
Code read `overall_assessment.get('corrected_score', 'N/A')` and
`overall_assessment.get('summary', 'N/A')`. Neither key exists in the LLM's JSON
response — the actual fields are `overall_score` and `quality_grade`.

**Fix:**
Updated to read `overall_score`, `quality_grade`, `production_readiness`, and
`primary_concerns` from the correct locations. Now prints a meaningful one-line summary.

---

### 6. Project Cleanup

**Removed:**
- 80+ stale `debug_output/` files (accumulated over many runs)
- 27 old `logs/` files (kept 3 most recent)
- Stale CSVs from old e-commerce dataset run (addresses, orders, products, etc.)
- All `__pycache__/` and `.pytest_cache/` directories

**Organised:**
- Moved 7 root-level `test_*.py` files into `tests/` subfolder

**Added:**
- `.gitignore` preventing `debug_output/`, `logs/`, and `output/` files from being
  committed to git
- `.gitkeep` placeholders to preserve empty directory structure

---

### 7. Redesigned Complex Nested Test Dataset

**File:** `input/input_data.json`

Replaced the simple 6-table HR dataset with a realistic 4-level deeply nested
organisation dataset:

```
organization
├── divisions[]
│   ├── departments[]
│   │   └── employees[]
│   │       └── performance_reviews[]
│   └── projects[]
│       └── tasks[]
├── products[]
│   ├── pricing_tiers[]
│   └── inventory_locations[]
└── clients[]
    ├── contacts[]
    └── contracts[]
        └── contract_items[]
```

**13 tables, 50 records, 22+ intentional data quality issues:**

| Table | Key Issues Planted |
|---|---|
| employees | Empty name, invalid email, negative salary, future hire date, exact duplicate row, completely null row |
| performance_reviews | Rating = 12 (scale is 0–5), empty comments |
| projects | Negative budget, end_date before start_date |
| tasks | Duplicate task_id across projects, negative hours_actual, null assignee, FK to non-existent employee |
| pricing_tiers | Negative price, null max_users |
| inventory_locations | Duplicate record, negative stock count, future audit date (2030) |
| contracts | Non-existent product FK, inverted start/end dates, negative contract value |
| contract_items | Null SKU, negative qty, discount_pct = 150% (impossible), duplicate item_id, negative discount |
| contacts | Invalid email format, invalid phone format |
| products | Full null row (all fields null) |

---

### 8. Rewrote the JSON Flattening Algorithm

**File:** `file_loader/loader.py` — `_flatten_nested_json()`

**Old algorithm problems:**
1. Same-name tables from sibling records created separate tables:
   `departments` from division 1 and division 2 became `departments` + `departments_1`
2. No FK linking child rows back to their parent record
3. Embedded objects (e.g., `address: {street, city}`) kept as raw dict
4. Primitive arrays (`skills: ["Python", "Go"]`) were silently dropped
5. A plain flat JSON object (no arrays anywhere) returned empty `{}`

**New algorithm (based on pandas json_normalize + dbt/Airbyte FK-injection pattern):**

| Feature | Behaviour |
|---|---|
| Same-name tables from siblings | **Merged** into one table |
| Different tables with same name (different path) | Path-qualified: `customers__orders` vs `suppliers__orders` |
| FK injection | `_ref_<parent>_id` column injected into every child row |
| Embedded objects | Flattened inline: `address__street`, `address__city` |
| Primitive arrays | Joined as string: `"Python, Go, Leadership"` |
| Empty arrays | Skipped — no artifact column created |
| Null values for dict/list fields | Skipped — avoids artifact null columns |
| Plain flat JSON at root | Wrapped as `{"main_table": [data]}` |

**Also fixed:** `_get_record_id()` was picking up `_ref_*` FK columns as the record's
own ID. Added `not key.startswith('_ref_')` guard to the fallback loop.

---

### 9. Comprehensive Testing — 7 Bugs Found and Fixed

Ran 40 tests across all components. Started at 33/40 passing.

| Bug | Severity | Root Cause | Fix |
|---|---|---|---|
| Test mocks using outdated return type | Medium | `load_file` now returns `dict`, tests mocked it returning `list` | Updated all 4 affected mocks |
| `Config.validate_config()` missing | Medium | Method expected by tests never existed | Added method returning `{field: message}` issues dict |
| `test_load_file_json` wrong assertion | Medium | Test asserted raw dict but loader runs flatten step | Fixed assertion to check `result['main_table'][0]['key']` |
| Plain flat JSON → empty `{}` | Medium | `_flatten_nested_json` had no fallback for no-array dicts | Added root-level fallback to wrap as `{"main_table": [data]}` |
| `summary['total_tables']` always None | Low | Count was nested under `multi_table_info`, not top-level | Fixed summary dict path |
| CSV missing values as `float('nan')` | Low | Pandas `read_csv` fills blanks with NaN, extractor checks `is None` | Added `df.where(df.notna(), other=None)` after read |
| Artifact columns inflating null rates | **High** | After flattening `address: {dict}`, records with `address: null` created a spurious 100%-null `address` column. `products.pricing_tiers = ''` (empty array) created similar artifact. `products` score was 43/100 due entirely to these fake nulls | Pre-scan all records to classify each field's dominant type; null/empty values for dict/list_of_dicts fields are skipped from the flat record |

**Final result: 40/40 tests passing.**

---

## Issues Still Open / Known Limitations

1. **Duplicate rows not flagged as anomaly** — the extractor tracks duplicates in
   `data_profiling.duplicate_analysis` but does not raise an `anomaly_flag` or count
   them in `total_anomalies`.

2. **Future dates not detected** — a date of "2099-01-01" or "2027-01-01" is not
   flagged by `_detect_anomalies()`. No temporal boundary check exists in Step 1.

3. **Mixed types in same column not flagged** — a column containing `[1, "hello", None,
   3.14, True]` infers a type but does not raise a type-inconsistency anomaly.

4. **"STEP 2" banner prints many times in terminal** — appears to be a Windows
   PowerShell / VS Code terminal buffer-rendering artifact during the long LLM wait.
   No code bug identified; all print statements are correct.

5. **Output directory not cleaned between runs** — stale CSV/Excel files from a previous
   input dataset persist alongside new ones. No auto-cleanup on pipeline start.

6. **`_ref_*` FK columns analysed as regular data** — the extractor treats FK columns
   like `_ref_departments_id` as regular attributes, including them in null rate and
   anomaly calculations. These should ideally be excluded or treated differently.

---

## Files Changed Today

| File | Change |
|---|---|
| `step1_metadata_extractor/extractor.py` | Quality scoring overhaul |
| `main.py` | Table detection fix, dataset_overview translation layer, LLM summary fix, table name patching |
| `step2_llm_enricher/llm_agent.py` | Record count lookup fix in `_detect_table_structure` |
| `file_loader/loader.py` | Full algorithm rewrite of `_flatten_nested_json`, CSV NaN fix |
| `config.py` | Added `validate_config()` method |
| `input/input_data.json` | Replaced with complex 4-level nested dataset |
| `tests/test_config.py` | Updated assertions for new `validate_config()` signature |
| `tests/test_file_loader.py` | Updated `test_load_file_json` assertion |
| `tests/test_main.py` | Updated pipeline test mocks |
| `tests/test_integration.py` | Updated integration test mocks |
| `.gitignore` | Created |
| `WORKLOG.md` | Created (this file) |
