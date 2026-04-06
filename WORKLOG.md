# ParseIQ — Work Log

Development sessions, decisions, and technical notes in reverse-chronological order.

---

## Session: 2026-04-06 — Multi-Provider LLM + Output Sheet Overhaul

### Overview
Added full multi-provider LLM support (Anthropic, Gemini, Perplexity + existing OpenRouter/OpenAI/Ollama/Azure).
Overhauled the Meta and Quality Excel sheets to expose all available attribute statistics.
Uploaded v0.0.2 to TestPyPI.

### LLM Provider Routing (`parseiq/step2_llm_enricher/llm_agent.py`)

Added `_detect_provider()` — infers provider from explicit `_provider` attribute or model name:
- `claude-*` model prefix → `anthropic`
- `gemini-*` model prefix → `gemini`
- everything else → `openai_compatible`

Refactored `_make_api_request()` into a dispatcher that routes to:
- `_make_api_request_anthropic()` — uses `anthropic` SDK
- `_make_api_request_gemini()` — uses `google-generativeai` SDK
- `_make_api_request_openai_compatible()` — existing OpenRouter/OpenAI REST path (renamed from `_make_api_request`)

Added `_PROVIDER_BASE_URLS` class constant — auto-sets `base_url` per provider when `enrich_metadata()` is called with `llm_provider`:
```python
'openrouter': 'https://openrouter.ai/api/v1',
'openai':     'https://api.openai.com/v1',
'perplexity': 'https://api.perplexity.ai',
```

Added credit exhaustion detection — 402 response now prints free-model alternatives:
```
[ParseIQ] Credits exhausted on your current plan.
  Free alternatives you can use right now:
    nvidia/nemotron-3-super-120b-a12b:free  (via openrouter.ai — free account)
    ...
```

### CLI Changes (`parseiq/_cli.py`)

- `--llm-provider` choices expanded: `openrouter openai anthropic claude gemini perplexity azure ollama`
- `_get_api_key_from_env(provider)` — picks correct env var per provider (`ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `PERPLEXITY_API_KEY`, etc.)
- `parseiq init` — 7-option provider wizard with per-provider model menus
- `parseiq models` — full breakdown by provider with install instructions
- `parseiq config` — shows all 5 provider API keys detected
- Ollama sets placeholder key (`'ollama'`) automatically — no `--llm-api-key` needed
- Interactive model list in `parseiq analyze` adapts to selected `--llm-provider`
- Footer changed from `Open the Excel report: ...` to `For a more detailed report, refer to: <absolute path>`
- `_print_banner()` reads version dynamically from `__version__`
- `sys.stdout.reconfigure(encoding='utf-8')` at startup — prevents cp1252 encoding errors on Windows

### Meta Sheet Overhaul (`parseiq/pipeline.py`)

Old: 7 columns (Table, Attribute, Data_Type, Null_Pct, Unique_Ratio, Quality_Score, Anomalies)
New: 30 columns — full attribute profile including:
- Core stats: Total_Records, Present_Count, Missing_Count, Missing_Percentage
- String stats: Min/Max/Avg/Median_Length, Most_Common_Values, Character_Distribution
- Anomaly: Anomaly_Count, Anomaly_Types, Has_Outliers, Recognized_Patterns
- Numeric stats: Min/Max/Mean/Median_Value, Std_Deviation, Outliers_Count
- Boolean stats: True_Count, False_Count, True_Percentage, False_Percentage
Type-specific columns are `None` for other data types.

### Quality Sheet Overhaul (`parseiq/pipeline.py`)

Old: 6-column wide format (Table, Attribute, Quality_Score, Null_Penalty, Anomaly_Count, Status)
New: Long format — one metric row per attribute:
- Columns: Table_Name, Quality_Category, Metric_Name, Metric_Value, Status, Description
- Table-level rows: Overall (quality score), Structure (attribute count), Volume (record count)
- Per-attribute rows: Attribute Quality, Uniqueness, Completeness; optional Anomalies + Outliers rows

### Dependencies (`pyproject.toml`)
- Added `anthropic = ["anthropic>=0.25.0"]` optional extra
- Added `gemini = ["google-generativeai>=0.5.0"]` optional extra
- Both included in `all` extra

### Files Changed

| File | Change |
|---|---|
| `parseiq/step2_llm_enricher/llm_agent.py` | Provider routing, Anthropic/Gemini backends, 402 detection |
| `parseiq/_cli.py` | Expanded providers, init wizard, models list, config, footer |
| `parseiq/pipeline.py` | Meta sheet (30 cols), Quality sheet (long format) |
| `parseiq/__init__.py` | Version → 0.0.2 |
| `pyproject.toml` | Version → 0.0.2, anthropic + gemini optional deps |
| `README.md` | Full rewrite for deployment-ready state |
| `CHANGELOG.md` | 0.0.2 entry added |
| `WORKLOG.md` | This entry |
| `commands.md` | Updated for all new providers and features |

### Test Results
- 159/159 passing (unchanged)

---

## Session: 2026-04-05 — Package Release + LLM BYOK Fixes

### Overview
Prepared and published ParseIQ as a Python package on TestPyPI.
Fixed CSV/XML/Excel crashing the pipeline, LLM kwargs bug, and 6 Excel output issues.
Built and verified wheel in a fresh virtual environment. Published v0.0.1 then v0.0.2.

### Package Release

- `pyproject.toml` created with full metadata, optional extras, entry point
- `LICENSE` — MIT, Shriniwas Ahirrao 2026
- Author cleaned: removed any internal references
- GitHub URL set to: `https://github.com/ShriniwasAhirrao/ParseIQ-V0.0.1`
- README rewritten for `pip install parseiq` workflow
- Wheel built: `parseiq-0.0.1-py3-none-any.whl` → `parseiq-0.0.2-py3-none-any.whl`
- Uploaded to TestPyPI: https://test.pypi.org/project/parseiq/

### CSV/XML/Excel Pipeline Fix (`parseiq/connectors/file.py`)

`FileLoader.load_file()` returns `list` for CSV/XML/Excel but `Pipeline` calls `.items()` on it.
Fixed by wrapping list as `{stem: rows}` in `connectors/file.py`:
```python
if isinstance(raw, list):
    stem = os.path.splitext(os.path.basename(path))[0]
    return {stem: raw}
```

### LLM BYOK Fixes (`parseiq/step2_llm_enricher/llm_agent.py`)

1. `LLMEnricher.__init__` raised `ValueError` immediately if no key → deferred to `_make_api_request`
2. `enrich_metadata()` didn't accept `llm_provider`, `llm_api_key`, `llm_model`, `llm_base_url` kwargs → extended signature

### Excel Output Fixes (`parseiq/pipeline.py`)

| Issue | Fix |
|---|---|
| `main_table` in sheet names instead of filename stem | Renamed in loader after flatten |
| zscore RuntimeWarning crashing on constant columns | `warnings.catch_warnings()` + `nan_to_num` |
| Total records showing table count not row count | Fixed to use `pipeline_info.get('total_records')` |
| All Data sheets → All Meta sheets → All Quality sheets | Reordered to group per table: Data→Meta→Quality |
| LLM Assessment not in separate sheets | Added `01_LLM_Assessment` and `02_LLM_Recommendations` |
| Issues sheet had columns and weak descriptions | Rewrote with Priority/Business_Impact/Effort + `_describe_issue()` helper |

### CLI Additions (`parseiq/_cli.py`)

New commands: `init`, `validate`, `models`, `config`, `version`
New flags: `--llm-api-key`, `--quiet/-q`, `--fail-under SCORE`
Interactive model prompt when LLM=True and no model specified
Graceful no-key handling with yes/no prompt to run without LLM

### Fresh Venv Verification

Before each PyPI upload, verified in a clean venv:
```bash
python -m venv test_venv && test_venv\Scripts\activate
pip install dist/parseiq-0.0.2-py3-none-any.whl
parseiq version          # OK
parseiq validate ...     # OK
parseiq analyze ... --no-llm  # OK
```

### Files Changed

| File | Change |
|---|---|
| `parseiq/connectors/file.py` | list→dict normalisation for CSV/XML/Excel |
| `parseiq/step2_llm_enricher/llm_agent.py` | Deferred key validation, extended enrich_metadata signature |
| `parseiq/pipeline.py` | Sheet ordering, LLM sheets, Issues sheet, total_records fix |
| `parseiq/file_loader/loader.py` | main_table → filename stem rename, zscore fix |
| `parseiq/_cli.py` | Full rewrite — all commands and flags |
| `parseiq/__init__.py` | Version → 0.0.1 → 0.0.2 |
| `pyproject.toml` | Created from scratch |
| `LICENSE` | Created |
| `README.md` | Full rewrite |
| `tests/test_file_loader.py` | Updated main_table assertion |

---

## Session: 2026-04-04 — Bug Fixes + Comprehensive Testing

### Overview
Three confirmed bugs fixed. 109-test comprehensive suite written. 159/159 passing.

### Bug 1 — Data Sheets Coercing All Values to Strings

**File:** `pipeline.py` (was `main.py`)
```python
# Old (broken):
df_table[col] = df_table[col].astype(str).replace('nan', '')
# New (correct):
df_table = df_table.astype(object).where(df_table.notna(), other=None)
```
Every cell was becoming a string — integers, booleans, None all lost their type.
The `astype(object)` is required before `.where()` because float64 columns silently keep NaN even after `.where(None)`.

### Bug 2 — LOW_UNIQUENESS False Positive on Boolean Fields

**File:** `step1_metadata_extractor/extractor.py`
```python
# Old (buggy):
if unique_ratio < 0.1 and len(values) > 10:
# New (correct):
if unique_ratio < 0.1 and len(values) > 10 and attr_metadata.get("data_type") != "boolean":
```
Boolean columns always have ≤2 values → ratio always below 0.1 threshold → false positive.

### Bug 3 — `Affected_Tables` Hardcoded to `"Multiple"`

**File:** `pipeline.py`
```python
# Old:
"Affected_Tables": "Multiple",
# New (parses [table_name] prefix from issue string):
affected_table = issue[1:issue.find("]")] if issue.startswith("[") else "Multiple"
```

### Comprehensive Test Suite (`tests/test_comprehensive.py`)
109 new tests covering 21 test classes across all components.
Full details in the [2026-04-04 session entry] below.

---

## Session: 2026-04-04 — Detection Improvements

Added 4 new anomaly detections to `step1_metadata_extractor/extractor.py`:

1. **`_ref_*` column exclusion** — FK columns injected by the flattener skipped from profiling
2. **`DUPLICATE_ROWS_DETECTED`** — surfaced as anomaly + quality score deduction (2 pts per duplicate, max 20)
3. **`FUTURE_DATE_DETECTED`** — any ISO-8601 date beyond `datetime.today()` flagged
4. **`MIXED_DATA_TYPES`** — columns mixing bool/numeric/str/list/dict flagged (int+float grouped as `numeric` to avoid false positives)

---

## Session: 2026-04-04 — Output Reduction + Stress Test Dataset

### Output file reduction
Before: 13 tables × 3 CSVs + 2 summary + 1 Excel + 3 JSON = 45 files
After: 1 Excel + 2 CSV + 3 JSON = 6 files
Per-table CSVs removed (redundant — Excel already has all sheets).

### Stress test dataset (`input/stress_test_data.json` — 37 MB)
14-level deep nested JSON, 53,981 records across 14 tables.
Intentional anomalies: every anomaly type injected systematically.
Flattener performance: 14 tables extracted in 1.1 seconds.

---

## Session: 2026-04-03 — Initial Build + Algorithm Design

### Overview
Full debugging, testing, and stabilisation from scratch.
Started with 1 table / always 100/100 / LLM seeing 0 records.
Ended with 40/40 tests, 13 tables, realistic scores.

### Key fixes
1. Quality always 100/100 — removed uniqueness reward, increased anomaly penalty, added table-level deduction
2. Only 1 table in output — brittle root-key detection replaced with generic `"tables"` sub-key search
3. LLM receiving 0 records — translation layer between extractor format and LLM compressor format
4. Table names showing as `main_table` — patch `table_metadata['table_name']` after extraction
5. LLM summary showing N/A — read correct keys (`quality_grade`, `production_readiness`)
6. Project cleanup — removed 80+ stale debug/log files, organised tests into `tests/`

### JSON Flattener rewrite (`file_loader/loader.py`)
Full rewrite of `_flatten_nested_json()` — see CHANGELOG 0.0.1 for algorithm details.

### 7 bugs found and fixed during testing
See CHANGELOG 0.0.1 for full table.

### Test results
40/40 passing at session end.

---

## Known Issues Resolved Across Sessions

| Issue | Status | Session |
|---|---|---|
| Quality score always 100/100 | Fixed | 2026-04-03 |
| Only 1 table in output | Fixed | 2026-04-03 |
| LLM seeing 0 records | Fixed | 2026-04-03 |
| Data sheets all strings | Fixed | 2026-04-04 |
| LOW_UNIQUENESS on booleans | Fixed | 2026-04-04 |
| CSV/XML/Excel crashing pipeline | Fixed | 2026-04-05 |
| LLM kwargs not accepted | Fixed | 2026-04-05 |
| main_table in sheet names | Fixed | 2026-04-05 |
| zscore RuntimeWarning | Fixed | 2026-04-05 |
| Total records = table count | Fixed | 2026-04-05 |
| Sheet ordering (type-first not table-first) | Fixed | 2026-04-05 |
| Meta sheet missing most stats | Fixed | 2026-04-06 |
| Quality sheet wide format (hard to read) | Fixed | 2026-04-06 |
| Only OpenRouter supported | Fixed | 2026-04-06 |
| UnicodeEncodeError on Windows CLI | Fixed | 2026-04-06 |
