# ParseIQ — Work Log

Development sessions, decisions, and technical notes in reverse-chronological order.

---

## Session: 2026-04-10 — LLM Mode Output Fixes (v0.0.6)

### Overview
Testing revealed that in LLM mode the Excel `01_LLM_Assessment` sheet showed `N/A` for
**Overall Score** and **Model Used**, and that `corrected_score` was always `0` regardless
of the actual data quality. Four related bugs traced to three files.

### Bugs Fixed

**Bug 1 — corrected_score always 0** (`parseiq/step2_llm_enricher/llm_agent.py`)
`_calculate_corrected_quality_score()` looked for `original_score` at keys like
`data_quality_summary.overall_score` and `data_quality_score` — keys that do not exist
in the metadata dict the pipeline actually builds. Extended with two correct lookup paths:
- `dataset_overview.table_summaries[*].quality_score` (average across all tables)
- `tables[*].table_metadata.data_quality_score` (per-table extractor output)

**Bug 2 — Overall Score N/A in Excel** (`parseiq/pipeline.py`)
`_generate_outputs()` read `oa.get("overall_score", "N/A")` but the fallback and internal
paths only set `"corrected_score"`. Fixed with defensive lookup across both keys.

**Bug 3 — Model Used N/A in Excel** (`parseiq/step2_llm_enricher/llm_agent.py`)
`_create_fallback_enrichment()` omitted `"model_used"` from its `enrichment_metadata`.
Added `"model_used": self.model`.

**Bug 4 — Fallback missing key_strengths / primary_concerns / overall_score**
Same fallback method also lacked `"overall_score"`, `"key_strengths"`, `"primary_concerns"`
in `overall_assessment`. Added all three.

### Tests
6 regression tests added to `tests/test_llm_enricher.py`. 165/165 passing.

### Version bump
`0.0.5` → `0.0.6` in `pyproject.toml` and `parseiq/__init__.py`.

---

## Session: 2026-04-09 — Schema Polymorphism Detection (v0.0.5)

### Overview
User provided `apex_capital_dataset.json` and full ParseIQ output showing 63 total anomalies
of which ~58 were false positives caused by schema polymorphism — ParseIQ treating
heterogeneous records (RELIANCE Energy + HDFCBANK Banking) as one uniform schema.

Root cause identified: no entity-type clustering before per-attribute statistics. Fields absent
for a specific entity type were counted as nulls across all records → `HIGH_NULL_RATE` fires
on every type-conditional column.

### Fix: `_detect_schema_groups()` (`extractor.py`)

New method added before the attribute-analysis loop in `_analyze_table_detailed()`.

**Algorithm:**
1. For each column: check ≥70% presence, 2–10 unique values, categorical (non-numeric),
   not an ID/key column (filtered by `_id` suffix, `isin`, `name`, `ticker`, etc.)
2. Group records by discriminator value → compute per-group column presence purity
3. Score = fraction of variable-presence columns (15–85% overall presence) with ≥80%
   purity within each group
4. Accept if score > 0.25 → record as discriminator; derive `type_conditional_cols`

**Result on apex_capital:**
- `holdings`: discriminator = `sector` (Energy vs Financial Services) → 39 columns classified as type-conditional
- `portfolios`: 19 columns classified as type-conditional
- `departments`: 3 risk-framework columns classified as type-conditional
- Quality score: 95.66 (up from ~50s); 0 false-positive `HIGH_NULL_RATE` flags

### `TYPE_CONDITIONAL_FIELD` anomaly type

- Replaces `HIGH_NULL_RATE` for type-conditional columns
- 2pt quality penalty vs 15pt for real anomalies
- Null-rate quality deduction suppressed
- Excluded from `top_issues` surfacing and null-rate issue messages
- `schema_groups` block written to table metadata (discriminator, group_count, score, type_conditional_columns)
- `_describe_issue()` in `pipeline.py` updated with actionable guidance

### `_NEGATIVE_ALLOWED_PATTERNS` extended

Added `'_return'` and `'_yield'` tokens — fixes `predicted_return_1m_pct` and
`dividend_yield_pct` false positives (pattern `return_pct` didn't match when `_1m_`
separated `return` and `pct` in the column name).

### Version bump: 0.0.4 → 0.0.5

- `pyproject.toml`, `parseiq/__init__.py` updated
- `CHANGELOG.md`, `README.md`, `TODO.md`, `WORKLOG.md` updated
- Git tag `v0.0.5` pushed to GitHub
- `dist/parseiq-0.0.5.tar.gz` and `.whl` built (awaiting PyPI upload with user token)

### Tests: 159/159 passing

---

## Session: 2026-04-09 — Deep JSON Flattening + Quality Score + Bug Fixes

### Overview
Tested ParseIQ on `input/input_data.json` — a deeply-nested financial JSON with 5+ levels of
nesting (portfolios → holdings → financials → income_statement → fy2025 → margins).
Identified and fixed 6 bugs, overhauled the loader's dict-handling strategy, and wrote full
community-facing documentation (TODO.md, CONTRIBUTING.md) for every known issue.

### Bug 1 — Deep JSON Objects Stringified as Blobs (`parseiq/file_loader/loader.py`)

**Problem:** Any dict field that itself contained nested dicts or arrays was JSON-stringified
into a single blob column (`json.dumps(value, default=str)`).  This caused:
- `financials`, `ml_signals`, `risk_framework`, `head.performance` → single unreadable cells
- Child arrays inside those dicts never extracted as tables (`stress_scenarios`, `top_signals`)
- Excel Data sheets with extremely wide, garbled blob columns

**Fix:** Replaced the stringify branch with a two-step approach:
1. Recursive call to `_flatten_nested_json(value, path_elements + [key], tables)` — extracts
   any array-of-dicts found at any depth as a child table.
2. `_deep_flatten_scalars(value, key)` — new helper that walks the dict tree collecting every
   scalar leaf into the parent record with `__`-joined key paths.

```python
def _deep_flatten_scalars(self, obj, prefix):
    result = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            full_key = f"{prefix}__{k}"
            if isinstance(v, dict):
                result.update(self._deep_flatten_scalars(v, full_key))
            elif isinstance(v, list):
                if not (v and isinstance(v[0], dict)):  # primitive list → join
                    result[full_key] = ', '.join(str(i) for i in v) if v else ''
            else:
                result[full_key] = v
    return result
```

**Result:**
- `departments` went from 8 → 15 attributes (now includes `head__performance__fy2024__rating`, etc.)
- `holdings` went from 17 → 62 attributes (all `financials__*` columns properly extracted)
- `portfolios` went from 24 → 31 attributes (sector breakdown, ML signal fields)
- `stress_scenarios` and `top_signals` now extracted as proper tables (were missing entirely)

### Bug 2 — Quality Score Bottoming at 0 on Wide Tables (`parseiq/step1_metadata_extractor/extractor.py`)

**Problem:** `_calculate_quality_score()` used: `base_score -= total_anomalies × 3`.
After the deep-flatten fix, `holdings` had 62 columns with ~40 anomalies (HIGH_NULL_RATE on
asymmetric schema between RELIANCE and HDFCBANK). `40 × 3 = 120` → `max(0, 50 - 120) = 0`.

**Fix:** Rate-based penalty capped at 20 points:
```python
anomaly_rate = anomalous_attrs / total_attrs
base_score -= min(anomaly_rate * 20, 20)
```
Per-attribute scores already penalise each flag (15 pts/flag). The table-level penalty now
only signals the fraction of affected attributes, not the raw count.

**Result:** `holdings` 0 → 61, `portfolios` 15 → 62, `avg_quality` 76.7 → 87.4.

### Bug 3 — Duplicate Table Processing (`parseiq/pipeline.py`)

`visited_tables: set` guard added to both the Data/Meta/Quality sheet builder loop and the
`99_Issues` section loop. Pure defensive measure — Python dicts can't have duplicate keys,
but guards against any edge case where the same table name might be yielded twice.

### Bug 4 — Excel Blob Columns (`parseiq/pipeline.py`)

```python
def _truncate_blobs(val, limit=120):
    if isinstance(val, str) and len(val) > limit and (val.startswith("{") or val.startswith("[")):
        return val[:limit] + "…"
    return val

df = df.apply(lambda col: col.map(_truncate_blobs))
```
Used `df.apply(col.map(...))` instead of `applymap` to avoid the pandas 2.1 deprecation warning.

### Bug 5 — Context Bleed Between Tables

Fixed as a side-effect of Bug 1 fix. Root cause: complex dicts being stringified produced
blob strings that looked like attributes from other JSON contexts when the parent record was
added to a sibling table.

### Bug 6 — Privacy: Prompt Template Path Logged in Full (`parseiq/step2_llm_enricher/llm_agent.py`)

```python
# Before
self.logger.info(f"Loaded prompt template from {template_path}")
# After
self.logger.info(f"Loaded prompt template from {Path(template_path).name}")
```
Added `from pathlib import Path` import.

### Other Fixes

- `00_Summary` Top_Issues column: replaced `"None"` string → empty string for clean tables
- `tests/test_comprehensive.py` line 482: updated expected score for rate-based penalty

### Verification

Tested against `input/input_data.json` (deeply nested: org → dept → head → performance → fy2025):
- 10 tables extracted: `departments`, `team`, `stress_scenarios`, `portfolios`, `holdings`,
  `transactions`, `rebalancing_history`, `actions_taken`, `top_signals`, `changelog`
- 34 Excel sheets generated (10 × 3 + 4 overview)
- No blob strings in any Data_ sheet
- No duplicate table processing
- No context bleed between tables
- Quality scores: avg 87.4/100
- 159/159 tests passing

### Community Documentation Written

- `TODO.md` — "Known Bugs — Open for Community PRs" section with 5 issues (A–E):
  Duplicate loop, Context bleed, Inconsistent flattening, Excel blobs, pip env collision.
- `CONTRIBUTING.md` — "Good First Issues — Ready for PRs" section mirroring each bug
  with exact file/line and fix guidance.

### Files Changed

| File | Change |
|---|---|
| `parseiq/file_loader/loader.py` | Deep dict recursion + `_deep_flatten_scalars()` helper |
| `parseiq/step1_metadata_extractor/extractor.py` | Rate-based quality score penalty |
| `parseiq/step2_llm_enricher/llm_agent.py` | Privacy: log filename only, added Path import |
| `parseiq/pipeline.py` | `visited_tables` guard, `_truncate_blobs()`, `'None'` → `''` fix |
| `TODO.md` | Known Bugs section (5 issues A–E) for community PRs |
| `CONTRIBUTING.md` | Good First Issues section with fix guidance per bug |
| `tests/test_comprehensive.py` | Updated quality score assertion (87 → 70) |
| `CHANGELOG.md` | v0.0.3 entry |
| `WORKLOG.md` | This entry |

### Test Results
- 159/159 passing

---

## Session: 2026-04-09 — Fix All Open Issues (v0.0.4)

### Overview
Fixed all 6 open known issues (E–J) identified during the multi-domain test case analysis.
Implemented three distinct mechanisms: column-name heuristic suppression (F), automatic
cross-level range detection (G), and a user-defined YAML/JSON rules sidecar engine (H, I).
Issue J was verified to already be handled by the existing HIGH_NULL_RATE detector.

### Issue E — Relax Version Pins (`pyproject.toml`)

Lowered minimum required versions for all 6 core dependencies to the earliest versions
that are still compatible with ParseIQ's API usage. Prevents pip from conflict-resolving
user's existing environment into incompatible downgrades.

### Issue F — NEGATIVE_VALUES False Positives (`extractor.py`)

Added `_NEGATIVE_ALLOWED_PATTERNS` module-level tuple (13 tokens).  Updated
`_detect_anomalies(attr_metadata, values, attr_name="")` — new `attr_name` parameter
flows from `_analyze_attribute` (already has it).  NEGATIVE_VALUES check now skips
columns whose lowercased name contains any allowed pattern.

**Verified:** `max_drawdown_pct = [-18.2, -12.4]` → no flag.  `salary = [-5000]` → flag.

### Issue G — Cross-Level Range Violations (`extractor.py`)

New `_detect_cross_level_range_violations(tables)` method:
- Scans all tables for `*_range*` columns containing comma-separated `"lo, hi"` pairs.
- **Tier 1** (name match): strips `_range` from the column name, searches all other tables.
  `temp_range_c` → `temp_c`; finds breach 10.8 in `tracking_events`. ✅
- **Tier 2** (FK fallback): if no name match, checks direct FK child tables (`_ref_{parent}_id`).
  `normal_range` has no name match; `readings` is FK child of `sensors` with 1 numeric col (`value`);
  checks `value` against [0, 10]; finds breach 18.9. ✅
- Called in `_extract_multi_table_metadata()` after cross-table relationship analysis.
- Violations injected into child table `top_issues` + `anomaly_summary` as `RANGE_VIOLATION_DETECTED`.

### Issues H & I — YAML Rules Sidecar Engine (`pipeline.py`)

Four new module-level helpers added after `_describe_issue`:

| Function | Purpose |
|---|---|
| `_find_rules_file(source_arg)` | Looks for `parseiq_rules.yaml/yml/json` next to input file |
| `_load_rules(rules_path)` | Parses YAML (via pyyaml) or JSON; returns `rules` list |
| `_apply_rules(rules, tables, raw_metadata)` | Dispatches each rule to its handler |
| `_rule_max_value` / `_rule_min_value` | Flag column > max or < min (Issue I) |
| `_rule_cross_table` | Join left/right tables on FK, check inequality (Issue H) |

Rules are applied in `run()` after `raw_metadata` is built, before alert rules.
Violations are stored in `raw_metadata["rule_violations"]` and injected into each
affected table's `top_issues` and `anomaly_summary`.

Example rules files created: `test_cases/tc04_university_rules.yaml` (max_value),
`test_cases/tc09_insurance_rules.yaml` (cross_table_compare).

### Issue J — Missing Sibling Dict Key

Deep-flatten already produces null columns for missing fiscal year keys (e.g. `financials__fy2024__*`).
With 2 subsidiaries and 1 missing fy2024: null rate = 50 % > 30 % threshold → HIGH_NULL_RATE fires.
No code change. Added verification note to TODO.md and CHANGELOG.md.

### Files Changed

| File | Change |
|---|---|
| `pyproject.toml` | Relaxed 6 version pins; added `rules = ["pyyaml>=6.0"]` optional extra |
| `parseiq/step1_metadata_extractor/extractor.py` | Issues F + G: `_NEGATIVE_ALLOWED_PATTERNS`, `_detect_anomalies(attr_name)`, `_detect_cross_level_range_violations()`, `_col_is_numeric()` |
| `parseiq/pipeline.py` | Issues H + I: `_find_rules_file`, `_load_rules`, `_apply_rules`, `_rule_max_value`, `_rule_min_value`, `_rule_cross_table`; rules invocation in `run()` |
| `test_cases/tc04_university_rules.yaml` | Example max_value rule |
| `test_cases/tc09_insurance_rules.yaml` | Example cross_table_compare rule |
| `TODO.md` | Issues E–J marked as fixed (v0.0.4) |
| `CHANGELOG.md` | v0.0.4 entry |
| `WORKLOG.md` | This entry |

### Test Results
- 159/159 passing

---

## Session: 2026-04-09 — Multi-Domain Test Case Suite (TC-01 to TC-10)

### Overview
Created a 10-file test case suite covering real-world enterprise data domains to validate
ParseIQ's anomaly detection breadth.  Each file contains injected anomalies spanning the full
range of ParseIQ's 8 detection types plus several classes of anomaly that require LLM mode or
future rule-engine support.  Performed a systematic gap analysis: what ParseIQ --no-llm catches
vs. what it misses and why.

### Test Cases Created (`test_cases/`)

| File | Domain | Key Injected Anomalies |
|---|---|---|
| `tc01_ecommerce.json` | E-Commerce orders | NEGATIVE_VALUES (qty, weight), future date, mixed types |
| `tc02_hr.json` | HR / Payroll | NULL salary, negative bonus, duplicate employee_id |
| `tc03_hospital.json` | Hospital / EMR | Negative dosage, future admission date, missing fields |
| `tc04_university.json` | University marks | Total marks = 128 (scale violation, > 100 cap) |
| `tc05_supplychain.json` | Cold-chain logistics | Temp 10.8 °C breaches zone range [2, 8] (cross-level) |
| `tc06_banking.json` | Banking / KYC | Negative balance, future date, pattern inconsistency |
| `tc07_social_media.json` | Social media analytics | Negative likes, NULL engagement, high avg_length |
| `tc08_iot_manufacturing.json` | IoT sensor readings | Reading 18.9 breaches sensor normal_range [0, 10] |
| `tc09_insurance.json` | Insurance policies | claimed_amount > sum_assured (cross-table), missing field |
| `tc10_conglomerate.json` | Conglomerate financials | Missing fy2024 key for one subsidiary (structural gap) |

### Gap Analysis — What ParseIQ --no-llm Catches

| Anomaly Class | ParseIQ Detects? | Reason |
|---|---|---|
| NEGATIVE_VALUES (qty, weight, dosage, premium, likes) | ✅ Yes | Pure numeric check |
| HIGH_NULL_RATE (missing fields across records) | ✅ Yes | Key presence comparison |
| AVG_LENGTH_TOO_LONG (deep blob fields) | ✅ Yes | String length heuristic |
| Date format inconsistency (dd/mm/yyyy vs ISO) | ⚠️ LLM only | Needs pattern recognition |
| Cross-level range violation (temp > zone range) | ❌ No | No parent-child constraint join |
| Cross-table constraint violation (claim > sum_assured) | ❌ No | Tables not joined post-extraction |
| Scale/domain violation (marks total > 100) | ❌ No | No domain upper-bound concept |
| Missing sibling dict key (fy2024 absent) | ❌ No | Dict gaps not compared across records |
| Semantic fraud/ESG flags | ❌ No | LLM-only |

### Bugs Documented

Four new limitation issues added to `TODO.md` and `CONTRIBUTING.md`:

- **Issue G** — Cross-level range violations (TC-05, TC-08): `temp_range_c` and the breaching
  value live in different flattened tables; no link is maintained post-extraction.
- **Issue H** — Cross-table constraint violations (TC-09): `claimed_amount > sum_assured`
  requires a FK join that ParseIQ never performs in local mode.
- **Issue I** — Scale/domain violations (TC-04): marks total `128` on a 100-point system is
  arithmetically correct but semantically invalid; requires a domain upper-bound rule.
- **Issue J** — Missing sibling dict key (TC-10): structural gap where `fy2024` is absent for
  one record but present for others; dict key-set comparison not implemented.

### Files Changed

| File | Change |
|---|---|
| `test_cases/tc01_ecommerce.json` through `tc10_conglomerate.json` | New 10-file test suite |
| `test_cases/write_tc06_10.py` | Generator script for TC-06 to TC-10 |
| `TODO.md` | Issues G–J added (cross-level range, cross-table constraint, scale violation, missing key) |

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
| Nested dicts stringified as blob columns | Fixed | 2026-04-09 |
| stress_scenarios / top_signals not extracted | Fixed | 2026-04-09 |
| Quality score = 0 on wide tables | Fixed | 2026-04-09 |
| Duplicate table processing in Excel output | Fixed | 2026-04-09 |
| Context bleed between table analyses | Fixed | 2026-04-09 |
| Excel blob columns (unreadable wide cells) | Fixed | 2026-04-09 |
| 'None' string in 00_Summary Top_Issues | Fixed | 2026-04-09 |
| Prompt template full path logged | Fixed | 2026-04-09 |
