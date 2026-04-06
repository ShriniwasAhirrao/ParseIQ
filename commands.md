# ParseIQ — Command Reference

All commands start with `parseiq`. Run `parseiq --help` or `parseiq <command> --help` for inline docs.

---

## Quick Start (copy-paste)

```bash
# 1. Install
pip install parseiq

# 2. First-time setup
parseiq init

# 3. Quick file check (no full analysis)
parseiq validate data.json

# 4. Run analysis — local mode (no API key needed)
parseiq analyze data.json --no-llm

# 5. Run with AI enrichment (needs API key)
parseiq analyze data.json
```

---

## `parseiq init`

Interactive setup wizard. Run once after install.

```bash
parseiq init
```

What it does:
1. Shows all supported LLM providers — choose one
2. Asks for your API key (shows where to get one free)
3. Shows provider-specific model list — pick a default
4. Tests the API connection
5. Sets your preferred output directory
6. Saves everything to `.env`

**Supported providers in init:**

| # | Provider | Key env var | Free tier |
|---|---|---|---|
| 1 | openrouter | OPENROUTER_API_KEY | Yes — 100+ free models |
| 2 | openai | OPENAI_API_KEY | No |
| 3 | anthropic | ANTHROPIC_API_KEY | No (free trial credits on signup) |
| 4 | gemini | GEMINI_API_KEY | Yes — free key at aistudio.google.com |
| 5 | perplexity | PERPLEXITY_API_KEY | No |
| 6 | ollama | (none) | Yes — fully local |
| 7 | Skip | — | — |

---

## `parseiq analyze <file>`

Main command. Runs the full 3-step pipeline on any supported file.

```bash
parseiq analyze <file> [options]
```

### Basic usage

```bash
# Local mode (no API key, always works)
parseiq analyze data.json --no-llm
parseiq analyze data.csv --no-llm
parseiq analyze data.xml --no-llm
parseiq analyze data.xlsx --no-llm

# Custom output folder
parseiq analyze data.json --no-llm --output reports/june/

# Force full reprocess (ignore incremental cache)
parseiq analyze data.json --no-llm --force

# Quiet mode (no terminal output — for scripts/CI)
parseiq analyze data.json --no-llm --quiet

# CI quality gate — exit code 1 if avg quality below threshold
parseiq analyze data.json --no-llm --fail-under 80
```

### With LLM enrichment

```bash
# OpenRouter (default) — free models available
parseiq analyze data.json
parseiq analyze data.json --llm-provider openrouter \
  --llm-model nvidia/nemotron-3-super-120b-a12b:free

# OpenAI
parseiq analyze data.json \
  --llm-provider openai \
  --llm-model gpt-4o \
  --llm-api-key sk-...

# Anthropic / Claude  (requires: pip install anthropic)
parseiq analyze data.json \
  --llm-provider anthropic \
  --llm-model claude-sonnet-4-5 \
  --llm-api-key sk-ant-...

# Google Gemini  (requires: pip install google-generativeai)
parseiq analyze data.json \
  --llm-provider gemini \
  --llm-model gemini-1.5-pro \
  --llm-api-key AIza...

# Perplexity
parseiq analyze data.json \
  --llm-provider perplexity \
  --llm-model llama-3.1-sonar-large-128k-online \
  --llm-api-key pplx-...

# Azure OpenAI
parseiq analyze data.json \
  --llm-provider azure \
  --llm-model gpt-4o \
  --llm-api-key your-azure-key \
  --llm-base-url https://your-resource.openai.azure.com/

# Local Ollama (no API key, no cost, data stays on machine)
parseiq analyze data.json \
  --llm-provider ollama \
  --llm-model llama3

# Pass key inline (no env var needed)
parseiq analyze data.json --llm-api-key sk-or-v1-your-key-here
```

### All flags

| Flag | Short | Default | Description |
|---|---|---|---|
| `--output` | `-o` | `output/` | Output directory |
| `--no-llm` | | off | Skip LLM — pure local mode |
| `--llm-provider` | | `openrouter` | `openrouter` `openai` `anthropic` `claude` `gemini` `perplexity` `azure` `ollama` |
| `--llm-model` | | provider default | Model name |
| `--llm-api-key` | | env var | API key (overrides env var) |
| `--llm-base-url` | | provider default | Custom URL for Azure or local Ollama |
| `--force` | | off | Reprocess all tables even if unchanged |
| `--quiet` | `-q` | off | Suppress all terminal output |
| `--fail-under` | | off | Exit code 1 if avg quality < SCORE |

### Output at end of run

```
=======================================================
ANALYSIS COMPLETE
=======================================================
  Tables analysed : 14
  Total records   : 53,981
  Avg quality     : 72.4/100
  Total anomalies : 48
  LLM grade       : B
  Output folder   : output/llm_test/
  Files written   : 5

Per-table quality scores:
  continents     ██████████ 100.0/100  ✓
  employees      ███░░░░░░░  37.6/100  ⚠
  ...

For a more detailed report, refer to:
  D:\your\path\output\complete_data_analysis.xlsx
```

---

## `parseiq validate <file>`

Quick file check — no full analysis, no API key needed.

```bash
parseiq validate data.json
parseiq validate export.csv
parseiq validate report.xlsx
```

Output:
```
Validating: data.json
  Status    : OK
  Tables    : 14
  Records   : 53,981
  [employees]  960 rows  12 columns
    Columns : emp_id, name, email, age, salary, hire_date, ...
  ...
File is valid. Run the full analysis with:
  parseiq analyze data.json --no-llm
```

---

## `parseiq models`

List all available LLM models, grouped by provider.

```bash
parseiq models
```

Shows for each provider:
- Model IDs you can pass to `--llm-model`
- Which are free vs paid
- Install command if an extra SDK is needed
- API key env var name
- Link to get a key

---

## `parseiq config`

Show current configuration and all detected API keys.

```bash
parseiq config
```

Output:
```
=======================================================
  ParseIQ - AI-Powered Data Quality Agent  v0.0.2
=======================================================

Configuration Summary:
==================================================
  Model      : nvidia/nemotron-3-super-120b-a12b:free
  Max Tokens : 4096
  Temperature: 0.1
  Timeout    : 240s
  .env file  : found

  API keys detected:
    OpenRouter           SET (sk-or-v1-5be...)
    Anthropic/Claude     SET (sk-ant-...)
    (others not set)
==================================================
```

---

## `parseiq version`

Print the installed version.

```bash
parseiq version
# parseiq 0.0.2
```

---

## Output Files

Every `parseiq analyze` run produces:

| File | Description |
|---|---|
| `complete_data_analysis.xlsx` | Master Excel workbook (see sheet structure below) |
| `overall_dataset_summary.csv` | One row per table: records, quality score, anomaly count |
| `combined_issues_and_recommendations.csv` | All flagged issues with recommended fixes |
| `raw_metadata.json` | Full Step 1 technical metadata (JSON) |
| `enriched_metadata.json` | Step 1 + LLM insights merged (JSON) |

### Excel sheet structure

```
complete_data_analysis.xlsx
├── 00_Summary                   <- dataset overview, one row per table
├── 01_LLM_Assessment            <- LLM grade, production readiness, primary concerns
├── 02_LLM_Recommendations       <- prioritised action plan from LLM
│
├── Data_<table>                 <- raw data rows (actual types preserved)
├── Meta_<table>                 <- 30-column attribute profile:
│                                   Table_Name, Attribute_Name, Data_Type,
│                                   Total_Records, Present_Count, Missing_Count,
│                                   Missing_Percentage, Unique_Values, Unique_Ratio,
│                                   Quality_Score, Min_Length, Max_Length, Avg_Length,
│                                   Median_Length, Most_Common_Values,
│                                   Character_Distribution, Anomaly_Count,
│                                   Anomaly_Types, Has_Outliers, Recognized_Patterns,
│                                   Min_Value, Max_Value, Mean_Value, Median_Value,
│                                   Std_Deviation, Outliers_Count, True_Count,
│                                   False_Count, True_Percentage, False_Percentage
├── Quality_<table>              <- long-format quality metrics:
│                                   Table_Name | Quality_Category | Metric_Name |
│                                   Metric_Value | Status | Description
│                                   (one row per metric per attribute)
│
├── Data_<next_table>
├── Meta_<next_table>
├── Quality_<next_table>
│   ... (3 sheets per table, grouped by table)
│
└── 99_Issues_Recommendations    <- all issues sorted CRITICAL→HIGH→MEDIUM→LOW:
                                    Priority | Source | Table | Column |
                                    Issue_Type | Column_Quality | Description |
                                    Business_Impact | Recommended_Fix | Effort | Stats
```

---

## Environment Variables

Set these to avoid passing flags every time:

```bash
# Windows (PowerShell)
$env:OPENROUTER_API_KEY  = "sk-or-v1-..."
$env:OPENAI_API_KEY      = "sk-..."
$env:ANTHROPIC_API_KEY   = "sk-ant-..."
$env:GEMINI_API_KEY      = "AIza..."
$env:PERPLEXITY_API_KEY  = "pplx-..."
$env:PARSEIQ_MODEL       = "nvidia/nemotron-3-super-120b-a12b:free"

# Windows (Command Prompt)
set OPENROUTER_API_KEY=sk-or-v1-...

# Linux / macOS / Git Bash
export OPENROUTER_API_KEY=sk-or-v1-...
```

Or save to `.env` in your project root (auto-loaded when `python-dotenv` is installed):

```
OPENROUTER_API_KEY=sk-or-v1-...
ANTHROPIC_API_KEY=sk-ant-...
PARSEIQ_MODEL=nvidia/nemotron-3-super-120b-a12b:free
PARSEIQ_OUTPUT_DIR=output/
```

---

## Common Scenarios

### Scenario 1 — First time, verify the tool works
```bash
parseiq validate input/input_data.json
parseiq analyze input/input_data.json --no-llm
```

### Scenario 2 — Free OpenRouter model (recommended starting point)
```bash
# One-time setup
parseiq init    # choose openrouter, paste key from openrouter.ai

# Every run
parseiq analyze data.json
```

### Scenario 3 — Use Claude instead of OpenRouter
```bash
pip install anthropic
export ANTHROPIC_API_KEY=sk-ant-...
parseiq analyze data.json --llm-provider anthropic --llm-model claude-sonnet-4-5
```

### Scenario 4 — Use Gemini with free key
```bash
pip install google-generativeai
# Get free key: https://aistudio.google.com/app/apikey
export GEMINI_API_KEY=AIza...
parseiq analyze data.json --llm-provider gemini --llm-model gemini-1.5-flash
```

### Scenario 5 — Fully offline (Ollama)
```bash
# Install ollama from https://ollama.com, then:
ollama pull llama3
ollama serve   # keep this running

parseiq analyze data.json --llm-provider ollama --llm-model llama3
```

### Scenario 6 — CI pipeline (GitHub Actions / Jenkins)
```bash
# Quality gate — fail build if avg quality below 75
parseiq analyze data.json --no-llm --quiet --fail-under 75
echo "Exit code: $?"
```

GitHub Actions example:
```yaml
- name: Data Quality Check
  run: |
    pip install parseiq
    parseiq analyze data/export.json --no-llm --quiet --fail-under 75
  env:
    OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
```

### Scenario 7 — CSV export from a database
```bash
parseiq analyze export.csv --no-llm --output reports/$(date +%Y%m%d)/
```

### Scenario 8 — Re-run after data update (incremental)
```bash
# First run — all 14 tables analysed
parseiq analyze data.json --no-llm

# Second run — only changed tables re-analysed (hash cache)
parseiq analyze data.json --no-llm

# Force full reprocess
parseiq analyze data.json --no-llm --force
```

### Scenario 9 — Credits exhausted mid-run
ParseIQ detects 402 errors automatically and prints:
```
[ParseIQ] Credits exhausted on your current plan.
  Free alternatives you can use right now:
    nvidia/nemotron-3-super-120b-a12b:free  (via openrouter.ai)
    ...
  Re-run with: parseiq analyze <file> --llm-provider openrouter \
               --llm-model nvidia/nemotron-3-super-120b-a12b:free
```

---

## Python API

```python
from parseiq import Pipeline

# Local mode
result = Pipeline("data.json").run(llm=False)

# OpenRouter (default)
result = Pipeline("data.json").run(
    llm=True,
    llm_provider="openrouter",
    llm_api_key="sk-or-v1-...",
    llm_model="nvidia/nemotron-3-super-120b-a12b:free",
)

# Anthropic
result = Pipeline("data.json").run(
    llm=True,
    llm_provider="anthropic",
    llm_api_key="sk-ant-...",
    llm_model="claude-sonnet-4-5",
)

# Gemini
result = Pipeline("data.json").run(
    llm=True,
    llm_provider="gemini",
    llm_api_key="AIza...",
    llm_model="gemini-1.5-pro",
)

# Ollama (no key needed)
result = Pipeline("data.json").run(
    llm=True,
    llm_provider="ollama",
    llm_model="llama3",
)

# Different source types
result = Pipeline.from_file("data.json").run(llm=False)
result = Pipeline.from_url("https://api.example.com/data.json").run(llm=False)
result = Pipeline.from_s3("s3://bucket/data.json").run(llm=False)
result = Pipeline.from_postgres("postgresql://user:pass@host/db", "SELECT * FROM orders").run(llm=False)
result = Pipeline.from_mongodb("mongodb://localhost:27017", "customers").run(llm=False)

# Inspect results
print(result.tables)                # ["employees", "departments", ...]
print(result.quality_scores)        # {"employees": 37.6, "departments": 93.3}
print(result.overall_quality_score) # 72.4
print(result.total_anomalies)       # 48
print(result.llm_grade)             # "B" or None
print(result.output_files)          # list of written file paths
print(result.anomalies)             # {table: {col: [flags]}}
print(result.alerts_fired)          # list of triggered alert rules

# Alert rules
from parseiq.alerts import slack_webhook

result = Pipeline("data.json").run(
    llm=False,
    alert_rules={
        "employees.salary": {"negative_values": True},
        "employees.email":  {"null_rate_gt": 0.05},
        "orders":           {"quality_score_lt": 70},
    },
    on_alert=slack_webhook("https://hooks.slack.com/services/..."),
)
```

---

## Supported File Formats

| Format | Extension | Notes |
|---|---|---|
| JSON | `.json` | Any nesting depth — arrays of objects become separate tables |
| CSV | `.csv` | Auto-detects delimiter (comma, semicolon, tab) and encoding |
| XML | `.xml` | Converted via xmltodict |
| Excel | `.xlsx` `.xls` | Each sheet becomes a separate table |

---

## Anomaly Types Reference

| Flag | Meaning | Score impact |
|---|---|---|
| `HIGH_NULL_RATE` | >30% of values are null | -15 pts |
| `LOW_UNIQUENESS` | Unique ratio <10% with >10 rows (booleans exempt) | -15 pts |
| `MIXED_DATA_TYPES` | Column mixes incompatible types | -15 pts |
| `FUTURE_DATE_DETECTED` | Date value is beyond today | -15 pts |
| `NUMERIC_OUTLIERS_DETECTED` | Z-score or IQR outlier in numeric column | -15 pts |
| `NEGATIVE_VALUES_DETECTED` | Numeric column has negative values | -15 pts |
| `PATTERN_INCONSISTENCY` | Dominant pattern (email/phone/URL) but 10–50% don't match | -15 pts |
| `DUPLICATE_ROWS_DETECTED` | Exact duplicate rows at table level | -2 per duplicate (max -20) |

Quality status thresholds:

| Score | Status |
|---|---|
| 90–100 | Excellent |
| 80–89 | Good |
| 60–79 | Warning |
| 0–59 | Critical |

---

## Troubleshooting

### "No API key found"
```bash
parseiq init           # interactive setup
# or
parseiq analyze data.json --llm-api-key sk-or-v1-...
# or
parseiq analyze data.json --no-llm   # skip LLM entirely
```

### "Credits exhausted"
ParseIQ detects this automatically and suggests free alternatives.
Fastest fix:
```bash
parseiq analyze data.json \
  --llm-provider openrouter \
  --llm-model nvidia/nemotron-3-super-120b-a12b:free
```

### "anthropic / google-generativeai not installed"
```bash
pip install anthropic              # for Claude
pip install google-generativeai    # for Gemini
# or
pip install parseiq[anthropic]
pip install parseiq[gemini]
pip install parseiq[all]
```

### Rate limited (429)
ParseIQ backs off automatically and retries up to 3 times.
If it persists, the free tier is exhausted for the hour — switch models:
```bash
parseiq analyze data.json \
  --llm-model mistralai/mistral-small-3.1-24b-instruct:free
```

### File too large
Max file size is 100 MB. For larger files, split by table or use the Python API
with `from_postgres()` / `from_mongodb()` which stream data in chunks.

### Output folder locked (Windows)
Close any open Excel files from the output folder, then re-run.
