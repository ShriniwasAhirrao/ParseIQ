# ParseIQ — Command Reference

All commands start with `parseiq`. Run `parseiq --help` or `parseiq <command> --help` for inline docs.

---

## Quick Start (copy-paste)

```bash
# 1. Install
pip install -e .

# 2. First-time setup (sets API key, model, etc.)
parseiq init

# 3. Quick file check (no analysis yet)
parseiq validate data.json

# 4. Run full analysis — local mode (no API key needed)
parseiq analyze data.json --no-llm

# 5. Run with AI enrichment (needs API key)
parseiq analyze data.json
```

---

## `parseiq init`

Interactive setup wizard — run this once when you first install ParseIQ.

```bash
parseiq init
```

What it does:
- Asks for your OpenRouter API key and saves it to `.env`
- Lets you pick a default model from the free tier list
- Tests the API connection before you run anything
- Sets your preferred output directory

---

## `parseiq analyze <file>`

Main command. Runs the full 3-step pipeline on any supported file.

```bash
# Syntax
parseiq analyze <file> [options]
```

### Basic usage

```bash
# JSON file, local mode (no API key, always works)
parseiq analyze data.json --no-llm

# CSV file
parseiq analyze data.csv --no-llm

# XML file
parseiq analyze data.xml --no-llm

# Excel file
parseiq analyze data.xlsx --no-llm
```

### With LLM enrichment (needs API key)

```bash
# Use your OPENROUTER_API_KEY from environment
parseiq analyze data.json

# Pass key directly (useful in scripts)
parseiq analyze data.json --llm-api-key sk-or-v1-your-key-here

# Choose a specific model
parseiq analyze data.json --llm-model mistralai/mistral-small-3.1-24b-instruct:free

# Use OpenAI instead of OpenRouter
parseiq analyze data.json --llm-provider openai --llm-api-key sk-your-openai-key

# Use local Ollama (no API key needed)
parseiq analyze data.json --llm-provider ollama --llm-model llama3 --llm-base-url http://localhost:11434/v1
```

### Output options

```bash
# Custom output folder
parseiq analyze data.json --no-llm --output my_reports/

# Quiet mode (no progress output — good for scripts/CI)
parseiq analyze data.json --no-llm --quiet

# Force reprocess (ignore incremental cache)
parseiq analyze data.json --no-llm --force
```

### CI / Quality gate

```bash
# Exit code 1 if avg quality score is below 80
parseiq analyze data.json --no-llm --fail-under 80 && echo "Quality OK" || echo "Quality too low"
```

### All options

| Flag | Short | Default | Description |
|---|---|---|---|
| `--output` | `-o` | `output/` | Output directory for all generated files |
| `--no-llm` | | off | Skip LLM — pure local mode, no API key needed |
| `--llm-provider` | | `openrouter` | `openrouter` / `openai` / `azure` / `ollama` |
| `--llm-model` | | config default | Model name, e.g. `gpt-4o`, `llama3`, `mistral` |
| `--llm-api-key` | | env var | API key (overrides env var) |
| `--llm-base-url` | | provider default | Custom URL for Azure OpenAI or local Ollama |
| `--force` | | off | Reprocess all tables even if data unchanged |
| `--quiet` | `-q` | off | Suppress all output except errors |
| `--fail-under` | | off | Exit code 1 if avg quality < SCORE (CI gate) |

---

## `parseiq validate <file>`

Quick file check — no full analysis. Use this to confirm a file will load correctly before running the full pipeline.

```bash
parseiq validate data.json
parseiq validate data.csv
parseiq validate data.xlsx
```

Output shows:
- File status (OK / FAILED)
- Number of tables discovered
- Record count per table
- Column names (first 8)

---

## `parseiq models`

List all supported LLM models — free, paid, and local.

```bash
parseiq models
```

Useful when choosing a `--llm-model` value.

---

## `parseiq config`

Show current configuration — model, tokens, temperature, API key status.

```bash
parseiq config
```

---

## `parseiq version`

Print the installed version.

```bash
parseiq version
# parseiq 0.0.1
```

---

## Output Files

Every `parseiq analyze` run produces these files in the output directory:

| File | Description |
|---|---|
| `complete_data_analysis.xlsx` | Master Excel workbook — Data / Metadata / Quality sheets per table + 2 summary tabs |
| `overall_dataset_summary.csv` | One-line summary per table: records, quality score, anomaly count |
| `combined_issues_and_recommendations.csv` | All flagged issues with recommended fixes |
| `raw_metadata.json` | Full Step 1 technical metadata |
| `enriched_metadata.json` | Step 1 + LLM insights merged |
| `llm_insights.json` | Raw LLM response (only when `--no-llm` is not set) |

---

## Environment Variables

Set these to avoid passing flags every time:

```bash
# Windows (PowerShell)
$env:OPENROUTER_API_KEY = "sk-or-v1-your-key-here"
$env:PARSEIQ_MODEL = "mistralai/mistral-small-3.1-24b-instruct:free"

# Windows (Command Prompt)
set OPENROUTER_API_KEY=sk-or-v1-your-key-here

# Linux / macOS / Git Bash
export OPENROUTER_API_KEY=sk-or-v1-your-key-here
export PARSEIQ_MODEL=mistralai/mistral-small-3.1-24b-instruct:free
```

Or save to a `.env` file in the project root (auto-loaded if `python-dotenv` is installed):

```
OPENROUTER_API_KEY=sk-or-v1-your-key-here
PARSEIQ_MODEL=nvidia/nemotron-3-super-120b-a12b:free
```

---

## Common Scenarios

### Scenario 1 — First time, just checking if the tool works
```bash
parseiq validate input/input_data.json
parseiq analyze input/input_data.json --no-llm
```

### Scenario 2 — Regular use with free OpenRouter model
```bash
export OPENROUTER_API_KEY=sk-or-v1-...
parseiq analyze data.json
```

### Scenario 3 — CSV file from a database export
```bash
parseiq analyze export.csv --no-llm --output reports/export_$(date +%Y%m%d)/
```

### Scenario 4 — Run in CI pipeline (GitHub Actions / Jenkins)
```bash
parseiq analyze data.json --no-llm --quiet --fail-under 75
```

### Scenario 5 — Re-run after data update (skip unchanged tables)
```bash
parseiq analyze data.json --no-llm
# Next run — only changed tables are re-analysed
parseiq analyze data.json --no-llm
# Force full re-analysis
parseiq analyze data.json --no-llm --force
```

### Scenario 6 — Use local Ollama (no internet, no API cost)
```bash
# First: install ollama + pull a model
# ollama pull llama3
parseiq analyze data.json --llm-provider ollama --llm-model llama3
```

### Scenario 7 — Use OpenAI GPT-4o
```bash
export OPENAI_API_KEY=sk-your-openai-key
parseiq analyze data.json --llm-provider openai --llm-model gpt-4o
```

---

## Programmatic Use (Python API)

```python
from parseiq import Pipeline

# Local mode
result = Pipeline("data.json").run(llm=False)

# With LLM
result = Pipeline("data.json").run(
    llm=True,
    llm_provider="openrouter",
    llm_api_key="sk-or-v1-...",
)

# From CSV / XML / Excel — same API
result = Pipeline("data.csv").run(llm=False)

# Check results
print(result.tables)               # list of table names
print(result.quality_scores)       # {"employees": 37.6, ...}
print(result.overall_quality_score) # 72.4
print(result.total_anomalies)      # 48
print(result.llm_grade)            # "B" or None
print(result.output_files)         # list of file paths written

# With alert rules
from parseiq.alerts import slack_webhook

result = Pipeline("data.json").run(
    llm=False,
    alert_rules={
        "employees.salary": {"negative_values": True},
        "employees.email":  {"null_rate_gt": 0.05},
    },
    on_alert=slack_webhook("https://hooks.slack.com/services/..."),
)
```

---

## Supported File Formats

| Format | Extension | Notes |
|---|---|---|
| JSON | `.json` | Nested/hierarchical JSON auto-flattened into multiple tables |
| CSV | `.csv` | Auto-detects delimiter and encoding |
| XML | `.xml` | Converted to dict structure |
| Excel | `.xlsx`, `.xls` | Each sheet becomes a separate table |
