# Contributing to ParseIQ

Thank you for your interest in contributing!

## How to Contribute

### Report a Bug
Open an issue at https://github.com/ShriniwasAhirrao/ParseIQ/issues with:
- What you ran (command or code)
- What you expected vs what happened
- Python version and OS

### Suggest a Feature
Open an issue with the `enhancement` label describing your use case.

### Submit a Pull Request

1. Fork the repo
2. Create a branch: `git checkout -b fix/your-fix` or `feat/your-feature`
3. Make your changes
4. Run the test suite — all 159 tests must pass:
   ```bash
   pip install -e ".[dev]"
   pytest
   ```
5. Open a PR against `master` with a clear description of what and why

## Code Style
- Follow existing patterns in the codebase
- No new dependencies without discussion
- Keep functions focused — one responsibility per function

## Development Setup

```bash
git clone https://github.com/ShriniwasAhirrao/ParseIQ.git
cd ParseIQ
python -m venv venv
venv\Scripts\activate       # Windows
pip install -e ".[dev]"
pytest
```

## Good First Issues — Ready for PRs

The issues below are open and need community help. Bugs A–D from the original list were
fixed in v0.0.3 — see `CHANGELOG.md` and `TODO.md` for details.

---

### Issue E — pip Environment Collision  *(open)*
**File:** `pyproject.toml`, `README.md`
**What happens:** `pip install parseiq` into an existing project venv can downgrade or
conflict with pinned dependencies (pandas, openpyxl, etc.).
**Fix options:**
1. Relax version pins in `pyproject.toml` to wide ranges (`pandas>=1.5`, `openpyxl>=3.0`)
2. Add isolated-install guidance (pipx / dedicated venv) to the README Install section
3. Add a `pip check` call in CI that installs alongside common data science packages
**Label:** `enhancement` `documentation`

---

### Issue F — NEGATIVE_VALUES_DETECTED False Positives in Financial Data  *(open)*
**File:** `parseiq/step1_metadata_extractor/extractor.py`
**What happens:** Columns like `var_1d_99_pct`, `max_drawdown_pct`, `equity_shock`,
`cfi_cr` are legitimately negative in finance (VaR, drawdown, cash outflows) but are
flagged as `NEGATIVE_VALUES_DETECTED` anomalies, inflating issue counts and reducing
quality scores.
**Fix options:**
1. Add a `--allow-negatives "pattern1,pattern2"` CLI flag
2. Add a built-in heuristic — suppress the flag when the column name contains keywords
   like `var`, `drawdown`, `shock`, `cfi`, `cff`, `pnl` (common financial negative-OK terms)
3. Expose a `negative_ok_columns` parameter on `Pipeline.run()`
**Tests to update:** `tests/test_comprehensive.py` — `TestAnomalyDetection`
**Label:** `enhancement` `good first issue`

---

## Previously Fixed Bugs (v0.0.3) — Closed

| Bug | Status | Fixed in |
|-----|--------|----------|
| A — Duplicate table analysis loop | ✅ Fixed | `parseiq/pipeline.py` |
| B — Attribute context bleeding between tables | ✅ Fixed | `parseiq/file_loader/loader.py` |
| C — Nested objects kept as blobs | ✅ Fixed | `parseiq/file_loader/loader.py` |
| D — Excel blob columns (unreadable wide cells) | ✅ Fixed | `parseiq/pipeline.py` |

See `CHANGELOG.md` → `[0.0.3]` for full details.

---

## Questions?
Open a [Discussion](https://github.com/ShriniwasAhirrao/ParseIQ/discussions) or an Issue.
