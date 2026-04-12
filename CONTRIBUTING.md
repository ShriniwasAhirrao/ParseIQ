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
4. Run the test suite — all 165 tests must pass:
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

All original Issues A–K and the LLM output bugs have been fixed (see `CHANGELOG.md`).
New issues welcome — open one at https://github.com/ShriniwasAhirrao/ParseIQ/issues.

### Potential areas for contribution

- **PDF report export** — generate a PDF version of the Excel quality report
- **Batch processing** — `parseiq analyze-all data/` for a folder of files
- **Cross-table FK violation detection** — flag orphaned `_ref_*` values
- **XML + Excel test coverage** — `_load_xml()` and `_load_excel()` have no dedicated tests
- **`conftest.py`** — shared test fixtures to reduce repetition across test files

---

## Previously Fixed Bugs — Closed

| Bug / Issue | Status | Fixed in |
|-------------|--------|----------|
| A — Duplicate table analysis loop | ✅ Fixed | v0.0.3 |
| B — Attribute context bleeding | ✅ Fixed | v0.0.3 |
| C — Nested objects kept as blobs | ✅ Fixed | v0.0.3 |
| D — Excel blob columns | ✅ Fixed | v0.0.3 |
| E — pip environment collision | ✅ Fixed | v0.0.4 |
| F — NEGATIVE_VALUES false positives | ✅ Fixed | v0.0.4 |
| G — Cross-level range violations | ✅ Fixed | v0.0.4 |
| H — Cross-table constraint validation | ✅ Fixed | v0.0.4 |
| I — Scale/domain violations | ✅ Fixed | v0.0.4 |
| J — Missing sibling dict key | ✅ Verified | v0.0.4 |
| K — Schema polymorphism false positives | ✅ Fixed | v0.0.5 |
| LLM mode N/A outputs + score=0 | ✅ Fixed | v0.0.6 |

See `CHANGELOG.md` for full details per version.

---

## Questions?
Open a [Discussion](https://github.com/ShriniwasAhirrao/ParseIQ/discussions) or an Issue.
