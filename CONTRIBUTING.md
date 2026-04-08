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

## Questions?
Open a [Discussion](https://github.com/ShriniwasAhirrao/ParseIQ/discussions) or an Issue.
