# Security Policy

## Supported Versions

| Version | Supported |
|---|---|
| 0.0.2 | Yes — current release |
| 0.0.1 | No |

## Reporting a Vulnerability

**Do not open a public issue for security vulnerabilities.**

Report security issues by opening a [GitHub Security Advisory](https://github.com/ShriniwasAhirrao/ParseIQ/security/advisories/new) (private disclosure).

Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

You will receive a response within 48 hours. If confirmed, a patch will be released as soon as possible.

## Security Notes

- ParseIQ runs locally — your data files never leave your machine unless you enable LLM mode
- In LLM mode, metadata (column names, sample statistics) is sent to your chosen LLM provider using **your own API key**
- No data is sent to any ParseIQ servers — there are none
- API keys are read from environment variables or `.env` files — never hardcoded
