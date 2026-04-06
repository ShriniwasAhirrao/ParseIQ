"""
parseiq CLI entry point.

Commands
--------
  parseiq init                          Interactive first-time setup
  parseiq analyze <file>                Analyse a data file
  parseiq validate <file>              Quick check (file loads, table count, no full run)
  parseiq models                        List available LLM models
  parseiq version                       Print version and exit
  parseiq config                        Show current configuration

Examples
--------
  parseiq analyze data.json --no-llm
  parseiq analyze data.csv --output report/
  parseiq analyze data.json --llm-provider openai --llm-model gpt-4o
  parseiq analyze data.json --force --quiet
"""
from __future__ import annotations

import argparse
import os
import sys


# --────────────────────────────────────────────────────────────────────────────
# Helpers
# --────────────────────────────────────────────────────────────────────────────

def _get_api_key_from_env(provider: str = None) -> str | None:
    """Return the best API key for the given provider, falling back to any known key."""
    _provider_env = {
        'openrouter':  'OPENROUTER_API_KEY',
        'openai':      'OPENAI_API_KEY',
        'anthropic':   'ANTHROPIC_API_KEY',
        'claude':      'ANTHROPIC_API_KEY',
        'gemini':      'GEMINI_API_KEY',
        'perplexity':  'PERPLEXITY_API_KEY',
    }
    if provider and provider in _provider_env:
        key = os.getenv(_provider_env[provider])
        if key:
            return key
    # Fallback: first key found across all providers
    for env_var in _provider_env.values():
        key = os.getenv(env_var)
        if key:
            return key
    return None


def _print_banner():
    from parseiq import __version__
    print("=" * 55)
    print(f"  ParseIQ - AI-Powered Data Quality Agent  v{__version__}")
    print("=" * 55)


def _ask(prompt: str, default: str = "") -> str:
    try:
        val = input(prompt).strip()
        return val if val else default
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled.")
        sys.exit(0)


def _ask_choice(prompt: str, choices: list, default: str) -> str:
    display = "/".join(f"[{c}]" if c == default else c for c in choices)
    while True:
        val = _ask(f"{prompt} ({display}): ", default)
        if val in choices:
            return val
        print(f"  Please choose one of: {', '.join(choices)}")


def _save_env(key: str, value: str, env_file: str = ".env"):
    lines = []
    replaced = False
    if os.path.exists(env_file):
        with open(env_file, encoding="utf-8") as f:
            for line in f:
                if line.startswith(f"{key}="):
                    lines.append(f"{key}={value}\n")
                    replaced = True
                else:
                    lines.append(line)
    if not replaced:
        lines.append(f"{key}={value}\n")
    with open(env_file, "w", encoding="utf-8") as f:
        f.writelines(lines)


# --────────────────────────────────────────────────────────────────────────────
# Commands
# --────────────────────────────────────────────────────────────────────────────

def cmd_version(_args):
    from parseiq import __version__
    print(f"parseiq {__version__}")


def cmd_config(_args):
    from parseiq.config import Config
    _print_banner()
    key = _get_api_key_from_env()
    print("\nConfiguration Summary:")
    print("=" * 50)
    print(f"  Model      : {Config.MODEL_NAME}")
    print(f"  Max Tokens : {Config.LLM_SETTINGS['max_tokens']}")
    print(f"  Temperature: {Config.LLM_SETTINGS['temperature']}")
    print(f"  Timeout    : {Config.LLM_SETTINGS['timeout']}s")
    print(f"  .env file  : {'found' if os.path.exists('.env') else 'not found'}")
    print("\n  API keys detected:")
    _env_vars = [
        ("OPENROUTER_API_KEY",  "OpenRouter"),
        ("OPENAI_API_KEY",      "OpenAI"),
        ("ANTHROPIC_API_KEY",   "Anthropic/Claude"),
        ("GEMINI_API_KEY",      "Google Gemini"),
        ("PERPLEXITY_API_KEY",  "Perplexity"),
    ]
    any_key = False
    for env_var, label in _env_vars:
        k = os.getenv(env_var)
        if k:
            print(f"    {label:<20} SET ({k[:12]}...)")
            any_key = True
    if not any_key:
        print("    (none found — run 'parseiq init' to configure)")
    print("=" * 50)
    issues = Config.validate(require_llm_key=False)
    if issues:
        print("\nConfig issues:")
        for field, msg in issues.items():
            print(f"  {field}: {msg}")
    else:
        print("\nConfiguration OK.")
    print("\nTo change settings: parseiq init")


def cmd_models(_args):
    _print_banner()

    print("\n--OpenRouter (free tier, one account covers all models) --")
    print("  Sign up free: https://openrouter.ai")
    free_or = [
        ("nvidia/nemotron-3-super-120b-a12b:free",       "120B, strong reasoning  [recommended default]"),
        ("mistralai/mistral-small-3.1-24b-instruct:free","24B, fast responses"),
        ("google/gemma-3-27b-it:free",                   "27B, good structured output"),
        ("meta-llama/llama-3.3-70b-instruct:free",       "70B LLaMA, well-rounded"),
        ("deepseek/deepseek-r1:free",                    "Reasoning model, thorough"),
    ]
    for model, desc in free_or:
        print(f"    {model}")
        print(f"      {desc}")
    print("  Usage: parseiq analyze data.json --llm-provider openrouter --llm-model <model>")

    print("\n--OpenAI (OPENAI_API_KEY) --")
    openai_models = [
        ("gpt-4o",      "Best overall quality"),
        ("gpt-4o-mini", "Fast, cost-efficient"),
        ("gpt-4-turbo", "Large context, strong reasoning"),
    ]
    for model, desc in openai_models:
        print(f"    {model}  — {desc}")
    print("  Usage: parseiq analyze data.json --llm-provider openai --llm-model gpt-4o")
    print("         Set key: set OPENAI_API_KEY=sk-...")

    print("\n--Anthropic / Claude (ANTHROPIC_API_KEY) --")
    claude_models = [
        ("claude-opus-4-5",        "Most capable, best for complex analysis"),
        ("claude-sonnet-4-5",      "Balanced speed and quality  [recommended]"),
        ("claude-haiku-4-5",       "Fastest, lowest cost"),
        ("claude-3-5-sonnet-20241022", "Previous gen, still excellent"),
    ]
    for model, desc in claude_models:
        print(f"    {model}  — {desc}")
    print("  Usage: parseiq analyze data.json --llm-provider anthropic --llm-model claude-sonnet-4-5")
    print("         Set key: set ANTHROPIC_API_KEY=sk-ant-...")
    print("         Install:  pip install anthropic")

    print("\n--Google Gemini (GEMINI_API_KEY) --")
    gemini_models = [
        ("gemini-1.5-pro",   "Large context (1M tokens), multimodal"),
        ("gemini-1.5-flash", "Fast, cost-efficient"),
        ("gemini-2.0-flash", "Latest, fast generation"),
    ]
    for model, desc in gemini_models:
        print(f"    {model}  — {desc}")
    print("  Usage: parseiq analyze data.json --llm-provider gemini --llm-model gemini-1.5-pro")
    print("         Set key: set GEMINI_API_KEY=AIza...")
    print("         Install:  pip install google-generativeai")
    print("         Get free key: https://aistudio.google.com/app/apikey")

    print("\n--Perplexity (PERPLEXITY_API_KEY) --")
    perplexity_models = [
        ("llama-3.1-sonar-large-128k-online", "Online search + reasoning"),
        ("llama-3.1-sonar-small-128k-online", "Faster online model"),
        ("llama-3.1-8b-instruct",             "Offline, lightweight"),
    ]
    for model, desc in perplexity_models:
        print(f"    {model}  — {desc}")
    print("  Usage: parseiq analyze data.json --llm-provider perplexity --llm-model llama-3.1-sonar-large-128k-online")
    print("         Set key: set PERPLEXITY_API_KEY=pplx-...")

    print("\n--Ollama (local, no API key needed) --")
    local = [
        ("llama3",   "Meta LLaMA 3 8B"),
        ("mistral",  "Mistral 7B"),
        ("phi3",     "Microsoft Phi-3 Mini (very fast)"),
        ("gemma2",   "Google Gemma 2 9B"),
    ]
    for model, desc in local:
        print(f"    {model}  — {desc}")
    print("  Usage: parseiq analyze data.json --llm-provider ollama --llm-model llama3")
    print("         Start server first: ollama serve")

    print("\n--Credit exhaustion? --")
    print("  ParseIQ detects 402 errors and suggests free alternatives automatically.")
    print("  Or run without LLM at any time:")
    print("    parseiq analyze data.json --no-llm")


def cmd_init(_args):
    _print_banner()
    print("\nFirst-time setup — press Enter to keep any default shown in brackets.\n")

    # --Choose provider --
    print("  Supported LLM providers:")
    print("    1. openrouter  — free tier available, access to 100+ models (recommended)")
    print("    2. openai      — OpenAI GPT-4o, GPT-4o-mini  (OPENAI_API_KEY)")
    print("    3. anthropic   — Claude models  (ANTHROPIC_API_KEY, pip install anthropic)")
    print("    4. gemini      — Google Gemini  (GEMINI_API_KEY, pip install google-generativeai)")
    print("    5. perplexity  — Perplexity AI  (PERPLEXITY_API_KEY)")
    print("    6. ollama      — Local models, no key needed  (ollama serve)")
    print("    7. Skip LLM setup")
    provider_choice = _ask("  Choose provider [1-7] (default 1): ", "1")

    _provider_info = {
        "1": ("openrouter",  "OPENROUTER_API_KEY",  "https://openrouter.ai  (sign up → Keys tab)"),
        "2": ("openai",      "OPENAI_API_KEY",      "https://platform.openai.com/api-keys"),
        "3": ("anthropic",   "ANTHROPIC_API_KEY",   "https://console.anthropic.com/settings/keys"),
        "4": ("gemini",      "GEMINI_API_KEY",      "https://aistudio.google.com/app/apikey  (free)"),
        "5": ("perplexity",  "PERPLEXITY_API_KEY",  "https://www.perplexity.ai/settings/api"),
        "6": ("ollama",      None,                  None),
        "7": (None,          None,                  None),
    }
    provider_name, env_var, key_url = _provider_info.get(provider_choice, (None, None, None))

    api_key = None
    if provider_name == "ollama":
        print("  Ollama: make sure 'ollama serve' is running locally.")
        print("  Start server and install a model:  ollama pull llama3")
        _save_env("PARSEIQ_PROVIDER", "ollama")
    elif provider_name and env_var:
        existing_key = _get_api_key_from_env(provider_name)
        if existing_key:
            print(f"  {env_var} already set ({existing_key[:12]}...)")
            change = _ask_choice("  Change it?", ["yes", "no"], "no")
            if change == "no":
                api_key = existing_key
            else:
                api_key = _ask(f"  Paste your {provider_name} API key: ")
        else:
            if key_url:
                print(f"  Get your API key at: {key_url}")
            api_key = _ask(f"  Paste your {provider_name} API key (or Enter to skip): ")

        if api_key:
            _save_env(env_var, api_key)
            _save_env("PARSEIQ_PROVIDER", provider_name)
            print(f"  Saved to .env")

    # --Model selection --
    print()
    _model_menus = {
        "openrouter": [
            ("1", "nvidia/nemotron-3-super-120b-a12b:free",        "Best free model (recommended)"),
            ("2", "mistralai/mistral-small-3.1-24b-instruct:free", "Fast free model"),
            ("3", "meta-llama/llama-3.3-70b-instruct:free",        "LLaMA 70B free"),
            ("4", "openai/gpt-4o",                                  "GPT-4o via OpenRouter (paid)"),
            ("5", "anthropic/claude-3-5-sonnet",                    "Claude via OpenRouter (paid)"),
        ],
        "openai": [
            ("1", "gpt-4o",      "Best quality"),
            ("2", "gpt-4o-mini", "Fast, cost-efficient"),
            ("3", "gpt-4-turbo", "Large context"),
        ],
        "anthropic": [
            ("1", "claude-sonnet-4-5",           "Recommended — balanced speed/quality"),
            ("2", "claude-opus-4-5",             "Most capable"),
            ("3", "claude-haiku-4-5",            "Fastest, lowest cost"),
            ("4", "claude-3-5-sonnet-20241022",  "Previous gen, still excellent"),
        ],
        "gemini": [
            ("1", "gemini-1.5-pro",   "Large context, multimodal"),
            ("2", "gemini-1.5-flash", "Fast, cost-efficient"),
            ("3", "gemini-2.0-flash", "Latest flash model"),
        ],
        "perplexity": [
            ("1", "llama-3.1-sonar-large-128k-online", "Online search + reasoning"),
            ("2", "llama-3.1-sonar-small-128k-online", "Faster online model"),
        ],
        "ollama": [
            ("1", "llama3",   "LLaMA 3 8B"),
            ("2", "mistral",  "Mistral 7B"),
            ("3", "phi3",     "Phi-3 Mini (very fast)"),
            ("4", "gemma2",   "Gemma 2 9B"),
        ],
    }

    chosen_model = ""
    if provider_name and provider_name in _model_menus:
        print(f"  Available {provider_name} models:")
        menu = _model_menus[provider_name]
        for num, model_id, desc in menu:
            default_tag = "  [default]" if num == "1" else ""
            print(f"    {num}. {model_id}  — {desc}{default_tag}")
        last = str(len(menu) + 1)
        print(f"    {last}. Keep current default")
        choice = _ask(f"  Choose model [1-{last}] (default 1): ", "1")
        model_map = {num: mid for num, mid, _ in menu}
        chosen_model = model_map.get(choice, "")
        if chosen_model:
            _save_env("PARSEIQ_MODEL", chosen_model)
            print(f"  Model set to: {chosen_model}")

    # --Output dir --
    print()
    out = _ask("  Default output directory [output]: ", "output")
    if out and out != "output":
        _save_env("PARSEIQ_OUTPUT_DIR", out)

    # --Test connection --
    if api_key and provider_name not in (None, "ollama"):
        print()
        test = _ask_choice("  Test API connection now?", ["yes", "no"], "yes")
        if test == "yes":
            print("  Testing connection...")
            try:
                from parseiq.config import Config
                from parseiq.step2_llm_enricher.llm_agent import LLMEnricher
                cfg = Config.get_llm_config()
                cfg["api_key"] = api_key
                if chosen_model:
                    cfg["model"] = chosen_model
                enricher = LLMEnricher(cfg)
                enricher._provider = provider_name
                ok = enricher.test_connection()
                if ok:
                    print("  Connection OK — LLM is reachable.")
                else:
                    print("  Connection FAILED. Check your API key.")
            except Exception as e:
                print(f"  Connection error: {e}")

    print()
    print("Setup complete. Run:")
    print(f"  parseiq analyze <your-file.json>")
    if provider_name:
        print(f"  parseiq analyze <your-file.json> --llm-provider {provider_name}")
    print("  parseiq analyze <your-file.json> --no-llm   (skip AI, always works)")


def cmd_validate(args):
    """Quick file check — loads the file, prints table summary, no full analysis."""
    if not os.path.exists(args.file):
        print(f"Error: file not found: {args.file}")
        sys.exit(1)

    print(f"Validating: {args.file}")
    try:
        from parseiq.connectors.file import load
        tables = load(args.file)
        total = sum(len(r) for r in tables.values())
        print(f"  Status    : OK")
        print(f"  Tables    : {len(tables)}")
        print(f"  Records   : {total:,}")
        for name, rows in tables.items():
            cols = list(rows[0].keys()) if rows else []
            print(f"  [{name}]  {len(rows):,} rows  {len(cols)} columns")
            if cols:
                print(f"    Columns : {', '.join(cols[:8])}{'...' if len(cols) > 8 else ''}")
        print("\nFile is valid. Run the full analysis with:")
        print(f"  parseiq analyze {args.file} --no-llm")
    except Exception as e:
        print(f"  Status    : FAILED")
        print(f"  Error     : {e}")
        sys.exit(1)


def cmd_analyze(args):
    if not os.path.exists(args.file):
        print(f"Error: file not found: {args.file}")
        sys.exit(1)

    # --Resolve API key --
    llm_api_key = args.llm_api_key or _get_api_key_from_env(args.llm_provider)

    use_llm = not args.no_llm
    # Ollama doesn't need an API key — use a placeholder so key checks pass
    if use_llm and args.llm_provider == 'ollama' and not llm_api_key:
        llm_api_key = 'ollama'  # Ollama ignores the Authorization header

    if use_llm and not llm_api_key:
        print("No API key found for provider:", args.llm_provider)
        print("  Option 1: parseiq init                    (interactive setup)")
        print("  Option 2: set the appropriate env var:")
        print("              OPENROUTER_API_KEY / OPENAI_API_KEY /")
        print("              ANTHROPIC_API_KEY / GEMINI_API_KEY / PERPLEXITY_API_KEY")
        print("  Option 3: parseiq analyze <file> --no-llm   (skip AI)")
        print()
        if sys.stdin.isatty():
            choice = _ask_choice("Run without LLM now?", ["yes", "no"], "yes")
            if choice == "yes":
                use_llm = False
            else:
                sys.exit(1)
        else:
            print("Running in local mode (--no-llm).")
            use_llm = False

    # --Interactive model selection if LLM but no model specified --
    llm_model = args.llm_model
    _provider_defaults = {
        'openrouter':  'nvidia/nemotron-3-super-120b-a12b:free',
        'openai':      'gpt-4o-mini',
        'anthropic':   'claude-sonnet-4-5',
        'claude':      'claude-sonnet-4-5',
        'gemini':      'gemini-1.5-flash',
        'perplexity':  'llama-3.1-sonar-large-128k-online',
        'azure':       'gpt-4o',
        'ollama':      'llama3',
    }
    if use_llm and not llm_model and sys.stdin.isatty() and not args.quiet:
        from parseiq.config import Config
        default_model = _provider_defaults.get(args.llm_provider, Config.MODEL_NAME)
        print(f"\nUsing model: {default_model}")
        change = _ask_choice("Use a different model?", ["yes", "no"], "no")
        if change == "yes":
            print(f"  Common {args.llm_provider} models (run 'parseiq models' for full list):")
            _quick_models = {
                'openrouter':  ['nvidia/nemotron-3-super-120b-a12b:free',
                                'mistralai/mistral-small-3.1-24b-instruct:free',
                                'meta-llama/llama-3.3-70b-instruct:free'],
                'openai':      ['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo'],
                'anthropic':   ['claude-opus-4-5', 'claude-sonnet-4-5', 'claude-haiku-4-5'],
                'claude':      ['claude-opus-4-5', 'claude-sonnet-4-5', 'claude-haiku-4-5'],
                'gemini':      ['gemini-1.5-pro', 'gemini-1.5-flash', 'gemini-2.0-flash'],
                'perplexity':  ['llama-3.1-sonar-large-128k-online', 'llama-3.1-8b-instruct'],
                'ollama':      ['llama3', 'mistral', 'phi3', 'gemma2'],
            }
            for m in _quick_models.get(args.llm_provider, []):
                print(f"    {m}")
            llm_model = _ask(f"  Model name [{default_model}]: ", default_model)

    if not args.quiet:
        _print_banner()
        print(f"\nFile     : {args.file}")
        print(f"Output   : {args.output}/")
        print(f"LLM mode : {'enabled (' + (args.llm_provider) + ')' if use_llm else 'disabled (local only)'}")
        if use_llm and llm_model:
            print(f"Model    : {llm_model}")
        if args.force:
            print(f"Force    : yes (incremental cache ignored)")
        print()

    from parseiq import Pipeline
    result = Pipeline.from_file(args.file, output_dir=args.output).run(
        llm=use_llm,
        llm_provider=args.llm_provider,
        llm_api_key=llm_api_key,
        llm_model=llm_model,
        llm_base_url=args.llm_base_url,
        force=args.force,
    )

    if not args.quiet:
        print("\n" + "=" * 55)
        print("ANALYSIS COMPLETE")
        print("=" * 55)
        print(f"  Tables analysed : {len(result.tables)}")
        print(f"  Total records   : {result.pipeline_info.get('total_records', 0):,}")
        print(f"  Avg quality     : {result.overall_quality_score}/100")
        print(f"  Total anomalies : {result.total_anomalies}")
        print(f"  LLM grade       : {result.llm_grade or 'N/A (local mode)'}")
        print(f"  Output folder   : {args.output}/")
        print(f"  Files written   : {len(result.output_files)}")
        print()
        print("Per-table quality scores:")
        for tname, score in result.quality_scores.items():
            bar = "█" * int(score / 10) + "░" * (10 - int(score / 10))
            flag = "  ⚠" if score < 60 else ("  ✓" if score >= 90 else "")
            print(f"  {tname:<28} {bar} {score:5.1f}/100{flag}")
        print()
        if result.total_anomalies > 0:
            print(f"Anomalies detected in these tables:")
            for tname, cols in result.anomalies.items():
                if cols:
                    for col, flags in cols.items():
                        print(f"  [{tname}.{col}] {', '.join(flags)}")
        print()
        xlsx_path = os.path.abspath(os.path.join(args.output, "complete_data_analysis.xlsx"))
        print(f"For a more detailed report, refer to:")
        print(f"  {xlsx_path}")

    # Machine-readable exit code: 0 = OK, 1 = quality below threshold
    if args.fail_under and result.overall_quality_score < args.fail_under:
        sys.exit(1)


# --────────────────────────────────────────────────────────────────────────────
# Argument parser
# --────────────────────────────────────────────────────────────────────────────

def main():
    # Ensure UTF-8 output on Windows consoles that default to cp1252
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except AttributeError:
            pass  # Python < 3.7

    parser = argparse.ArgumentParser(
        prog="parseiq",
        description="ParseIQ — AI-powered data quality analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  parseiq init                                   First-time setup\n"
            "  parseiq analyze data.json --no-llm            Local mode, no API key needed\n"
            "  parseiq analyze data.csv --output report/     CSV with custom output folder\n"
            "  parseiq analyze data.json                     With LLM (needs API key)\n"
            "  parseiq validate data.json                    Quick file check\n"
            "  parseiq models                                 List available LLM models\n"
            "  parseiq config                                 Show current settings\n"
        ),
    )
    sub = parser.add_subparsers(dest="command", metavar="command")

    # --init --
    sub.add_parser("init", help="Interactive first-time setup (API key, model, output dir)")

    # --version --
    sub.add_parser("version", help="Print version and exit")

    # --config --
    sub.add_parser("config", help="Show current configuration and environment")

    # --models --
    sub.add_parser("models", help="List available LLM models (free, paid, local)")

    # --validate --
    val = sub.add_parser("validate", help="Quick file check — tables/columns/record count, no full analysis")
    val.add_argument("file", help="Path to input file")

    # --analyze --
    analyze = sub.add_parser("analyze", help="Analyse a data file for quality issues")
    analyze.add_argument("file", help="Path to input file (JSON, CSV, XML, Excel)")
    analyze.add_argument("--output", "-o", default="output",
                         help="Output directory (default: output/)")
    analyze.add_argument("--no-llm", action="store_true",
                         help="Skip LLM enrichment — pure local mode, no API key needed")
    analyze.add_argument("--llm-provider", default="openrouter",
                         choices=["openrouter", "openai", "anthropic", "claude",
                                  "gemini", "perplexity", "azure", "ollama"],
                         help="LLM provider (default: openrouter)")
    analyze.add_argument("--llm-model", default=None,
                         help="Model name, e.g. gpt-4o, llama3, mistral")
    analyze.add_argument("--llm-api-key", default=None,
                         help="API key (overrides OPENROUTER_API_KEY env var)")
    analyze.add_argument("--llm-base-url", default=None,
                         help="Custom base URL for Azure OpenAI or local Ollama")
    analyze.add_argument("--force", action="store_true",
                         help="Reprocess all tables even if unchanged (ignore cache)")
    analyze.add_argument("--quiet", "-q", action="store_true",
                         help="Suppress all output except errors (useful in scripts/CI)")
    analyze.add_argument("--fail-under", type=float, default=None, metavar="SCORE",
                         help="Exit with code 1 if avg quality score is below SCORE (CI gate)")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    dispatch = {
        "init":     cmd_init,
        "version":  cmd_version,
        "config":   cmd_config,
        "models":   cmd_models,
        "validate": cmd_validate,
        "analyze":  cmd_analyze,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
