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


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _get_api_key_from_env() -> str | None:
    return (
        os.getenv("OPENROUTER_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("ANTHROPIC_API_KEY")
    )


def _print_banner():
    print("=" * 55)
    print("  ParseIQ — AI-Powered Data Quality Agent  v0.0.1")
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


# ──────────────────────────────────────────────────────────────────────────────
# Commands
# ──────────────────────────────────────────────────────────────────────────────

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
    print(f"  API Key    : {'SET (' + key[:12] + '...)' if key else 'NOT SET'}")
    print(f"  .env file  : {'found' if os.path.exists('.env') else 'not found'}")
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
    print("\nFREE models (no cost — require OpenRouter account):")
    free = [
        ("nvidia/nemotron-3-super-120b-a12b:free", "120B param, strong reasoning  [default]"),
        ("mistralai/mistral-small-3.1-24b-instruct:free", "24B, fast responses"),
        ("google/gemma-3-27b-it:free",              "27B, good for structured output"),
        ("meta-llama/llama-3.3-70b-instruct:free",  "70B LLaMA, well-rounded"),
        ("deepseek/deepseek-r1:free",               "Reasoning model, slower but thorough"),
    ]
    for model, desc in free:
        print(f"  {model}")
        print(f"    {desc}")

    print("\nPAID models (need credits on your provider account):")
    paid = [
        ("openai/gpt-4o",          "openai",     "Best overall quality"),
        ("openai/gpt-4o-mini",     "openai",     "Fast, cheap, good quality"),
        ("anthropic/claude-3-5-sonnet", "openrouter", "Excellent at structured analysis"),
        ("google/gemini-pro-1.5",  "openrouter", "Large context, multimodal"),
    ]
    for model, provider, desc in paid:
        print(f"  {model}  [{provider}]")
        print(f"    {desc}")

    print("\nLocal models (Ollama — no API key needed):")
    local = [
        ("llama3",    "Meta LLaMA 3 8B"),
        ("mistral",   "Mistral 7B"),
        ("phi3",      "Microsoft Phi-3 Mini (very fast)"),
    ]
    for model, desc in local:
        print(f"  {model}")
        print(f"    {desc}")

    print("\nTo use a model:")
    print("  parseiq analyze data.json --llm-provider openrouter --llm-model <model-id>")
    print("  parseiq analyze data.json --llm-provider ollama --llm-model llama3")


def cmd_init(_args):
    _print_banner()
    print("\nFirst-time setup — press Enter to keep any default shown in brackets.\n")

    # ── API Key ──
    existing_key = _get_api_key_from_env()
    if existing_key:
        print(f"  API key already set in environment ({existing_key[:12]}...)")
        change = _ask_choice("  Change it?", ["yes", "no"], "no")
        if change == "no":
            api_key = existing_key
        else:
            api_key = _ask("  Paste your new API key: ")
    else:
        print("  Get a FREE key at: https://openrouter.ai  (sign up → Keys tab)")
        api_key = _ask("  Paste your OpenRouter API key (or press Enter to skip LLM): ")

    if api_key:
        _save_env("OPENROUTER_API_KEY", api_key)
        print("  Saved to .env")

    # ── Model ──
    print()
    print("  Available free models:")
    print("    1. nvidia/nemotron-3-super-120b-a12b:free  (default, best quality)")
    print("    2. mistralai/mistral-small-3.1-24b-instruct:free  (faster)")
    print("    3. meta-llama/llama-3.3-70b-instruct:free")
    print("    4. Keep default")
    choice = _ask("  Choose model [1-4] (default 1): ", "1")
    model_map = {
        "1": "nvidia/nemotron-3-super-120b-a12b:free",
        "2": "mistralai/mistral-small-3.1-24b-instruct:free",
        "3": "meta-llama/llama-3.3-70b-instruct:free",
        "4": "",
    }
    chosen_model = model_map.get(choice, "")
    if chosen_model:
        _save_env("PARSEIQ_MODEL", chosen_model)
        print(f"  Model set to: {chosen_model}")

    # ── Output dir ──
    print()
    out = _ask("  Default output directory [output]: ", "output")
    if out and out != "output":
        _save_env("PARSEIQ_OUTPUT_DIR", out)

    # ── Test connection ──
    if api_key:
        print()
        test = _ask_choice("  Test API connection now?", ["yes", "no"], "yes")
        if test == "yes":
            print("  Testing connection...")
            try:
                os.environ["OPENROUTER_API_KEY"] = api_key
                from parseiq.config import Config
                from parseiq.step2_llm_enricher.llm_agent import LLMEnricher
                cfg = Config.get_llm_config()
                cfg["api_key"] = api_key
                enricher = LLMEnricher(cfg)
                ok = enricher.test_connection()
                if ok:
                    print("  Connection OK — LLM is reachable.")
                else:
                    print("  Connection FAILED. Check your API key.")
            except Exception as e:
                print(f"  Connection error: {e}")

    print()
    print("Setup complete. Run:")
    print("  parseiq analyze <your-file.json>")
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

    # ── Resolve API key ──
    llm_api_key = args.llm_api_key or _get_api_key_from_env()

    use_llm = not args.no_llm
    if use_llm and not llm_api_key:
        print("No API key found.")
        print("  Option 1: parseiq init              (interactive setup)")
        print("  Option 2: set OPENROUTER_API_KEY=sk-or-v1-...  (env var)")
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

    # ── Interactive model selection if LLM but no model specified ──
    llm_model = args.llm_model
    if use_llm and not llm_model and sys.stdin.isatty() and not args.quiet:
        from parseiq.config import Config
        default_model = Config.MODEL_NAME
        print(f"\nUsing model: {default_model}")
        change = _ask_choice("Use a different model?", ["yes", "no"], "no")
        if change == "yes":
            print("  Common choices:")
            print("    nvidia/nemotron-3-super-120b-a12b:free")
            print("    mistralai/mistral-small-3.1-24b-instruct:free")
            print("    meta-llama/llama-3.3-70b-instruct:free")
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
        print(f"  Total records   : {sum(1 for _ in result.tables):,}" )
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
        print(f"Open the Excel report: {args.output}/complete_data_analysis.xlsx")

    # Machine-readable exit code: 0 = OK, 1 = quality below threshold
    if args.fail_under and result.overall_quality_score < args.fail_under:
        sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# Argument parser
# ──────────────────────────────────────────────────────────────────────────────

def main():
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

    # ── init ──
    sub.add_parser("init", help="Interactive first-time setup (API key, model, output dir)")

    # ── version ──
    sub.add_parser("version", help="Print version and exit")

    # ── config ──
    sub.add_parser("config", help="Show current configuration and environment")

    # ── models ──
    sub.add_parser("models", help="List available LLM models (free, paid, local)")

    # ── validate ──
    val = sub.add_parser("validate", help="Quick file check — tables/columns/record count, no full analysis")
    val.add_argument("file", help="Path to input file")

    # ── analyze ──
    analyze = sub.add_parser("analyze", help="Analyse a data file for quality issues")
    analyze.add_argument("file", help="Path to input file (JSON, CSV, XML, Excel)")
    analyze.add_argument("--output", "-o", default="output",
                         help="Output directory (default: output/)")
    analyze.add_argument("--no-llm", action="store_true",
                         help="Skip LLM enrichment — pure local mode, no API key needed")
    analyze.add_argument("--llm-provider", default="openrouter",
                         choices=["openrouter", "openai", "azure", "ollama"],
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
