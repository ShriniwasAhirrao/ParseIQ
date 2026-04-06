"""
ParseIQ configuration.

Priority order (highest → lowest):
  1. Keyword params passed directly to Pipeline.run()
  2. Environment variables
  3. .env file (loaded automatically when python-dotenv is installed)
  4. Built-in defaults

The root-level config.py re-exports this class for backward compatibility.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

# Load .env file if python-dotenv is available (optional dependency)
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv()
except ImportError:
    pass


class Config:
    """Centralised configuration for ParseIQ.

    Sensitive values are read exclusively from environment variables — never
    hardcoded.  Call ``Config.validate()`` to get a human-readable list of any
    missing required settings before running.
    """

    # LLM / API
    OPENROUTER_API_KEY: Optional[str] = os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_URL: str = "https://openrouter.ai/api/v1/chat/completions"
    MODEL_NAME: str = os.getenv("PARSEIQ_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")
    FALLBACK_MODELS: List[str] = []

    SITE_URL: str = os.getenv("SITE_URL", "http://localhost:3000")
    SITE_NAME: str = os.getenv("SITE_NAME", "ParseIQ")

    # Provider base URLs
    PROVIDER_BASE_URLS: Dict[str, str] = {
        "openrouter": "https://openrouter.ai/api/v1",
        "openai": "https://api.openai.com/v1",
        "ollama": "http://localhost:11434/v1",
        # azure: user must supply llm_base_url explicitly
    }

    # Data quality thresholds
    ANOMALY_THRESHOLDS: Dict[str, Any] = {
        "high_null_rate": 30.0,
        "avg_length_too_short": 2,
        "avg_length_too_long": 500,
        "z_score_threshold": 3.0,
        "iqr_multiplier": 1.5,
        "min_unique_ratio": 0.1,
        "max_null_percentage": 50.0,
    }

    STATISTICAL_SETTINGS: Dict[str, Any] = {
        "calculate_percentiles": [25, 50, 75, 90, 95, 99],
        "regex_patterns": {
            "email": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
            "phone": r"^\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}$",
            "date_iso": r"^\d{4}-\d{2}-\d{2}$",
            "url": r"^https?://[^\s]+$",
            "ip_address": r"^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$",
        },
    }

    LLM_SETTINGS: Dict[str, Any] = {
        "max_tokens": 4096,
        "temperature": 0.1,
        "top_p": 0.9,
        "timeout": 240,
        "retry_attempts": 3,
        "retry_delay": 2,
    }

    FILE_SETTINGS: Dict[str, Any] = {
        "max_file_size_mb": 100,
        "supported_formats": [".json", ".csv", ".xml", ".xlsx"],
        "encoding": "utf-8",
        "csv_delimiter_detection": True,
    }

    LOGGING_SETTINGS: Dict[str, Any] = {
        "log_level": "INFO",
        "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "log_to_file": True,
        "log_to_console": True,
        "max_log_files": 10,
        "max_log_size_mb": 50,
    }

    @classmethod
    def validate(cls, *, require_llm_key: bool = True) -> Dict[str, str]:
        """Return ``{field: problem}`` for any missing / invalid config.

        Pass ``require_llm_key=False`` when running in pure local mode (llm=False).

        Example::

            issues = Config.validate()
            if issues:
                for field, msg in issues.items():
                    print(f"  {field}: {msg}")
        """
        issues: Dict[str, str] = {}
        if require_llm_key and not cls.OPENROUTER_API_KEY:
            issues["api_key"] = (
                "OPENROUTER_API_KEY not set. "
                "Export it: export OPENROUTER_API_KEY='sk-or-v1-...' "
                "or add it to a .env file."
            )
        if not cls.MODEL_NAME:
            issues["PARSEIQ_MODEL"] = "MODEL_NAME is empty."
        if cls.LLM_SETTINGS.get("max_tokens", 0) <= 0:
            issues["max_tokens"] = "Must be a positive integer."
        if cls.LLM_SETTINGS.get("timeout", 0) <= 0:
            issues["timeout"] = "Must be a positive number (seconds)."
        return issues

    # Kept for backward compat with existing tests
    @classmethod
    def validate_config(cls) -> Dict[str, str]:
        return cls.validate(require_llm_key=True)

    @classmethod
    def get_api_key(cls) -> str:
        """Return the API key, or prompt interactively (dev-only fallback).

        In library / programmatic usage, pass ``llm_api_key`` to
        ``Pipeline.run()`` instead of calling this.
        """
        api_key = cls.OPENROUTER_API_KEY
        if not api_key:
            print("\nOpenRouter API key not found in environment variables.")
            print("  1. Set OPENROUTER_API_KEY env var (recommended)")
            print("  2. Enter it now (dev only)")
            choice = input("\nChoice (1 or 2): ").strip()
            if choice == "2":
                api_key = input("Enter your OpenRouter API key: ").strip()
                if not api_key.startswith("sk-or-v1-"):
                    print("Warning: API key format may be incorrect")
                    if input("Continue? (y/N): ").strip().lower() != "y":
                        sys.exit(1)
            else:
                print("Please set OPENROUTER_API_KEY and restart.")
                sys.exit(1)
        return api_key

    @classmethod
    def get_llm_config(cls) -> Dict[str, Any]:
        """Return a complete LLM config dict (used by LLMEnricher)."""
        return {
            "api_key": cls.get_api_key(),
            "base_url": "https://openrouter.ai/api/v1",
            "model": cls.MODEL_NAME,
            "fallback_models": cls.FALLBACK_MODELS,
            "max_tokens": cls.LLM_SETTINGS["max_tokens"],
            "temperature": cls.LLM_SETTINGS["temperature"],
            "top_p": cls.LLM_SETTINGS["top_p"],
            "timeout": cls.LLM_SETTINGS["timeout"],
            "retry_attempts": cls.LLM_SETTINGS.get("retry_attempts", 3),
            "retry_delay": cls.LLM_SETTINGS.get("retry_delay", 2),
            "site_url": cls.SITE_URL,
            "site_name": cls.SITE_NAME,
        }

    @classmethod
    def print_config_summary(cls) -> None:
        print("\nConfiguration Summary:")
        print("=" * 50)
        print(f"  Model: {cls.MODEL_NAME}")
        print(f"  Max Tokens: {cls.LLM_SETTINGS['max_tokens']}")
        print(f"  Temperature: {cls.LLM_SETTINGS['temperature']}")
        print(f"  Timeout: {cls.LLM_SETTINGS['timeout']}s")
        print(f"  API Key: {'✅ Set' if cls.OPENROUTER_API_KEY else '❌ Not Set'}")
        if cls.OPENROUTER_API_KEY:
            masked = cls.OPENROUTER_API_KEY[:10] + "..." + cls.OPENROUTER_API_KEY[-4:]
            print(f"  API Key Preview: {masked}")
        print("=" * 50)

    @classmethod
    def create_prompt_template_path(cls) -> str:
        """Return the absolute path to the bundled prompt template."""
        return os.path.join(os.path.dirname(__file__), "step2_llm_enricher", "prompt_template.txt")

    @classmethod
    def ensure_directories(cls, base: str = ".") -> List[str]:
        """Ensure runtime directories exist under *base*."""
        directories = ["input", "output", "logs", "debug_output", "step2_llm_enricher"]
        for d in directories:
            os.makedirs(os.path.join(base, d), exist_ok=True)
        return directories
