"""
parseiq.pipeline — The public Pipeline API.

Quickstart::

    from parseiq import Pipeline

    # Local file, no LLM
    result = Pipeline("data.json").run(llm=False)

    # With LLM (BYOK)
    result = Pipeline("data.json").run(
        llm=True,
        llm_provider="openrouter",
        llm_api_key="sk-or-v1-...",
    )

    # Other sources
    result = Pipeline.from_s3("s3://bucket/data.json").run(llm=False)
    result = Pipeline.from_url("https://api.example.com/data").run(llm=False)

    # Alert rules
    from parseiq.alerts import slack_webhook
    result = Pipeline("data.json").run(
        llm=False,
        alert_rules={"employees.salary": {"negative_values": True}},
        on_alert=slack_webhook("https://hooks.slack.com/services/..."),
    )
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from .config import Config
from .result import PipelineResult

logger = logging.getLogger(__name__)


class Pipeline:
    """Analyse a dataset for data quality issues in a 3-step pipeline.

    Pass a file path directly or use a class-method constructor:

    .. code-block:: python

        Pipeline("data.json")
        Pipeline.from_file("data.json")
        Pipeline.from_url("https://api.example.com/data.json")
        Pipeline.from_s3("s3://bucket/data.json")
        Pipeline.from_postgres(conn, "SELECT * FROM orders")
        Pipeline.from_mongodb(conn, "customers")

    Each ``Pipeline`` instance is fully isolated — parallel runs with different
    ``output_dir`` values never conflict (Concurrency guarantee, item 6).
    """

    def __init__(self, source: str = None, *, output_dir: str = "output"):
        self._source_type = "file"
        self._source_arg: Any = source
        self._source_kwargs: Dict[str, Any] = {}
        self._output_dir = os.path.abspath(output_dir)

    # ------------------------------------------------------------------
    # Class-method constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_file(cls, path: str, *, output_dir: str = "output") -> "Pipeline":
        """Load from a local file (JSON, CSV, XML, Excel)."""
        p = cls(output_dir=output_dir); p._source_type = "file"; p._source_arg = path; return p

    @classmethod
    def from_url(cls, url: str, headers: Optional[Dict[str, str]] = None, *, output_dir: str = "output") -> "Pipeline":
        """Fetch data from an HTTP/HTTPS URL."""
        p = cls(output_dir=output_dir); p._source_type = "url"; p._source_arg = url
        p._source_kwargs = {"headers": headers}; return p

    @classmethod
    def from_s3(cls, s3_uri: str, *, aws_access_key_id=None, aws_secret_access_key=None,
                region_name=None, output_dir: str = "output") -> "Pipeline":
        """Download and analyse a file stored in Amazon S3."""
        p = cls(output_dir=output_dir); p._source_type = "s3"; p._source_arg = s3_uri
        p._source_kwargs = {"aws_access_key_id": aws_access_key_id,
                            "aws_secret_access_key": aws_secret_access_key, "region_name": region_name}
        return p

    @classmethod
    def from_postgres(cls, conn_string: str, query: str, table_name: str = "query_result",
                      *, output_dir: str = "output") -> "Pipeline":
        """Run a SQL query against PostgreSQL and analyse the result."""
        p = cls(output_dir=output_dir); p._source_type = "postgres"; p._source_arg = conn_string
        p._source_kwargs = {"query": query, "table_name": table_name}; return p

    @classmethod
    def from_mongodb(cls, conn_string: str, collection_name: str, database_name=None,
                     limit: int = 0, *, output_dir: str = "output") -> "Pipeline":
        """Read a MongoDB collection and analyse it."""
        p = cls(output_dir=output_dir); p._source_type = "mongodb"; p._source_arg = conn_string
        p._source_kwargs = {"collection_name": collection_name,
                            "database_name": database_name, "limit": limit}
        return p

    # ------------------------------------------------------------------
    # run()
    # ------------------------------------------------------------------

    def run(
        self,
        *,
        llm: bool = True,
        llm_provider: str = "openrouter",
        llm_api_key: Optional[str] = None,
        llm_model: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        force: bool = False,
        alert_rules: Optional[Dict[str, Dict[str, Any]]] = None,
        on_alert: Optional[Callable] = None,
    ) -> PipelineResult:
        """Execute the 3-step pipeline and return a :class:`PipelineResult`.

        Parameters
        ----------
        llm:
            ``True`` — call an LLM for business-level enrichment.
            ``False`` — pure local mode: no API call, data never leaves machine.
        llm_provider:
            ``"openrouter"`` | ``"openai"`` | ``"azure"`` | ``"ollama"``
        llm_api_key:
            Your API key — overrides ``OPENROUTER_API_KEY`` / ``OPENAI_API_KEY`` env vars.
        llm_model:
            Model name, e.g. ``"gpt-4o"``, ``"llama3"``.
        llm_base_url:
            Custom base URL for Azure OpenAI or self-hosted Ollama.
        force:
            ``True`` — reprocess every table even if the hash matches the last run.
            Default ``False`` (incremental mode: unchanged tables reuse cached Step-1 results).
        alert_rules:
            ``{"table"`` or ``"table.column": {rule_type: threshold}}``
        on_alert:
            Callback fired for every matched alert.
            Signature: ``(rule_key, table, column_or_metric, actual_value) → None``
        """
        self._configure_logging()
        os.makedirs(self._output_dir, exist_ok=True)

        t0 = time.time()
        created_files: List[str] = []

        # ── Load data ──────────────────────────────────────────────────────
        print(f"Loading data ({self._source_type})...")
        tables = self._load_data()
        print(f"  {len(tables)} table(s): {list(tables.keys())}")

        # ── Incremental state ──────────────────────────────────────────────
        state_path = os.path.join(self._output_dir, ".parseiq_state.json")
        state = _load_state(state_path)
        hashes = {name: _hash_table(rows) for name, rows in tables.items()}

        # ── STEP 1: Metadata extraction ────────────────────────────────────
        print("\nSTEP 1: Extracting metadata...")
        t1 = time.time()
        from .step1_metadata_extractor.extractor import MetadataExtractor
        extractor = MetadataExtractor()
        all_metadata: Dict[str, Any] = {}
        cached: List[str] = []

        for tname, rows in tables.items():
            if not force and _is_cached(tname, hashes[tname], state):
                print(f"  {tname}: unchanged — reusing cached result")
                all_metadata[tname] = state["tables"][tname]["metadata"]
                cached.append(tname)
                continue
            print(f"  {tname}: analysing ({len(rows)} records)...")
            meta = extractor.extract_metadata(rows)
            if isinstance(meta, dict) and "table_metadata" in meta:
                meta["table_metadata"]["table_name"] = tname
            all_metadata[tname] = meta

        total_records = sum(len(r) for r in tables.values())
        raw_metadata = {
            "tables": all_metadata,
            "summary": {
                "total_tables": len(tables), "total_records": total_records,
                "table_names": list(tables.keys()),
                "table_record_counts": {n: len(r) for n, r in tables.items()},
            },
            "dataset_overview": _build_dataset_overview(all_metadata),
            "pipeline_metadata": {
                "extraction_timestamp": datetime.now().isoformat(),
                "step1_duration": round(time.time() - t1, 2),
                "cached_tables": cached,
            },
        }
        raw_path = os.path.join(self._output_dir, "raw_metadata.json")
        _write_json(raw_metadata, raw_path)
        created_files.append(raw_path)
        print(f"  raw_metadata.json written  ({time.time() - t1:.1f}s)")

        # ── User-defined rules (Issues H, I) ──────────────────────────────
        rule_violations: List[Dict[str, Any]] = []
        if self._source_type == "file" and self._source_arg:
            rules_path = _find_rules_file(str(self._source_arg))
            if rules_path:
                print(f"  Applying rules: {os.path.basename(rules_path)}...")
                rules = _load_rules(rules_path)
                rule_violations = _apply_rules(rules, tables, raw_metadata)
                if rule_violations:
                    print(f"  {len(rule_violations)} rule violation(s) detected.")
                    raw_metadata["rule_violations"] = rule_violations

        # ── Alert rules (post Step 1) ──────────────────────────────────────
        alerts_fired: List[Dict[str, Any]] = []
        if alert_rules:
            from .alerts import evaluate_rules
            alerts_fired = evaluate_rules(alert_rules, raw_metadata, on_alert)
            if alerts_fired:
                print(f"  {len(alerts_fired)} alert(s) fired.")

        # ── STEP 2: LLM enrichment ─────────────────────────────────────────
        llm_insights: Optional[Dict[str, Any]] = None
        if llm:
            print(f"\nSTEP 2: LLM enrichment ({llm_provider})...")
            t2 = time.time()
            llm_insights = self._run_llm(raw_metadata, llm_provider=llm_provider,
                                         llm_api_key=llm_api_key, llm_model=llm_model,
                                         llm_base_url=llm_base_url)
            llm_path = os.path.join(self._output_dir, "llm_insights.json")
            _write_json(llm_insights, llm_path)
            created_files.append(llm_path)
            print(f"  llm_insights.json written  ({time.time() - t2:.1f}s)")
        else:
            print("\nSTEP 2: skipped (llm=False) — local mode.")
            llm_insights = _fallback_enrichment()

        # ── STEP 3: Output generation ──────────────────────────────────────
        print("\nSTEP 3: Writing outputs...")
        t3 = time.time()
        enriched = {
            "pipeline_info": {
                "timestamp": datetime.now().isoformat(),
                "source_type": self._source_type,
                "total_tables": len(tables),
                "total_records": total_records,
                "table_summary": raw_metadata["summary"]["table_record_counts"],
                "llm_used": llm,
                "llm_provider": llm_provider if llm else None,
                "total_duration": round(time.time() - t0, 2),
            },
            "raw_metadata": raw_metadata,
            "llm_insights": llm_insights,
        }
        enriched_path = os.path.join(self._output_dir, "enriched_metadata.json")
        _write_json(enriched, enriched_path)
        created_files.append(enriched_path)

        output_files = _generate_outputs(tables, raw_metadata, enriched, self._output_dir)
        created_files.extend(output_files)
        print(f"  Done.  ({time.time() - t3:.1f}s)")

        # ── Save incremental state ─────────────────────────────────────────
        _save_state(state_path, state, hashes, all_metadata)

        # ── Build PipelineResult ───────────────────────────────────────────
        quality_scores = _extract_quality_scores(all_metadata)
        anomalies = _extract_anomalies(all_metadata)
        duration = round(time.time() - t0, 1)
        avg_q = round(sum(quality_scores.values()) / max(len(quality_scores), 1), 1)
        print(f"\nDone in {duration}s  |  tables={len(tables)}  records={total_records}  avg_quality={avg_q}")

        return PipelineResult(
            tables=list(tables.keys()),
            quality_scores=quality_scores,
            anomalies=anomalies,
            output_files=created_files,
            llm_insights=llm_insights if llm else None,
            alerts_fired=alerts_fired,
            raw_metadata=raw_metadata,
            pipeline_info=enriched["pipeline_info"],
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_data(self) -> Dict[str, List[Dict]]:
        if self._source_type == "file":
            from .connectors.file import load; return load(self._source_arg)
        if self._source_type == "url":
            from .connectors.url import load; return load(self._source_arg, **self._source_kwargs)
        if self._source_type == "s3":
            from .connectors.s3 import load; return load(self._source_arg, **self._source_kwargs)
        if self._source_type == "postgres":
            from .connectors.postgres import load; return load(self._source_arg, **self._source_kwargs)
        if self._source_type == "mongodb":
            from .connectors.mongodb import load; return load(self._source_arg, **self._source_kwargs)
        raise ValueError(f"Unknown source type: {self._source_type!r}")

    def _run_llm(self, raw_metadata, *, llm_provider, llm_api_key, llm_model, llm_base_url):
        from .step2_llm_enricher.llm_agent import LLMEnricher
        cfg = {
            "api_key": llm_api_key or Config.OPENROUTER_API_KEY or "",
            "base_url": Config.PROVIDER_BASE_URLS.get(llm_provider, "https://openrouter.ai/api/v1"),
            "model": llm_model or Config.MODEL_NAME,
            "max_tokens": Config.LLM_SETTINGS["max_tokens"],
            "temperature": Config.LLM_SETTINGS["temperature"],
            "timeout": Config.LLM_SETTINGS["timeout"],
            "site_url": Config.SITE_URL,
            "site_name": Config.SITE_NAME,
            "debug": False,
            "prompt_template_path": Config.create_prompt_template_path(),
        }
        if llm_base_url:
            cfg["base_url"] = llm_base_url
        enricher = LLMEnricher(cfg)
        try:
            return enricher.enrich_metadata(raw_metadata, llm_provider=llm_provider,
                                            llm_api_key=llm_api_key, llm_model=llm_model,
                                            llm_base_url=llm_base_url)
        except Exception as exc:
            logger.warning("LLM enrichment failed (%s) — using fallback.", exc)
            print(f"  LLM failed ({exc.__class__.__name__}: {exc}) — falling back.")
            return _fallback_enrichment()

    @staticmethod
    def _configure_logging() -> None:
        if hasattr(sys.stdout, "reconfigure"):
            try: sys.stdout.reconfigure(encoding="utf-8")
            except Exception: pass
        root = logging.getLogger()
        if not root.handlers:
            os.makedirs("logs", exist_ok=True)
            fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            ch = logging.StreamHandler(sys.stdout); ch.setFormatter(fmt)
            fh = logging.FileHandler("logs/app.log", encoding="utf-8"); fh.setFormatter(fmt)
            root.setLevel(logging.INFO); root.addHandler(ch); root.addHandler(fh)


# ---------------------------------------------------------------------------
# MetadataEnrichmentAgent — backward-compat shim for existing tests/code
# ---------------------------------------------------------------------------

class MetadataEnrichmentAgent:
    """Thin shim so existing code that imports ``MetadataEnrichmentAgent`` still works.

    New code should use :class:`Pipeline` directly.
    """

    def __init__(self, debug: bool = True):
        Pipeline._configure_logging()
        self.config = Config()
        self.debug = debug
        from .file_loader.loader import FileLoader
        from .step1_metadata_extractor.extractor import MetadataExtractor
        from .step2_llm_enricher.llm_agent import LLMEnricher
        self.file_loader = FileLoader()
        self.metadata_extractor = MetadataExtractor()
        cfg = {
            "api_key": self.config.OPENROUTER_API_KEY or "",
            "base_url": "https://openrouter.ai/api/v1",
            "model": self.config.MODEL_NAME,
            "max_tokens": self.config.LLM_SETTINGS["max_tokens"],
            "temperature": self.config.LLM_SETTINGS["temperature"],
            "timeout": self.config.LLM_SETTINGS["timeout"],
            "site_url": self.config.SITE_URL,
            "site_name": self.config.SITE_NAME,
            "debug": debug,
            "prompt_template_path": self.config.create_prompt_template_path(),
        }
        self.llm_enricher = LLMEnricher(cfg)
        self.llm_connection_ok = True
        self.supported_models = {"default": self.config.MODEL_NAME}
        self.selected_model_key = "default"
        for d in ["output", "input", "logs", "debug_output"]:
            os.makedirs(d, exist_ok=True)

    def _create_fallback_enrichment(self, raw_metadata):
        return _fallback_enrichment()

    def run_pipeline(self, input_file_path: str, skip_llm: bool = False, selected_model: str = None):
        import glob as _g
        for ext in ("*.csv", "*.xlsx"):
            for f in _g.glob(os.path.join("output", ext)):
                try: os.remove(f)
                except OSError: pass

        t0 = time.time()
        tables = self.file_loader.load_file(input_file_path)
        total_records = sum(len(t) for t in tables.values())
        all_metadata: Dict[str, Any] = {}
        for tname, trows in tables.items():
            meta = self.metadata_extractor.extract_metadata(trows)
            if isinstance(meta, dict) and "table_metadata" in meta:
                meta["table_metadata"]["table_name"] = tname
            all_metadata[tname] = meta

        raw_metadata = {
            "tables": all_metadata,
            "summary": {
                "total_tables": len(tables), "total_records": total_records,
                "table_names": list(tables.keys()),
                "table_record_counts": {n: len(r) for n, r in tables.items()},
            },
            "dataset_overview": _build_dataset_overview(all_metadata),
            "pipeline_metadata": {"input_file": input_file_path},
        }
        _write_json(raw_metadata, "output/raw_metadata.json")

        model_name = selected_model or self.config.MODEL_NAME
        enriched_insights: Optional[Dict[str, Any]] = None
        if not skip_llm and self.llm_connection_ok:
            try:
                enriched_insights = self.llm_enricher.enrich_metadata(raw_metadata, model=model_name)
                _write_json(enriched_insights, "output/llm_insights.json")
            except Exception as exc:
                print(f"LLM enrichment failed: {exc} — using fallback")
                enriched_insights = _fallback_enrichment()
        else:
            enriched_insights = _fallback_enrichment()

        final_output = {
            "pipeline_info": {
                "timestamp": datetime.now().isoformat(),
                "input_file": input_file_path,
                "total_tables": len(tables),
                "total_records": total_records,
                "table_summary": raw_metadata["summary"]["table_record_counts"],
                "llm_used": not skip_llm and self.llm_connection_ok,
                "total_duration": round(time.time() - t0, 2),
                "model_used": model_name,
                "debug_mode": self.debug,
            },
            "raw_metadata": raw_metadata,
            "llm_insights": enriched_insights,
            "summary": self._generate_summary(raw_metadata, enriched_insights),
        }
        _write_json(final_output, "output/enriched_metadata.json")
        return final_output

    def _generate_summary(self, raw_metadata, llm_insights):
        qs = _extract_quality_scores(raw_metadata.get("tables", {}))
        avg = round(sum(qs.values()) / max(len(qs), 1), 2)
        anomalies = _extract_anomalies(raw_metadata.get("tables", {}))
        total_anomalies = sum(len(f) for cm in anomalies.values() for f in cm.values())
        top_issues: List[str] = []
        for tname, tmeta in raw_metadata.get("tables", {}).items():
            inner = tmeta.get("table_metadata", tmeta)
            for issue in inner.get("top_issues", [])[:3]:
                top_issues.append(f"[{tname}] {issue}")
        return {
            "total_attributes": sum(len(t.get("table_metadata", t).get("attributes", {}))
                                    for t in raw_metadata.get("tables", {}).values()),
            "total_anomalies": total_anomalies,
            "data_quality_score": avg,
            "top_issues": top_issues[:10],
            "llm_recommendations_count": len((llm_insights or {}).get("recommendations", [])),
            "processing_mode": "llm_enhanced" if llm_insights else "basic_fallback",
            "total_records": raw_metadata.get("summary", {}).get("total_records", 0),
            "multi_table_info": {
                "total_tables": len(raw_metadata.get("tables", {})),
                "table_names": raw_metadata.get("summary", {}).get("table_names", []),
                "table_record_counts": raw_metadata.get("summary", {}).get("table_record_counts", {}),
            },
        }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _fallback_enrichment() -> Dict[str, Any]:
    return {
        "overall_assessment": {
            "quality_grade": "C", "confidence_score": 50, "overall_score": 70,
            "production_readiness": "Needs Review", "primary_concerns": [],
            "key_strengths": ["Data structure is parseable"],
        },
        "critical_issues": [],
        "recommendations": [
            {"priority": "MEDIUM", "action": "Enable LLM analysis for full assessment.",
             "category": "process", "estimated_effort": "Low"},
        ],
        "risk_assessment": {"overall_risk_level": "MEDIUM"},
        "enrichment_metadata": {
            "timestamp": datetime.now().isoformat(),
            "model_used": "local_fallback", "llm_used": False,
        },
    }


def _describe_issue(flag: str, col: str, table: str, dtype: str,
                    null_pct: float, mn: Any, mx: Any,
                    unique_ratio: float, rec_count: int):
    """Return (description, business_impact, recommended_fix, effort) for an anomaly flag."""
    col_ref = f"'{col}' in '{table}'"

    if flag == "HIGH_NULL_RATE":
        desc   = f"{null_pct}% of values in {col_ref} are missing ({int(null_pct * rec_count / 100)} of {rec_count} records)"
        impact = ("High null rate will break aggregations, averages, and joins that depend on this column. "
                  "Downstream reports will silently undercount.")
        fix    = (f"1. Run: SELECT COUNT(*) FROM {table} WHERE {col} IS NULL  "
                  f"2. Determine if nulls are expected (optional field) or data gaps.  "
                  f"3. Add a NOT NULL constraint or default value at the source if required.  "
                  f"4. Document nullability in data dictionary.")
        effort = "Medium"

    elif flag == "NEGATIVE_VALUES_DETECTED":
        desc   = f"Negative values found in {col_ref} (min: {mn}, max: {mx}, type: {dtype})"
        impact = ("Negative values in numeric fields corrupt SUM/AVG aggregations and may indicate "
                  "sign-flip bugs, refunds coded incorrectly, or test data contamination.")
        fix    = (f"1. Run: SELECT * FROM {table} WHERE {col} < 0  "
                  f"2. Decide: are negatives valid (e.g. credit adjustments) or errors?  "
                  f"3. If errors: add CHECK ({col} >= 0) constraint at source.  "
                  f"4. If valid: document the business meaning and add a 'credit_flag' column for clarity.")
        effort = "Low"

    elif flag == "FUTURE_DATE_DETECTED":
        desc   = f"Future dates detected in {col_ref} — values beyond today's date (max: {mx})"
        impact = ("Future dates break time-series analyses, age/tenure calculations, and SLA reporting. "
                  "They usually indicate placeholder, test, or incorrectly formatted records.")
        fix    = (f"1. Run: SELECT * FROM {table} WHERE {col} > CURRENT_DATE  "
                  f"2. Determine if future dates are intentionally scheduled (e.g. due_date) or errors.  "
                  f"3. For event columns (hire_date, created_at): add validation to reject future values at ingestion.  "
                  f"4. Flag or quarantine affected rows for manual review.")
        effort = "Low"

    elif flag == "NUMERIC_OUTLIERS_DETECTED":
        desc   = f"Statistical outliers detected in {col_ref} — values far outside normal range (min: {mn}, max: {mx})"
        impact = ("Outliers skew mean/std calculations and can mask real trends. "
                  "They may indicate data entry errors, unit mismatches (e.g. dollars vs cents), or genuine edge cases.")
        fix    = (f"1. Run: SELECT * FROM {table} WHERE {col} < [lower_bound] OR {col} > [upper_bound]  "
                  f"2. Review distribution — use percentiles not mean for skewed data.  "
                  f"3. Validate unit consistency (e.g. all values in same currency/unit).  "
                  f"4. Cap extreme values or move to a separate 'outlier' table if they are genuine but rare.")
        effort = "Medium"

    elif flag == "MIXED_DATA_TYPES":
        desc   = f"Column {col_ref} contains mixed data types — both {dtype} and non-{dtype} values detected"
        impact = ("Mixed types cause type-cast errors in ETL pipelines, break schema inference, "
                  "and produce incorrect aggregations when numeric strings are sorted alphabetically.")
        fix    = (f"1. Run: SELECT DISTINCT typeof({col}) FROM {table} (SQLite) or check pg_typeof()  "
                  f"2. Identify which records have the wrong type.  "
                  f"3. Cast or clean at source — enforce a single type at ingestion.  "
                  f"4. Add schema validation (e.g. Great Expectations, dbt tests) to catch future regressions.")
        effort = "High"

    elif flag == "LOW_UNIQUENESS":
        desc   = (f"Column {col_ref} has very low uniqueness ({unique_ratio}% unique values) — "
                  f"behaves like a near-constant or low-cardinality field")
        impact = ("Low-uniqueness columns add little analytical value and may indicate data truncation, "
                  "incorrect default values, or a poorly normalised schema.")
        fix    = (f"1. Review value distribution: SELECT {col}, COUNT(*) FROM {table} GROUP BY {col} ORDER BY 2 DESC  "
                  f"2. If intended (e.g. status flags): document valid values and add an enum constraint.  "
                  f"3. If unintended: trace back to source system for the default/fallback causing low variance.  "
                  f"4. Consider normalising into a lookup/reference table.")
        effort = "Low"

    elif flag == "PATTERN_INCONSISTENCY":
        desc   = (f"Column {col_ref} has a dominant format pattern but {100 - unique_ratio:.0f}% of values "
                  f"deviate — likely a mix of valid and malformed entries")
        impact = ("Pattern inconsistency breaks regex-based validation, email delivery, phone parsing, "
                  "and any downstream system that expects a consistent format.")
        fix    = (f"1. Run a regex scan: SELECT * FROM {table} WHERE {col} NOT LIKE '[expected pattern]'  "
                  f"2. Identify the dominant pattern (email, phone, ID format) and document it.  "
                  f"3. Add a format validator at the ingestion layer.  "
                  f"4. Clean existing non-conforming values or flag them with a 'format_valid' boolean column.")
        effort = "Medium"

    elif flag == "DUPLICATE_ROWS_DETECTED":
        desc   = f"Exact duplicate rows detected in table '{table}'"
        impact = ("Duplicates inflate record counts, corrupt SUM/COUNT metrics, and cause fan-out "
                  "in joins — downstream reports will overcount. Especially critical for fact tables.")
        fix    = (f"1. Run: SELECT *, COUNT(*) FROM {table} GROUP BY [all columns] HAVING COUNT(*) > 1  "
                  f"2. Determine deduplication key (natural key or surrogate).  "
                  f"3. Add a UNIQUE constraint or PRIMARY KEY at source.  "
                  f"4. Use DISTINCT or ROW_NUMBER() deduplication in your ETL before loading.")
        effort = "Medium"

    elif flag == "TYPE_CONDITIONAL_FIELD":
        desc   = (f"Column {col_ref} is type-conditional — it is absent for some entity types "
                  f"within this table ({null_pct}% null overall)")
        impact = ("This is an informational flag, not a data quality error. The column is only "
                  "applicable to a subset of entity types (detected via schema polymorphism). "
                  "No fix needed unless the column should be normalised into separate sub-type tables.")
        fix    = (f"1. Verify that the discriminator column (entity type) correctly partitions records.  "
                  f"2. If strict normalisation is required: split '{table}' into per-type sub-tables.  "
                  f"3. Otherwise: document which entity types populate '{col}' in your data dictionary.  "
                  f"4. Consider adding a JSON/JSONB variant column for optional type-specific attributes.")
        effort = "Low"

    else:
        desc   = f"Issue '{flag}' detected in {col_ref}"
        impact = "Data quality concern that may affect downstream analysis accuracy."
        fix    = f"Investigate column {col_ref} for the flagged condition and apply appropriate data cleaning."
        effort = "Medium"

    return desc, impact, fix, effort


# ---------------------------------------------------------------------------
# Rules engine (Issues H & I) — reads parseiq_rules.yaml/.json sidecar
# ---------------------------------------------------------------------------

def _find_rules_file(source_arg: str) -> Optional[str]:
    """Return path of a rules sidecar next to the input file.

    Looks for (in priority order):
      1. parseiq_rules.yaml / .yml / .json  — shared, one per directory
      2. <input_stem>_rules.yaml / .yml / .json — file-specific sidecar
         e.g. tc04_university_rules.yaml for tc04_university.json
    """
    if not os.path.isfile(source_arg):
        return None
    abs_path = os.path.abspath(source_arg)
    base_dir = os.path.dirname(abs_path)
    stem = os.path.splitext(os.path.basename(abs_path))[0]
    candidates = [
        "parseiq_rules.yaml", "parseiq_rules.yml", "parseiq_rules.json",
        f"{stem}_rules.yaml", f"{stem}_rules.yml", f"{stem}_rules.json",
    ]
    for name in candidates:
        candidate = os.path.join(base_dir, name)
        if os.path.isfile(candidate):
            return candidate
    return None


def _load_rules(rules_path: str) -> List[Dict[str, Any]]:
    """Parse a rules sidecar file. Supports YAML (if pyyaml installed) and JSON."""
    ext = os.path.splitext(rules_path)[1].lower()
    with open(rules_path, "r", encoding="utf-8") as fh:
        content = fh.read()
    if ext in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
            data = yaml.safe_load(content)
        except ImportError:
            import json as _json
            data = _json.loads(content)
    else:
        import json as _json
        data = _json.loads(content)
    return data.get("rules", []) if isinstance(data, dict) else []


def _inject_violation(raw_metadata: Dict, table_name: str,
                      anomaly_key: str, message: str) -> None:
    """Inject a rule violation into the named table's metadata top_issues."""
    tbl = raw_metadata.get("tables", {}).get(table_name)
    if not tbl:
        return
    inner = tbl.get("table_metadata", tbl)
    issues = inner.setdefault("top_issues", [])
    if message not in issues:
        issues.append(message)
    asum = inner.setdefault("anomaly_summary", {"total_anomalies": 0, "anomaly_types": {}})
    asum["total_anomalies"] = asum.get("total_anomalies", 0) + 1
    asum.setdefault("anomaly_types", {})[anomaly_key] = (
        asum["anomaly_types"].get(anomaly_key, 0) + 1
    )


def _apply_rules(rules: List[Dict[str, Any]],
                 tables: Dict[str, List[Dict]],
                 raw_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Dispatch each rule to its handler and return all violations."""
    violations: List[Dict[str, Any]] = []
    for rule in rules:
        rtype = rule.get("type", "")
        if rtype == "max_value":
            violations.extend(_rule_max_value(rule, tables, raw_metadata))
        elif rtype == "min_value":
            violations.extend(_rule_min_value(rule, tables, raw_metadata))
        elif rtype == "cross_table_compare":
            violations.extend(_rule_cross_table(rule, tables, raw_metadata))
    return violations


def _rule_max_value(rule: Dict, tables: Dict, raw_metadata: Dict) -> List[Dict]:
    """Flag rows where column > max."""
    violations: List[Dict] = []
    tname = rule.get("table", "")
    col   = rule.get("column", "")
    bound = rule.get("max")
    msg_tmpl = rule.get("message", f"{col} exceeds max {{val}} (limit {bound})")
    if tname not in tables or not col or bound is None:
        return violations
    for rec in tables[tname]:
        val = rec.get(col)
        if val is None:
            continue
        try:
            if float(val) > float(bound):
                v = {"rule_type": "max_value", "table": tname, "column": col,
                     "value": val, "max": bound, "anomaly": "SCALE_VIOLATION_DETECTED",
                     "message": msg_tmpl.format(val=val)}
                violations.append(v)
                _inject_violation(
                    raw_metadata, tname, "SCALE_VIOLATION_DETECTED",
                    f"SCALE_VIOLATION_DETECTED: {col}={val} exceeds max {bound}",
                )
        except (ValueError, TypeError):
            pass
    return violations


def _rule_min_value(rule: Dict, tables: Dict, raw_metadata: Dict) -> List[Dict]:
    """Flag rows where column < min."""
    violations: List[Dict] = []
    tname = rule.get("table", "")
    col   = rule.get("column", "")
    bound = rule.get("min")
    msg_tmpl = rule.get("message", f"{col} below min {{val}} (floor {bound})")
    if tname not in tables or not col or bound is None:
        return violations
    for rec in tables[tname]:
        val = rec.get(col)
        if val is None:
            continue
        try:
            if float(val) < float(bound):
                v = {"rule_type": "min_value", "table": tname, "column": col,
                     "value": val, "min": bound, "anomaly": "SCALE_VIOLATION_DETECTED",
                     "message": msg_tmpl.format(val=val)}
                violations.append(v)
                _inject_violation(
                    raw_metadata, tname, "SCALE_VIOLATION_DETECTED",
                    f"SCALE_VIOLATION_DETECTED: {col}={val} below min {bound}",
                )
        except (ValueError, TypeError):
            pass
    return violations


def _rule_cross_table(rule: Dict, tables: Dict, raw_metadata: Dict) -> List[Dict]:
    """Flag rows where left_table.left_col violates an inequality vs right_table.right_col.

    The right table is the 'parent' (holds the limit value).  The FK from
    left_table back to right_table is inferred as _ref_{right_table}_id unless
    'fk_key' is specified.  'join_key' names the PK column in right_table.
    """
    violations: List[Dict] = []
    ltname = rule.get("left_table", "")
    lcol   = rule.get("left_col", "")
    rtname = rule.get("right_table", "")
    rcol   = rule.get("right_col", "")
    jkey   = rule.get("join_key", "")         # PK col in right_table
    fk_key = rule.get("fk_key", f"_ref_{rtname}_id")   # FK col in left_table
    expr   = rule.get("rule", "left <= right")
    msg    = rule.get("message",
                      f"{ltname}.{lcol} violates '{expr}' vs {rtname}.{rcol}")

    if not all([ltname, lcol, rtname, rcol]) or ltname not in tables or rtname not in tables:
        return violations

    # Build lookup: right PK value → right col value
    right_lookup: Dict[str, float] = {}
    for rec in tables[rtname]:
        pk = rec.get(jkey) if jkey else None
        rv = rec.get(rcol)
        if pk is not None and rv is not None:
            try:
                right_lookup[str(pk)] = float(rv)
            except (ValueError, TypeError):
                pass

    _ops = {
        "left <= right": lambda l, r: l > r,
        "left < right":  lambda l, r: l >= r,
        "left >= right": lambda l, r: l < r,
        "left > right":  lambda l, r: l <= r,
        "left == right": lambda l, r: l != r,
    }
    violates = _ops.get(expr, lambda l, r: False)

    for rec in tables[ltname]:
        lv = rec.get(lcol)
        fk = rec.get(fk_key)
        if lv is None or fk is None:
            continue
        rv = right_lookup.get(str(fk))
        if rv is None:
            continue
        try:
            lf = float(lv)
            if violates(lf, rv):
                v = {"rule_type": "cross_table_compare",
                     "left_table": ltname, "left_col": lcol, "left_value": lf,
                     "right_table": rtname, "right_col": rcol, "right_value": rv,
                     "join_key": str(fk), "anomaly": "CONSTRAINT_VIOLATION_DETECTED",
                     "message": msg}
                violations.append(v)
                _inject_violation(
                    raw_metadata, ltname, "CONSTRAINT_VIOLATION_DETECTED",
                    f"CONSTRAINT_VIOLATION_DETECTED: {lcol}={lf} violates '{expr}' "
                    f"vs {rtname}.{rcol}={rv}",
                )
        except (ValueError, TypeError):
            pass
    return violations


def _generate_outputs(tables, raw_metadata, enriched, out_dir) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    created: List[str] = []
    excel_sheets: Dict[str, pd.DataFrame] = {}
    used: set = set()
    _INV = str.maketrans({'[': '_', ']': '_', ':': '_', '*': '_', '?': '_', '/': '_', '\\': '_'})

    def _safe(prefix, tname):
        s = tname.translate(_INV).strip("'").strip()
        allowed = 31 - len(prefix)
        base = s[:allowed] if len(s) <= allowed else s[:allowed - 1] + "~"
        cand = prefix + base
        if cand in used:
            for i in range(2, 100):
                sf = str(i); cand = prefix + base[:allowed - len(sf)] + sf
                if cand not in used: break
        used.add(cand); return cand

    visited_tables: set = set()
    for tname, rows in tables.items():
        if not rows: continue
        if tname in visited_tables: continue
        visited_tables.add(tname)
        df_raw = pd.DataFrame(rows)
        df = df_raw.astype(object).where(df_raw.notna(), other=None)
        # Truncate JSON-blob strings so Data_ sheet columns stay readable
        def _truncate_blobs(val, limit=120):
            if isinstance(val, str) and len(val) > limit and (val.startswith("{") or val.startswith("[")):
                return val[:limit] + "…"
            return val
        df = df.apply(lambda col: col.map(_truncate_blobs))
        excel_sheets[_safe("Data_", tname)] = df

        tbl_meta = raw_metadata.get("tables", {}).get(tname, {})
        inner = tbl_meta.get("table_metadata", tbl_meta)
        attrs = inner.get("attributes", {})

        total_records = inner.get("dataset_info", {}).get("total_records", len(rows))
        table_score   = round(inner.get("data_quality_score", 0), 1)

        # ── Meta sheet — full attribute profile ───────────────────────────────
        meta_rows = []
        for a, info in attrs.items():
            if not isinstance(info, dict):
                continue
            dtype     = info.get("data_type", "unknown")
            present   = info.get("present_count", 0)
            missing   = info.get("null_count", 0)
            miss_pct  = info.get("null_percentage", 0.0)
            unique_v  = info.get("unique_count", 0)
            unique_r  = round(info.get("unique_ratio", 0), 3)
            qscore    = f"{round(info.get('quality_score', 0))}/100"
            flags     = info.get("anomaly_flags", [])
            outliers  = info.get("outliers", {})
            patterns  = info.get("pattern_analysis", {}).get("regex_patterns", {})

            # Recognised patterns (any pattern with > 0 match)
            recog = [p for p, v in patterns.items()
                     if isinstance(v, dict) and v.get("matches", 0) > 0]
            recog_str = ", ".join(recog) if recog else None

            # Most common values (string / categorical)
            common_vals = info.get("common_values", {})
            common_str = str(list(common_vals.keys())[:5]) if common_vals else None

            # Character distribution (strings)
            charset = info.get("charset_analysis", {})
            char_dist = None
            if charset:
                parts = []
                if charset.get("alphabetic", 0):
                    parts.append(f"Alpha: {charset['alphabetic']:.1f}%")
                if charset.get("numeric", 0):
                    parts.append(f"Numeric: {charset['numeric']:.1f}%")
                if charset.get("special_chars", 0):
                    parts.append(f"Special: {charset['special_chars']:.1f}%")
                char_dist = ", ".join(parts) if parts else None

            row = {
                "Table_Name":            tname,
                "Attribute_Name":        a,
                "Data_Type":             dtype,
                "Total_Records":         total_records,
                "Present_Count":         present,
                "Missing_Count":         missing,
                "Missing_Percentage":    f"{miss_pct:.1f}%",
                "Unique_Values":         unique_v,
                "Unique_Ratio":          unique_r,
                "Quality_Score":         qscore,
                # String-specific
                "Min_Length":            info.get("min_length") if dtype == "string" else None,
                "Max_Length":            info.get("max_length") if dtype == "string" else None,
                "Avg_Length":            round(info.get("avg_length", 0), 2) if dtype == "string" else None,
                "Median_Length":         info.get("median_length") if dtype == "string" else None,
                "Most_Common_Values":    common_str,
                "Character_Distribution":char_dist,
                "Anomaly_Count":         len(flags),
                "Anomaly_Types":         ", ".join(flags) if flags else None,
                "Has_Outliers":          "Yes" if outliers.get("count", 0) > 0 else "No",
                "Recognized_Patterns":   recog_str,
                # Numeric-specific
                "Min_Value":             info.get("min_value") if dtype in ("integer", "float") else None,
                "Max_Value":             info.get("max_value") if dtype in ("integer", "float") else None,
                "Mean_Value":            round(info.get("mean", 0), 4) if dtype in ("integer", "float") else None,
                "Median_Value":          info.get("median") if dtype in ("integer", "float") else None,
                "Std_Deviation":         round(info.get("std", 0), 4) if dtype in ("integer", "float") else None,
                "Outliers_Count":        outliers.get("count", 0) if dtype in ("integer", "float") else None,
                # Boolean-specific
                "True_Count":            info.get("true_count") if dtype == "boolean" else None,
                "False_Count":           info.get("false_count") if dtype == "boolean" else None,
                "True_Percentage":       f"{info.get('true_percentage', 0):.1f}%" if dtype == "boolean" else None,
                "False_Percentage":      f"{info.get('false_percentage', 0):.1f}%" if dtype == "boolean" else None,
            }
            meta_rows.append(row)
        if meta_rows:
            excel_sheets[_safe("Meta_", tname)] = pd.DataFrame(meta_rows)

        # ── Quality sheet — long format, one metric row per attribute ─────────
        def _qs_status(score):
            if score >= 90: return "Excellent"
            if score >= 80: return "Good"
            if score >= 60: return "Warning"
            return "Critical"

        qual_rows = []
        # Table-level header rows
        qual_rows.append({
            "Table_Name":      tname,
            "Quality_Category":"Overall",
            "Metric_Name":     "Table Quality Score",
            "Metric_Value":    f"{table_score}/100",
            "Status":          _qs_status(table_score),
            "Description":     "Overall data quality assessment across all attributes",
        })
        qual_rows.append({
            "Table_Name":      tname,
            "Quality_Category":"Structure",
            "Metric_Name":     "Total Attributes",
            "Metric_Value":    len(attrs),
            "Status":          "Info",
            "Description":     "Number of columns/attributes in the table",
        })
        qual_rows.append({
            "Table_Name":      tname,
            "Quality_Category":"Volume",
            "Metric_Name":     "Total Records",
            "Metric_Value":    total_records,
            "Status":          "Info",
            "Description":     "Number of rows/records in the table",
        })
        # Per-attribute rows
        for a, info in attrs.items():
            if not isinstance(info, dict):
                continue
            ascore     = round(info.get("quality_score", 0), 1)
            null_pct   = info.get("null_percentage", 0.0)
            unique_r   = round(info.get("unique_ratio", 0), 3)
            a_flags    = info.get("anomaly_flags", [])
            outlier_ct = info.get("outliers", {}).get("count", 0)

            qual_rows.append({
                "Table_Name":      tname,
                "Quality_Category":"Attribute Quality",
                "Metric_Name":     f"{a} - Overall Quality",
                "Metric_Value":    f"{round(ascore)}/100",
                "Status":          _qs_status(ascore),
                "Description":     f"Overall quality score for {a} attribute",
            })
            qual_rows.append({
                "Table_Name":      tname,
                "Quality_Category":"Uniqueness",
                "Metric_Name":     f"{a} - Unique Ratio",
                "Metric_Value":    unique_r,
                "Status":          "Good" if unique_r > 0.9 else ("Warning" if unique_r > 0.5 else "Info"),
                "Description":     f"Ratio of unique values in {a} (1.0 = all unique)",
            })
            qual_rows.append({
                "Table_Name":      tname,
                "Quality_Category":"Completeness",
                "Metric_Name":     f"{a} - Missing %",
                "Metric_Value":    f"{null_pct:.1f}%",
                "Status":          "Excellent" if null_pct == 0 else ("Warning" if null_pct < 10 else "Critical"),
                "Description":     f"Percentage of missing/null values in {a}",
            })
            if a_flags:
                qual_rows.append({
                    "Table_Name":      tname,
                    "Quality_Category":"Anomalies",
                    "Metric_Name":     f"{a} - Anomaly Types",
                    "Metric_Value":    ", ".join(a_flags),
                    "Status":          "Critical" if len(a_flags) > 2 else "Warning",
                    "Description":     f"Detected anomaly types in {a}",
                })
            if outlier_ct > 0:
                qual_rows.append({
                    "Table_Name":      tname,
                    "Quality_Category":"Outliers",
                    "Metric_Name":     f"{a} - Outlier Count",
                    "Metric_Value":    outlier_ct,
                    "Status":          "Warning",
                    "Description":     f"Statistical outliers detected in {a}",
                })
        if qual_rows:
            excel_sheets[_safe("Quality_", tname)] = pd.DataFrame(qual_rows)

    pi = enriched.get("pipeline_info", {})
    llm_ins = enriched.get("llm_insights", {}) or {}
    qs = _extract_quality_scores(raw_metadata.get("tables", {}))
    avg_q = round(sum(qs.values()) / max(len(qs), 1), 1)
    oa = llm_ins.get("overall_assessment", {})

    # ── 00_Summary — one row per table overview ────────────────────────────
    tbl_summaries = raw_metadata.get("dataset_overview", {}).get("table_summaries", {})
    summary_overview = []
    for tname in tables:
        tmeta = raw_metadata.get("tables", {}).get(tname, {})
        inner = tmeta.get("table_metadata", tmeta)
        attrs = inner.get("attributes", {})
        anomaly_cols = [col for col, info in attrs.items()
                        if isinstance(info, dict) and info.get("anomaly_flags")]
        summary_overview.append({
            "Table": tname,
            "Records": len(tables[tname]),
            "Columns": len(attrs),
            "Quality_Score": round(qs.get(tname, 0), 1),
            "Anomaly_Columns": len(anomaly_cols),
            "Top_Issues": "; ".join(inner.get("top_issues", [])[:3]) or "",
        })
    pipeline_meta_rows = [
        {"Table": "— PIPELINE SUMMARY —", "Records": "", "Columns": "",
         "Quality_Score": avg_q, "Anomaly_Columns": sum(r["Anomaly_Columns"] for r in summary_overview),
         "Top_Issues": f"LLM Grade: {oa.get('quality_grade','N/A')} | "
                       f"Readiness: {oa.get('production_readiness','N/A')} | "
                       f"Duration: {pi.get('total_duration',0):.1f}s"},
    ]
    excel_sheets["00_Summary"] = pd.DataFrame(pipeline_meta_rows + summary_overview)

    # ── 01_LLM_Assessment — full LLM output in table form ─────────────────
    if oa:
        assess_rows = [
            {"Field": "Quality Grade",        "Value": oa.get("quality_grade", "N/A")},
            {"Field": "Overall Score",        "Value": next((oa[k] for k in ("overall_score", "corrected_score") if k in oa), "N/A")},
            {"Field": "Production Readiness", "Value": oa.get("production_readiness", "N/A")},
            {"Field": "Risk Level",           "Value": llm_ins.get("risk_assessment", {}).get("overall_risk_level", "N/A")},
            {"Field": "Model Used",           "Value": llm_ins.get("enrichment_metadata", {}).get("model_used", "N/A")},
            {"Field": "Analysis Timestamp",   "Value": llm_ins.get("enrichment_metadata", {}).get("timestamp", "N/A")},
            {"Field": "— Key Strengths —",    "Value": ""},
        ]
        for s in (oa.get("key_strengths") or []):
            assess_rows.append({"Field": "Strength", "Value": str(s)})
        assess_rows.append({"Field": "— Primary Concerns —", "Value": ""})
        for c in (oa.get("primary_concerns") or []):
            assess_rows.append({"Field": "Concern", "Value": str(c)})
        excel_sheets["01_LLM_Assessment"] = pd.DataFrame(assess_rows)

    # ── 02_LLM_Recommendations — prioritised action list ──────────────────
    rec_rows = []
    for rec in (llm_ins.get("critical_issues") or []):
        if isinstance(rec, dict):
            rec_rows.append({
                "Priority": "CRITICAL", "Type": "Issue",
                "Table": rec.get("table", "All"),
                "Description": rec.get("issue", ""),
                "Action": rec.get("specific_fix", ""),
                "Effort": "",
            })
    for rec in (llm_ins.get("recommendations") or []):
        if isinstance(rec, dict):
            rec_rows.append({
                "Priority": rec.get("priority", ""),
                "Type": "Recommendation",
                "Table": "All",
                "Description": rec.get("category", ""),
                "Action": rec.get("action", ""),
                "Effort": rec.get("estimated_effort", ""),
            })
    if rec_rows:
        excel_sheets["02_LLM_Recommendations"] = pd.DataFrame(rec_rows)

    # ── 99_Issues_Recommendations — rich, actionable combined sheet ─────────
    combined_rows = []

    # Section 1: Step 1 column-level anomalies with human descriptions + fixes
    _seen_issue_tables: set = set()
    for tname, _rows in tables.items():
        if tname in _seen_issue_tables: continue
        _seen_issue_tables.add(tname)
        tmeta = raw_metadata.get("tables", {}).get(tname, {})
        inner = tmeta.get("table_metadata", tmeta)
        rec_count = len(_rows)
        for col, info in inner.get("attributes", {}).items():
            if not isinstance(info, dict) or not info.get("anomaly_flags"):
                continue
            q_score = info.get("quality_score", 100)
            null_pct = round(info.get("null_percentage", 0), 1)
            dtype = info.get("data_type", "unknown")
            mn = info.get("min_value", "")
            mx = info.get("max_value", "")
            unique_ratio = round(info.get("unique_ratio", 1) * 100, 1)
            priority = "CRITICAL" if q_score < 40 else "HIGH" if q_score < 70 else "MEDIUM"

            for flag in info["anomaly_flags"]:
                desc, impact, fix, effort = _describe_issue(
                    flag, col, tname, dtype, null_pct, mn, mx, unique_ratio, rec_count
                )
                combined_rows.append({
                    "Priority":           priority,
                    "Source":             "Automated Analysis",
                    "Table":              tname,
                    "Column":             col,
                    "Issue_Type":         flag,
                    "Column_Quality":     f"{q_score}/100",
                    "Description":        desc,
                    "Business_Impact":    impact,
                    "Recommended_Fix":    fix,
                    "Effort":             effort,
                    "Stats":              f"null={null_pct}% | unique={unique_ratio}% | min={mn} | max={mx}",
                })

    # Section 2: LLM critical issues (only when LLM was used)
    for rec in (llm_ins.get("critical_issues") or []):
        if isinstance(rec, dict):
            combined_rows.append({
                "Priority":        rec.get("severity", "HIGH"),
                "Source":          "LLM Analysis",
                "Table":           rec.get("table", "All"),
                "Column":          rec.get("column", ""),
                "Issue_Type":      "LLM_Critical_Issue",
                "Column_Quality":  "",
                "Description":     rec.get("issue", ""),
                "Business_Impact": rec.get("business_impact", "Data reliability risk identified by AI review"),
                "Recommended_Fix": rec.get("specific_fix", ""),
                "Effort":          rec.get("effort", "Medium"),
                "Stats":           "",
            })

    # Section 3: LLM recommendations (only when LLM was used)
    for rec in (llm_ins.get("recommendations") or []):
        if isinstance(rec, dict):
            combined_rows.append({
                "Priority":        rec.get("priority", "MEDIUM"),
                "Source":          "LLM Analysis",
                "Table":           "All Tables",
                "Column":          "",
                "Issue_Type":      "LLM_Recommendation",
                "Column_Quality":  "",
                "Description":     f"[{rec.get('category', '')}] {rec.get('action', '')}",
                "Business_Impact": "Process or quality improvement identified by AI",
                "Recommended_Fix": rec.get("action", ""),
                "Effort":          rec.get("estimated_effort", ""),
                "Stats":           "",
            })

    if combined_rows:
        # Sort: CRITICAL first, then HIGH, MEDIUM, LOW; within same priority sort by table
        _priority_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        combined_rows.sort(key=lambda r: (
            _priority_order.get(r["Priority"], 9), r["Table"], r["Column"]
        ))
        excel_sheets["99_Issues_Recommendations"] = pd.DataFrame(combined_rows)

    # ── Write Excel — grouped per table: Data → Meta → Quality ────────────
    excel_path = os.path.join(out_dir, "complete_data_analysis.xlsx")
    try:
        # Build ordered sheet list: overview sheets first, then per-table groups
        order = ["00_Summary"]
        if "01_LLM_Assessment" in excel_sheets:    order.append("01_LLM_Assessment")
        if "02_LLM_Recommendations" in excel_sheets: order.append("02_LLM_Recommendations")

        # Group Data/Meta/Quality by table in table order
        for tname in tables:
            for prefix in ("Data_", "Meta_", "Quality_"):
                key = _safe(prefix, tname)  # reuse same _safe function
                # find the actual key that was used (safe name may differ)
                for k in excel_sheets:
                    if k.startswith(prefix) and tname[:min(len(tname), 20)] in k:
                        if k not in order:
                            order.append(k)
                        break

        if "99_Issues_Recommendations" in excel_sheets: order.append("99_Issues_Recommendations")

        # Catch any sheet not yet in order
        for k in excel_sheets:
            if k not in order:
                order.append(k)

        with pd.ExcelWriter(excel_path, engine="openpyxl") as w:
            for sn in order:
                if sn in excel_sheets:
                    excel_sheets[sn].to_excel(w, sheet_name=sn, index=False)
        created.append(excel_path)
        print(f"  Excel ({len(order)} sheets) → {excel_path}")
    except Exception as exc:
        logger.warning("Excel generation failed: %s", exc)

    # ── CSV summaries ──────────────────────────────────────────────────────
    sum_path = os.path.join(out_dir, "overall_dataset_summary.csv")
    excel_sheets["00_Summary"].to_csv(sum_path, index=False)
    created.append(sum_path)

    if "99_Issues_Recommendations" in excel_sheets:
        iss_path = os.path.join(out_dir, "combined_issues_and_recommendations.csv")
        excel_sheets["99_Issues_Recommendations"].to_csv(iss_path, index=False)
        created.append(iss_path)

    return created


def _build_dataset_overview(all_metadata):
    ov: Dict[str, Any] = {"table_summaries": {}}
    for tname, tmeta in all_metadata.items():
        inner = tmeta.get("table_metadata", tmeta)
        attrs = inner.get("attributes", {})
        ov["table_summaries"][tname] = {
            "record_count": inner.get("dataset_info", {}).get("total_records", 0),
            "field_analysis": {
                a: {"data_type": i.get("data_type"), "null_percentage": i.get("null_percentage", 0),
                    "anomalies": i.get("anomaly_flags", []), "outlier_count": i.get("outliers", {}).get("count", 0),
                    "unique_percentage": round(i.get("unique_ratio", 0) * 100, 1),
                    "min_value": i.get("min_value"), "max_value": i.get("max_value"), "mean": i.get("mean")}
                for a, i in attrs.items()
            },
            "completeness_rate": inner.get("data_profiling", {}).get("record_completeness", {}).get("avg_completeness", 100),
            "duplicate_count": inner.get("data_profiling", {}).get("duplicate_analysis", {}).get("duplicate_rows", 0),
            "quality_score": inner.get("data_quality_score", 0),
            "top_issues": inner.get("top_issues", []),
        }
    return ov


def _extract_quality_scores(all_metadata):
    return {tname: round(t.get("table_metadata", t).get("data_quality_score", 0), 2)
            for tname, t in all_metadata.items()}


def _extract_anomalies(all_metadata):
    result: Dict[str, Dict[str, List[str]]] = {}
    for tname, tmeta in all_metadata.items():
        inner = tmeta.get("table_metadata", tmeta)
        result[tname] = {
            col: info.get("anomaly_flags", [])
            for col, info in inner.get("attributes", {}).items()
            if isinstance(info, dict) and info.get("anomaly_flags")
        }
    return result


def _hash_table(rows: List[Dict]) -> str:
    return hashlib.sha256(json.dumps(rows, sort_keys=True, default=str).encode()).hexdigest()


def _is_cached(tname: str, current_hash: str, state: Dict[str, Any]) -> bool:
    entry = state.get("tables", {}).get(tname)
    return bool(entry and entry.get("hash") == current_hash)


def _load_state(path: str) -> Dict[str, Any]:
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f: return json.load(f)
        except Exception: pass
    return {"tables": {}}


def _save_state(path, existing, hashes, all_metadata):
    new = {**existing, "tables": {**existing.get("tables", {})}}
    for tname, h in hashes.items():
        new["tables"][tname] = {"hash": h, "last_run": datetime.now().isoformat(),
                                "metadata": all_metadata.get(tname, {})}
    try:
        with open(path, "w", encoding="utf-8") as f: json.dump(new, f, indent=2, default=str)
    except Exception as exc: logger.warning("Could not save state: %s", exc)


def _write_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f: json.dump(obj, f, indent=2, default=str)
