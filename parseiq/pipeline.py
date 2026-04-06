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

    for tname, rows in tables.items():
        if not rows: continue
        df_raw = pd.DataFrame(rows)
        df = df_raw.astype(object).where(df_raw.notna(), other=None)
        excel_sheets[_safe("Data_", tname)] = df

        tbl_meta = raw_metadata.get("tables", {}).get(tname, {})
        inner = tbl_meta.get("table_metadata", tbl_meta)
        attrs = inner.get("attributes", {})

        meta_rows = [{
            "Table": tname, "Attribute": a,
            "Data_Type": info.get("data_type", "unknown"),
            "Null_Pct": round(info.get("null_percentage", 0), 1),
            "Unique_Ratio": round(info.get("unique_ratio", 0), 3),
            "Quality_Score": round(info.get("quality_score", 0), 1),
            "Anomalies": ", ".join(info.get("anomaly_flags", [])) or "None",
        } for a, info in attrs.items() if isinstance(info, dict)]
        if meta_rows: excel_sheets[_safe("Meta_", tname)] = pd.DataFrame(meta_rows)

        qual_rows = [{
            "Table": tname, "Attribute": a,
            "Quality_Score": round(info.get("quality_score", 0), 1),
            "Null_Penalty": round(min(info.get("null_percentage", 0) * 0.5, 30), 1),
            "Anomaly_Count": len(info.get("anomaly_flags", [])),
            "Status": ("Excellent" if info.get("quality_score", 0) >= 90 else
                       "Good" if info.get("quality_score", 0) >= 80 else
                       "Warning" if info.get("quality_score", 0) >= 60 else "Critical"),
        } for a, info in attrs.items() if isinstance(info, dict)]
        if qual_rows: excel_sheets[_safe("Quality_", tname)] = pd.DataFrame(qual_rows)

    # Summary sheet
    pi = enriched.get("pipeline_info", {})
    llm_ins = enriched.get("llm_insights", {}) or {}
    qs = _extract_quality_scores(raw_metadata.get("tables", {}))
    avg_q = round(sum(qs.values()) / max(len(qs), 1), 1)
    summary_rows = [
        {"Metric": "Total Tables",          "Value": str(len(tables))},
        {"Metric": "Total Records",         "Value": str(pi.get("total_records", 0))},
        {"Metric": "Avg Quality Score",     "Value": f"{avg_q}/100"},
        {"Metric": "LLM Grade",             "Value": llm_ins.get("overall_assessment", {}).get("quality_grade", "N/A")},
        {"Metric": "LLM Used",              "Value": str(pi.get("llm_used", False))},
        {"Metric": "Processing Time",       "Value": f"{pi.get('total_duration', 0):.1f}s"},
    ]
    excel_sheets["00_Overall_Summary"] = pd.DataFrame(summary_rows)

    # Issues sheet
    issues_rows = []
    for rec in (llm_ins.get("critical_issues") or [])[:15]:
        if isinstance(rec, dict):
            issues_rows.append({"Severity": rec.get("severity", ""), "Table": rec.get("table", "All"),
                                "Issue": rec.get("issue", ""), "Fix": rec.get("specific_fix", "")})
    for rec in (llm_ins.get("recommendations") or [])[:10]:
        if isinstance(rec, dict):
            issues_rows.append({"Severity": rec.get("priority", ""), "Table": "All",
                                "Issue": rec.get("category", ""), "Fix": rec.get("action", "")})
    if issues_rows: excel_sheets["99_Issues_Recommendations"] = pd.DataFrame(issues_rows)

    # Write Excel
    excel_path = os.path.join(out_dir, "complete_data_analysis.xlsx")
    try:
        order = ["00_Overall_Summary"]
        order += sorted(s for s in excel_sheets if s.startswith(("Data_", "Meta_", "Quality_")))
        if "99_Issues_Recommendations" in excel_sheets: order.append("99_Issues_Recommendations")
        with pd.ExcelWriter(excel_path, engine="openpyxl") as w:
            for sn in order:
                if sn in excel_sheets: excel_sheets[sn].to_excel(w, sheet_name=sn, index=False)
        created.append(excel_path)
        print(f"  Excel ({len(excel_sheets)} sheets) → {excel_path}")
    except Exception as exc:
        logger.warning("Excel generation failed: %s", exc)

    sum_path = os.path.join(out_dir, "overall_dataset_summary.csv")
    excel_sheets["00_Overall_Summary"].to_csv(sum_path, index=False); created.append(sum_path)

    if "99_Issues_Recommendations" in excel_sheets:
        iss_path = os.path.join(out_dir, "combined_issues_and_recommendations.csv")
        excel_sheets["99_Issues_Recommendations"].to_csv(iss_path, index=False); created.append(iss_path)

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
