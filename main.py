#!/usr/bin/env python3
"""
main.py — thin entry point for running ParseIQ as a script.

For programmatic use, import from the package:

    from parseiq import Pipeline
    result = Pipeline("input/input_data.json").run(llm=True)

This file is kept for backward compatibility:
    python main.py
"""
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Module-level imports — kept at module scope so @patch('main.FileLoader') etc. works in tests
from parseiq.config import Config                      # noqa: F401
from parseiq.file_loader.loader import FileLoader      # noqa: F401
from parseiq.step1_metadata_extractor.extractor import MetadataExtractor  # noqa: F401
from parseiq.step2_llm_enricher.llm_agent import LLMEnricher              # noqa: F401
from parseiq.pipeline import (                         # noqa: F401
    Pipeline,
    _fallback_enrichment,
    _build_dataset_overview,
    _extract_quality_scores,
    _extract_anomalies,
    _write_json,
)


class MetadataEnrichmentAgent:
    """Backward-compat shim — lives in main.py so @patch('main.FileLoader') tests work."""

    def __init__(self, debug: bool = True):
        from parseiq.pipeline import Pipeline as _Pipeline
        _Pipeline._configure_logging()
        self.config = Config()
        self.debug = debug
        # Use module-level names so test patches via @patch('main.FileLoader') take effect
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
        import time
        from datetime import datetime
        for ext in ("*.csv", "*.xlsx"):
            for f in _g.glob(os.path.join("output", ext)):
                try: os.remove(f)
                except OSError: pass

        t0 = time.time()
        tables = self.file_loader.load_file(input_file_path)
        total_records = sum(len(t) for t in tables.values())
        all_metadata: dict = {}
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
        enriched_insights = None
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
        top_issues = []
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


def convert_enriched_json_to_csv(
    input_data_path="input/input_data.json",
    raw_path="output/raw_metadata.json",
    enriched_path="output/enriched_metadata.json",
    output_dir="output/",
    create_excel=True,
    csv_per_table=False,
):
    """Backward-compat wrapper: generate Excel + CSV from the JSON outputs."""
    import json
    from parseiq.pipeline import _generate_outputs
    from parseiq.file_loader.loader import FileLoader as _FL

    with open(input_data_path, encoding="utf-8") as f:
        input_data = json.load(f)
    with open(raw_path, encoding="utf-8") as f:
        raw_metadata = json.load(f)
    with open(enriched_path, encoding="utf-8") as f:
        enriched_metadata = json.load(f)

    tables = _FL()._flatten_nested_json(input_data)
    created = _generate_outputs(tables, raw_metadata, enriched_metadata, output_dir.rstrip("/\\"))

    return {
        "success": True,
        "tables_processed": len(tables),
        "total_records": sum(len(r) for r in tables.values()),
        "csv_files_created": len([f for f in created if f.endswith(".csv")]),
        "excel_created": any(f.endswith(".xlsx") for f in created),
        "output_directory": output_dir,
        "created_files": created,
        "table_names": list(tables.keys()),
        "excel_sheets": len(created),
    }


def main():
    """Run the full pipeline on input/input_data.json."""
    for d in ["input", "output", "logs", "debug_output"]:
        os.makedirs(d, exist_ok=True)

    try:
        import openpyxl  # noqa: F401
    except ImportError:
        print("openpyxl is required: pip install openpyxl"); return

    agent = MetadataEnrichmentAgent(debug=False)
    print(f"Model: {agent.config.MODEL_NAME}")

    input_file = os.path.join("input", "input_data.json")
    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}"); return

    result = agent.run_pipeline(input_file, selected_model=agent.config.MODEL_NAME)

    if result:
        print("\nSuccess! Check the output/ directory for results.")
        try:
            convert_result = convert_enriched_json_to_csv(
                input_data_path=input_file,
                raw_path=os.path.join("output", "raw_metadata.json"),
                enriched_path=os.path.join("output", "enriched_metadata.json"),
                output_dir="output",
                create_excel=True,
            )
            if convert_result.get("success"):
                print(f"Generated {convert_result['csv_files_created']} analysis files")
        except Exception as e:
            print(f"CSV/Excel generation failed: {e}"); traceback.print_exc()
    else:
        print("Pipeline failed. Check logs for details.")


if __name__ == "__main__":
    main()
