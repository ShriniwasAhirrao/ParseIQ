"""Pipeline runner service — manages jobs, runs ParseIQ in background threads."""
from __future__ import annotations

import datetime as _dt
import io
import json
import os
import re
import sys
import uuid
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure the project root is importable so `from parseiq import Pipeline` works
_project_root = str(Path(__file__).resolve().parents[3])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from parseiq import Pipeline  # noqa: E402


@dataclass
class Job:
    job_id: str
    filename: str
    file_path: str
    output_dir: str
    status: str = "pending"       # pending | running | completed | failed
    step: Optional[str] = None    # extract | enrich | output
    progress: int = 0
    error: Optional[str] = None
    result: Optional[Any] = None
    use_llm: bool = False
    llm_config: Optional[Dict[str, Any]] = None
    events: List[Dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    # Cached loaded tables (populated lazily — avoids re-parsing the file
    # on every /results/.../table/... request)
    _loaded_tables: Optional[Dict[str, List[Dict[str, Any]]]] = None
    _hierarchy: Optional[Dict[str, Optional[str]]] = None


_jobs: Dict[str, Job] = {}
_jobs_lock = threading.Lock()
_stdout_lock = threading.Lock()
_job_semaphore = threading.Semaphore(4)  # max 4 concurrent analyses
_TTL_SECONDS = 3600


def _redact_key(msg: str, job: Job) -> str:
    """Replace API keys in log messages with a redacted form."""
    if job.llm_config and job.llm_config.get("api_key"):
        key = job.llm_config["api_key"]
        if len(key) > 8:
            msg = msg.replace(key, key[:4] + "****" + key[-4:])
        else:
            msg = msg.replace(key, "****")
    return msg


def _push_event(job: Job, level: str, message: str) -> None:
    """Append a thought-process event to the job timeline."""
    msg = message.strip()
    if not msg:
        return
    msg = _redact_key(msg, job)
    job.events.append({
        "ts": time.time(),
        "level": level,
        "message": msg,
    })
    # Keep memory bounded
    if len(job.events) > 500:
        del job.events[:100]


class _TeeStream(io.TextIOBase):
    """Mirror writes to the original stream *and* push lines into job.events."""

    def __init__(self, job: Job, original):
        self._job = job
        self._orig = original
        self._buf = ""

    def write(self, s):
        try:
            self._orig.write(s)
        except Exception:
            pass
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                _push_event(self._job, "info", line)
        return len(s)

    def flush(self):
        try:
            self._orig.flush()
        except Exception:
            pass
        if self._buf.strip():
            _push_event(self._job, "info", self._buf)
            self._buf = ""


def create_job(file_path: str, filename: str, use_llm: bool = False) -> Job:
    job_id = uuid.uuid4().hex[:12]
    output_dir = tempfile.mkdtemp(prefix=f"parseiq_{job_id}_")
    job = Job(
        job_id=job_id,
        filename=filename,
        file_path=file_path,
        output_dir=output_dir,
        use_llm=use_llm,
    )
    with _jobs_lock:
        _jobs[job_id] = job
    _cleanup_stale_jobs()
    return job


def get_job(job_id: str) -> Optional[Job]:
    with _jobs_lock:
        return _jobs.get(job_id)


def set_llm_config(job: Job, config: Optional[Dict[str, Any]]) -> None:
    job.llm_config = config or None


def run_analysis(job_id: str) -> None:
    """Run the ParseIQ pipeline in a background thread."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        return

    def _worker():
        _job_semaphore.acquire()
        tee = _TeeStream(job, sys.__stdout__)
        try:
            with _stdout_lock:
                old_stdout = sys.stdout
                sys.stdout = tee

            job.status = "running"
            job.step = "extract"
            job.progress = 5
            _push_event(job, "stage", f"Queued analysis for {job.filename}")

            _push_event(job, "stage", "Step 1 — extracting metadata")
            pipeline = Pipeline.from_file(job.file_path, output_dir=job.output_dir)
            job.progress = 20

            llm_kwargs: Dict[str, Any] = {"llm": job.use_llm}
            if job.use_llm and job.llm_config:
                cfg = job.llm_config
                if cfg.get("provider"):
                    llm_kwargs["llm_provider"] = cfg["provider"]
                if cfg.get("api_key"):
                    llm_kwargs["llm_api_key"] = cfg["api_key"]
                if cfg.get("model"):
                    llm_kwargs["llm_model"] = cfg["model"]
                if cfg.get("base_url"):
                    llm_kwargs["llm_base_url"] = cfg["base_url"]

            if job.use_llm:
                _push_event(job, "stage", f"Step 2 — LLM enrichment ({job.llm_config.get('provider') if job.llm_config else 'openrouter'})")
                job.step = "enrich"
            else:
                _push_event(job, "stage", "Step 2 — skipped (LLM off, local mode)")

            job.progress = 45
            result = pipeline.run(**llm_kwargs)

            job.step = "output"
            job.progress = 92
            _push_event(job, "stage", "Step 3 — writing outputs")

            job.result = result
            job.status = "completed"
            job.progress = 100
            job.step = None
            _push_event(job, "stage", "Analysis complete")

        except Exception as exc:
            job.status = "failed"
            job.error = str(exc)
            job.progress = 0
            _push_event(job, "error", f"Failed: {exc}")
        finally:
            with _stdout_lock:
                sys.stdout = old_stdout
            tee.flush()
            _job_semaphore.release()

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()


def _get_loaded_tables(job: Job) -> Dict[str, List[Dict[str, Any]]]:
    """Load the source file once per job and cache the result."""
    if job._loaded_tables is not None:
        return job._loaded_tables
    try:
        from parseiq.connectors.file import load as load_file
        job._loaded_tables = load_file(job.file_path)
    except Exception as exc:
        _push_event(job, "error", f"Table load failed: {exc}")
        job._loaded_tables = {}
    return job._loaded_tables


def _infer_hierarchy(job: Job, table_names: List[str]) -> Dict[str, Optional[str]]:
    """Return {table_name: parent_table_name or None}, cached on the job.

    Uses the FK convention from file_loader: child tables get a
    `_ref_<parent>_id` column injected when they sit inside a parent table.
    """
    if job._hierarchy is not None:
        return job._hierarchy

    parents: Dict[str, Optional[str]] = {t: None for t in table_names}
    loaded = _get_loaded_tables(job)
    name_set = set(table_names)
    for tname, rows in loaded.items():
        if tname not in name_set or not rows:
            continue
        sample = rows[0]
        for col in sample:
            if not isinstance(col, str) or not col.startswith("_ref_") or not col.endswith("_id"):
                continue
            parent = col[len("_ref_"):-len("_id")]
            if parent in name_set and parent != tname:
                parents[tname] = parent
                break
    job._hierarchy = parents
    return parents


def result_to_dict(job: Job) -> Dict[str, Any]:
    r = job.result
    if r is None:
        return {}

    raw = r.raw_metadata or {}
    tables_meta = raw.get("tables", {})
    summary = raw.get("summary", {})
    record_counts = summary.get("table_record_counts", {})

    parents = _infer_hierarchy(job, list(r.tables))
    children_map: Dict[str, List[str]] = {t: [] for t in r.tables}
    for child, parent in parents.items():
        if parent:
            children_map.setdefault(parent, []).append(child)

    table_summaries = []
    for tname in r.tables:
        tmeta = tables_meta.get(tname, {})
        tmd = tmeta.get("table_metadata", {})
        attrs = tmeta.get("attributes", {})
        anomaly_count = sum(
            len(flags) for flags in r.anomalies.get(tname, {}).values()
        )
        table_summaries.append({
            "name": tname,
            "record_count": record_counts.get(tname, tmd.get("total_records", 0)),
            "attribute_count": tmd.get("total_attributes", len(attrs)),
            "quality_score": r.quality_scores.get(tname, 0),
            "anomaly_count": anomaly_count,
            "parent": parents.get(tname),
            "children": sorted(children_map.get(tname, [])),
        })

    return {
        "job_id": job.job_id,
        "tables": r.tables,
        "quality_scores": r.quality_scores,
        "anomalies": r.anomalies,
        "overall_quality_score": r.overall_quality_score,
        "total_anomalies": r.total_anomalies,
        "table_summaries": table_summaries,
        "output_files": [os.path.basename(f) for f in r.output_files],
        "llm_insights": r.llm_insights,
        "raw_metadata": raw,
        "pipeline_info": r.pipeline_info,
    }


def get_table_detail(job: Job, table_name: str) -> Optional[Dict[str, Any]]:
    r = job.result
    if r is None:
        return None

    raw = r.raw_metadata or {}
    tables_meta = raw.get("tables", {})
    if table_name not in tables_meta:
        return None

    tmeta = tables_meta[table_name]
    tmd = tmeta.get("table_metadata", {})
    attrs = tmeta.get("attributes", {})
    anomalies = r.anomalies.get(table_name, {})

    data_preview = _load_data_preview(job, table_name)

    parents = _infer_hierarchy(job, list(r.tables))
    parent = parents.get(table_name)
    children = sorted([c for c, p in parents.items() if p == table_name])

    # Build child summaries so the detail view can render drilldown cards
    raw_tables = raw.get("tables", {})
    child_summaries = []
    for c in children:
        ctmd = raw_tables.get(c, {}).get("table_metadata", {})
        c_anoms = sum(len(v) for v in r.anomalies.get(c, {}).values())
        child_summaries.append({
            "name": c,
            "record_count": ctmd.get("total_records", 0),
            "attribute_count": ctmd.get("total_attributes", 0),
            "quality_score": r.quality_scores.get(c, 0),
            "anomaly_count": c_anoms,
        })

    return {
        "name": table_name,
        "record_count": tmd.get("total_records", 0),
        "quality_score": r.quality_scores.get(table_name, 0),
        "attributes": attrs,
        "anomalies": anomalies,
        "data_preview": data_preview,
        "parent": parent,
        "children": child_summaries,
    }


def get_download_path(job: Job) -> Optional[str]:
    if not job.result:
        return None
    for fp in job.result.output_files:
        if fp.endswith(".xlsx"):
            return fp
    return None


def _load_data_preview(job: Job, table_name: str, max_rows: int = 50) -> list:
    """Return first N rows for a table from the cached loaded tables."""
    tables = _get_loaded_tables(job)
    rows = tables.get(table_name)
    if not rows:
        lower = {k.lower(): v for k, v in tables.items()}
        rows = lower.get(table_name.lower())
    if not rows:
        return []
    preview = rows[:max_rows]
    out = []
    for r in preview:
        if isinstance(r, dict):
            out.append({k: _sanitize(v) for k, v in r.items()})
    return out


def _sanitize(v):
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, (_dt.datetime, _dt.date)):
        return v.isoformat()
    if isinstance(v, (list, dict)):
        try:
            return json.dumps(v, default=str)
        except Exception:
            return str(v)
    return str(v)


def _cleanup_stale_jobs():
    now = time.time()
    with _jobs_lock:
        stale = [jid for jid, j in _jobs.items() if now - j.created_at > _TTL_SECONDS]
        stale_jobs = [_jobs.pop(jid) for jid in stale if jid in _jobs]
    for job in stale_jobs:
        shutil.rmtree(job.output_dir, ignore_errors=True)
        if os.path.exists(job.file_path):
            os.remove(job.file_path)
