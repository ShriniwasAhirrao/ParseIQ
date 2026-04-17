"""Analyze route — trigger pipeline, poll status."""
from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel

from ..services.pipeline_runner import get_job, run_analysis, set_llm_config

router = APIRouter(prefix="/api", tags=["analyze"])


class LLMSettings(BaseModel):
    provider: Optional[str] = None
    api_key: Optional[str] = None
    model: Optional[str] = None
    base_url: Optional[str] = None


class AnalyzeBody(BaseModel):
    llm: Optional[LLMSettings] = None


@router.post("/analyze/{job_id}")
async def start_analysis(job_id: str, body: Optional[AnalyzeBody] = Body(default=None)):
    """Kick off the ParseIQ pipeline for an uploaded file."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status not in ("pending",):
        raise HTTPException(status_code=409, detail=f"Job already {job.status}")

    if body and body.llm:
        set_llm_config(job, body.llm.model_dump(exclude_none=True))

    run_analysis(job_id)

    return {
        "success": True,
        "data": {"job_id": job_id, "status": "running"},
    }


@router.get("/status/{job_id}")
async def job_status(job_id: str, since: float = 0):
    """Poll analysis progress. Returns events with ts > `since`."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    events = [e for e in job.events if e["ts"] > since] if since else list(job.events)
    last_ts = job.events[-1]["ts"] if job.events else 0

    return {
        "success": True,
        "data": {
            "job_id": job.job_id,
            "status": job.status,
            "step": job.step,
            "progress": job.progress,
            "error": job.error,
            "events": events,
            "last_event_ts": last_ts,
        },
    }
