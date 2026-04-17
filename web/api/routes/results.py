"""Results route — fetch analysis results, download reports."""
from __future__ import annotations

import os
import re

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ..services.pipeline_runner import (
    get_download_path,
    get_job,
    get_table_detail,
    result_to_dict,
)

router = APIRouter(prefix="/api", tags=["results"])


@router.get("/results/{job_id}")
async def get_results(job_id: str):
    """Full analysis results."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "completed":
        raise HTTPException(
            status_code=409,
            detail=f"Job not ready (status: {job.status})",
        )

    return {"success": True, "data": result_to_dict(job)}


@router.get("/results/{job_id}/table/{table_name:path}")
async def get_table(job_id: str, table_name: str):
    """Detail view for a single table."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "completed":
        raise HTTPException(status_code=409, detail="Job not ready")

    detail = get_table_detail(job, table_name)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"Table '{table_name}' not found")

    return {"success": True, "data": detail}


@router.get("/results/{job_id}/download")
async def download_report(job_id: str):
    """Download the Excel report."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "completed":
        raise HTTPException(status_code=409, detail="Job not ready")

    path = get_download_path(job)
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Report file not found")

    # Validate the path is within the job's output directory
    real_path = os.path.realpath(path)
    real_output = os.path.realpath(job.output_dir)
    if not real_path.startswith(real_output):
        raise HTTPException(status_code=403, detail="Access denied")

    # Sanitize filename for Content-Disposition header
    safe_name = re.sub(r'[^\w\s\-.]', '_', job.filename or 'data')

    return FileResponse(
        path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=f"ParseIQ_{safe_name}_report.xlsx",
    )
