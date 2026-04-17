"""Upload route — accept file, create job."""
from __future__ import annotations

import os
import tempfile

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ..services.pipeline_runner import create_job

router = APIRouter(prefix="/api", tags=["upload"])

ALLOWED_EXTENSIONS = {".json", ".csv", ".xml", ".xlsx", ".xls"}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB


@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    use_llm: bool = Form(False),
):
    """Upload a data file and create an analysis job."""
    # Validate extension
    _, ext = os.path.splitext(file.filename or "")
    if ext.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # Stream file to disk in chunks — fail early if too large
    tmp = tempfile.NamedTemporaryFile(
        delete=False, suffix=ext, prefix="parseiq_upload_"
    )
    file_size = 0
    try:
        while chunk := await file.read(1024 * 1024):  # 1 MB chunks
            file_size += len(chunk)
            if file_size > MAX_FILE_SIZE:
                tmp.close()
                os.remove(tmp.name)
                raise HTTPException(status_code=400, detail="File too large (max 100 MB)")
            tmp.write(chunk)
    finally:
        tmp.close()

    if file_size == 0:
        os.remove(tmp.name)
        raise HTTPException(status_code=400, detail="File is empty")

    job = create_job(file_path=tmp.name, filename=file.filename, use_llm=use_llm)

    return {
        "success": True,
        "data": {
            "job_id": job.job_id,
            "filename": file.filename,
            "file_size": file_size,
        },
    }
