"""Pydantic models for API request/response schemas."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel


# ── Responses ──────────────────────────────────────────────

class APIResponse(BaseModel):
    success: bool
    data: Any = None
    error: Optional[str] = None


class UploadResponse(BaseModel):
    job_id: str
    filename: str
    file_size: int


class JobStatus(BaseModel):
    job_id: str
    status: str  # pending | running | completed | failed
    step: Optional[str] = None  # extract | enrich | output
    progress: int = 0  # 0-100
    error: Optional[str] = None


class TableSummary(BaseModel):
    name: str
    record_count: int
    attribute_count: int
    quality_score: float
    anomaly_count: int


class AnalysisResult(BaseModel):
    job_id: str
    tables: List[str]
    quality_scores: Dict[str, float]
    anomalies: Dict[str, Dict[str, List[str]]]
    overall_quality_score: float
    total_anomalies: int
    table_summaries: List[TableSummary]
    output_files: List[str]
    llm_insights: Optional[Dict[str, Any]] = None
    raw_metadata: Dict[str, Any] = {}
    pipeline_info: Dict[str, Any] = {}


class TableDetail(BaseModel):
    name: str
    record_count: int
    quality_score: float
    attributes: Dict[str, Any]
    anomalies: Dict[str, List[str]]
    data_preview: List[Dict[str, Any]]  # first N rows
