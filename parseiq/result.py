"""PipelineResult — structured return value from Pipeline.run()."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class PipelineResult:
    """Structured result returned by ``Pipeline.run()``.

    Attributes
    ----------
    tables:
        Names of every table extracted from the source.
    quality_scores:
        ``{table_name: score}`` where score is 0–100.
    anomalies:
        ``{table_name: {column: [flag, ...]}}`` — only columns with flags.
    output_files:
        Absolute paths of every file written to the output directory.
    llm_insights:
        Raw LLM assessment dict when ``llm=True``, else ``None``.
    alerts_fired:
        Every alert that fired during this run (empty list when no rules matched).
    raw_metadata:
        Full Step-1 technical metadata (same structure as raw_metadata.json).
    pipeline_info:
        Timing, model, and processing-mode metadata.
    """

    tables: List[str]
    quality_scores: Dict[str, float]
    anomalies: Dict[str, Dict[str, List[str]]]
    output_files: List[str]
    llm_insights: Optional[Dict[str, Any]]
    alerts_fired: List[Dict[str, Any]] = field(default_factory=list)
    raw_metadata: Dict[str, Any] = field(default_factory=dict)
    pipeline_info: Dict[str, Any] = field(default_factory=dict)

    @property
    def overall_quality_score(self) -> float:
        """Average quality score across all tables (0–100)."""
        if not self.quality_scores:
            return 0.0
        return round(sum(self.quality_scores.values()) / len(self.quality_scores), 2)

    @property
    def total_anomalies(self) -> int:
        """Total anomaly flag count across all tables and columns."""
        return sum(
            len(flags)
            for col_map in self.anomalies.values()
            for flags in col_map.values()
        )

    @property
    def llm_grade(self) -> Optional[str]:
        """Quality grade from LLM (e.g. ``'B'``) or ``None`` when LLM was skipped."""
        if self.llm_insights:
            return self.llm_insights.get("overall_assessment", {}).get("quality_grade")
        return None

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"PipelineResult("
            f"tables={self.tables}, "
            f"overall_quality={self.overall_quality_score}, "
            f"total_anomalies={self.total_anomalies}, "
            f"llm_grade={self.llm_grade!r})"
        )
