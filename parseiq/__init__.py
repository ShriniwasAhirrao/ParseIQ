"""
ParseIQ — AI-powered data quality and metadata analysis.

Quickstart::

    from parseiq import Pipeline

    result = Pipeline("data.json").run(llm=False)
    print(result.overall_quality_score)
    print(result.output_files)
"""
from .pipeline import Pipeline, MetadataEnrichmentAgent  # noqa: F401
from .result import PipelineResult                       # noqa: F401
from .config import Config                               # noqa: F401
from . import alerts                                     # noqa: F401

__version__ = "0.0.4"
__all__ = ["Pipeline", "PipelineResult", "Config", "alerts", "MetadataEnrichmentAgent"]
