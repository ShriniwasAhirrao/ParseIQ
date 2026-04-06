"""Connector: local file (JSON, CSV, XML, Excel)."""
from __future__ import annotations
import os
from typing import Any, Dict, List


def load(path: str) -> Dict[str, Any]:
    """Load a local file and always return Dict[str, List[Dict]].

    FileLoader returns a list for flat formats (CSV, XML, single-table Excel)
    and a dict-of-tables for nested JSON.  Pipeline always needs a dict, so we
    normalise here using the filename stem as the table name.
    """
    from parseiq.file_loader.loader import FileLoader
    raw = FileLoader().load_file(path)

    if isinstance(raw, dict):
        # Already {table_name: [rows, ...]} — pass through
        return raw

    if isinstance(raw, list):
        # Flat file — wrap as single table named after the file stem
        stem = os.path.splitext(os.path.basename(path))[0]
        return {stem: raw}

    # Unexpected type — wrap defensively
    return {"data": [raw]}
