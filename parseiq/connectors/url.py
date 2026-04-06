"""Connector: HTTP/HTTPS URL. Requires: requests (already a core dep)."""
from __future__ import annotations
import os, tempfile
from typing import Any, Dict, Optional

def load(url: str, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Fetch *url* (JSON/CSV/XML) and return the normalised table dict."""
    import requests as _r
    resp = _r.get(url, headers=headers or {}, timeout=30)
    resp.raise_for_status()
    ct = resp.headers.get("Content-Type", "application/json").lower()
    ext = ".csv" if "csv" in ct else ".xml" if "xml" in ct else ".json"
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False, mode="wb") as tmp:
        tmp.write(resp.content)
        tmp_path = tmp.name
    try:
        from parseiq.file_loader.loader import FileLoader
        return FileLoader().load_file(tmp_path)
    finally:
        try: os.unlink(tmp_path)
        except OSError: pass
