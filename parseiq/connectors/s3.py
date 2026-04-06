"""Connector: Amazon S3.  Requires: pip install parseiq[s3]"""
from __future__ import annotations
import os, tempfile
from pathlib import PurePosixPath
from typing import Any, Dict, Optional

def load(s3_uri: str, aws_access_key_id=None, aws_secret_access_key=None, region_name=None) -> Dict[str, Any]:
    try: import boto3
    except ImportError as e: raise ImportError("pip install parseiq[s3]") from e
    if not s3_uri.startswith("s3://"): raise ValueError(f"URI must start with s3://: {s3_uri!r}")
    bucket, _, key = s3_uri[5:].partition("/")
    ext = PurePosixPath(key).suffix or ".json"
    kw = {k: v for k, v in [("aws_access_key_id", aws_access_key_id),
                              ("aws_secret_access_key", aws_secret_access_key),
                              ("region_name", region_name)] if v}
    s3 = boto3.client("s3", **kw)
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        s3.download_fileobj(bucket, key, tmp); tmp_path = tmp.name
    try:
        from parseiq.file_loader.loader import FileLoader
        return FileLoader().load_file(tmp_path)
    finally:
        try: os.unlink(tmp_path)
        except OSError: pass
