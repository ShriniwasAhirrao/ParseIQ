"""Connector: MongoDB.  Requires: pip install parseiq[mongodb]"""
from __future__ import annotations
from typing import Any, Dict, Optional

def load(conn_string: str, collection_name: str, database_name: Optional[str] = None, limit: int = 0) -> Dict[str, Any]:
    try: import pymongo
    except ImportError as e: raise ImportError("pip install parseiq[mongodb]") from e
    client = pymongo.MongoClient(conn_string)
    try:
        db_name = database_name or client.get_default_database().name
        cur = client[db_name][collection_name].find({}, {"_id": 0})
        if limit: cur = cur.limit(limit)
        rows = list(cur)
    finally: client.close()
    return {collection_name: rows}
