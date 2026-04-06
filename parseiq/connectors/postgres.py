"""Connector: PostgreSQL.  Requires: pip install parseiq[postgres]"""
from __future__ import annotations
from typing import Any, Dict

def load(conn_string: str, query: str, table_name: str = "query_result") -> Dict[str, Any]:
    try: import psycopg2, psycopg2.extras
    except ImportError as e: raise ImportError("pip install parseiq[postgres]") from e
    conn = psycopg2.connect(conn_string)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query)
            rows = [dict(r) for r in cur.fetchall()]
    finally: conn.close()
    return {table_name: rows}
