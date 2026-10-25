"""
CloudVisionAI — Lightweight SQLite Persistence Service.

Stores lightweight run history and metadata:
- Run ID and timestamps
- Mode (live acquisition or cached baseline)
- AOI / Scene metadata (dates, CRS, shapes)
- Stage completion status and execution timings
- Safe output file paths (GeoTIFF on disk)
- Summary evaluation metrics (JSON string)

Strict constraints:
- NO raster arrays, GeoTIFF contents, or raw bytes stored in SQLite.
- NO credentials, API keys, or access tokens stored in SQLite.
- Safe initialization with parameterized queries and explicit connection closure.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_DB_DIR = Path(__file__).resolve().parent.parent.parent.parent / "acquisition" / "data_cache"
DEFAULT_DB_PATH = DEFAULT_DB_DIR / "cvai_history.db"


def get_db_path(custom_path: Optional[Path | str] = None) -> Path:
    if custom_path:
        p = Path(custom_path)
    else:
        p = DEFAULT_DB_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def init_db(db_path: Optional[Path | str] = None) -> None:
    """Initialize the SQLite schema if it doesn't already exist."""
    path = get_db_path(db_path)
    conn = sqlite3.connect(str(path))
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS run_history (
                run_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                mode TEXT NOT NULL,
                aoi_label TEXT,
                s2_date TEXT,
                s1_date TEXT,
                crs TEXT,
                dimensions TEXT,
                execution_time_s REAL,
                device TEXT,
                ecrformer_output_path TEXT,
                opensr_output_path TEXT,
                reference_classification TEXT,
                metrics_summary_json TEXT
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def record_run(
    run_id: str,
    mode: str,
    aoi_label: Optional[str] = None,
    s2_date: Optional[str] = None,
    s1_date: Optional[str] = None,
    crs: Optional[str] = None,
    dimensions: Optional[str] = None,
    execution_time_s: Optional[float] = None,
    device: Optional[str] = None,
    ecrformer_output_path: Optional[str] = None,
    opensr_output_path: Optional[str] = None,
    reference_classification: Optional[str] = None,
    metrics_summary: Optional[Dict[str, Any]] = None,
    db_path: Optional[Path | str] = None,
) -> None:
    """Safely record a run metadata entry."""
    init_db(db_path)
    path = get_db_path(db_path)
    now_str = datetime.now(timezone.utc).isoformat()
    metrics_json = json.dumps(metrics_summary) if metrics_summary else None

    conn = sqlite3.connect(str(path))
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO run_history (
                run_id, created_at, mode, aoi_label, s2_date, s1_date,
                crs, dimensions, execution_time_s, device,
                ecrformer_output_path, opensr_output_path,
                reference_classification, metrics_summary_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                now_str,
                mode,
                aoi_label,
                s2_date,
                s1_date,
                crs,
                dimensions,
                execution_time_s,
                device,
                ecrformer_output_path,
                opensr_output_path,
                reference_classification,
                metrics_json,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_recent_runs(limit: int = 10, db_path: Optional[Path | str] = None) -> List[Dict[str, Any]]:
    """Retrieve recent run history records."""
    init_db(db_path)
    path = get_db_path(db_path)
    conn = sqlite3.connect(str(path))
    try:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT run_id, created_at, mode, aoi_label, s2_date, s1_date,
                   crs, dimensions, execution_time_s, device,
                   ecrformer_output_path, opensr_output_path,
                   reference_classification, metrics_summary_json
            FROM run_history
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cursor.fetchall()

        results = []
        for r in rows:
            item = dict(r)
            if item.get("metrics_summary_json"):
                try:
                    item["metrics_summary"] = json.loads(item["metrics_summary_json"])
                except Exception:
                    item["metrics_summary"] = None
            results.append(item)
        return results
    finally:
        conn.close()
