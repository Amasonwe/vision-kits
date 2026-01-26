import os
import sqlite3
import json
from typing import Optional, Dict, Any
from datetime import datetime

DB_DIR = "tmp"
DB_PATH = os.path.join(DB_DIR, "detections.db")
os.makedirs(DB_DIR, exist_ok=True)


def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT,
            version TEXT,
            image_path TEXT,
            annotated_path TEXT,
            detections_json TEXT,
            created_at TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def save_detection(category: str, version: str, image_path: str, annotated_path: str, detections: Any) -> int:
    init_db()
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO detections (category, version, image_path, annotated_path, detections_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (category, version, image_path, annotated_path, json.dumps(detections, ensure_ascii=False), datetime.utcnow().isoformat())
    )
    conn.commit()
    rowid = cur.lastrowid
    conn.close()
    return rowid


def get_detection(record_id: int) -> Optional[Dict]:
    init_db()
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM detections WHERE id = ?", (record_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return {
        "id": row["id"],
        "category": row["category"],
        "version": row["version"],
        "image_path": row["image_path"],
        "annotated_path": row["annotated_path"],
        "detections": json.loads(row["detections_json"]),
        "created_at": row["created_at"]
    }
