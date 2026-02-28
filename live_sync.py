"""
CadenceWorks Analytics Engine
Live Sync Module

Watches a folder for new/updated booking files and scores them automatically.
Supports:
  - Folder watch (drop Excel/CSV files in → auto-scored)
  - Simulated API poll (extensible to real booking system APIs)

All scored bookings are stored in a local SQLite database so the
dashboard can display them live without re-reading files.
"""

import sqlite3
import json
import time
import hashlib
import pandas as pd
from pathlib import Path
from datetime import datetime

from engine.predictive import _build_features, _rule_based_score
from engine import ingestor

DB_PATH    = Path("cadenceworks_live.db")
WATCH_DIR  = Path("watch_folder")


# ── Database setup ─────────────────────────────────────────────────────────────

def init_db():
    """Create tables if they don't exist."""
    WATCH_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS scored_bookings (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            appointment_id  TEXT,
            provider        TEXT,
            patient_type    TEXT,
            channel         TEXT,
            day_of_week     TEXT,
            lead_time_days  INTEGER,
            appointment_type TEXT,
            is_prime_slot   INTEGER,
            fee             REAL,
            risk_score      REAL,
            risk_band       TEXT,
            recommended_action TEXT,
            status          TEXT DEFAULT 'Pending',
            source_file     TEXT,
            scored_at       TEXT,
            reminded        INTEGER DEFAULT 0
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS sync_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            event       TEXT,
            detail      TEXT,
            ts          TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS processed_files (
            filepath    TEXT PRIMARY KEY,
            file_hash   TEXT,
            processed_at TEXT,
            rows_scored INTEGER
        )
    """)

    conn.commit()
    conn.close()


def log_event(event, detail=""):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO sync_log (event, detail, ts) VALUES (?, ?, ?)",
        (event, detail, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )
    conn.commit()
    conn.close()


# ── Risk scoring ───────────────────────────────────────────────────────────────

def _band(s):
    if s >= 70:  return "High Risk"
    if s >= 45:  return "Medium Risk"
    return "Low Risk"

def _action(s):
    if s >= 70:  return "Send reminders at 72hr, 24hr & 4hr"
    if s >= 45:  return "Send reminder at 24hr"
    return "Standard 24hr reminder only"

def score_dataframe(df, source_file=""):
    """Score a UDM dataframe and return rows ready to insert into DB."""
    features = _build_features(df)
    scores   = features.apply(_rule_based_score, axis=1).values
    now      = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows     = []

    for i, (_, row) in enumerate(df.iterrows()):
        s = round(float(scores[i]), 1)
        rows.append({
            "appointment_id":    str(row.get("appointment_id", f"APT-{i+1:04d}")),
            "provider":          str(row.get("provider", "")),
            "patient_type":      str(row.get("patient_type", "")),
            "channel":           str(row.get("channel", "")),
            "day_of_week":       str(row.get("day_of_week", "")),
            "lead_time_days":    int(row.get("lead_time_days", 0)),
            "appointment_type":  str(row.get("appointment_type", "")),
            "is_prime_slot":     int(bool(row.get("is_prime_slot", False))),
            "fee":               float(row.get("fee", 0)),
            "risk_score":        s,
            "risk_band":         _band(s),
            "recommended_action": _action(s),
            "status":            str(row.get("status", "Pending")),
            "source_file":       source_file,
            "scored_at":         now,
            "reminded":          0,
        })
    return rows


def insert_bookings(rows):
    """Insert scored bookings, skipping duplicates by appointment_id + source."""
    if not rows:
        return 0
    conn   = sqlite3.connect(DB_PATH)
    c      = conn.cursor()
    inserted = 0
    for r in rows:
        existing = c.execute(
            "SELECT id FROM scored_bookings WHERE appointment_id=? AND source_file=?",
            (r["appointment_id"], r["source_file"])
        ).fetchone()
        if not existing:
            c.execute("""
                INSERT INTO scored_bookings
                (appointment_id, provider, patient_type, channel, day_of_week,
                 lead_time_days, appointment_type, is_prime_slot, fee,
                 risk_score, risk_band, recommended_action, status, source_file, scored_at, reminded)
                VALUES
                (:appointment_id, :provider, :patient_type, :channel, :day_of_week,
                 :lead_time_days, :appointment_type, :is_prime_slot, :fee,
                 :risk_score, :risk_band, :recommended_action, :status, :source_file, :scored_at, :reminded)
            """, r)
            inserted += 1
    conn.commit()
    conn.close()
    return inserted


# ── File watcher ───────────────────────────────────────────────────────────────

def _file_hash(filepath):
    """MD5 hash of file contents to detect changes."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def _already_processed(filepath, file_hash):
    conn = sqlite3.connect(DB_PATH)
    row  = conn.execute(
        "SELECT file_hash FROM processed_files WHERE filepath=?",
        (str(filepath),)
    ).fetchone()
    conn.close()
    return row and row[0] == file_hash


def _mark_processed(filepath, file_hash, rows_scored):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT OR REPLACE INTO processed_files (filepath, file_hash, processed_at, rows_scored)
        VALUES (?, ?, ?, ?)
    """, (str(filepath), file_hash, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), rows_scored))
    conn.commit()
    conn.close()


def scan_watch_folder():
    """
    Scan the watch folder for new/changed Excel or CSV files.
    Returns (files_processed, bookings_scored).
    """
    WATCH_DIR.mkdir(exist_ok=True)
    files = list(WATCH_DIR.glob("*.xlsx")) + \
            list(WATCH_DIR.glob("*.xls"))  + \
            list(WATCH_DIR.glob("*.csv"))

    files_processed = 0
    bookings_scored  = 0

    for fp in files:
        try:
            fh = _file_hash(fp)
            if _already_processed(fp, fh):
                continue

            df, _ = ingestor.ingest(fp)
            rows   = score_dataframe(df, source_file=fp.name)
            n      = insert_bookings(rows)
            _mark_processed(fp, fh, n)

            files_processed += 1
            bookings_scored += n
            log_event("FILE_PROCESSED", f"{fp.name} → {n} new bookings scored")

        except Exception as e:
            log_event("FILE_ERROR", f"{fp.name}: {str(e)}")

    return files_processed, bookings_scored


# ── Read back from DB ──────────────────────────────────────────────────────────

def get_live_bookings(limit=200):
    """Fetch all scored bookings from DB, most recent first."""
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql_query(
        "SELECT * FROM scored_bookings ORDER BY scored_at DESC, risk_score DESC LIMIT ?",
        conn, params=(limit,)
    )
    conn.close()
    return df


def get_sync_log(limit=20):
    """Fetch recent sync events."""
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql_query(
        "SELECT ts, event, detail FROM sync_log ORDER BY id DESC LIMIT ?",
        conn, params=(limit,)
    )
    conn.close()
    return df


def get_live_stats():
    """Summary stats from the live DB."""
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    stats = {}

    stats["total"]       = c.execute("SELECT COUNT(*) FROM scored_bookings").fetchone()[0]
    stats["high_risk"]   = c.execute("SELECT COUNT(*) FROM scored_bookings WHERE risk_band='High Risk'").fetchone()[0]
    stats["medium_risk"] = c.execute("SELECT COUNT(*) FROM scored_bookings WHERE risk_band='Medium Risk'").fetchone()[0]
    stats["low_risk"]    = c.execute("SELECT COUNT(*) FROM scored_bookings WHERE risk_band='Low Risk'").fetchone()[0]
    stats["files"]       = c.execute("SELECT COUNT(*) FROM processed_files").fetchone()[0]
    stats["last_sync"]   = c.execute("SELECT ts FROM sync_log ORDER BY id DESC LIMIT 1").fetchone()
    stats["last_sync"]   = stats["last_sync"][0] if stats["last_sync"] else "Never"
    rev = c.execute("SELECT SUM(fee) FROM scored_bookings WHERE risk_band='High Risk'").fetchone()[0]
    stats["revenue_at_risk"] = float(rev) if rev else 0.0

    conn.close()
    return stats


def mark_reminded(appointment_id):
    """Mark a booking as reminded."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "UPDATE scored_bookings SET reminded=1 WHERE appointment_id=?",
        (appointment_id,)
    )
    conn.commit()
    conn.close()


def clear_all():
    """Wipe the database — for testing."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM scored_bookings")
    conn.execute("DELETE FROM sync_log")
    conn.execute("DELETE FROM processed_files")
    conn.commit()
    conn.close()
    log_event("DB_CLEARED", "All data wiped")
