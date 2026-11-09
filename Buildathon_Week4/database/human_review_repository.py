"""Persistence for pending human reviews and their audit history."""
from datetime import datetime, timezone
from pathlib import Path
import sqlite3


def _connect(db_path: Path) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path), timeout=15)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout=15000")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS human_reviews (
            ticket_id TEXT PRIMARY KEY,
            status TEXT NOT NULL CHECK (status IN ('Pending','Approved','Rejected')),
            trigger_reason TEXT NOT NULL,
            created_at TEXT NOT NULL,
            reviewed_at TEXT,
            reviewer TEXT,
            reviewer_note TEXT
        );
        CREATE TABLE IF NOT EXISTS human_review_audit (
            audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticket_id TEXT NOT NULL,
            action TEXT NOT NULL,
            reviewer TEXT NOT NULL,
            note TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL
        );
        """
    )
    return connection


def register_pending_review(db_path: Path, ticket_id: str, reason: str) -> bool:
    if not ticket_id.strip() or not reason.strip():
        raise ValueError("A ticket ID and review reason are required")
    connection = _connect(db_path)
    try:
        cursor = connection.execute(
            """INSERT OR IGNORE INTO human_reviews
               (ticket_id, status, trigger_reason, created_at)
               VALUES (?, 'Pending', ?, ?)""",
            (ticket_id, reason, datetime.now(timezone.utc).isoformat()),
        )
        if cursor.rowcount == 1:
            connection.execute(
                """INSERT INTO human_review_audit
                   (ticket_id, action, reviewer, note, created_at)
                   VALUES (?, 'Pending', 'System', ?, ?)""",
                (ticket_id, reason, datetime.now(timezone.utc).isoformat()),
            )
        connection.commit()
        return cursor.rowcount == 1
    finally:
        connection.close()


def list_reviews(db_path: Path) -> list[dict]:
    connection = _connect(db_path)
    try:
        return [dict(row) for row in connection.execute(
            "SELECT * FROM human_reviews ORDER BY created_at DESC, ticket_id"
        ).fetchall()]
    finally:
        connection.close()


def list_review_audit(db_path: Path) -> list[dict]:
    connection = _connect(db_path)
    try:
        return [dict(row) for row in connection.execute(
            "SELECT * FROM human_review_audit ORDER BY created_at DESC, audit_id DESC"
        ).fetchall()]
    finally:
        connection.close()


def record_review_decision(
    db_path: Path,
    ticket_id: str,
    *,
    decision: str,
    reviewer: str,
    note: str,
) -> dict:
    if decision not in {"Approved", "Rejected"}:
        raise ValueError("Decision must be Approved or Rejected")
    if not reviewer.strip():
        raise ValueError("Reviewer name is required")
    connection = _connect(db_path)
    try:
        existing = connection.execute(
            "SELECT status FROM human_reviews WHERE ticket_id = ?", (ticket_id,)
        ).fetchone()
        if existing is None:
            raise LookupError("Human review request was not found")
        if existing["status"] != "Pending":
            raise ValueError("This review has already been decided")
        reviewed_at = datetime.now(timezone.utc).isoformat()
        cursor = connection.execute(
            """UPDATE human_reviews
               SET status = ?, reviewed_at = ?, reviewer = ?, reviewer_note = ?
               WHERE ticket_id = ? AND status = 'Pending'""",
            (decision, reviewed_at, reviewer.strip(), note.strip(), ticket_id),
        )
        if cursor.rowcount != 1:
            raise ValueError("This review has already been decided")
        connection.execute(
            """INSERT INTO human_review_audit
               (ticket_id, action, reviewer, note, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (ticket_id, decision, reviewer.strip(), note.strip(), reviewed_at),
        )
        connection.commit()
        return {
            "ticket_id": ticket_id,
            "status": decision,
            "reviewed_at": reviewed_at,
            "reviewer": reviewer.strip(),
            "reviewer_note": note.strip(),
        }
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()
