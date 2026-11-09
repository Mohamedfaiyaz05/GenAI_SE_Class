
import csv
import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from crewai.tools import tool


logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
AUDIT_FILE = BASE_DIR / "storage" / "fallback_audit.csv"

# Protect concurrent writes within one Python process
_write_lock = threading.Lock()

CSV_FIELDS = [
    "event_id",
    "timestamp_utc",
    "question",
    "fallback_reason",
    "source_urls",
    "fallback_status",
    "monitoring_verdict",
    "monitoring_reason",
]


def _safe_csv_value(value) -> str:
    """Convert values to safe CSV cell strings."""

    text = str(value) if value is not None else ""

    # Prevent spreadsheet formula injection
    if text.lstrip().startswith(("=", "+", "-", "@")):
        text = "'" + text

    return text


def write_fallback_audit(
    question: str,
    fallback_reason: str,
    source_urls: list[str],
    fallback_status: str,
    monitoring_verdict: str,
    monitoring_reason: str,
) -> str:
    """Append one fallback audit event and return its ID."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("A non-empty audit question is required")

    if not isinstance(source_urls, list):
        raise ValueError("source_urls must be a list")

    AUDIT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    event_id = str(uuid4())

    record = {
        "event_id": event_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "question": _safe_csv_value(question),
        "fallback_reason": _safe_csv_value(fallback_reason),
        "source_urls": json.dumps(source_urls),
        "fallback_status": _safe_csv_value(fallback_status),
        "monitoring_verdict": _safe_csv_value(monitoring_verdict),
        "monitoring_reason": _safe_csv_value(monitoring_reason),
    }

    with _write_lock:

        file_exists = (
            AUDIT_FILE.exists()
            and AUDIT_FILE.stat().st_size > 0
        )

        with AUDIT_FILE.open(
            mode="a",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=CSV_FIELDS
            )

            if not file_exists:
                writer.writeheader()

            writer.writerow(record)

    logger.info("Fallback audit recorded: %s", event_id)

    return event_id


@tool("record_fallback_audit")
def record_fallback_audit(
    question: str,
    fallback_reason: str,
    source_urls: list[str],
    fallback_status: str,
    monitoring_verdict: str,
    monitoring_reason: str,
) -> str:
    """
    Record a medical information fallback event in the CSV audit.

    Only use sanitized, non-sensitive question summaries.
    The CrewAI Flow should call the underlying audit function
    directly to guarantee that required events are logged.
    """

    event_id = write_fallback_audit(
        question=question,
        fallback_reason=fallback_reason,
        source_urls=source_urls,
        fallback_status=fallback_status,
        monitoring_verdict=monitoring_verdict,
        monitoring_reason=monitoring_reason,
    )

    return json.dumps({
        "status": "success",
        "event_id": event_id
    })
