import sqlite3
from contextlib import contextmanager
from pathlib import Path
from collections.abc import Generator
from config.settings import get_settings

SCHEMA_PATH = Path(__file__).with_name('schema.sql')

@contextmanager
def connect(db_path: Path | None = None) -> Generator[sqlite3.Connection, None, None]:
    path = db_path or get_settings().database_path
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=15)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute('PRAGMA foreign_keys=ON')
        conn.execute('PRAGMA busy_timeout=15000')
        yield conn
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()

def initialize_database(db_path: Path | None = None) -> None:
    with connect(db_path) as conn:
        conn.executescript(SCHEMA_PATH.read_text(encoding='utf-8'))
