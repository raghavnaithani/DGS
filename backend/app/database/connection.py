from __future__ import annotations

import sqlite3
from pathlib import Path

from .schema import apply_schema

DEFAULT_SQLITE_PATH = Path(__file__).resolve().with_name("dgs_phase1.sqlite3")


def get_connection(db_path: str | Path | None = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path is not None else DEFAULT_SQLITE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(path, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database(db_path: str | Path | None = None) -> sqlite3.Connection:
    connection = get_connection(db_path)
    apply_schema(connection)
    connection.commit()
    return connection

