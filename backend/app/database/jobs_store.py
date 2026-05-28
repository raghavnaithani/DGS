from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from uuid import uuid4

from ..models.jobs import IngestionRequest, JobRecord
from .connection import DEFAULT_SQLITE_PATH, get_connection, initialize_database


class SQLiteJobStore:
    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path) if db_path is not None else DEFAULT_SQLITE_PATH
        initialize_database(self.db_path)

    def _connect(self) -> sqlite3.Connection:
        return get_connection(self.db_path)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> JobRecord:
        request = json.loads(row["request_json"])
        result = json.loads(row["result_json"]) if row["result_json"] else None
        return JobRecord(
            id=row["id"],
            job_type=row["job_type"],
            request=request,
            status=row["status"],
            progress=row["progress"],
            current_step=row["current_step"],
            total_sources=row["total_sources"],
            scraped_sources=row["scraped_sources"],
            stored_chunks=row["stored_chunks"],
            result=result,
            error_message=row["error_message"],
            created_at=datetime.fromisoformat(row["created_at"].replace("Z", "+00:00")),
            updated_at=datetime.fromisoformat(row["updated_at"].replace("Z", "+00:00")),
        )

    def create_job(self, request: IngestionRequest) -> JobRecord:
        job_id = str(uuid4())
        payload_json = request.model_dump_json()
        now = self._now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO jobs (
                    id, job_type, request_json, status, progress, current_step,
                    total_sources, scraped_sources, stored_chunks, result_json,
                    error_message, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    "ingestion",
                    payload_json,
                    "queued",
                    0,
                    "queued",
                    0,
                    0,
                    0,
                    None,
                    None,
                    now,
                    now,
                ),
            )
            connection.commit()
        return self.get_job(job_id)

    def create_simulation_job(self, request: dict[str, object]) -> JobRecord:
        """Create a simulation job record. Request is a plain dict representing the simulation request."""
        job_id = str(uuid4())
        payload_json = json.dumps(request, ensure_ascii=False)
        now = self._now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO jobs (
                    id, job_type, request_json, status, progress, current_step,
                    total_sources, scraped_sources, stored_chunks, result_json,
                    error_message, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    "simulation",
                    payload_json,
                    "queued",
                    0,
                    "queued",
                    0,
                    0,
                    0,
                    None,
                    None,
                    now,
                    now,
                ),
            )
            connection.commit()
        return self.get_job(job_id)

    def update_job(self, job_id: str, **fields: object) -> JobRecord:
        if not fields:
            return self.get_job(job_id)
        allowed = {
            "status",
            "progress",
            "current_step",
            "total_sources",
            "scraped_sources",
            "stored_chunks",
            "result",
            "error_message",
        }
        unknown = sorted(set(fields) - allowed)
        if unknown:
            raise ValueError(f"Unsupported job fields: {', '.join(unknown)}")

        updates: list[str] = []
        values: list[object] = []
        for key, value in fields.items():
            column = "result_json" if key == "result" else key
            if key == "result":
                value = json.dumps(value, ensure_ascii=False)
            updates.append(f"{column} = ?")
            values.append(value)
        updates.append("updated_at = ?")
        values.append(self._now())
        values.append(job_id)

        with self._connect() as connection:
            connection.execute(f"UPDATE jobs SET {', '.join(updates)} WHERE id = ?", values)
            connection.commit()
        return self.get_job(job_id)

    def get_job(self, job_id: str) -> JobRecord:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return self._row_to_record(row)

    def claim_next_simulation_job(self) -> JobRecord | None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT *
                FROM jobs
                WHERE job_type = 'simulation' AND status = 'queued'
                ORDER BY created_at ASC
                LIMIT 1
                """
            ).fetchone()
            if row is None:
                connection.rollback()
                return None

            job_id = str(row["id"])
            now = self._now()
            connection.execute(
                """
                UPDATE jobs
                SET status = 'running', current_step = 'starting', progress = MAX(progress, 1), updated_at = ?
                WHERE id = ? AND status = 'queued'
                """,
                (now, job_id),
            )
            connection.commit()

            refreshed = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if refreshed is None:
                return None
            return self._row_to_record(refreshed)


def get_job_store(db_path: str | Path | None = None) -> SQLiteJobStore:
    return SQLiteJobStore(db_path)