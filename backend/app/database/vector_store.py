from __future__ import annotations

from functools import lru_cache
from pathlib import Path

try:
    import lancedb  # type: ignore
    import pyarrow as pa  # type: ignore
    _HAS_LANCEDB = True
except Exception:
    lancedb = None  # type: ignore
    pa = None  # type: ignore
    _HAS_LANCEDB = False

from ..config import settings
from ..models.knowledge import ChunkDocument


class _InMemoryTable:
    def __init__(self):
        self._rows: list[dict] = []

    def add(self, rows: list[dict]) -> None:
        self._rows.extend(rows)

    def count_rows(self) -> int:
        return len(self._rows)


class _InMemoryDB:
    def __init__(self):
        self._tables: dict[str, _InMemoryTable] = {}

    def open_table(self, name: str):
        if name not in self._tables:
            raise KeyError(name)
        return self._tables[name]

    def create_table(self, name: str, schema=None, mode=None, exist_ok=False):
        table = _InMemoryTable()
        self._tables[name] = table
        return table


def _chunk_schema() -> pa.Schema:
    if _HAS_LANCEDB and pa is not None:
        return pa.schema(
            [
                ("id", pa.string()),
                ("content", pa.string()),
                ("source_url", pa.string()),
                ("source_title", pa.string()),
                ("chunk_index", pa.int64()),
                ("parent_id", pa.string()),
                ("parent_content", pa.string()),
                ("section_title", pa.string()),
                ("embedding", pa.list_(pa.float32())),
                ("created_at", pa.string()),
                ("ttl_days", pa.int64()),
                ("verification_status", pa.string()),
                ("similarity_score", pa.float64()),
            ]
        )
    # Fallback: return None when pyarrow isn't available; callers should tolerate this in-memory mode.
    return None


class LanceChunkStore:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or settings.lancedb_path)
        self.path.mkdir(parents=True, exist_ok=True)
        if _HAS_LANCEDB and lancedb is not None:
            self._db = lancedb.connect(self.path)
        else:
            self._db = _InMemoryDB()
        self._table_name = "chunks"

    def _open_or_create_table(self):
        try:
            return self._db.open_table(self._table_name)
        except Exception:
            return self._db.create_table(self._table_name, schema=_chunk_schema(), mode="create", exist_ok=True)

    def store_chunks(self, chunks: list[ChunkDocument]) -> int:
        if not chunks:
            return 0
        table = self._open_or_create_table()
        table.add([chunk.model_dump(mode="json") for chunk in chunks])
        return len(chunks)

    def count_chunks(self) -> int:
        table = self._open_or_create_table()
        return table.count_rows()


@lru_cache(maxsize=1)
def get_vector_store() -> LanceChunkStore:
    return LanceChunkStore()
