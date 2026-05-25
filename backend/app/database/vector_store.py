from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import lancedb
import pyarrow as pa

from ..config import settings
from ..models.knowledge import ChunkDocument


def _chunk_schema() -> pa.Schema:
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


class LanceChunkStore:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or settings.lancedb_path)
        self.path.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(self.path)
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
