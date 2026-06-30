from __future__ import annotations

from functools import lru_cache
from math import sqrt
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
        self.indexes: list[dict] = []

    def add(self, rows: list[dict]) -> None:
        self._rows.extend(rows)

    def count_rows(self) -> int:
        return len(self._rows)

    def create_index(self, **kwargs) -> None:
        self.indexes.append(dict(kwargs))

    def top_k_by_cosine(self, query_embedding: list[float], limit: int) -> list[dict]:
        def _cosine(lhs: list[float], rhs: list[float]) -> float:
            lhs_len = len(lhs) if lhs is not None else 0
            rhs_len = len(rhs) if rhs is not None else 0
            if lhs_len == 0 or rhs_len == 0 or lhs_len != rhs_len:
                return 0.0
            dot = sum(a * b for a, b in zip(lhs, rhs))
            lhs_norm = sqrt(sum(a * a for a in lhs))
            rhs_norm = sqrt(sum(b * b for b in rhs))
            if lhs_norm == 0.0 or rhs_norm == 0.0:
                return 0.0
            return dot / (lhs_norm * rhs_norm)

        ranked: list[tuple[float, dict]] = []
        for row in self._rows:
            embedding = row.get("embedding") or []
            similarity = _cosine(query_embedding, embedding)
            ranked.append((similarity, row))

        ranked.sort(key=lambda item: item[0], reverse=True)
        output: list[dict] = []
        for similarity, row in ranked[:limit]:
            row_copy = dict(row)
            row_copy["_cosine_similarity"] = max(0.0, min(1.0, float(similarity)))
            output.append(row_copy)
        return output

    def get_row_by_id(self, chunk_id: str) -> dict | None:
        for row in self._rows:
            if str(row.get("id")) == chunk_id:
                return dict(row)
        return None


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
                ("actionability_score", pa.float64()),
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
        self._row_cache: dict[str, dict] = {}

    def _open_or_create_table(self):
        try:
            return self._db.open_table(self._table_name)
        except Exception:
            return self._db.create_table(self._table_name, schema=_chunk_schema(), mode="create", exist_ok=True)

    def store_chunks(self, chunks: list[ChunkDocument]) -> int:
        if not chunks:
            return 0
        table = self._open_or_create_table()
        rows = [chunk.model_dump(mode="json") for chunk in chunks]
        table.add(rows)
        for row in rows:
            chunk_id = str(row.get("id"))
            if chunk_id:
                self._row_cache[chunk_id] = dict(row)
        self._maybe_build_index(table)
        return len(chunks)

    def count_chunks(self) -> int:
        table = self._open_or_create_table()
        return table.count_rows()

    def _maybe_build_index(self, table) -> None:
        try:
            row_count = int(table.count_rows())
        except Exception:
            return
        if row_count < 256 or not hasattr(table, "create_index"):
            return
        try:
            num_partitions = max(1, min(256, row_count // 64))
            table.create_index(metric="cosine", num_partitions=num_partitions, num_sub_vectors=16, replace=True)
        except Exception:
            return

    @staticmethod
    def _cosine_similarity(lhs: list[float], rhs: list[float]) -> float:
        lhs_len = len(lhs) if lhs is not None else 0
        rhs_len = len(rhs) if rhs is not None else 0
        if lhs_len == 0 or rhs_len == 0 or lhs_len != rhs_len:
            return 0.0
        dot = sum(a * b for a, b in zip(lhs, rhs))
        lhs_norm = sqrt(sum(a * a for a in lhs))
        rhs_norm = sqrt(sum(b * b for b in rhs))
        if lhs_norm == 0.0 or rhs_norm == 0.0:
            return 0.0
        return dot / (lhs_norm * rhs_norm)

    def _cached_similarity_search(self, query_embedding: list[float], limit: int) -> list[dict] | None:
        if not self._row_cache:
            return None

        ranked: list[tuple[float, dict]] = []
        for row in self._row_cache.values():
            embedding = row.get("embedding")
            if embedding is None:
                embedding = []
            similarity = self._cosine_similarity(query_embedding, embedding)
            ranked.append((similarity, dict(row)))

        ranked.sort(key=lambda item: item[0], reverse=True)
        output: list[dict] = []
        for similarity, row in ranked[:limit]:
            row["_cosine_similarity"] = max(0.0, min(1.0, float(similarity)))
            output.append(row)
        return output

    def get_chunk_by_id(self, chunk_id: str) -> dict | None:
        if chunk_id in self._row_cache:
            return dict(self._row_cache[chunk_id])

        table = self._open_or_create_table()
        if hasattr(table, "get_row_by_id"):
            row = table.get_row_by_id(chunk_id)
            if row is not None:
                self._row_cache[chunk_id] = dict(row)
                return dict(row)

        for loader_name in ("to_pandas", "to_arrow", "to_list"):
            loader = getattr(table, loader_name, None)
            if loader is None:
                continue
            try:
                loaded = loader()
            except Exception:
                continue
            try:
                if loader_name == "to_pandas":
                    records = loaded.to_dict(orient="records")
                elif loader_name == "to_arrow":
                    records = loaded.to_pylist()
                else:
                    records = list(loaded)
            except Exception:
                continue
            for row in records:
                row_id = str(row.get("id", ""))
                if row_id:
                    self._row_cache[row_id] = dict(row)
            if chunk_id in self._row_cache:
                return dict(self._row_cache[chunk_id])

        return None

    def query_similar_chunks(self, query_embedding: list[float], limit: int = 20) -> list[dict]:
        if not query_embedding:
            return []

        cached_results = self._cached_similarity_search(query_embedding, limit)
        if cached_results is not None:
            return cached_results

        table = self._open_or_create_table()

        if hasattr(table, "top_k_by_cosine"):
            return table.top_k_by_cosine(query_embedding, limit)

        if not hasattr(table, "search"):
            return []

        # LanceDB returns distance by default; convert to cosine similarity where available.
        results: list[dict] = []
        search_candidates: list[dict] = []
        try:
            search_candidates = table.search(query_embedding).metric("cosine").limit(limit).to_list()
        except Exception:
            try:
                search_candidates = table.search(query_embedding).limit(limit).to_list()
            except Exception:
                return []

        for row in search_candidates:
            distance = row.get("_distance")
            if distance is not None:
                similarity = max(0.0, min(1.0, 1.0 - float(distance)))
            else:
                similarity = row.get("similarity_score")
                if similarity is None:
                    similarity = 0.0
                similarity = max(0.0, min(1.0, float(similarity)))
            row_copy = dict(row)
            row_copy["_cosine_similarity"] = similarity
            results.append(row_copy)

        return results


@lru_cache(maxsize=1)
def get_vector_store() -> LanceChunkStore:
    return LanceChunkStore()
