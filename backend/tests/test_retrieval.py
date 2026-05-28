from __future__ import annotations

import json
from datetime import datetime, timezone

from app.database.connection import get_connection, initialize_database
from app.database.vector_store import LanceChunkStore
from app.engines import retriever as retriever_module
from app.engines.retriever import HybridRetriever, RankedChunk
from app.models.knowledge import ChunkDocument


class FakeEmbedder:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        outputs: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            if "finance" in lowered or "cashflow" in lowered:
                outputs.append([1.0, 0.0, 0.0])
            elif "health" in lowered or "hospital" in lowered:
                outputs.append([0.0, 1.0, 0.0])
            else:
                outputs.append([0.0, 0.0, 1.0])
        return outputs


def _chunk(
    *,
    chunk_id: str,
    content: str,
    source_url: str,
    source_title: str,
    chunk_index: int,
    embedding: list[float],
    parent_id: str | None = None,
    parent_content: str | None = None,
    section_title: str | None = None,
) -> ChunkDocument:
    return ChunkDocument(
        id=chunk_id,
        content=content,
        source_url=source_url,
        source_title=source_title,
        chunk_index=chunk_index,
        parent_id=parent_id,
        parent_content=parent_content,
        section_title=section_title,
        embedding=embedding,
        created_at=datetime.now(timezone.utc),
        verification_status="verified",
    )


class FakeIndexTable:
    def __init__(self):
        self.rows: list[dict] = []
        self.index_calls: list[dict] = []

    def add(self, rows: list[dict]) -> None:
        self.rows.extend(rows)

    def count_rows(self) -> int:
        return len(self.rows)

    def create_index(self, **kwargs) -> None:
        self.index_calls.append(dict(kwargs))


def _seed_sqlite_chunks(db_path, chunks: list[ChunkDocument]) -> None:
    with get_connection(db_path) as connection:
        for chunk in chunks:
            connection.execute(
                """
                INSERT INTO chunks (
                    id, session_id, content, source_url, source_title, chunk_index,
                    embedding_json, created_at, ttl_days, verification_status, similarity_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk.id,
                    None,
                    chunk.content,
                    chunk.source_url,
                    chunk.source_title,
                    chunk.chunk_index,
                    json.dumps(chunk.embedding),
                    chunk.created_at.isoformat(),
                    chunk.ttl_days,
                    chunk.verification_status,
                    chunk.similarity_score,
                ),
            )
        connection.commit()


def _seed_retriever(tmp_path):
    db_path = tmp_path / "retrieval.sqlite3"
    initialize_database(db_path)
    vector_store = LanceChunkStore(path=tmp_path / "lancedb")

    chunks = [
        _chunk(
            chunk_id="chunk-finance",
            content="Finance report on cashflow optimization and budget planning.",
            source_url="https://example.com/finance",
            source_title="Finance Report",
            chunk_index=0,
            embedding=[1.0, 0.0, 0.0],
        ),
        _chunk(
            chunk_id="chunk-health",
            content="Healthcare policy updates for hospitals and providers.",
            source_url="https://example.com/health",
            source_title="Health Policy",
            chunk_index=1,
            embedding=[0.0, 1.0, 0.0],
        ),
        _chunk(
            chunk_id="chunk-mixed",
            content="Budget allocation for healthcare systems and hospital finance teams.",
            source_url="https://example.com/mixed",
            source_title="Mixed Domain",
            chunk_index=2,
            embedding=[0.7, 0.3, 0.0],
        ),
    ]

    vector_store.store_chunks(chunks)
    _seed_sqlite_chunks(db_path, chunks)

    retriever = HybridRetriever(vector_store=vector_store, embedder=FakeEmbedder(), db_path=db_path)
    return retriever


def test_dense_and_bm25_return_expected_chunks(tmp_path):
    retriever = _seed_retriever(tmp_path)

    dense = retriever.dense_search("finance cashflow budget", limit=20, min_similarity=0.7)
    bm25 = retriever.bm25_search("finance budget", limit=20)

    assert dense
    assert bm25
    assert dense[0].id == "chunk-finance"
    assert any(item.id == "chunk-finance" for item in bm25)


def test_reranked_output_is_subset_of_input_candidates(tmp_path, monkeypatch):
    retriever = _seed_retriever(tmp_path)

    dense_ranked = [
        RankedChunk(
            id="chunk-finance",
            content="Finance report on cashflow optimization and budget planning.",
            source_url="https://example.com/finance",
            source_title="Finance Report",
            chunk_index=0,
            dense_similarity=0.95,
        ),
        RankedChunk(
            id="chunk-health",
            content="Healthcare policy updates for hospitals and providers.",
            source_url="https://example.com/health",
            source_title="Health Policy",
            chunk_index=1,
            dense_similarity=0.9,
        ),
    ]
    bm25_ranked = [
        RankedChunk(
            id="chunk-mixed",
            content="Budget allocation for healthcare systems and hospital finance teams.",
            source_url="https://example.com/mixed",
            source_title="Mixed Domain",
            chunk_index=2,
            bm25_score=0.8,
        )
    ]

    class FakeReranker:
        def rerank(self, query=None, passages=None, **kwargs):
            return [passages[2], passages[0]]

    monkeypatch.setattr(retriever, "dense_search", lambda *args, **kwargs: dense_ranked)
    monkeypatch.setattr(retriever, "bm25_search", lambda *args, **kwargs: bm25_ranked)
    monkeypatch.setattr(retriever_module, "get_flashrank_reranker", lambda: FakeReranker())

    response = retriever.assemble_evidence("finance budget", top_k=2)

    candidate_ids = {chunk.id for chunk in dense_ranked + bm25_ranked}
    assert len(response.evidence) == 2
    assert {item.id for item in response.evidence}.issubset(candidate_ids)
    assert all(item.context_type == "evidence" for item in response.evidence)


def test_bm25_fallback_when_dense_returns_nothing(tmp_path):
    retriever = _seed_retriever(tmp_path)

    dense = retriever.dense_search("hospital policy", limit=20, min_similarity=1.01)
    bm25 = retriever.bm25_search("hospital policy", limit=20)
    fused = retriever.fuse_rrf(dense, bm25, k=60)

    assert dense == []
    assert bm25
    assert fused
    assert fused[0].id == "chunk-health"


def test_bm25_strong_signal_skips_dense_search(tmp_path, monkeypatch):
    retriever = _seed_retriever(tmp_path)

    bm25_ranked = [
        RankedChunk(
            id="chunk-finance",
            content="Finance report on cashflow optimization and budget planning.",
            source_url="https://example.com/finance",
            source_title="Finance Report",
            chunk_index=0,
            bm25_score=0.95,
        ),
        RankedChunk(
            id="chunk-mixed",
            content="Budget allocation for healthcare systems and hospital finance teams.",
            source_url="https://example.com/mixed",
            source_title="Mixed Domain",
            chunk_index=2,
            bm25_score=0.7,
        ),
    ]

    def fail_dense(*args, **kwargs):
        raise AssertionError("dense_search should not be called for a strong BM25 signal")

    monkeypatch.setattr(retriever, "bm25_search", lambda *args, **kwargs: bm25_ranked)
    monkeypatch.setattr(retriever, "dense_search", fail_dense)

    response = retriever.assemble_evidence("finance budget", top_k=2)

    assert [item.id for item in response.evidence] == ["chunk-finance", "chunk-mixed"]
    assert response.expanded_context == []


def test_rrf_fusion_combines_rankings_and_preserves_scores(tmp_path):
    retriever = HybridRetriever(vector_store=LanceChunkStore(path=tmp_path / "lancedb"), embedder=FakeEmbedder())

    dense_ranked = [
        RankedChunk(
            id="a",
            content="A",
            source_url="https://example.com/a",
            source_title="A",
            chunk_index=0,
            dense_similarity=0.91,
        ),
        RankedChunk(
            id="b",
            content="B",
            source_url="https://example.com/b",
            source_title="B",
            chunk_index=1,
            dense_similarity=0.87,
        ),
    ]
    bm25_ranked = [
        RankedChunk(
            id="b",
            content="B",
            source_url="https://example.com/b",
            source_title="B",
            chunk_index=1,
            bm25_score=-4.2,
        ),
        RankedChunk(
            id="c",
            content="C",
            source_url="https://example.com/c",
            source_title="C",
            chunk_index=2,
            bm25_score=-3.5,
        ),
    ]

    fused = retriever.fuse_rrf(dense_ranked, bm25_ranked, k=60)

    assert [item.id for item in fused] == ["b", "a", "c"]
    assert fused[0].dense_similarity == 0.87
    assert fused[0].bm25_score == -4.2


def test_ann_index_is_created_after_threshold_crossed(tmp_path):
    store = LanceChunkStore(path=tmp_path / "lancedb")
    fake_table = FakeIndexTable()
    store._open_or_create_table = lambda: fake_table  # type: ignore[method-assign]

    chunks = [
        _chunk(
            chunk_id=f"chunk-{index}",
            content=f"Chunk {index} content about retrieval indexing.",
            source_url=f"https://example.com/{index}",
            source_title=f"Chunk {index}",
            chunk_index=index,
            embedding=[0.1, 0.2, 0.3],
        )
        for index in range(256)
    ]

    stored = store.store_chunks(chunks)

    assert stored == 256
    assert fake_table.index_calls
    assert fake_table.index_calls[0]["metric"] == "cosine"
    assert fake_table.index_calls[0]["num_partitions"] == 4
    assert fake_table.index_calls[0]["num_sub_vectors"] == 16
    assert fake_table.index_calls[0]["replace"] is True


def test_assemble_evidence_respects_top_k(tmp_path):
    retriever = _seed_retriever(tmp_path)

    evidence = retriever.assemble_evidence("finance healthcare budget", top_k=2)

    assert len(evidence.evidence) == 2
    assert all(item.citation for item in evidence.evidence)
    assert all(item.rrf_score > 0 for item in evidence.evidence)


def test_parent_context_is_returned_separately(tmp_path, monkeypatch):
    job_store_path = tmp_path / "retrieval.sqlite3"
    initialize_database(job_store_path)
    store = LanceChunkStore(path=tmp_path / "lancedb")

    parent = _chunk(
        chunk_id="chunk-parent",
        content="Parent section covering the broader budget strategy.",
        source_url="https://example.com/parent",
        source_title="Parent Source",
        chunk_index=0,
        embedding=[1.0, 0.0, 0.0],
    )
    child = _chunk(
        chunk_id="chunk-child",
        content="Child chunk with a specific finance detail.",
        source_url="https://example.com/child",
        source_title="Child Source",
        chunk_index=1,
        embedding=[0.9, 0.1, 0.0],
        parent_id="chunk-parent",
        parent_content="Parent section covering the broader budget strategy.",
        section_title="Finance Detail",
    )

    store.store_chunks([parent, child])
    retriever = HybridRetriever(vector_store=store, embedder=FakeEmbedder(), db_path=job_store_path)

    monkeypatch.setattr(retriever, "bm25_search", lambda *args, **kwargs: [])
    monkeypatch.setattr(retriever, "dense_search", lambda *args, **kwargs: [retriever._row_to_ranked_chunk(child.model_dump(mode="json"), dense_similarity=0.97)])

    response = retriever.assemble_evidence("finance detail", top_k=1)

    assert len(response.evidence) == 1
    assert response.evidence[0].id == "chunk-child"
    assert len(response.expanded_context) == 1
    assert response.expanded_context[0].id == "chunk-parent"
    assert response.expanded_context[0].context_type == "parent"
