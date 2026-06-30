from __future__ import annotations

from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

from ..config import settings
from ..database.connection import DEFAULT_SQLITE_PATH, get_connection
from ..database.vector_store import LanceChunkStore
from ..engines.embedder import ChunkEmbedder, get_embedder
from ..models.knowledge import RetrievedEvidenceChunk, RetrievalResponse

try:
    from flashrank import Ranker, RerankRequest  # type: ignore
except Exception:
    Ranker = None  # type: ignore
    RerankRequest = None  # type: ignore


@dataclass(slots=True)
class RankedChunk:
    id: str
    content: str
    source_url: str
    source_title: str | None
    chunk_index: int
    parent_id: str | None = None
    parent_content: str | None = None
    section_title: str | None = None
    dense_similarity: float | None = None
    bm25_score: float | None = None
    bm25_raw_score: float | None = None
    rrf_score: float = 0.0
    actionability_score: float = 0.0
    context_type: str = "evidence"

    def to_response(self) -> RetrievedEvidenceChunk:
        title = self.source_title.strip() if self.source_title else ""
        citation_title = title if title else self.source_url
        return RetrievedEvidenceChunk(
            id=self.id,
            content=self.content,
            source_url=self.source_url,
            source_title=self.source_title,
            chunk_index=self.chunk_index,
            parent_id=self.parent_id,
            parent_content=self.parent_content,
            section_title=self.section_title,
            citation=f"{citation_title} ({self.source_url})",
            rrf_score=self.rrf_score,
            dense_similarity=self.dense_similarity,
            bm25_score=self.bm25_score,
            actionability_score=self.actionability_score,
            context_type=self.context_type,
        )


def _candidate_id(candidate: Any) -> str | None:
    if isinstance(candidate, dict):
        if candidate.get("id"):
            return str(candidate["id"])
        metadata = candidate.get("metadata")
        if isinstance(metadata, dict) and metadata.get("id"):
            return str(metadata["id"])
        if candidate.get("passage_id"):
            return str(candidate["passage_id"])
    for attr in ("id", "passage_id"):
        value = getattr(candidate, attr, None)
        if value:
            return str(value)
    metadata = getattr(candidate, "metadata", None)
    if isinstance(metadata, dict) and metadata.get("id"):
        return str(metadata["id"])
    return None


def _rank_bonus(rank_index: int, top_rank_bonus: float) -> float:
    if rank_index == 1:
        return top_rank_bonus
    if rank_index in (2, 3):
        return top_rank_bonus * 0.4
    return 0.0


@lru_cache(maxsize=1)
def get_flashrank_reranker():
    if not settings.retrieval_enable_reranking:
        return None
    if Ranker is None:
        return None
    try:
        return Ranker()
    except Exception:
        return None


class HybridRetriever:
    def __init__(
        self,
        *,
        vector_store: LanceChunkStore,
        embedder: ChunkEmbedder | None = None,
        db_path: str | Path | None = None,
    ):
        self.vector_store = vector_store
        self.embedder = embedder or get_embedder()
        self.db_path = Path(db_path) if db_path is not None else DEFAULT_SQLITE_PATH

    def dense_search(
        self,
        query: str,
        *,
        limit: int | None = None,
        min_similarity: float | None = None,
    ) -> list[RankedChunk]:
        dense_limit = limit or settings.retrieval_dense_limit
        threshold = min_similarity if min_similarity is not None else settings.retrieval_similarity_threshold
        query_embedding = self.embedder.embed_texts([query])[0]
        candidates = self.vector_store.query_similar_chunks(query_embedding, limit=dense_limit)

        results: list[RankedChunk] = []
        for row in candidates:
            similarity = float(row.get("_cosine_similarity", 0.0))
            if similarity < threshold:
                continue
            results.append(
                self._row_to_ranked_chunk(row, dense_similarity=similarity)
            )

        return [chunk for chunk in results if chunk.id and chunk.content and chunk.source_url][:dense_limit]

    def bm25_search(self, query: str, *, limit: int | None = None) -> list[RankedChunk]:
        bm25_limit = limit or settings.retrieval_bm25_limit
        safe_query = self._safe_fts_query(query)
        if not safe_query:
            return []
        with get_connection(self.db_path) as connection:
            rows = connection.execute(
                """
                SELECT
                    c.id,
                    c.content,
                    c.source_url,
                    c.source_title,
                    c.chunk_index,
                    bm25(chunks_fts) AS bm25_score
                FROM chunks_fts
                JOIN chunks c ON c.id = chunks_fts.id
                WHERE chunks_fts MATCH ?
                ORDER BY bm25_score ASC
                LIMIT ?
                """,
                (safe_query, bm25_limit),
            ).fetchall()

        output: list[RankedChunk] = []
        for row in rows:
            normalized_score = 1.0 / (1.0 + abs(float(row["bm25_score"])))
            enriched = self.vector_store.get_chunk_by_id(str(row["id"])) or {}
            merged_row = {
                **enriched,
                "id": row["id"],
                "content": row["content"],
                "source_url": row["source_url"],
                "source_title": row["source_title"],
                "chunk_index": row["chunk_index"],
                "bm25_score": normalized_score,
                "bm25_raw_score": float(row["bm25_score"]),
            }
            output.append(
                self._row_to_ranked_chunk(merged_row, bm25_score=normalized_score, bm25_raw_score=float(row["bm25_score"]))
            )
        return output

    @staticmethod
    def _safe_fts_query(query: str) -> str:
        terms = re.findall(r"[A-Za-z0-9]+", query)
        if not terms:
            return ""
        return " AND ".join(f'"{term}"' for term in terms)

    def fuse_rrf(
        self,
        dense_ranked: list[RankedChunk],
        bm25_ranked: list[RankedChunk],
        *,
        k: int | None = None,
        weights: dict[str, float] | None = None,
        top_rank_bonus: float | None = None,
        query_weight_multiplier: float = 2.0,
    ) -> list[RankedChunk]:
        rrf_k = int(k or settings.retrieval_rrf_k)
        active_weights = weights or {
            "dense": float(settings.retrieval_rrf_dense_weight),
            "bm25": float(settings.retrieval_rrf_bm25_weight),
        }
        active_top_rank_bonus = settings.retrieval_rrf_top_rank_bonus if top_rank_bonus is None else float(top_rank_bonus)
        merged: dict[str, RankedChunk] = {}

        def _upsert(chunk: RankedChunk) -> RankedChunk:
            existing = merged.get(chunk.id)
            if existing is None:
                merged[chunk.id] = chunk
                return chunk
            if existing.dense_similarity is None and chunk.dense_similarity is not None:
                existing.dense_similarity = chunk.dense_similarity
            if existing.bm25_score is None and chunk.bm25_score is not None:
                existing.bm25_score = chunk.bm25_score
            if not existing.content and chunk.content:
                existing.content = chunk.content
            if not existing.source_url and chunk.source_url:
                existing.source_url = chunk.source_url
            if existing.source_title is None and chunk.source_title is not None:
                existing.source_title = chunk.source_title
            if existing.parent_id is None and chunk.parent_id is not None:
                existing.parent_id = chunk.parent_id
            if existing.parent_content is None and chunk.parent_content is not None:
                existing.parent_content = chunk.parent_content
            if existing.section_title is None and chunk.section_title is not None:
                existing.section_title = chunk.section_title
            return existing

        for rank_index, chunk in enumerate(dense_ranked, start=1):
            target = _upsert(chunk)
            target.rrf_score += query_weight_multiplier * active_weights.get("dense", 1.0) * (
                1.0 / (rrf_k + rank_index) + _rank_bonus(rank_index, active_top_rank_bonus)
            )

        for rank_index, chunk in enumerate(bm25_ranked, start=1):
            target = _upsert(chunk)
            target.rrf_score += query_weight_multiplier * active_weights.get("bm25", 1.0) * (
                1.0 / (rrf_k + rank_index) + _rank_bonus(rank_index, active_top_rank_bonus)
            )

        ranked = sorted(
            merged.values(),
            key=lambda item: (item.rrf_score, item.dense_similarity or 0.0, item.actionability_score),
            reverse=True,
        )
        return ranked

    def assemble_evidence(self, query: str, top_k: int = 10) -> RetrievalResponse:
        bm25_results = self.bm25_search(query, limit=settings.retrieval_bm25_limit)
        if self._should_skip_dense(bm25_results):
            selected = bm25_results[:top_k]
            return self._build_response(query=query, top_k=top_k, evidence=selected, reranked=False)

        dense_results = self.dense_search(query, limit=settings.retrieval_dense_limit)
        fused = self.fuse_rrf(
            dense_results,
            bm25_results,
            k=settings.retrieval_rrf_k,
            weights={
                "dense": float(settings.retrieval_rrf_dense_weight),
                "bm25": float(settings.retrieval_rrf_bm25_weight),
            },
            top_rank_bonus=float(settings.retrieval_rrf_top_rank_bonus),
            query_weight_multiplier=2.0,
        )
        candidates = fused[:30]
        selected = self._rerank_candidates(query, candidates)[:top_k]
        return self._build_response(query=query, top_k=top_k, evidence=selected, reranked=True)

    def _build_response(self, *, query: str, top_k: int, evidence: list[RankedChunk], reranked: bool) -> RetrievalResponse:
        expanded_context = self._expand_parent_context(evidence, top_k=top_k) if settings.retrieval_expand_parents else []
        if reranked:
            for chunk in evidence:
                chunk.context_type = "evidence"
        return RetrievalResponse(
            query=query,
            top_k=top_k,
            evidence=[chunk.to_response() for chunk in evidence],
            expanded_context=expanded_context,
        )

    def _should_skip_dense(self, bm25_results: list[RankedChunk]) -> bool:
        if len(bm25_results) < 2:
            return False
        top_score = bm25_results[0].bm25_score or 0.0
        second_score = bm25_results[1].bm25_score or 0.0
        gap = top_score - second_score
        return top_score >= float(settings.retrieval_bm25_skip_threshold) and gap >= float(settings.retrieval_bm25_skip_gap)

    def _rerank_candidates(self, query: str, candidates: list[RankedChunk]) -> list[RankedChunk]:
        if not settings.retrieval_enable_reranking or not candidates:
            return candidates

        reranker = get_flashrank_reranker()
        if reranker is None:
            return candidates

        passages = [
            {
                "id": chunk.id,
                "text": chunk.content,
                "metadata": {"id": chunk.id, "source_url": chunk.source_url},
            }
            for chunk in candidates
        ]

        try:
            if RerankRequest is not None:
                request = RerankRequest(query=query, passages=passages)
                result = reranker.rerank(request)
            else:
                result = reranker.rerank(query=query, passages=passages)
        except TypeError:
            try:
                result = reranker.rerank(query, passages)
            except Exception:
                return candidates
        except Exception:
            return candidates

        ranked_ids: list[str] = []
        for item in result or []:
            candidate_id = _candidate_id(item)
            if candidate_id and candidate_id not in ranked_ids:
                ranked_ids.append(candidate_id)

        if not ranked_ids:
            return candidates

        candidate_map = {chunk.id: chunk for chunk in candidates}
        reranked = [candidate_map[candidate_id] for candidate_id in ranked_ids if candidate_id in candidate_map]
        if not reranked:
            return candidates
        return reranked

    def _expand_parent_context(self, evidence: list[RankedChunk], *, top_k: int) -> list[RetrievedEvidenceChunk]:
        evidence_ids = {chunk.id for chunk in evidence}
        expanded: list[RetrievedEvidenceChunk] = []
        for chunk in evidence:
            if not chunk.parent_id:
                continue
            if len(expanded) >= top_k:
                break
            if chunk.parent_id in evidence_ids or any(parent.id == chunk.parent_id for parent in expanded):
                continue
            parent_row = self.vector_store.get_chunk_by_id(chunk.parent_id)
            if not parent_row:
                continue
            parent_chunk = self._row_to_ranked_chunk(parent_row, context_type="parent")
            if parent_chunk.id in evidence_ids or any(parent.id == parent_chunk.id for parent in expanded):
                continue
            expanded.append(parent_chunk.to_response())
        return expanded

    def _row_to_ranked_chunk(
        self,
        row: dict[str, Any],
        *,
        dense_similarity: float | None = None,
        bm25_score: float | None = None,
        bm25_raw_score: float | None = None,
        context_type: str = "evidence",
    ) -> RankedChunk:
        return RankedChunk(
            id=str(row.get("id", "")),
            content=str(row.get("content", "")),
            source_url=str(row.get("source_url", "")),
            source_title=row.get("source_title"),
            chunk_index=int(row.get("chunk_index", 0) or 0),
            parent_id=row.get("parent_id"),
            parent_content=row.get("parent_content"),
            section_title=row.get("section_title"),
            dense_similarity=dense_similarity,
            bm25_score=bm25_score,
            bm25_raw_score=bm25_raw_score,
            actionability_score=float(row.get("actionability_score") or 0.0),
            context_type=context_type,
        )


def assemble_evidence(query: str, top_k: int = 10, retriever: HybridRetriever | None = None) -> RetrievalResponse:
    active_retriever = retriever or HybridRetriever(vector_store=LanceChunkStore())
    return active_retriever.assemble_evidence(query=query, top_k=top_k)
