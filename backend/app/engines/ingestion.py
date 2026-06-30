from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from urllib.parse import urlparse

from ..config import settings
from ..database.jobs_store import SQLiteJobStore
from ..database.vector_store import LanceChunkStore
from ..models.jobs import IngestionRequest, JobRecord
from ..models.knowledge import ChunkDocument
from .chunker import chunk_markdown
from .embedder import get_embedder
from .scraper import scrape_urls
from .search import filter_candidates, search_web


@dataclass(slots=True)
class IngestionResult:
    query: str | None
    requested_urls: list[str]
    sources: list[dict[str, str]]
    scraped_sources: int
    failed_sources: int
    stored_chunks: int
    chunk_count: int


def _source_from_url(url: str) -> dict[str, str]:
    parsed = urlparse(url)
    domain = parsed.netloc.lower().removeprefix("www.")
    return {
        "title": domain or url,
        "url": url,
        "snippet": "",
        "domain": domain,
    }


class IngestionService:
    def __init__(self, *, job_store: SQLiteJobStore, vector_store: LanceChunkStore):
        self.job_store = job_store
        self.vector_store = vector_store
        self._background_tasks: set[asyncio.Task] = set()

    async def submit(self, payload: IngestionRequest) -> JobRecord:
        job = self.job_store.create_job(payload)
        task = asyncio.create_task(self._run_job(job.id, payload))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return job

    async def _run_job(self, job_id: str, payload: IngestionRequest) -> None:
        try:
            self.job_store.update_job(job_id, status="running", progress=5, current_step="starting")

            if payload.query:
                self.job_store.update_job(job_id, current_step="searching sources", progress=10)
                market_aware_query = f"{payload.query} trends 2026 market outlook salary industry growth"
                candidates = await search_web(market_aware_query, limit=settings.search_max_results)
                filtered_sources = filter_candidates(candidates)
                if not filtered_sources and candidates:
                    filtered_sources = candidates[:10]
            else:
                filtered_sources = [_source_from_url(str(url)) for url in payload.urls]

            if len(filtered_sources) < 1:
                raise RuntimeError("No sources were found for ingestion")

            self.job_store.update_job(
                job_id,
                current_step="scraping sources",
                progress=25,
                total_sources=len(filtered_sources),
            )
            scrape_targets = [source["url"] for source in filtered_sources]
            scrape_timeout = max(
                30.0,
                float(settings.scrape_timeout_seconds) * float(max(1, len(scrape_targets))) + 20.0,
            )
            scraped_pages = await asyncio.wait_for(
                scrape_urls(
                    scrape_targets,
                    max_workers=self._resolve_scrape_workers(),
                    disable_live_scraping=settings.disable_live_scraping,
                ),
                timeout=scrape_timeout,
            )
            successful_pages = [page for page in scraped_pages if page.status == "success" and page.markdown.strip()]
            failed_sources = max(0, len(scraped_pages) - len(successful_pages))

            # Fail-fast guard: abort the job early if too many sources failed to scrape.
            total_sources = len(scrape_targets)
            try:
                ratio = float(failed_sources) / float(max(1, total_sources))
            except Exception:
                ratio = 0.0
            if ratio > float(getattr(settings, "scrape_fail_fast_ratio", 0.5)):
                message = f"Too many sources failed to scrape ({failed_sources}/{total_sources}); aborting ingestion"
                self.job_store.update_job(job_id, status="failed", progress=100, current_step="failed", error_message=message)
                return

            self.job_store.update_job(
                job_id,
                current_step="chunking sources",
                progress=60,
                scraped_sources=len(successful_pages),
            )

            chunks: list[ChunkDocument] = []
            for page in successful_pages:
                chunks.extend(
                    chunk_markdown(
                        page.markdown,
                        source_url=str(page.url),
                        source_title=page.title,
                    )
                )

            if not chunks:
                raise RuntimeError("No chunks were produced from scraped pages")

            self.job_store.update_job(job_id, current_step="embedding chunks", progress=80, stored_chunks=0)
            embeddings = await self._embed_chunk_contents([chunk.content for chunk in chunks])
            for chunk, embedding in zip(chunks, embeddings, strict=True):
                chunk.embedding = embedding
                chunk.verification_status = "verified"

            self.job_store.update_job(job_id, current_step="storing chunks", progress=90)
            stored_count = self.vector_store.store_chunks(chunks)
            self._store_sqlite_chunks(chunks)

            result = IngestionResult(
                query=payload.query,
                requested_urls=[str(url) for url in payload.urls],
                sources=filtered_sources,
                scraped_sources=len(successful_pages),
                failed_sources=failed_sources,
                stored_chunks=stored_count,
                chunk_count=len(chunks),
            )
            self.job_store.update_job(
                job_id,
                status="completed",
                progress=100,
                current_step="completed",
                stored_chunks=stored_count,
                result={
                    "query": result.query,
                    "requested_urls": result.requested_urls,
                    "sources": result.sources,
                    "scraped_sources": result.scraped_sources,
                    "failed_sources": result.failed_sources,
                    "stored_chunks": result.stored_chunks,
                    "chunk_count": result.chunk_count,
                },
            )
        except Exception as exc:
            # Debug: surface full exception to test output to diagnose failures
            try:
                import traceback

                traceback.print_exc()
            except Exception:
                pass
            message = str(exc).strip() or f"Unhandled ingestion error ({type(exc).__name__})"
            self.job_store.update_job(job_id, status="failed", progress=100, current_step="failed", error_message=message)

    @staticmethod
    def _resolve_scrape_workers() -> int:
        configured = settings.scrape_max_workers or settings.ingestion_max_workers
        return max(1, configured)

    async def _embed_chunk_contents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        embedder = get_embedder()
        return await asyncio.to_thread(embedder.embed_texts, texts)

    def _store_sqlite_chunks(self, chunks: list[ChunkDocument]) -> None:
        from ..database.connection import get_connection

        with get_connection() as connection:
            for chunk in chunks:
                connection.execute(
                    """
                    INSERT INTO chunks (
                        id, session_id, content, source_url, source_title, chunk_index,
                        embedding_json, created_at, ttl_days, verification_status, similarity_score, actionability_score
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        chunk.actionability_score,
                    ),
                )
            connection.commit()
