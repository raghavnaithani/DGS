from __future__ import annotations

from app.models.jobs import IngestionRequest
from app.models.knowledge import ChunkDocument, ScrapedPage
from app.models.knowledge import ChunkDocument
from app.api import knowledge as knowledge_api
from app.engines import retriever as retriever_module
import time
from app.database.vector_store import LanceChunkStore
from app.database.connection import get_connection, initialize_database
from pathlib import Path
from types import SimpleNamespace
import asyncio
from app.database.jobs_store import SQLiteJobStore
from fastapi.testclient import TestClient
from app.main import app
from app.engines import chunker, embedder as embedder_module, ingestion, search
import json
from app.engines.retriever import HybridRetriever, RankedChunk
from datetime import datetime, timezone

client = TestClient(app)

class FakeArray:
    def __init__(self, data):
        self._data = data

    def tolist(self):
        return self._data

def _build_ddg_html(count: int = 16) -> str:
    blocks: list[str] = []
    for index in range(count):
        blocks.append(
            f"""
            <div class="result">
              <a class="result__a" href="https://example{index}.com/report-{index}">Example Title {index}</a>
              <div class="result__snippet">AI labor market report 2026 {index}</div>
            </div>
            """
        )
    return "\n".join(blocks)

def _long_markdown() -> str:
    sentence = (
        "This section provides a detailed, repeated explanation of the changing AI job market, the kinds of roles that are growing, "
        "the kinds of tasks that are being automated, and the practical constraints teams face when adopting new systems."
    )
    body = "\n\n".join([sentence] * 4)
    table = "\n".join(
        [
            "| Segment | Trend |",
            "| --- | --- |",
            "| Entry-level | Mixed |",
            "| Senior-level | Strong |",
        ]
    )
    return f"# AI job market trends 2026\n\n{body}\n\n## Data snapshot\n\n{table}\n\n{body}"

def test_search_returns_results(monkeypatch):
    html = _build_ddg_html(16)

    async def fake_fetch_search_page(client, query, offset):
        return html

    monkeypatch.setattr(search, "_fetch_search_page", fake_fetch_search_page)

    results = asyncio.run(search.search_web("AI job market trends 2026", limit=20))

    assert 16 <= len(results) <= 20
    assert all(result["title"] and result["url"] and result["snippet"] and result["domain"] for result in results)

def test_ingestion_uses_search_max_results(tmp_path, monkeypatch):
    job_store = SQLiteJobStore(tmp_path / "jobs.sqlite3")
    vector_store = LanceChunkStore(tmp_path / "lancedb")
    service = ingestion.IngestionService(job_store=job_store, vector_store=vector_store)

    recorded_limits: list[int] = []

    async def fake_search_web(query, *, limit):
        recorded_limits.append(limit)
        return [
            {"title": "Title", "url": "https://example.com/page", "snippet": "Trend report 2026", "domain": "example.com"},
        ]

    def fake_filter_candidates(items, *, min_sources=5, max_sources=10):
        return items

    async def fake_scrape_urls(urls, *, max_workers=None, disable_live_scraping=False):
        return [
            ScrapedPage(
                title="Example",
                url=urls[0],
                domain="example.com",
                markdown=_long_markdown(),
                status="success",
            )
        ]

    class DummyEmbedder:
        def embed_texts(self, texts):
            return [[0.1, 0.2, 0.3] for _ in texts]

    monkeypatch.setattr(ingestion, "search_web", fake_search_web)
    monkeypatch.setattr(ingestion, "filter_candidates", fake_filter_candidates)
    monkeypatch.setattr(ingestion, "scrape_urls", fake_scrape_urls)
    monkeypatch.setattr(ingestion, "get_embedder", lambda: DummyEmbedder())

    async def run_pipeline():
        job = await service.submit(IngestionRequest(query="AI job market trends 2026"))
        await asyncio.gather(*service._background_tasks)
        return job

    asyncio.run(run_pipeline())

    assert recorded_limits == [ingestion.settings.search_max_results]

def test_filter_reduces_list(monkeypatch):
    candidates = [
        {"title": f"Title {index}", "url": f"https://example{index}.com/page", "snippet": f"Report 2026 {index}", "domain": f"example{index}.com"}
        for index in range(12)
    ]
    candidates.extend(
        [
            {"title": "Empty snippet", "url": "https://pinterest.com/ignore", "snippet": "", "domain": "pinterest.com"},
            {"title": "Blacklisted", "url": "https://facebook.com/ignore", "snippet": "2026 market update", "domain": "facebook.com"},
        ]
    )

    filtered = search.filter_candidates(candidates, min_sources=5, max_sources=10)

    assert 5 <= len(filtered) <= 10
    assert all(item["snippet"].strip() for item in filtered)
    assert all("pinterest" not in item["domain"] and "facebook" not in item["domain"] for item in filtered)

def test_chunker_produces_correct_sizes():
    markdown = _long_markdown()
    chunks = chunker.chunk_markdown(markdown, source_url="https://example.com/article", source_title="Example Article")

    assert chunks
    assert all(chunk.parent_id for chunk in chunks)
    assert all(chunk.parent_content for chunk in chunks)
    assert all(0 <= chunk.chunk_index for chunk in chunks)
    assert all(len(chunk.content) <= 700 for chunk in chunks)
    assert any(len(chunk.content) >= 400 for chunk in chunks)

def test_embedder_returns_vectors_of_expected_dimensions(monkeypatch):
    class DummyModel:
        def encode(self, texts, **kwargs):
            assert kwargs["batch_size"] == embedder_module.settings.embedding_batch_size
            return FakeArray([[float(index + column) for column in range(3)] for index, _ in enumerate(texts)])

        def get_sentence_embedding_dimension(self):
            return 3

    monkeypatch.setattr(embedder_module.ChunkEmbedder, "_load_model", lambda self: DummyModel())

    service = embedder_module.ChunkEmbedder(model_name="dummy")
    vectors = service.embed_texts(["alpha", "beta"])

    assert service.dimension() == 3
    assert len(vectors) == 2
    assert all(len(vector) == 3 for vector in vectors)

def test_lancedb_storage_works(tmp_path):
    store = LanceChunkStore(path=tmp_path / "lancedb")
    chunks = [
        ChunkDocument(
            id="chunk-1",
            content="A detailed source chunk about the AI job market.",
            source_url="https://example.com/a",
            source_title="Example A",
            chunk_index=0,
            parent_id="parent-1",
            parent_content="Parent section content",
            section_title="Overview",
            embedding=[0.1, 0.2, 0.3],
            created_at=datetime.now(timezone.utc),
            ttl_days=30,
            verification_status="verified",
        ),
        ChunkDocument(
            id="chunk-2",
            content="Another source chunk with useful context.",
            source_url="https://example.com/b",
            source_title="Example B",
            chunk_index=1,
            parent_id="parent-1",
            parent_content="Parent section content",
            section_title="Overview",
            embedding=[0.3, 0.2, 0.1],
            created_at=datetime.now(timezone.utc),
            ttl_days=30,
            verification_status="verified",
        ),
    ]

    stored = store.store_chunks(chunks)

    assert stored == 2
    assert store.count_chunks() == 2

def test_background_ingestion_pipeline_completes(tmp_path, monkeypatch):
    job_store = SQLiteJobStore(tmp_path / "jobs.sqlite3")
    vector_store = LanceChunkStore(tmp_path / "lancedb")
    service = ingestion.IngestionService(job_store=job_store, vector_store=vector_store)

    candidates = [
        {"title": f"Title {index}", "url": f"https://example{index}.com/page", "snippet": f"Trend report 2026 {index}", "domain": f"example{index}.com"}
        for index in range(6)
    ]
    scraped_pages = [
        ScrapedPage(
            title=f"Example {index}",
            url=f"https://example{index}.com/page",
            domain=f"example{index}.com",
            markdown=_long_markdown(),
            status="success",
        )
        for index in range(5)
    ]

    async def fake_search_web(query, *, limit=None):
        return candidates

    def fake_filter_candidates(items, *, min_sources=5, max_sources=10):
        return candidates[:5]

    async def fake_scrape_urls(urls, *, max_workers=None, disable_live_scraping=False):
        return scraped_pages[: len(urls)]

    class DummyEmbedder:
        def embed_texts(self, texts):
            return [[0.1, 0.2, 0.3] for _ in texts]

    monkeypatch.setattr(ingestion, "search_web", fake_search_web)
    monkeypatch.setattr(ingestion, "filter_candidates", fake_filter_candidates)
    monkeypatch.setattr(ingestion, "scrape_urls", fake_scrape_urls)
    monkeypatch.setattr(ingestion, "get_embedder", lambda: DummyEmbedder())

    async def run_pipeline():
        job = await service.submit(IngestionRequest(query="AI job market trends 2026"))
        await asyncio.gather(*service._background_tasks)
        return job

    job = asyncio.run(run_pipeline())
    record = job_store.get_job(job.id)

    assert record.status == "completed"
    assert record.progress == 100
    assert record.stored_chunks > 0
    assert vector_store.count_chunks() == record.stored_chunks

def test_background_ingestion_continues_when_some_scrapes_fail(tmp_path, monkeypatch):
    job_store = SQLiteJobStore(tmp_path / "jobs.sqlite3")
    vector_store = LanceChunkStore(tmp_path / "lancedb")
    service = ingestion.IngestionService(job_store=job_store, vector_store=vector_store)

    candidates = [
        {"title": "Working source", "url": "https://example.com/ok", "snippet": "Trend report 2026", "domain": "example.com"},
        {"title": "Broken source", "url": "https://example.org/fail", "snippet": "Trend report 2026", "domain": "example.org"},
    ]

    async def fake_search_web(query, *, limit=None):
        return candidates

    def fake_filter_candidates(items, *, min_sources=5, max_sources=10):
        return candidates

    async def fake_scrape_urls(urls, *, max_workers=None, disable_live_scraping=False):
        return [
            ScrapedPage(
                title="Working source",
                url="https://example.com/ok",
                domain="example.com",
                markdown=_long_markdown(),
                status="success",
            ),
            ScrapedPage(
                title="Broken source",
                url="https://example.org/fail",
                domain="example.org",
                markdown="",
                status="failed",
                error_message="Timeout",
            ),
        ]

    class DummyEmbedder:
        def embed_texts(self, texts):
            return [[0.1, 0.2, 0.3] for _ in texts]

    monkeypatch.setattr(ingestion, "search_web", fake_search_web)
    monkeypatch.setattr(ingestion, "filter_candidates", fake_filter_candidates)
    monkeypatch.setattr(ingestion, "scrape_urls", fake_scrape_urls)
    monkeypatch.setattr(ingestion, "get_embedder", lambda: DummyEmbedder())

    async def run_pipeline():
        job = await service.submit(IngestionRequest(query="AI job market trends 2026"))
        await asyncio.gather(*service._background_tasks)
        return job

    job = asyncio.run(run_pipeline())
    record = job_store.get_job(job.id)

    assert record.status == "completed"
    assert record.scraped_sources == 1
    assert record.stored_chunks > 0
    assert record.result is not None
    assert record.result["scraped_sources"] == 1
    assert record.result["stored_chunks"] == record.stored_chunks

def test_fail_fast_aborts_job_when_too_many_scrapes_fail(tmp_path, monkeypatch):
    job_store = SQLiteJobStore(tmp_path / "jobs.sqlite3")
    vector_store = LanceChunkStore(tmp_path / "lancedb")
    service = ingestion.IngestionService(job_store=job_store, vector_store=vector_store)

    candidates = [
        {"title": f"Title {i}", "url": f"https://example{i}.com/page", "snippet": "Trend", "domain": f"example{i}.com"}
        for i in range(5)
    ]

    async def fake_search_web(query, *, limit=None):
        return candidates

    def fake_filter_candidates(items, *, min_sources=5, max_sources=10):
        return items

    # Simulate 4 failures and 1 success -> failure ratio = 0.8
    async def fake_scrape_urls(urls, *, max_workers=None, disable_live_scraping=False):
        pages = []
        for i, url in enumerate(urls):
            if i == 0:
                pages.append(
                    ScrapedPage(title="OK", url=url, domain="example.com", markdown=_long_markdown(), status="success")
                )
            else:
                pages.append(
                    ScrapedPage(title="Broken", url=url, domain="example.com", markdown="", status="failed", error_message="Timeout")
                )
        return pages

    monkeypatch.setattr(ingestion, "search_web", fake_search_web)
    monkeypatch.setattr(ingestion, "filter_candidates", fake_filter_candidates)
    monkeypatch.setattr(ingestion, "scrape_urls", fake_scrape_urls)
    monkeypatch.setattr(ingestion.settings, "scrape_fail_fast_ratio", 0.6, raising=False)

    async def run_pipeline():
        job = await service.submit(IngestionRequest(query="AI job market trends 2026"))
        await asyncio.gather(*service._background_tasks)
        return job

    job = asyncio.run(run_pipeline())
    record = job_store.get_job(job.id)

    assert record.status == "failed"
    assert record.error_message and "Too many sources failed to scrape" in record.error_message

def test_chunker_handles_empty_and_list_only_and_long_paragraphs():
    assert chunker.chunk_markdown("", source_url="https://example.com/empty") == []

    list_only = "\n".join([
        "- First item",
        "- Second item",
        "- Third item",
        "- Fourth item",
        "- Fifth item",
    ])
    list_chunks = chunker.chunk_markdown(list_only, source_url="https://example.com/list")
    assert list_chunks == [] or all(chunk.content.strip() for chunk in list_chunks)

    long_paragraph = " ".join(["word"] * 2500)
    long_chunks = chunker.chunk_markdown(long_paragraph, source_url="https://example.com/long")
    assert long_chunks
    assert all(len(chunk.content) <= 700 for chunk in long_chunks)

def test_duplicate_content_does_not_crash_chunking_embedding_and_storage(tmp_path, monkeypatch):
    store = LanceChunkStore(path=tmp_path / "lancedb")
    markdown = _long_markdown()

    chunks_a = chunker.chunk_markdown(markdown, source_url="https://example.com/a", source_title="A")
    chunks_b = chunker.chunk_markdown(markdown, source_url="https://example.com/b", source_title="B")

    assert chunks_a and chunks_b
    assert chunks_a[0].content == chunks_b[0].content

    class DummyModel:
        def encode(self, texts, **kwargs):
            return FakeArray([[0.1, 0.2, 0.3] for _ in texts])

        def get_sentence_embedding_dimension(self):
            return 3

    monkeypatch.setattr(embedder_module.ChunkEmbedder, "_load_model", lambda self: DummyModel())
    service = embedder_module.ChunkEmbedder(model_name="dummy")
    vectors = service.embed_texts([chunk.content for chunk in chunks_a + chunks_b])
    for chunk, vector in zip(chunks_a + chunks_b, vectors, strict=True):
        chunk.embedding = vector

    stored = store.store_chunks(chunks_a + chunks_b)
    assert stored == len(chunks_a) + len(chunks_b)
    assert store.count_chunks() == stored

def test_chunk_defaults_ttl_and_verification_status():
    chunks = chunker.chunk_markdown(_long_markdown(), source_url="https://example.com/defaults")

    assert chunks
    assert all(chunk.ttl_days == 30 for chunk in chunks)
    assert all(chunk.verification_status == "unverified" for chunk in chunks)

def test_two_ingestion_jobs_can_run_concurrently(tmp_path, monkeypatch):
    job_store = SQLiteJobStore(tmp_path / "jobs.sqlite3")
    vector_store = LanceChunkStore(tmp_path / "lancedb")
    service = ingestion.IngestionService(job_store=job_store, vector_store=vector_store)

    async def fake_search_web(query, *, limit=None):
        return [
            {"title": f"Title {query}", "url": f"https://example.com/{query}", "snippet": "Trend report 2026", "domain": "example.com"},
        ]

    def fake_filter_candidates(items, *, min_sources=5, max_sources=10):
        return items

    async def fake_scrape_urls(urls, *, max_workers=None, disable_live_scraping=False):
        return [
            ScrapedPage(
                title="Example",
                url=urls[0],
                domain="example.com",
                markdown=_long_markdown(),
                status="success",
            )
        ]

    class DummyEmbedder:
        def embed_texts(self, texts):
            return [[0.1, 0.2, 0.3] for _ in texts]

    monkeypatch.setattr(ingestion, "search_web", fake_search_web)
    monkeypatch.setattr(ingestion, "filter_candidates", fake_filter_candidates)
    monkeypatch.setattr(ingestion, "scrape_urls", fake_scrape_urls)
    monkeypatch.setattr(ingestion, "get_embedder", lambda: DummyEmbedder())

    async def run_two_jobs():
        job_a = await service.submit(IngestionRequest(query="alpha"))
        job_b = await service.submit(IngestionRequest(query="beta"))
        await asyncio.gather(*service._background_tasks)
        return job_a, job_b

    job_a, job_b = asyncio.run(run_two_jobs())
    record_a = job_store.get_job(job_a.id)
    record_b = job_store.get_job(job_b.id)

    assert job_a.id != job_b.id
    assert record_a.status == "completed"
    assert record_b.status == "completed"
    assert record_a.stored_chunks > 0
    assert record_b.stored_chunks > 0
    assert vector_store.count_chunks() == record_a.stored_chunks + record_b.stored_chunks

def test_ingestion_benchmark_throughput_settings_reduce_runtime(tmp_path, monkeypatch):
    job_store = SQLiteJobStore(tmp_path / "jobs.sqlite3")
    vector_store = LanceChunkStore(tmp_path / "lancedb")
    service = ingestion.IngestionService(job_store=job_store, vector_store=vector_store)

    async def fake_search_web(query, *, limit=None):
        return [
            {"title": f"Title {index}", "url": f"https://example{index}.com/page", "snippet": "Trend report 2026", "domain": f"example{index}.com"}
            for index in range(6)
        ]

    def fake_filter_candidates(items, *, min_sources=5, max_sources=10):
        return list(items)[:6]

    async def fake_scrape_urls(urls, *, max_workers=None, disable_live_scraping=False):
        workers = max(1, max_workers or 1)
        await asyncio.sleep(0.06 / workers)
        return [
            ScrapedPage(
                title=f"Example {index}",
                url=url,
                domain=f"example{index}.com",
                markdown=_long_markdown(),
                status="success",
            )
            for index, url in enumerate(urls)
        ]

    class BenchmarkEmbedder:
        def embed_texts(self, texts):
            batch = max(1, ingestion.settings.embedding_batch_size)
            simulated_calls = max(1, (len(texts) + batch - 1) // batch)
            time.sleep(0.01 * simulated_calls)
            return [[0.1, 0.2, 0.3] for _ in texts]

    monkeypatch.setattr(ingestion, "search_web", fake_search_web)
    monkeypatch.setattr(ingestion, "filter_candidates", fake_filter_candidates)
    monkeypatch.setattr(ingestion, "scrape_urls", fake_scrape_urls)
    monkeypatch.setattr(ingestion, "get_embedder", lambda: BenchmarkEmbedder())

    async def run_profile(*, scrape_workers: int, embedding_workers: int, embedding_batch: int, search_limit: int) -> float:
        monkeypatch.setattr(ingestion.settings, "scrape_max_workers", scrape_workers, raising=False)
        monkeypatch.setattr(ingestion.settings, "embedding_max_workers", embedding_workers, raising=False)
        monkeypatch.setattr(ingestion.settings, "embedding_batch_size", embedding_batch, raising=False)
        monkeypatch.setattr(ingestion.settings, "search_max_results", search_limit, raising=False)

        started = time.perf_counter()
        job = await service.submit(IngestionRequest(query="AI job market trends 2026"))
        await asyncio.gather(*service._background_tasks)
        elapsed = time.perf_counter() - started

        record = job_store.get_job(job.id)
        assert record.status == "completed"
        assert record.stored_chunks > 0
        return elapsed

    baseline = asyncio.run(
        run_profile(scrape_workers=1, embedding_workers=1, embedding_batch=4, search_limit=15)
    )
    tuned = asyncio.run(
        run_profile(scrape_workers=6, embedding_workers=2, embedding_batch=16, search_limit=30)
    )

    assert tuned < baseline

def test_knowledge_ingest_endpoint_returns_job_id(monkeypatch, tmp_path):
    job_store = SQLiteJobStore(tmp_path / "jobs.sqlite3")
    app.state.job_store = job_store

    class FakeJob:
        def __init__(self, job_id: str):
            self.id = job_id

    class FakeService:
        async def submit(self, payload):
            return FakeJob("job-123")

    app.state.ingestion_service = FakeService()

    response = client.post("/v1/knowledge/ingest", json={"query": "AI job market trends 2026"})

    assert response.status_code == 202
    assert response.json() == {"job_id": "job-123", "status": "queued"}

def test_job_status_endpoint_returns_record(tmp_path):
    job_store = SQLiteJobStore(tmp_path / "jobs.sqlite3")
    app.state.job_store = job_store
    job = job_store.create_job(IngestionRequest(query="AI job market trends 2026"))

    response = client.get(f"/v1/jobs/{job.id}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["id"] == job.id
    assert payload["status"] == "queued"

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

def test_retrieval_pipeline(tmp_path, monkeypatch):
    retriever = _seed_retriever(tmp_path)

    # 1. Dense and BM25 return expected chunks
    dense = retriever.dense_search("finance cashflow budget", limit=20, min_similarity=0.7)
    bm25 = retriever.bm25_search("finance budget", limit=20)
    assert dense
    assert bm25
    assert dense[0].id == "chunk-finance"
    assert any(item.id == "chunk-finance" for item in bm25)

    # 2. Reranked output is subset of input candidates
    dense_ranked = [
        RankedChunk(id="chunk-finance", content="Finance report on cashflow optimization and budget planning.", source_url="https://example.com/finance", source_title="Finance Report", chunk_index=0, dense_similarity=0.95),
        RankedChunk(id="chunk-health", content="Healthcare policy updates for hospitals and providers.", source_url="https://example.com/health", source_title="Health Policy", chunk_index=1, dense_similarity=0.9),
    ]
    bm25_ranked = [
        RankedChunk(id="chunk-mixed", content="Budget allocation for healthcare systems and hospital finance teams.", source_url="https://example.com/mixed", source_title="Mixed Domain", chunk_index=2, bm25_score=0.8)
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

    # 3. BM25 fallback when dense returns nothing
    monkeypatch.undo() # reset dense/bm25 mock
    dense = retriever.dense_search("hospital policy", limit=20, min_similarity=1.01)
    bm25 = retriever.bm25_search("hospital policy", limit=20)
    fused = retriever.fuse_rrf(dense, bm25, k=60)
    assert dense == []
    assert bm25
    assert fused
    assert fused[0].id == "chunk-health"

    # 4. BM25 strong signal skips dense search
    def fail_dense(*args, **kwargs):
        raise AssertionError("dense_search should not be called for a strong BM25 signal")
    bm25_ranked_strong = [
        RankedChunk(id="chunk-finance", content="Finance", source_url="a", source_title="a", chunk_index=0, bm25_score=0.95),
        RankedChunk(id="chunk-mixed", content="Mixed", source_url="b", source_title="b", chunk_index=2, bm25_score=0.7),
    ]
    monkeypatch.setattr(retriever, "bm25_search", lambda *args, **kwargs: bm25_ranked_strong)
    monkeypatch.setattr(retriever, "dense_search", fail_dense)
    response = retriever.assemble_evidence("finance budget", top_k=2)
    assert [item.id for item in response.evidence] == ["chunk-finance", "chunk-mixed"]
    assert response.expanded_context == []

    # 5. RRF fusion combines rankings and preserves scores
    dense_rrf = [
        RankedChunk(id="a", content="A", source_url="a", source_title="A", chunk_index=0, dense_similarity=0.91),
        RankedChunk(id="b", content="B", source_url="b", source_title="B", chunk_index=1, dense_similarity=0.87),
    ]
    bm25_rrf = [
        RankedChunk(id="b", content="B", source_url="b", source_title="B", chunk_index=1, bm25_score=-4.2),
        RankedChunk(id="c", content="C", source_url="c", source_title="C", chunk_index=2, bm25_score=-3.5),
    ]
    fused = retriever.fuse_rrf(dense_rrf, bm25_rrf, k=60)
    assert [item.id for item in fused] == ["b", "a", "c"]
    assert fused[0].dense_similarity == 0.87
    assert fused[0].bm25_score == -4.2

    # 6. ANN index is created after threshold crossed
    store = LanceChunkStore(path=tmp_path / "lancedb2")
    fake_table = FakeIndexTable()
    store._open_or_create_table = lambda: fake_table
    chunks = [
        _chunk(chunk_id=f"chunk-{index}", content=f"Chunk {index} content.", source_url=f"https://example.com/{index}", source_title=f"Chunk {index}", chunk_index=index, embedding=[0.1, 0.2, 0.3])
        for index in range(256)
    ]
    stored = store.store_chunks(chunks)
    assert stored == 256
    assert fake_table.index_calls
    assert fake_table.index_calls[0]["metric"] == "cosine"

    # 7. Parent context is returned separately
    job_store_path = tmp_path / "retrieval3.sqlite3"
    initialize_database(job_store_path)
    store3 = LanceChunkStore(path=tmp_path / "lancedb3")
    parent = _chunk(chunk_id="chunk-parent", content="Parent section.", source_url="a", source_title="A", chunk_index=0, embedding=[1.0, 0.0, 0.0])
    child = _chunk(chunk_id="chunk-child", content="Child chunk.", source_url="b", source_title="B", chunk_index=1, embedding=[0.9, 0.1, 0.0], parent_id="chunk-parent", parent_content="Parent section.", section_title="Finance Detail")
    store3.store_chunks([parent, child])
    retriever3 = HybridRetriever(vector_store=store3, embedder=FakeEmbedder(), db_path=job_store_path)
    monkeypatch.setattr(retriever3, "bm25_search", lambda *args, **kwargs: [])
    monkeypatch.setattr(retriever3, "dense_search", lambda *args, **kwargs: [retriever3._row_to_ranked_chunk(child.model_dump(mode="json"), dense_similarity=0.97)])
    response = retriever3.assemble_evidence("finance detail", top_k=1)
    assert len(response.evidence) == 1
    assert response.evidence[0].id == "chunk-child"
    assert len(response.expanded_context) == 1
    assert response.expanded_context[0].id == "chunk-parent"
    assert response.expanded_context[0].context_type == "parent"
