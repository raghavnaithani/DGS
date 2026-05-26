import asyncio
import sys
from types import SimpleNamespace

# Inject a minimal lancedb stub so we can run the debug script without installing lancedb
if 'lancedb' not in sys.modules:
    sys.modules['lancedb'] = SimpleNamespace()
if 'pyarrow' not in sys.modules:
    sys.modules['pyarrow'] = SimpleNamespace()

from app.database.jobs_store import SQLiteJobStore
from app.database.vector_store import LanceChunkStore
from app.engines import ingestion
from app.models.jobs import IngestionRequest
from app.models.knowledge import ScrapedPage


def _long_markdown():
    sentence = (
        "This section provides a detailed, repeated explanation of the changing AI job market, the kinds of roles that are growing, "
        "the kinds of tasks that are being automated, and the practical constraints teams face when adopting new systems."
    )
    body = "\n\n".join([sentence] * 4)
    table = "\n".join([
        "| Segment | Trend |",
        "| --- | --- |",
        "| Entry-level | Mixed |",
        "| Senior-level | Strong |",
    ])
    return f"# AI job market trends 2026\n\n{body}\n\n## Data snapshot\n\n{table}\n\n{body}"


async def main():
    tmp = 'e:/DEVELOPMENT/PROJECTS/ACTIVE/DGS/debug_runs/tmp1'
    job_store = SQLiteJobStore(tmp + '/jobs.sqlite3')
    vector_store = LanceChunkStore(path=tmp + '/lancedb')
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

    ingestion.search_web = fake_search_web
    ingestion.filter_candidates = fake_filter_candidates
    ingestion.scrape_urls = fake_scrape_urls
    ingestion.get_embedder = lambda: DummyEmbedder()

    job = await service.submit(IngestionRequest(query="AI job market trends 2026"))
    await asyncio.gather(*service._background_tasks)
    record = job_store.get_job(job.id)
    print('JOB STATUS:', record.status)
    print('ERROR MESSAGE:', record.error_message)
    print('PROGRESS:', record.progress)
    print('STORED CHUNKS:', record.stored_chunks)


if __name__ == '__main__':
    asyncio.run(main())
