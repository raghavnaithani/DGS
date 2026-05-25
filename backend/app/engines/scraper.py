from __future__ import annotations

import asyncio
from functools import lru_cache
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig

from ..config import settings
from ..models.knowledge import ScrapedPage


def _domain_base(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}"


@lru_cache(maxsize=128)
def _cached_robot_parser(base_url: str) -> RobotFileParser:
    parser = RobotFileParser()
    parser.set_url(f"{base_url}/robots.txt")
    parser.read()
    return parser


async def _robots_allow(url: str, user_agent: str) -> bool:
    try:
        parser = await asyncio.to_thread(_cached_robot_parser, _domain_base(url))
        return await asyncio.to_thread(parser.can_fetch, user_agent, url)
    except Exception:
        return True


def _extract_markdown(result) -> str:
    markdown = getattr(result, "markdown", None)
    if markdown is None:
        return (getattr(result, "extracted_content", "") or "").strip()
    for candidate in (
        getattr(markdown, "fit_markdown", None),
        getattr(markdown, "raw_markdown", None),
        getattr(markdown, "markdown_with_citations", None),
        getattr(markdown, "references_markdown", None),
    ):
        if candidate and str(candidate).strip():
            return str(candidate).strip()
    return (getattr(result, "extracted_content", "") or "").strip()


def _extract_title(result, url: str) -> str:
    metadata = getattr(result, "metadata", None) or {}
    for key in ("title", "og:title", "twitter:title"):
        value = metadata.get(key)
        if value:
            return str(value).strip()
    return urlparse(url).netloc.removeprefix("www.") or url


async def _scrape_one(crawler: AsyncWebCrawler, url: str) -> ScrapedPage:
    user_agent = BrowserConfig().user_agent
    if not await _robots_allow(url, user_agent):
        return ScrapedPage(
            title=urlparse(url).netloc,
            url=url,
            domain=urlparse(url).netloc,
            markdown="",
            status="skipped",
            error_message="Disallowed by robots.txt",
        )

    run_config = CrawlerRunConfig(
        check_robots_txt=True,
        page_timeout=int(settings.scrape_timeout_seconds * 1000),
        wait_until="domcontentloaded",
        remove_consent_popups=True,
        verbose=False,
        stream=False,
    )
    try:
        # Protect the underlying crawler run with an asyncio timeout so
        # a misbehaving page or the crawler doesn't hang the ingestion job.
        # Add a small buffer to the configured page timeout for safety.
        timeout = max(1.0, float(settings.scrape_timeout_seconds) + 5.0)
        result = await asyncio.wait_for(crawler.arun(url, config=run_config), timeout=timeout)
        markdown = _extract_markdown(result)
        title = _extract_title(result, url)
        return ScrapedPage(title=title, url=url, domain=urlparse(url).netloc, markdown=markdown, status="success")
    except Exception as exc:
        return ScrapedPage(
            title=urlparse(url).netloc,
            url=url,
            domain=urlparse(url).netloc,
            markdown="",
            status="failed",
            error_message=str(exc),
        )


async def scrape_urls(urls: list[str], *, max_workers: int | None = None, disable_live_scraping: bool = False) -> list[ScrapedPage]:
    if disable_live_scraping:
        return [
            ScrapedPage(
                title=urlparse(url).netloc,
                url=url,
                domain=urlparse(url).netloc,
                markdown="",
                status="failed",
                error_message="Live scraping disabled by configuration",
            )
            for url in urls
        ]

    limit = max_workers or settings.ingestion_max_workers
    semaphore = asyncio.Semaphore(limit)
    browser_config = BrowserConfig(headless=True, text_mode=True, verbose=False)
    async with AsyncWebCrawler(config=browser_config, thread_safe=True) as crawler:
        async def run(url: str) -> ScrapedPage:
            async with semaphore:
                return await _scrape_one(crawler, url)

        return await asyncio.gather(*(run(url) for url in urls))
