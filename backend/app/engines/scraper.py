from __future__ import annotations

import asyncio
from functools import lru_cache
import sys
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser
from urllib.request import urlopen

import httpx
from bs4 import BeautifulSoup

try:
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
    _HAS_CRAWL4AI = True
except Exception:
    # Provide lightweight stubs so the module can be imported in environments
    # where crawl4ai is not installed (tests often monkeypatch `scrape_urls`).
    _HAS_CRAWL4AI = False

    class BrowserConfig:
        def __init__(self, *args, **kwargs):
            self.user_agent = "debug-agent"

    class CrawlerRunConfig:
        def __init__(self, *args, **kwargs):
            pass

    class AsyncWebCrawler:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def arun(self, url, config=None):
            raise RuntimeError("crawl4ai not available in this environment")

from ..config import settings
from ..models.knowledge import ScrapedPage

DEFAULT_HTTP_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


def _domain_base(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}"


@lru_cache(maxsize=128)
def _cached_robot_parser(base_url: str) -> RobotFileParser:
    parser = RobotFileParser()
    robots_url = f"{base_url}/robots.txt"
    parser.set_url(robots_url)
    with urlopen(robots_url, timeout=min(10, int(settings.scrape_timeout_seconds))) as response:
        content = response.read().decode("utf-8", errors="ignore")
    parser.parse(content.splitlines())
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


def _extract_text_from_html(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "lxml")
    for element in soup(["script", "style", "noscript"]):
        element.decompose()

    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    if not title:
        heading = soup.find("h1")
        if heading:
            title = heading.get_text(" ", strip=True)

    paragraphs = [paragraph.get_text(" ", strip=True) for paragraph in soup.find_all(["p", "li"])]
    markdown = "\n\n".join(part for part in paragraphs if part)
    if not markdown:
        markdown = soup.get_text("\n", strip=True)

    return title, markdown.strip()


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
        message = str(exc).strip() or f"Scrape error ({type(exc).__name__})"
        return ScrapedPage(
            title=urlparse(url).netloc,
            url=url,
            domain=urlparse(url).netloc,
            markdown="",
            status="failed",
            error_message=message,
        )


async def _scrape_urls_async(urls: list[str], *, max_workers: int | None = None) -> list[ScrapedPage]:
    limit = max_workers or settings.ingestion_max_workers
    semaphore = asyncio.Semaphore(limit)
    browser_config = BrowserConfig(headless=True, text_mode=True, verbose=False)
    async with AsyncWebCrawler(config=browser_config, thread_safe=True) as crawler:
        async def run(url: str) -> ScrapedPage:
            async with semaphore:
                return await _scrape_one(crawler, url)

        return await asyncio.gather(*(run(url) for url in urls))


async def _scrape_http_one(client: httpx.AsyncClient, url: str) -> ScrapedPage:
    user_agent = DEFAULT_HTTP_USER_AGENT
    if not await _robots_allow(url, user_agent):
        return ScrapedPage(
            title=urlparse(url).netloc,
            url=url,
            domain=urlparse(url).netloc,
            markdown="",
            status="skipped",
            error_message="Disallowed by robots.txt",
        )

    for attempt in range(2):
        try:
            try:
                response = await client.get(
                    url,
                    headers={
                        "User-Agent": user_agent,
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                        "Accept-Language": "en-US,en;q=0.9",
                        "Cache-Control": "no-cache",
                        "Pragma": "no-cache",
                        "Referer": "https://www.google.com/",
                    },
                )
                
                if response.status_code in (401, 403, 406, 429):
                    raise httpx.HTTPStatusError("Blocked status code", request=response.request, response=response)
                    
                response.raise_for_status()
                title, markdown = _extract_text_from_html(response.text)
                if not markdown:
                    raise ValueError(f"No extractable content from {url}")
                    
                return ScrapedPage(
                    title=title or _extract_title(response, url),
                    url=url,
                    domain=urlparse(url).netloc,
                    markdown=markdown,
                    status="success",
                )
            except (httpx.TimeoutException, httpx.HTTPStatusError, ValueError) as exc:
                is_timeout = isinstance(exc, httpx.TimeoutException)
                is_blocked = isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in (401, 403, 406, 429)
                is_no_content = isinstance(exc, ValueError)
                
                if is_timeout or is_blocked or is_no_content:
                    proxy_url = f"https://r.jina.ai/{url}"
                    proxy_response = await client.get(proxy_url, headers={"User-Agent": user_agent})
                    proxy_response.raise_for_status()
                    proxy_text = proxy_response.text.strip()
                    if not proxy_text:
                        raise RuntimeError(f"Reader proxy returned empty content for {url}")
                    return ScrapedPage(
                        title=urlparse(url).netloc,
                        url=url,
                        domain=urlparse(url).netloc,
                        markdown=proxy_text,
                        status="success",
                    )
                raise exc
        except Exception as exc:
            if attempt == 0:
                continue
            return ScrapedPage(
                title=urlparse(url).netloc,
                url=url,
                domain=urlparse(url).netloc,
                markdown="",
                status="failed",
                error_message=str(exc),
            )


async def _scrape_urls_http_async(urls: list[str], *, max_workers: int | None = None) -> list[ScrapedPage]:
    limit = max_workers or settings.ingestion_max_workers
    semaphore = asyncio.Semaphore(limit)
    timeout = httpx.Timeout(settings.scrape_timeout_seconds)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        async def run(url: str) -> ScrapedPage:
            async with semaphore:
                return await _scrape_http_one(client, url)

        return await asyncio.gather(*(run(url) for url in urls))


def _scrape_urls_sync(urls: list[str], *, max_workers: int | None = None) -> list[ScrapedPage]:
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    return asyncio.run(scrape_urls(urls, max_workers=max_workers))


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

    if sys.platform == "win32":
        return await _scrape_urls_http_async(urls, max_workers=max_workers)

    return await _scrape_urls_async(urls, max_workers=max_workers)
