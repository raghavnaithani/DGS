from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable
from urllib.parse import parse_qs, unquote, urlparse

import httpx
from bs4 import BeautifulSoup
from ddgs import DDGS

from ..config import settings

DDG_HTML_URL = "https://html.duckduckgo.com/html/"
SEARCH_RESULTS_PER_PAGE = 10
MAX_CANDIDATES = 30

TRUSTED_DOMAIN_SUFFIXES = {
    "gov": 6,
    "edu": 5,
    "arxiv.org": 6,
    "nature.com": 6,
    "science.org": 6,
    "nih.gov": 6,
    "who.int": 6,
    "oecd.org": 5,
    "worldbank.org": 5,
    "microsoft.com": 4,
    "google.com": 4,
    "openai.com": 4,
    "wikipedia.org": 2,
}

BLACKLIST_DOMAIN_SUFFIXES = {
    "pinterest.com",
    "facebook.com",
    "instagram.com",
    "tiktok.com",
    "x.com",
    "reddit.com",
    "linktr.ee",
}


@dataclass(slots=True)
class SearchCandidate:
    title: str
    url: str
    snippet: str
    domain: str
    published_at: str | None = None
    score: float = 0.0

    def as_dict(self) -> dict[str, str]:
        payload = {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "domain": self.domain,
        }
        if self.published_at:
            payload["published_at"] = self.published_at
        return payload


def _domain_from_url(url: str) -> str:
    return urlparse(url).netloc.lower().removeprefix("www.")


def _normalize_ddg_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.query:
        query = parse_qs(parsed.query)
        if "uddg" in query and query["uddg"]:
            return unquote(query["uddg"][0])
    return url


def _extract_year(text: str) -> int | None:
    matches = re.findall(r"\b(20\d{2})\b", text)
    if not matches:
        return None
    years = [int(match) for match in matches]
    return max(years)


def _domain_score(domain: str) -> float:
    domain = domain.lower().removeprefix("www.")
    if any(domain == banned or domain.endswith(f".{banned}") for banned in BLACKLIST_DOMAIN_SUFFIXES):
        return -10.0
    score = 0.0
    for suffix, weight in TRUSTED_DOMAIN_SUFFIXES.items():
        if domain == suffix or domain.endswith(f".{suffix}"):
            score = max(score, float(weight))
    return score


def _recency_score(candidate: SearchCandidate) -> float:
    text = " ".join(filter(None, [candidate.title, candidate.snippet, candidate.url]))
    year = _extract_year(text)
    if year is None:
        return 0.0
    current_year = datetime.now().year
    if year >= current_year:
        return 4.0
    if year >= current_year - 1:
        return 3.0
    if year >= current_year - 2:
        return 2.0
    return 0.0


def _snippet_score(snippet: str) -> float:
    return 1.5 if snippet.strip() else -5.0


def _score_candidate(candidate: SearchCandidate) -> float:
    return _domain_score(candidate.domain) + _recency_score(candidate) + _snippet_score(candidate.snippet)


def _parse_ddg_results(html: str) -> list[SearchCandidate]:
    soup = BeautifulSoup(html, "html.parser")
    candidates: list[SearchCandidate] = []
    for result in soup.select("div.result, article, div[data-testid='result']"):
        anchor = result.select_one("a.result__a, a[data-testid='result-title-a'], h2 a")
        if anchor is None:
            continue
        title = anchor.get_text(" ", strip=True)
        url = _normalize_ddg_url(anchor.get("href", ""))
        if not url:
            continue
        snippet_el = result.select_one(".result__snippet, [data-result='snippet'], [data-testid='result-snippet']")
        snippet = snippet_el.get_text(" ", strip=True) if snippet_el else ""
        domain = _domain_from_url(url)
        candidates.append(SearchCandidate(title=title, url=url, snippet=snippet, domain=domain))
    return candidates


async def _fetch_search_page(client: httpx.AsyncClient, query: str, offset: int) -> str:
    response = await client.get(DDG_HTML_URL, params={"q": query, "s": offset})
    response.raise_for_status()
    return response.text


def _fetch_ddgs_results(query: str, limit: int) -> list[SearchCandidate]:
    results: list[SearchCandidate] = []
    try:
        with DDGS(timeout=20) as ddgs:
            fallback_results = ddgs.text(query, backend="lite", max_results=limit)
        for item in fallback_results:
            url = str(item.get("href") or item.get("url") or "").strip()
            title = str(item.get("title") or "").strip()
            snippet = str(item.get("body") or item.get("snippet") or "").strip()
            if not url or not title or not snippet:
                continue
            normalized_url = _normalize_ddg_url(url)
            candidate = SearchCandidate(
                title=title,
                url=normalized_url,
                snippet=snippet,
                domain=_domain_from_url(normalized_url),
                published_at=str(item.get("date") or "").strip() or None,
            )
            candidate.score = _score_candidate(candidate)
            results.append(candidate)
    except Exception:
        return []
    return results


async def search_web(query: str, *, limit: int = MAX_CANDIDATES) -> list[dict[str, str]]:
    query = query.strip()
    if not query:
        raise ValueError("query cannot be empty")

    seen_urls: set[str] = set()
    candidates: list[SearchCandidate] = _fetch_ddgs_results(query, limit)
    for candidate in candidates:
        seen_urls.add(candidate.url)

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    }
    if len(candidates) < limit:
        async with httpx.AsyncClient(headers=headers, timeout=20.0, follow_redirects=True) as client:
            for page_index in range(0, 3):
                html = await _fetch_search_page(client, query, page_index * SEARCH_RESULTS_PER_PAGE)
                for candidate in _parse_ddg_results(html):
                    if candidate.url in seen_urls:
                        continue
                    candidate.score = _score_candidate(candidate)
                    seen_urls.add(candidate.url)
                    candidates.append(candidate)
                    if len(candidates) >= limit:
                        break
                if len(candidates) >= limit:
                    break
                if page_index < 2:
                    await asyncio.sleep(settings.search_page_delay_seconds)

    candidates.sort(key=lambda item: (item.score, len(item.snippet), item.domain), reverse=True)
    return [candidate.as_dict() for candidate in candidates[:limit]]


def filter_candidates(candidates: Iterable[dict[str, str]], *, min_sources: int = 5, max_sources: int = 10) -> list[dict[str, str]]:
    scored: list[SearchCandidate] = []
    for candidate in candidates:
        title = str(candidate.get("title", "")).strip()
        url = str(candidate.get("url", "")).strip()
        snippet = str(candidate.get("snippet", "")).strip()
        domain = str(candidate.get("domain", "") or _domain_from_url(url))
        if not title or not url or not snippet:
            continue
        modeled = SearchCandidate(title=title, url=url, snippet=snippet, domain=domain, published_at=candidate.get("published_at"))
        modeled.score = _score_candidate(modeled)
        if modeled.score <= -5:
            continue
        scored.append(modeled)

    scored.sort(key=lambda item: (item.score, len(item.snippet), item.domain), reverse=True)

    filtered: list[dict[str, str]] = []
    seen_domains: set[str] = set()
    for candidate in scored:
        if candidate.domain in seen_domains and len(filtered) >= min_sources:
            continue
        filtered.append(candidate.as_dict())
        seen_domains.add(candidate.domain)
        if len(filtered) >= max_sources:
            break

    return filtered[:max_sources] if len(filtered) >= min_sources else [candidate.as_dict() for candidate in scored[:max_sources]]
