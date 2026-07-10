import asyncio
import os
import sys
from pathlib import Path
from pprint import pprint

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from backend.app.engines.search import search_web
from backend.app.engines.scraper import scrape_urls

async def test_search():
    query = "startup AI step by step tools OR software site:reddit.com"
    print(f"Searching: {query}")
    candidates = await search_web(query, limit=5)
    print(f"Candidates found: {len(candidates)}")
    for c in candidates:
        print(f"- {c['title']} ({c['url']})")
    
    if not candidates:
        print("No candidates found, search failed!")
        return

    print("\nScraping candidates...")
    urls = [c['url'] for c in candidates]
    pages = await scrape_urls(urls, max_workers=3)
    
    for page in pages:
        print(f"\nURL: {page.url}")
        print(f"Status: {page.status}")
        if page.status == "failed":
            print(f"Error: {page.error_message}")
        else:
            print(f"Markdown length: {len(page.markdown)} chars")
            print(f"Preview: {page.markdown[:100]}...")

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(test_search())
