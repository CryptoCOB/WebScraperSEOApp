import asyncio
import aiohttp
from aiohttp import ClientSession, ClientError, ClientTimeoutError, HTTPStatusError
from bs4 import BeautifulSoup
import logging

logging.basicConfig(level=logging.INFO)

class WebScraper:
    def __init__(self):
        self.session = None

    async def __aenter__(self):
        self.session = await aiohttp.ClientSession().__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.session.__aexit__(exc_type, exc_val, exc_tb)

    async def init_session(self):
        if not self.session:
            self.session = aiohttp.ClientSession()

    async def close_session(self):
        await self.session.close()

    async def fetch(self, url):
        try:
            async with self.session.get(url, timeout=30) as response:
                response.raise_for_status()
                return await response.text()
        except ClientTimeoutError:
            logging.error(f"Timeout while fetching {url}")
        except HTTPStatusError as e:
            logging.error(f"HTTP error {e.status} while fetching {url}")
        except ClientError as e:
            logging.error(f"Client error when fetching {url}: {e}")
        return None

    def parse_html(self, html_content):
        return BeautifulSoup(html_content, 'html.parser')

    def extract_links(self, soup):
        links = [a['href'] for a in soup.find_all('a', href=True)]
        return links

async def process_urls(scraper, urls):
    tasks = [scraper.fetch(url) for url in urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results

async def test_scraper():
    async with WebScraper() as scraper:
        urls = [
            'https://example.com',
            'https://openai.com',
            'https://nonexistentwebsite.org'
        ]
        results = await process_urls(scraper, urls)
        for url, html_content in zip(urls, results):
            if html_content and not isinstance(html_content, Exception):
                soup = scraper.parse_html(html_content)
                links = scraper.extract_links(soup)
                logging.info(f"Extracted {len(links)} links from {url}")
            else:
                logging.error(f"Failed to fetch content from {url}")

asyncio.run(test_scraper())
