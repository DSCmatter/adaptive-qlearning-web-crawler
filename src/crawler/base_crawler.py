"""Base crawler implementation"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
from typing import List, Set, Optional


class BaseCrawler:
    """Basic web crawler with politeness and URL filtering"""
    
    def __init__(self, max_pages: int = 100, delay: float = 1.0):
        self.max_pages = max_pages
        self.delay = delay
        self.visited: Set[str] = set()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def crawl(self, seed_url: str):
        """Main crawl loop - to be overridden by subclasses"""
        raise NotImplementedError
    
    def fetch_page(self, url: str, timeout: int = 10) -> Optional[str]:
        """Fetch a single page"""
        try:
            response = self.session.get(url, timeout=timeout)
            response.raise_for_status()
            time.sleep(self.delay)  # Politeness delay
            return response.text
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return None
    
    def extract_links(self, html: str, base_url: str) -> List[str]:
        """Extract and normalize links from HTML"""
        soup = BeautifulSoup(html, 'html.parser')
        links = []
        
        for a in soup.find_all('a', href=True):
            link = urljoin(base_url, a['href'])
            if self.is_valid_url(link):
                links.append(link)
        
        return links
    
    def is_valid_url(self, url: str) -> bool:
        """Check if URL should be crawled"""
        try:
            parsed = urlparse(url)
            return (
                parsed.scheme in ['http', 'https'] and
                len(url) < 500 and
                not url.endswith(('.pdf', '.jpg', '.png', '.zip', '.mp4')) and
                '#' not in url
            )
        except:
            return False
