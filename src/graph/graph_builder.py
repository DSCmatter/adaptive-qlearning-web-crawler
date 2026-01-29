"""Bootstrap initial graph structure"""

import requests
from bs4 import BeautifulSoup
from collections import deque
from .web_graph import WebGraph
from typing import List
import time


def bootstrap_initial_graph(seed_urls: List[str], max_pages: int = 500) -> WebGraph:
    """
    Build initial graph with simple BFS crawl
    Budget-friendly: 500 pages in ~10 minutes
    
    Args:
        seed_urls: Starting URLs
        max_pages: Maximum pages to crawl
    
    Returns:
        Populated WebGraph
    """
    graph = WebGraph()
    queue = deque(seed_urls)
    visited = set()
    
    print(f"Bootstrapping graph with {max_pages} pages...")
    
    while queue and len(visited) < max_pages:
        url = queue.popleft()
        
        if url in visited:
            continue
        
        try:
            # Fetch page
            response = requests.get(
                url, 
                timeout=5,
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            html = response.text
            
            # Add to graph (features will be computed later)
            graph.add_page(url, html)
            
            # Extract outgoing links
            soup = BeautifulSoup(html, 'html.parser')
            for a in soup.find_all('a', href=True)[:10]:  # Limit breadth
                link = a['href']
                if link.startswith('http'):
                    graph.add_link(url, link, anchor_text=a.get_text()[:100])
                    queue.append(link)
            
            visited.add(url)
            
            if len(visited) % 50 == 0:
                print(f"  Crawled {len(visited)}/{max_pages} pages...")
            
            time.sleep(0.5)  # Politeness delay
            
        except Exception as e:
            print(f"  Failed to fetch {url}: {e}")
            continue
    
    print(f"Bootstrap complete: {graph.num_nodes()} nodes, {graph.num_edges()} edges")
    return graph
