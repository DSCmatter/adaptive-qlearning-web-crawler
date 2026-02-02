"""Bootstrap initial graph structure"""

import requests
from bs4 import BeautifulSoup
from collections import deque
from urllib.parse import urljoin, urlparse
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
                timeout=10,  # Increased timeout
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            response.raise_for_status()
            html = response.text
            
            # Add to graph (features will be computed later)
            graph.add_page(url, html)
            
            # Extract outgoing links
            soup = BeautifulSoup(html, 'html.parser')
            links_found = soup.find_all('a', href=True)
            links_added = 0
            http_links = 0
            
            for a in links_found[:30]:  # Limit breadth - increased to reach 500 pages
                link = a['href']
                # Convert relative URLs to absolute
                absolute_link = urljoin(url, link)
                
                if absolute_link.startswith('http'):
                    http_links += 1
                    # Add link to graph and queue
                    graph.add_link(url, absolute_link, anchor_text=a.get_text()[:100])
                    if absolute_link not in visited and absolute_link not in queue:
                        queue.append(absolute_link)
                        links_added += 1
            
            if len(visited) == 1 or len(visited) % 50 == 0:
                print(f"  [{len(visited)}/{max_pages}] Found {len(links_found)} links ({http_links} HTTP), queued {links_added}, queue size: {len(queue)}")
            
            visited.add(url)
            
            time.sleep(0.2)  # Politeness delay (reduced for faster bootstrap)
            
        except requests.exceptions.Timeout:
            print(f"  Timeout: {url[:60]}")
            visited.add(url)
            continue
        except requests.exceptions.RequestException as e:
            print(f"  Request failed: {url[:60]} - {type(e).__name__}")
            visited.add(url)
            continue
            continue
    
    print(f"Bootstrap complete: {graph.num_nodes()} nodes, {graph.num_edges()} edges")
    return graph
