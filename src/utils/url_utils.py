"""URL manipulation utilities"""

from urllib.parse import urlparse, urljoin, urldefrag
from typing import List


def normalize_url(url: str) -> str:
    """Normalize URL (remove fragments, trailing slashes, etc.)"""
    url, _ = urldefrag(url)  # Remove fragment
    url = url.rstrip('/')  # Remove trailing slash
    return url


def is_same_domain(url1: str, url2: str) -> bool:
    """Check if two URLs belong to same domain"""
    domain1 = urlparse(url1).netloc
    domain2 = urlparse(url2).netloc
    return domain1 == domain2


def get_domain(url: str) -> str:
    """Extract domain from URL"""
    return urlparse(url).netloc


def filter_candidate_links(
    links: List[str], 
    current_url: str, 
    max_candidates: int = 50
) -> List[str]:
    """
    Filter and limit candidate links for efficiency
    
    Heuristics:
    - Prefer same domain
    - Prefer shorter URLs
    - Remove duplicates
    - Limit to max_candidates
    """
    current_domain = get_domain(current_url)
    
    # Score each link
    scored_links = []
    seen = set()
    
    for link in links:
        normalized = normalize_url(link)
        if normalized in seen:
            continue
        seen.add(normalized)
        
        score = 0.0
        
        # Prefer same domain
        if is_same_domain(link, current_url):
            score += 10.0
        
        # Prefer shorter URLs (simpler pages)
        score -= len(link) / 100.0
        
        # Prefer not too deep
        depth = link.count('/')
        score -= depth * 0.5
        
        scored_links.append((score, normalized))
    
    # Sort by score and take top candidates
    scored_links.sort(reverse=True, key=lambda x: x[0])
    return [link for _, link in scored_links[:max_candidates]]
