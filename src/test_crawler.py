import requests
from bs4 import BeautifulSoup
from collections import defaultdict
import numpy as np
from urllib.parse import urljoin, urlparse
import time

class MinimalSmartCrawler:
    """Absolute minimum viable crawler"""
    
    def __init__(self):
        self.visited = set()
        self.link_rewards = defaultdict(lambda: {'sum': 0, 'count': 0})
        self.link_queue = []  # Keep track of unvisited links
    
    def is_relevant(self, html):
        """Check if page is about machine learning/AI"""
        keywords = [
            'machine learning', 'artificial intelligence', 'neural network',
            'deep learning', 'supervised learning', 'unsupervised learning',
            'reinforcement learning', 'data science', 'algorithm', 'model training'
        ]
        html_lower = html.lower()
        matches = sum(1 for kw in keywords if kw in html_lower)
        return matches >= 2  # Need at least 2 keywords
    
    def pick_link(self, links):
        """Simple UCB with exploration bonus"""
        scores = []
        total_visits = sum(self.link_rewards[l]['count'] for l in self.link_rewards)
        
        for link in links:
            if link not in self.link_rewards:
                scores.append(999)  # Explore new links
            else:
                stats = self.link_rewards[link]
                avg_reward = stats['sum'] / stats['count']
                
                # UCB: avg reward + exploration bonus
                if total_visits > 0:
                    exploration = 1.5 * np.sqrt(np.log(total_visits) / stats['count'])
                else:
                    exploration = 0
                    
                scores.append(avg_reward + exploration)
        
        return links[np.argmax(scores)]
    
    def crawl(self, seed, max_pages=50):
        """Main loop"""
        url = seed
        relevant_count = 0
        
        for i in range(max_pages):
            # Skip if already visited
            if url in self.visited:
                # Get next unvisited URL from queue
                while self.link_queue:
                    url = self.link_queue.pop(0)
                    if url not in self.visited:
                        break
                else:
                    print("No more unvisited URLs")
                    break
            
            try:
                # Add headers so Wikipedia doesn't block us
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                response = requests.get(url, timeout=10, headers=headers)  # Increased timeout
                html = response.text
                self.visited.add(url)
                
                print(f"  Fetched page: {len(html)} bytes")
                
                # Be polite - wait between requests
                time.sleep(1)
                
                # Score page
                is_rel = self.is_relevant(html)
                reward = 10 if is_rel else -2
                if is_rel:
                    print(f"  ✓ RELEVANT PAGE found!")
                relevant_count += is_rel
                
                # Update stats
                self.link_rewards[url]['sum'] += reward
                self.link_rewards[url]['count'] += 1
                
                # Get next links
                soup = BeautifulSoup(html, 'html.parser')
                all_links = soup.find_all('a', href=True)
                
                links = []
                for a in all_links[:200]:  # Check more links
                    link = urljoin(url, a['href'])  # Handle relative URLs
                    if self._is_valid_url(link) and link not in self.visited:
                        links.append(link)
                        # Add to global queue if not already there
                        if link not in self.link_queue:
                            self.link_queue.append(link)
                
                print(f"  Found {len(all_links)} links → {len(links)} valid after filtering")
                
                if not links:
                    # Try to get from global queue (skip visited)
                    while self.link_queue:
                        next_url = self.link_queue.pop(0)
                        if next_url not in self.visited:
                            url = next_url
                            print(f"  No links here, picking from queue: {url[:60]}...")
                            break
                    else:
                        print(f"No more valid links at page {i+1}")
                        break
                
                url = self.pick_link(links)
                print(f"Page {i+1}: Crawling {url[:60]}... (Relevant so far: {relevant_count})")
                
            except requests.Timeout:
                print(f"  Timeout on page {i+1}, trying next link...")
                # Get next URL from queue
                while self.link_queue:
                    url = self.link_queue.pop(0)
                    if url not in self.visited:
                        break
                else:
                    break
            except Exception as e:
                print(f"  Error at page {i+1}: {type(e).__name__}")
                # Get next URL from queue
                while self.link_queue:
                    url = self.link_queue.pop(0)
                    if url not in self.visited:
                        break
                else:
                    break
        
        print(f"\n=== Results ===")
        print(f"Pages crawled: {len(self.visited)}")
        print(f"Relevant pages: {relevant_count}")
        print(f"Harvest Rate: {relevant_count/len(self.visited):.1%}")
    
    def _is_valid_url(self, url):
        """Check if URL is valid"""
        try:
            parsed = urlparse(url)
            path = parsed.path
            
            # Only English Wikipedia articles
            return (parsed.scheme in ['http', 'https'] and 
                    'en.wikipedia.org' in url and
                    len(url) < 500 and
                    not url.endswith(('.pdf', '.jpg', '.png', '.zip', '.mp4')) and
                    '#' not in url and
                    '/wiki/Special:' not in path and  # Block Special namespace
                    '/wiki/File:' not in path and
                    '/wiki/Help:' not in path and
                    '/wiki/Wikipedia:' not in path and  # Block Wikipedia namespace
                    '/wiki/Portal:' not in path and
                    '/wiki/Talk:' not in path and
                    '/wiki/Category:' not in path and
                    '/wiki/Template:' not in path and  # Skip templates
                    'donate.wikimedia.org' not in url and
                    '/wiki/' in url)  # Only actual articles
        except:
            return False


# Run it!
if __name__ == '__main__':
    print("Starting Minimal Smart Crawler...\n")
    
    crawler = MinimalSmartCrawler()
    
    # Try a site about machine learning
    seed = 'https://en.wikipedia.org/wiki/Machine_learning'
    
    print(f"Seed URL: {seed}")
    print(f"Looking for keywords: machine learning, AI, neural\n")
    
    crawler.crawl(seed, max_pages=50)  # Try 50 pages to see real performance
