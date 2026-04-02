"""Baseline crawler implementations for Phase 6 evaluation."""

import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from .base_crawler import BaseCrawler
try:
    from environment.reward_function import RewardFunction
    from graph.web_graph import WebGraph
    from utils.metrics import precision_at_k
except ImportError:  # pragma: no cover - fallback for package-style imports
    from src.environment.reward_function import RewardFunction
    from src.graph.web_graph import WebGraph
    from src.utils.metrics import precision_at_k


class BaselineCrawler(BaseCrawler):
    """Shared evaluation loop for heuristic baseline crawlers."""

    KEYWORDS = (
        'machine', 'learning', 'artificial', 'intelligence', 'neural',
        'network', 'deep', 'data', 'science', 'model', 'climate',
        'blockchain', 'crypto', 'carbon', 'energy'
    )

    def __init__(
        self,
        relevance_fn=None,
        random_seed: int = 42,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.relevance_fn = relevance_fn
        self.random = random.Random(random_seed)
        self.reward_fn = RewardFunction()
        self.web_graph = WebGraph()
        self.page_history: List[bool] = []
        self.total_reward = 0.0
        self.trace: List[Dict[str, object]] = []

    def _reset_runtime(self):
        self.visited = set()
        self.web_graph = WebGraph()
        self.page_history = []
        self.total_reward = 0.0
        self.trace = []

    def _is_relevant(self, url: str, html: str) -> bool:
        if self.relevance_fn is not None:
            return bool(self.relevance_fn(url, html))

        text = f"{url} {html or ''}".lower()
        matches = sum(1 for keyword in self.KEYWORDS if keyword in text)
        return matches > 0

    def _candidate_details(
        self,
        candidate_url: str,
        frontier_meta: Dict[str, Dict[str, str]]
    ) -> Dict[str, str]:
        return frontier_meta.get(
            candidate_url,
            {'anchor_text': '', 'source_url': ''}
        )

    def _select_candidate(
        self,
        candidates: List[str],
        frontier_meta: Dict[str, Dict[str, str]],
        current_url: str
    ) -> Tuple[str, Optional[np.ndarray]]:
        raise NotImplementedError

    def crawl(self, seed_url: str) -> dict:
        """Run a baseline crawl from a single seed URL."""
        self._reset_runtime()
        frontier = [seed_url]
        frontier_meta: Dict[str, Dict[str, str]] = {
            seed_url: {'anchor_text': '', 'source_url': ''}
        }
        crawl_start = time.time()
        current_url = seed_url

        while len(self.visited) < self.max_pages and frontier:
            candidates = [
                url for url in frontier
                if url not in self.visited
            ][:self.max_candidates]
            if not candidates:
                break

            next_url, _ = self._select_candidate(candidates, frontier_meta, current_url)
            frontier.remove(next_url)
            metadata = frontier_meta.pop(next_url, {'anchor_text': '', 'source_url': ''})

            self.visited.add(next_url)
            current_url = next_url

            start_time = time.time()
            html = self.fetch_page(next_url, timeout=self.timeout)
            fetch_time = time.time() - start_time

            is_relevant = False
            if html:
                is_relevant = self._is_relevant(next_url, html)
                link_candidates = self.extract_link_candidates(html, next_url)
                self.web_graph.add_page(next_url, html=html, label=int(is_relevant))

                for candidate in link_candidates:
                    child_url = candidate['url']
                    self.web_graph.add_link(next_url, child_url, candidate['anchor_text'])
                    if child_url not in self.visited and child_url not in frontier:
                        frontier.append(child_url)
                        frontier_meta[child_url] = {
                            'anchor_text': candidate['anchor_text'],
                            'source_url': next_url,
                        }

            reward = self.reward_fn.compute_reward(
                is_relevant=is_relevant,
                relevance_score=1.0 if is_relevant else 0.1,
                depth=len(self.visited),
                fetch_time=fetch_time,
                is_new_domain=True,
                is_duplicate=False
            )
            self.total_reward += reward
            self.page_history.append(is_relevant)
            self.trace.append({
                'url': next_url,
                'is_relevant': is_relevant,
                'reward': reward,
                'source_url': metadata.get('source_url', ''),
                'anchor_text': metadata.get('anchor_text', ''),
            })

        crawl_time = time.time() - crawl_start
        total_crawled = len(self.visited)
        relevant_found = sum(self.page_history)
        avg_reward = self.total_reward / max(1, total_crawled)

        return {
            'total_crawled': total_crawled,
            'relevant_found': relevant_found,
            'harvest_rate': relevant_found / max(1, total_crawled),
            'precision_at_10': precision_at_k(self.page_history, min(10, total_crawled)),
            'precision_at_20': precision_at_k(self.page_history, min(20, total_crawled)),
            'total_reward': self.total_reward,
            'avg_reward': avg_reward,
            'crawl_time': crawl_time,
            'page_history': list(self.page_history),
            'trace': list(self.trace),
        }


class RandomCrawler(BaselineCrawler):
    """Baseline that samples uniformly from the current frontier."""

    def _select_candidate(
        self,
        candidates: List[str],
        frontier_meta: Dict[str, Dict[str, str]],
        current_url: str
    ) -> Tuple[str, Optional[np.ndarray]]:
        return self.random.choice(candidates), None


class BestFirstCrawler(BaselineCrawler):
    """Greedy baseline using keyword matches over URL and anchor text."""

    def _heuristic_score(self, url: str, anchor_text: str) -> float:
        text = f"{url} {anchor_text}".lower()
        matches = sum(1 for keyword in self.KEYWORDS if keyword in text)
        return float(matches)

    def _select_candidate(
        self,
        candidates: List[str],
        frontier_meta: Dict[str, Dict[str, str]],
        current_url: str
    ) -> Tuple[str, Optional[np.ndarray]]:
        scored = []
        for candidate in candidates:
            details = self._candidate_details(candidate, frontier_meta)
            score = self._heuristic_score(candidate, details.get('anchor_text', ''))
            scored.append((score, candidate))

        scored.sort(key=lambda item: (-item[0], item[1]))
        return scored[0][1], None


class PageRankCrawler(BaselineCrawler):
    """Greedy baseline using the currently discovered graph PageRank."""

    def _pagerank_score(self, url: str) -> float:
        if not self.web_graph.has_node(url):
            return 0.0

        score = self.web_graph.get_pagerank(url)
        score += 0.05 * self.web_graph.get_in_degree(url)
        score -= 0.01 * self.web_graph.get_out_degree(url)
        return float(score)

    def _select_candidate(
        self,
        candidates: List[str],
        frontier_meta: Dict[str, Dict[str, str]],
        current_url: str
    ) -> Tuple[str, Optional[np.ndarray]]:
        scored = [
            (self._pagerank_score(candidate), candidate)
            for candidate in candidates
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))
        return scored[0][1], None
