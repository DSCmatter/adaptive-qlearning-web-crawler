"""Adaptive crawler with RL agents"""

import time
import torch
import numpy as np
from typing import Optional, Dict

from .base_crawler import BaseCrawler
try:
    from environment.reward_function import RewardFunction
    from graph.web_graph import WebGraph
    from utils.metrics import precision_at_k
except ImportError:  # pragma: no cover - fallback for package-style imports
    from src.environment.reward_function import RewardFunction
    from src.graph.web_graph import WebGraph
    from src.utils.metrics import precision_at_k


class AdaptiveCrawler(BaseCrawler):
    """
    Hybrid RL crawler combining:
    - Q-Learning for high-level navigation
    - Contextual Bandit for link selection
    - GNN for graph structure encoding
    """

    def __init__(
        self,
        gnn_encoder=None,
        qlearning_agent=None,
        bandit=None,
        feature_extractor=None,
        relevance_fn=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.gnn_encoder = gnn_encoder
        self.qlearning_agent = qlearning_agent
        self.bandit = bandit
        self.feature_extractor = feature_extractor
        self.relevance_fn = relevance_fn
        self.reward_fn = RewardFunction()
        
        # Runtime variables
        self.web_graph = WebGraph()
        self.relevant_pages_found = 0
        self.total_reward = 0.0
        self.page_history = []
        self.trace = []

    def _simulate_relevance(self, url: str, html: str) -> bool:
        """Heuristic relevance for real-world Phase 5 crawl testing"""
        if self.relevance_fn is not None:
            return bool(self.relevance_fn(url, html))

        # Could use proper NLP model, for student budget, keyword matches suffice inside the text/url
        keywords = ['machine', 'learning', 'artificial', 'intelligence', 'neural', 'network', 'deep', 'data', 'science', 'model', 'climate', 'blockchain', 'crypto']
        text = (url + " " + (html or "")).lower()
        return sum(1 for kw in keywords if kw in text) > 0

    def _build_metrics(self) -> Dict[str, float]:
        pages_crawled = len(self.visited)
        avg_reward = self.total_reward / max(1, pages_crawled)
        return {
            'budget_remaining': self.max_pages - pages_crawled,
            'relevant_found': self.relevant_pages_found,
            'current_depth': pages_crawled,
            'avg_reward': avg_reward,
            'exploration_rate': self.qlearning_agent.epsilon if self.qlearning_agent else 0.1
        }

    def _get_gnn_embedding(self, url: str) -> np.ndarray:
        if not self.gnn_encoder or not self.feature_extractor:
            return np.zeros(64)
            
        init_context = self.feature_extractor.build_context_vector(url, "", "", np.zeros(64), self.web_graph)
        x = torch.tensor(init_context, dtype=torch.float).unsqueeze(0)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        
        with torch.no_grad():
            embedding = self.gnn_encoder(x, edge_index)[0].numpy()
        return embedding

    def _build_q_state(self, current_url: str) -> np.ndarray:
        metrics = self._build_metrics()
        gnn_embed = self._get_gnn_embedding(current_url)
        
        state = np.zeros(69)
        state[:64] = gnn_embed
        state[64] = metrics['budget_remaining'] / 200.0
        state[65] = metrics['relevant_found'] / 200.0
        state[66] = metrics['current_depth'] / 200.0
        state[67] = np.clip(metrics['avg_reward'] / 10.0, -1, 1)
        state[68] = metrics['exploration_rate']
        return state

    def _build_candidate_context(self, url: str, frontier_meta: Dict[str, Dict[str, str]]) -> np.ndarray:
        """Build a richer bandit context for an unfetched candidate link."""
        if not self.feature_extractor:
            return np.zeros(174)

        metadata = frontier_meta.get(url, {})
        anchor_text = metadata.get('anchor_text', '')
        # For online link selection we do not know the target HTML yet, so we
        # reuse a bounded slice of the source page as topical context.
        source_html = metadata.get('source_html', '')
        gnn_embedding = self._get_gnn_embedding(url) if self.gnn_encoder else np.zeros(64)

        return self.feature_extractor.build_context_vector(
            url=url,
            html=source_html[:2000],
            anchor_text=anchor_text,
            gnn_embedding=gnn_embedding,
            graph=self.web_graph,
        )

    def crawl(self, seed_url: str) -> dict:
        """Main adaptive crawl loop"""
        print(f"Starting adaptive crawl from {seed_url}")
        
        # Reset state
        self.visited = set()
        self.web_graph = WebGraph()
        self.relevant_pages_found = 0
        self.total_reward = 0.0
        self.page_history = []
        self.trace = []
        frontier = [seed_url]
        current_url = seed_url
        frontier_meta = {
            seed_url: {'anchor_text': '', 'source_url': ''}
        }
        crawl_start = time.time()
        
        # Crawl loop
        while len(self.visited) < self.max_pages and frontier:
            # 1. State extraction and high-level Q-Learning action
            if self.qlearning_agent and len(self.visited) > 0:
                q_state = self._build_q_state(current_url)
                action = self.qlearning_agent.get_action(q_state, [0, 1])
                
                if action == 0:  # STOP condition learned by Q-Agent
                    print(f"Q-Agent initiated early stop at depth {len(self.visited)}")
                    break
            else:
                q_state = None
                action = 1
                
            # Filter and bound candidates (max 50 candidates directly bounding search space)
            candidates = list(dict.fromkeys(
                u for u in frontier if u not in self.visited
            ))[:self.max_candidates]
            if not candidates:
                print("No valid candidates left.")
                break
                
            # 2. Link selection (Contextual Bandit)
            if self.bandit and len(candidates) > 1 and self.feature_extractor:
                contexts = []
                for link in candidates:
                    vec = self._build_candidate_context(link, frontier_meta)
                    contexts.append(vec)
                best_link, best_idx = self.bandit.select_link(candidates, contexts)
                context_used = contexts[best_idx]
            else:
                best_link = candidates[0]
                context_used = self._build_candidate_context(best_link, frontier_meta)

            # Execution
            frontier.remove(best_link)
            metadata = frontier_meta.pop(best_link, {'anchor_text': '', 'source_url': ''})
            self.visited.add(best_link)
            current_url = best_link
            
            print(f"Fetching: {best_link[:80]}...")
            start_time = time.time()
            html = self.fetch_page(best_link)
            fetch_time = time.time() - start_time
            
            is_rel = False
            if html:
                is_rel = self._simulate_relevance(best_link, html)
                if is_rel:
                    self.relevant_pages_found += 1
                self.web_graph.add_page(best_link, html=html, label=int(is_rel))
                
                # Extract links on page naturally updating Web Graph nodes/edges
                new_links = self.extract_link_candidates(html, best_link)
                for link in new_links:
                    child_url = link['url']
                    self.web_graph.add_link(best_link, child_url, link['anchor_text'])
                    if child_url not in self.visited and child_url not in frontier:
                        frontier.append(child_url)
                        frontier_meta[child_url] = {
                            'anchor_text': link['anchor_text'],
                            'source_url': best_link,
                            'source_html': html[:4000],
                        }

            # 3. Environment Step Reward 
            reward = self.reward_fn.compute_reward(
                is_relevant=is_rel,
                relevance_score=1.0 if is_rel else 0.1,
                depth=len(self.visited),
                fetch_time=fetch_time,
                is_new_domain=True, # Simplified assumption
                is_duplicate=False  # Handled earlier
            )
            self.total_reward += reward
            self.page_history.append(is_rel)
            self.trace.append({
                'url': best_link,
                'is_relevant': is_rel,
                'reward': reward,
                'source_url': metadata.get('source_url', ''),
                'anchor_text': metadata.get('anchor_text', ''),
            })

            # 4. Updates (Bandit Matrix & Q-Network iterations)
            if self.bandit and self.feature_extractor:
                self.bandit.update(best_link, context_used, reward)
                
            if self.qlearning_agent and q_state is not None:
                next_q_state = self._build_q_state(current_url)
                self.qlearning_agent.update(q_state, action, reward, next_q_state, done=False)

        print(f"\\nCrawl complete. Visited {len(self.visited)} pages. Relevant targets mapped: {self.relevant_pages_found}")
        crawl_time = time.time() - crawl_start
        total_crawled = len(self.visited)
        return {
            "total_crawled": total_crawled,
            "relevant_found": self.relevant_pages_found,
            "harvest_rate": (self.relevant_pages_found / max(1, total_crawled)),
            "precision_at_10": precision_at_k(self.page_history, min(10, total_crawled)),
            "precision_at_20": precision_at_k(self.page_history, min(20, total_crawled)),
            "total_reward": self.total_reward,
            "avg_reward": self.total_reward / max(1, total_crawled),
            "crawl_time": crawl_time,
            "page_history": list(self.page_history),
            "trace": list(self.trace),
        }
