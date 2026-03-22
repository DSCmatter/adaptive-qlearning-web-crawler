"""Adaptive crawler with RL agents"""

import time
import torch
import numpy as np
from typing import Optional, Dict

from .base_crawler import BaseCrawler
from environment.reward_function import RewardFunction
from graph.web_graph import WebGraph


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
        **kwargs
    ):
        super().__init__(**kwargs)
        self.gnn_encoder = gnn_encoder
        self.qlearning_agent = qlearning_agent
        self.bandit = bandit
        self.feature_extractor = feature_extractor
        self.reward_fn = RewardFunction()
        
        # Runtime variables
        self.web_graph = WebGraph()
        self.relevant_pages_found = 0
        self.total_reward = 0.0

    def _simulate_relevance(self, url: str, html: str) -> bool:
        """Heuristic relevance for real-world Phase 5 crawl testing"""
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

    def crawl(self, seed_url: str) -> dict:
        """Main adaptive crawl loop"""
        print(f"Starting adaptive crawl from {seed_url}")
        
        # Reset state
        self.visited = set()
        self.web_graph = WebGraph()
        self.relevant_pages_found = 0
        self.total_reward = 0.0
        frontier = [seed_url]
        current_url = seed_url
        
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
            candidates = list(set([u for u in frontier if u not in self.visited]))[:50]
            if not candidates:
                print("No valid candidates left.")
                break
                
            # 2. Link selection (Contextual Bandit)
            if self.bandit and len(candidates) > 1 and self.feature_extractor:
                contexts = []
                for link in candidates:
                    vec = self.feature_extractor.build_context_vector(link, "", "", np.zeros(64), self.web_graph)
                    contexts.append(vec)
                best_link, best_idx = self.bandit.select_link(candidates, contexts)
                context_used = contexts[best_idx]
            else:
                best_link = candidates[0]
                context_used = self.feature_extractor.build_context_vector(best_link, "", "", np.zeros(64), self.web_graph) if self.feature_extractor else np.zeros(174)

            # Execution
            frontier.remove(best_link)
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
                
                # Extract links on page naturally updating Web Graph nodes/edges
                new_links = self.extract_links(html, best_link)
                for link in new_links:
                    self.web_graph.add_link(best_link, link)
                    if link not in self.visited:
                        frontier.append(link)

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

            # 4. Updates (Bandit Matrix & Q-Network iterations)
            if self.bandit and self.feature_extractor:
                self.bandit.update(best_link, context_used, reward)
                
            if self.qlearning_agent and q_state is not None:
                next_q_state = self._build_q_state(current_url)
                self.qlearning_agent.update(q_state, action, reward, next_q_state, done=False)

        print(f"\\nCrawl complete. Visited {len(self.visited)} pages. Relevant targets mapped: {self.relevant_pages_found}")
        return {
            "total_crawled": len(self.visited),
            "relevant_found": self.relevant_pages_found,
            "harvest_rate": (self.relevant_pages_found / max(1, len(self.visited))),
            "total_reward": self.total_reward
        }
