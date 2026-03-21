"""
Phase 4: Simulated crawl environment based on Bootstrap Graph
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from graph.web_graph import WebGraph
from environment.reward_function import RewardFunction

class SimulatedCrawlEnv:
    """
    Offline GYM-like simulated RL environment for student-budget training.
    We crawl solely through the pre-built Phase 2 `bootstrap_graph.pkl` rather
    than fetching actual data live. (No Network I/O = 0 cost & high speed)
    """
    
    def __init__(self, graph_path: Path):
        self.graph_path = graph_path
        print("Loading offline graph for simulated environment...")
        self.web_graph = WebGraph.load(self.graph_path)
        self.reset()
        self.reward_fn = RewardFunction()
        
    def reset(self):
        """Reset state."""
        self.visited = set()
        self.current_url = None
        self.pages_crawled = 0
        self.relevant_found = 0
        self.total_reward = 0.0
        
        # Pick random starting node
        nodes = list(self.web_graph.graph.nodes())
        if nodes:
            self.current_url = np.random.choice(nodes)
        
        return self._get_state_metrics()

    def get_candidate_links(self, url: str):
        """Get outgoing nodes from current URL in the loaded graph."""
        if not self.web_graph.has_node(url):
            return []
        
        all_links = self.web_graph.get_neighbors(url)
        # Filter unvisited ones
        unvisited = [link for link in all_links if link not in self.visited]
        return unvisited

    def step(self, action_url: str):
        """
        Transition function: moving to a new node. Returns reward, whether done, and metrics.
        """
        self.pages_crawled += 1
        
        is_duplicate = action_url in self.visited
        self.visited.add(action_url)
        self.current_url = action_url
        
        # In a real environment, relevance would be computed dynamically.
        # Here we look up the truth mapping from the static graph or simulate it.
        # For simulation, we check the label (if exists), or simulate using keyword matches from URL.
        is_rel = self._simulate_relevance(action_url)
        if is_rel:
            self.relevant_found += 1
            
        relevance_score = 1.0 if is_rel else 0.1
        
        # Compute Reward
        step_reward = self.reward_fn.compute_reward(
            is_relevant=is_rel,
            relevance_score=relevance_score,
            depth=self.pages_crawled,  # Simplification for episode length
            fetch_time=0.0,            # It's an offline simulation
            is_new_domain=True,        # Mock
            is_duplicate=is_duplicate
        )
        
        self.total_reward += step_reward
        
        return self._get_state_metrics(), step_reward, False

    def _simulate_relevance(self, url):
        """Hack simulate relevance using URL string since this is offline training."""
        keywords = ['machine', 'learning', 'artificial', 'intelligence', 'neural', 'network', 'deep', 'data', 'science', 'model', 'climate', 'blockchain', 'crypto']
        url_lower = url.lower()
        return sum(1 for kw in keywords if kw in url_lower) > 0
        
    def _get_state_metrics(self):
        """Return scalar metrics for Q-state."""
        avg_reward = self.total_reward / max(1, self.pages_crawled)
        return {
            'budget_remaining': 200 - self.pages_crawled,
            'relevant_found': self.relevant_found,
            'current_depth': self.pages_crawled,
            'avg_reward': avg_reward,
            'exploration_rate': 0.1 # This will be overridden by agent's current epsilon
        }
