"""Reward function for RL training"""


class RewardFunction:
    """Compute rewards for crawl actions"""
    
    def __init__(self, relevance_threshold: float = 0.5):
        self.relevance_threshold = relevance_threshold
    
    def compute_reward(
        self,
        is_relevant: bool,
        relevance_score: float = 0.0,
        depth: int = 0,
        fetch_time: float = 0.0,
        is_new_domain: bool = False,
        is_duplicate: bool = False
    ) -> float:
        """
        Compute reward for visiting a page
        
        Reward components:
        - Relevance: +10 (high), +5 (medium), 0 (low), -2 (irrelevant)
        - Novelty: +2 (new domain), +1 (new subdomain)
        - Cost: -0.1 * fetch_time
        - Depth penalty: -0.1 * depth
        - Duplicate: -5
        
        Args:
            is_relevant: Whether page is relevant
            relevance_score: Continuous relevance score [0, 1]
            depth: Crawl depth
            fetch_time: Time to fetch page
            is_new_domain: Whether domain is new
            is_duplicate: Whether page already visited
        
        Returns:
            Total reward
        """
        reward = 0.0
        
        # Relevance reward
        if is_duplicate:
            reward -= 5.0
        elif relevance_score > 0.8:
            reward += 10.0
        elif relevance_score > 0.5:
            reward += 5.0
        elif relevance_score > 0.3:
            reward += 0.0
        else:
            reward -= 2.0
        
        # Novelty bonus
        if is_new_domain:
            reward += 2.0
        
        # Cost penalty
        reward -= 0.1 * fetch_time
        
        # Depth penalty
        reward -= 0.1 * min(depth, 10)
        
        return reward
