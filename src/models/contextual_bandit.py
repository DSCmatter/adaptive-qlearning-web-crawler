"""Contextual bandit for link selection"""

import numpy as np
from typing import List, Tuple, Dict


class LinUCBBandit:
    """
    Linear Upper Confidence Bound bandit
    Uses context vectors to select links
    """
    
    def __init__(self, context_dim: int, alpha: float = 1.0):
        """
        Args:
            context_dim: Dimension of context vectors
            alpha: Exploration parameter (higher = more exploration)
        """
        self.context_dim = context_dim
        self.alpha = alpha
        
        # Arms stored as {link: (A, b)}
        self.arms: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        
        # Regularization for numerical stability
        self.ridge_lambda = 0.1
    
    def select_link(self, candidate_links: List[str], contexts: List[np.ndarray]) -> Tuple[str, int]:
        """
        Select best link using UCB criterion
        
        Args:
            candidate_links: List of URLs
            contexts: Context vector for each link
        
        Returns:
            (selected_link, index)
        """
        ucb_scores = []
        
        for link, context in zip(candidate_links, contexts):
            # Initialize arm if new
            if link not in self.arms:
                self.arms[link] = (
                    np.eye(self.context_dim) * self.ridge_lambda,  # A with ridge
                    np.zeros(self.context_dim)  # b
                )
            
            A, b = self.arms[link]
            
            # Solve for theta
            try:
                A_inv = np.linalg.inv(A)
                theta = A_inv @ b
                
                # UCB score: exploitation + exploration
                expected_reward = context @ theta
                uncertainty = np.sqrt(context @ A_inv @ context)
                ucb_score = expected_reward + self.alpha * uncertainty
                
            except np.linalg.LinAlgError:
                # Fallback: exploration bonus for unstable matrix
                ucb_score = 999.0
            
            ucb_scores.append(ucb_score)
        
        best_idx = np.argmax(ucb_scores)
        return candidate_links[best_idx], best_idx
    
    def update(self, link: str, context: np.ndarray, reward: float):
        """
        Update arm parameters after observing reward
        
        Args:
            link: Selected URL
            context: Context vector used
            reward: Observed reward
        """
        if link not in self.arms:
            return
        
        A, b = self.arms[link]
        
        # Update A and b
        A += np.outer(context, context)
        b += reward * context
        
        self.arms[link] = (A, b)
