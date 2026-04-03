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

    def _score_link(self, link: str, context: np.ndarray) -> float:
        """Compute UCB score for a single candidate link."""
        if link not in self.arms:
            self.arms[link] = (
                np.eye(self.context_dim) * self.ridge_lambda,
                np.zeros(self.context_dim)
            )

        A, b = self.arms[link]

        try:
            A_inv = np.linalg.inv(A)
            theta = A_inv @ b
            expected_reward = context @ theta
            uncertainty = np.sqrt(context @ A_inv @ context)
            return float(expected_reward + self.alpha * uncertainty)
        except np.linalg.LinAlgError:
            # Fall back to optimistic exploration when matrix inversion fails.
            return 999.0

    def score_candidates(self, candidate_links: List[str], contexts: List[np.ndarray]) -> List[float]:
        """Score every candidate link using current UCB parameters."""
        return [self._score_link(link, context) for link, context in zip(candidate_links, contexts)]
    
    def select_link(self, candidate_links: List[str], contexts: List[np.ndarray]) -> Tuple[str, int]:
        """
        Select best link using UCB criterion
        
        Args:
            candidate_links: List of URLs
            contexts: Context vector for each link
        
        Returns:
            (selected_link, index)
        """
        ucb_scores = self.score_candidates(candidate_links, contexts)
        
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
