"""Q-Learning agent for high-level navigation"""

import torch
import torch.nn as nn
import numpy as np
from typing import List


class QLearningAgent(nn.Module):
    """
    Lightweight Q-network for navigation decisions
    ~10K parameters, CPU-friendly
    """
    
    def __init__(
        self, 
        state_dim: int = 69, 
        action_dim: int = 2,
        hidden_dim: int = 64,
        learning_rate: float = 0.001,
        gamma: float = 0.95,
        epsilon: float = 0.1
    ):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
    
    def get_action(self, state: np.ndarray, valid_actions: List[int]) -> int:
        """
        Epsilon-greedy action selection
        
        Args:
            state: Current state vector
            valid_actions: List of valid action indices
        
        Returns:
            Selected action
        """
        if np.random.rand() < self.epsilon:
            return np.random.choice(valid_actions)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.network(state_tensor)[0]
        
        # Pick best valid action
        valid_q = [(a, q_values[a].item()) for a in valid_actions]
        best_action = max(valid_q, key=lambda x: x[1])[0]
        
        return best_action
    
    def update(self, state: np.ndarray, action: int, reward: float, 
               next_state: np.ndarray, done: bool):
        """
        Q-learning update with TD error
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode finished
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        
        # Current Q-value
        current_q = self.network(state_tensor)[0, action]
        
        # Target Q-value
        with torch.no_grad():
            if done:
                target_q = reward
            else:
                next_q = self.network(next_state_tensor).max()
                target_q = reward + self.gamma * next_q
        
        # MSE loss
        loss = (current_q - target_q) ** 2
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()
    
    def decay_epsilon(self, decay_rate: float = 0.995):
        """Decay exploration rate"""
        self.epsilon = max(0.01, self.epsilon * decay_rate)
