"""Main training script"""

import sys
sys.path.append('..')

# TODO: Implement training loop following WALKTHROUGH.md Phase 4.2


def train(config_path: str = '../configs/crawler_config.yaml'):
    """
    Train hybrid RL crawler
    
    Steps:
    1. Load configuration
    2. Bootstrap initial graph (one-time)
    3. Pre-train GNN (one-time, ~30 min)
    4. Train Q-learning + Bandit (500 episodes)
    5. Save trained models
    """
    print("Training not yet implemented")
    print("Follow WALKTHROUGH.md Phase 4.2 for implementation")


if __name__ == '__main__':
    train()
