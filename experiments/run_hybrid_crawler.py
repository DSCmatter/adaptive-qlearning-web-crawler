"""
Phase 5: Hybrid Crawler Integration Run
Tests the fully assembled Adaptive Crawler on live seeds.
"""

import sys
from pathlib import Path
import json
import torch
import yaml
import pickle

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from crawler.adaptive_crawler import AdaptiveCrawler
from models.feature_extractor import FeatureExtractor
from models.gnn_encoder import WebGraphEncoder
from models.qlearning_agent import QLearningAgent
from models.contextual_bandit import LinUCBBandit

def load_config():
    config_path = Path(__file__).parent.parent / 'configs' / 'crawler_config.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run():
    print("="*60)
    print("Phase 5: Live Hybrid Web Crawling Execution")
    print("="*60)

    config = load_config()
    base_dir = Path(__file__).parent.parent
    models_dir = base_dir / 'data' / 'models'

    # === 1. Load Pre-Trained Components ===
    
    # Feature Extractor
    print("Initializing Feature Extractor...")
    feature_extractor = FeatureExtractor()

    # GNN Encoder
    print("Loading Frozen GNN...")
    gnn_config = config.get('gnn', {})
    gnn_encoder = WebGraphEncoder(
        input_dim=gnn_config.get('input_dim', 174),
        hidden_dim=gnn_config.get('hidden_dim', 128),
        output_dim=gnn_config.get('output_dim', 64),
        num_layers=gnn_config.get('num_layers', 2)
    )
    gnn_encoder.load_state_dict(torch.load(models_dir / 'gnn_encoder_frozen.pt', weights_only=True))
    gnn_encoder.eval() 

    # Bandit Loader
    print("Loading Contextual Bandit...")
    bandit_config = config.get('bandit', {})
    bandit = LinUCBBandit(context_dim=bandit_config.get('context_dim', 174))
    bandit_params_path = models_dir / 'bandit_arms.pkl'
    if bandit_params_path.exists():
        with open(bandit_params_path, 'rb') as f:
            bandit.arms = pickle.load(f)

    # Q-Learning Agent Loader
    print("Loading Q-Learning Agent...")
    ql_config = config.get('qlearning', {})
    q_agent = QLearningAgent(
        state_dim=ql_config.get('state_dim', 69),
        action_dim=ql_config.get('action_dim', 2)
    )
    q_agent_path = models_dir / 'qlearning_agent.pt'
    if q_agent_path.exists():
        q_agent.load_state_dict(torch.load(q_agent_path, weights_only=True))
        q_agent.epsilon = 0.01  # Exploit learned strategy heavily

    # === 2. Build Hybrid Crawler ===
    crawler = AdaptiveCrawler(
        gnn_encoder=gnn_encoder,
        qlearning_agent=q_agent,
        bandit=bandit,
        feature_extractor=feature_extractor,
        max_pages=20, # Reduced bound for quick live testing
        delay=1.0 # Politeness constraint
    )

    # === 3. Run Live Crawl Tests ===
    # For Phase 5 we test integration against known topic seeds 
    seeds_path = base_dir / 'data' / 'seeds' / 'ml_seeds.json'
    
    with open(seeds_path, 'r') as f:
        seed_data = json.load(f)
        
    test_seed = seed_data["seeds"][0]  # Take the first Machine Learning seed
    print(f"\n[!] Beginning Live Crawl Execution for target: {test_seed}")
    
    results = crawler.crawl(test_seed)
    
    print("\n" + "="*60)
    print("PHASE 5 Live Run Results")
    print("="*60)
    print(f"Total Pages Crawled: {results['total_crawled']}")
    print(f"Relevant Targets:    {results['relevant_found']}")
    print(f"Harvest Rate:        {results['harvest_rate']:.3f}")
    print(f"Avg Session Reward:  {results['total_reward']:.2f}")
    
if __name__ == "__main__":
    run()