"""Main training script - Phase 4.2 Hybrid Agent Training"""

import sys
from pathlib import Path
import numpy as np
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.qlearning_agent import QLearningAgent
from models.contextual_bandit import LinUCBBandit
from models.feature_extractor import FeatureExtractor
from models.gnn_encoder import WebGraphEncoder
from simulated_env import SimulatedCrawlEnv

def load_config() -> dict:
    config_path = Path(__file__).parent.parent / 'configs' / 'crawler_config.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def build_q_state(gnn_embedding, metrics) -> np.ndarray:
    """Combine 64-dim GNN embedding with 5 scalar metrics into 69-dim state"""
    state = np.zeros(69)
    state[:64] = gnn_embedding
    
    # Normalize metrics based on config values
    state[64] = metrics['budget_remaining'] / 200.0
    state[65] = metrics['relevant_found'] / 200.0  # Max possible
    state[66] = metrics['current_depth'] / 200.0
    state[67] = np.clip(metrics['avg_reward'] / 10.0, -1, 1)
    state[68] = metrics['exploration_rate']
    
    return state

def get_gnn_embedding(encoder, feature_extractor, url, graph):
    """Dynamically get GNN Embeddings for a URL passing through the Frozen Network"""
    # 1. Provide zero-state for initialization to build generic context
    init_context = feature_extractor.build_context_vector(url, "", "", np.zeros(64), graph)
    
    # 2. Add structural node mappings if we mapped edges (for simulation we will treat node individually or map dummy edge index)
    x = torch.tensor(init_context, dtype=torch.float).unsqueeze(0)
    edge_index = torch.empty((2, 0), dtype=torch.long)
    
    # Run through frozen network
    with torch.no_grad():
        embedding = encoder(x, edge_index)[0].numpy()
        
    return embedding

def train():
    print("=" * 60)
    print("Phase 4: Hybrid Agent Training")
    print("=" * 60)
    
    config = load_config()
    
    base_dir = Path(__file__).parent.parent
    graph_path = base_dir / 'data' / 'graphs' / 'bootstrap_graph.pkl'
    models_dir = base_dir / 'data' / 'models'
    
    # 1. Initialize Environment
    env = SimulatedCrawlEnv(graph_path)
    
    # 2. Load Frozen GNN
    print("Loading Frozen GNN Encoder...")
    gnn_config = config.get('gnn', {})
    encoder = WebGraphEncoder(
        input_dim=gnn_config.get('input_dim', 174),
        hidden_dim=gnn_config.get('hidden_dim', 128),
        output_dim=gnn_config.get('output_dim', 64),
        num_layers=gnn_config.get('num_layers', 2)
    )
    encoder.load_state_dict(torch.load(models_dir / 'gnn_encoder_frozen.pt', weights_only=True))
    encoder.eval() # Must be frozen
    
    extractor = FeatureExtractor()
    
    # 3. Initialize RL Agents
    print("Initializing RL Agents...")
    ql_config = config.get('qlearning', {})
    q_agent = QLearningAgent(
        state_dim=ql_config.get('state_dim', 69),
        action_dim=ql_config.get('action_dim', 2),  # 0: STOP, 1: CONTINUE
        hidden_dim=ql_config.get('hidden_dim', 64),
        learning_rate=ql_config.get('learning_rate', 0.001),
        gamma=ql_config.get('gamma', 0.95),
        epsilon=ql_config.get('epsilon', 1.0) # Start high
    )
    
    bandit_config = config.get('bandit', {})
    bandit = LinUCBBandit(
        context_dim=bandit_config.get('context_dim', 174),
        alpha=bandit_config.get('alpha', 1.0)
    )
    
    num_episodes = config.get('training', {}).get('num_episodes', 500)
    max_pages = config.get('crawler', {}).get('max_pages', 200)
    
    stats_history = []
    
    print(f"\nStarting {num_episodes} training episodes...")
    for episode in range(num_episodes):
        metrics = env.reset()
        current_url = env.current_url
        
        episode_reward = 0
        pages_crawled = 0
        
        while metrics['budget_remaining'] > 0:
            if current_url is None:
                break
                
            metrics['exploration_rate'] = q_agent.epsilon
            
            # Feature pipeline
            gnn_embedding = get_gnn_embedding(encoder, extractor, current_url, env.web_graph)
            state = build_q_state(gnn_embedding, metrics)
            
            # Q-Agent Action
            # Actions: 0 = STOP, 1 = CONTINUE
            action = q_agent.get_action(state, [0, 1])
            
            if action == 0:  # STOP
                # Penalize early stopping without finding anything, reward if found things
                stop_reward = 10 if metrics['relevant_found'] > 5 else -10
                q_agent.update(state, action, stop_reward, state, True)
                break
                
            if action == 1:  # CONTINUE
                candidates = env.get_candidate_links(current_url)
                
                # If dead end, treat as stop
                if not candidates:
                    q_agent.update(state, action, -5, state, True)
                    break
                    
                # Downsample candidates to save compute
                if len(candidates) > config.get('crawler', {}).get('max_candidates', 50):
                    candidates = np.random.choice(candidates, config.get('crawler', {}).get('max_candidates', 50), replace=False).tolist()
                
                # Contextual Bandit selects link
                contexts = []
                for cand in candidates:
                    c_embed = get_gnn_embedding(encoder, extractor, cand, env.web_graph)
                    ctx = extractor.build_context_vector(cand, "", "", c_embed, env.web_graph)
                    contexts.append(ctx)
                    
                next_url, cand_idx = bandit.select_link(candidates, contexts)
                
                # Step environment
                next_metrics, step_reward, done = env.step(next_url)
                
                # Update Bandit
                bandit.update(next_url, contexts[cand_idx], step_reward)
                
                # Update Q-Learning Agent
                next_embed = get_gnn_embedding(encoder, extractor, next_url, env.web_graph)
                next_metrics['exploration_rate'] = q_agent.epsilon
                next_state = build_q_state(next_embed, next_metrics)
                
                q_agent.update(state, action, step_reward, next_state, done)
                
                current_url = next_url
                metrics = next_metrics
                episode_reward += step_reward
                pages_crawled += 1
                
        # End of Episode
        q_agent.decay_epsilon(ql_config.get('epsilon_decay', 0.995))
        
        stats = {
            'episode': episode,
            'reward': episode_reward,
            'pages_crawled': pages_crawled,
            'relevant_found': metrics['relevant_found'],
            'harvest_rate': metrics['relevant_found'] / max(1, pages_crawled),
            'epsilon': q_agent.epsilon
        }
        stats_history.append(stats)
        
        if (episode + 1) % 50 == 0:
            print(f"Episode {episode+1:03d} | Avg Reward: {episode_reward:5.2f} | Harvest: {stats['harvest_rate']:5.3f} | Eps: {q_agent.epsilon:5.3f}")
            
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Final Episode Harvest Rate: {stats_history[-1]['harvest_rate']:.3f}")
    
    # Save the models
    print("\nSaving Models...")
    torch.save(q_agent.state_dict(), models_dir / 'qlearning_agent.pt')
    
    # Because bandit uses standard numpy arrays/dictionaries
    import pickle
    with open(models_dir / 'bandit_arms.pkl', 'wb') as f:
        pickle.dump(bandit.arms, f)
        
    print(f"Q-Network saved to {models_dir / 'qlearning_agent.pt'}")
    print(f"Bandit internal state saved to {models_dir / 'bandit_arms.pkl'}")
    print("=" * 60)

if __name__ == '__main__':
    train()
