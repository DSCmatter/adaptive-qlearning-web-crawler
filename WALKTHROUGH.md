# Project Walkthrough: Adaptive Q-Learning Web Crawler with Contextual Bandits and GNNs

## Executive Summary

This project implements a novel hybrid approach to focused web crawling that combines:
- **Q-Learning**: For long-term reward optimization in navigation strategy
- **Contextual Bandits**: For intelligent exploration-exploitation balance in link selection
- **Graph Neural Networks (GNNs)**: For understanding web graph structure and link importance

This fusion addresses limitations in existing RL-based crawlers by leveraging graph topology information and contextual features for more informed decision-making.

## Innovation & Research Contribution

### Novel Aspects
1. **Hybrid RL Architecture**: Combines value-based (Q-learning) and bandit-based approaches
2. **GNN-Enhanced State Representation**: Uses graph structure for richer state embeddings
3. **Contextual Link Scoring**: Bandits use page content, anchor text, and graph features as context
4. **Dynamic Exploration Strategy**: Adapts exploration rate based on crawl progress and graph density

### Differentiation from Prior Work
- **vs. Traditional Q-Learning Crawlers**: We add contextual information and graph structure awareness
- **vs. Deep RL Crawlers**: We use bandits for faster convergence on link selection with less data
- **vs. Heuristic-Based Crawlers**: We learn optimal policies rather than relying on hand-crafted rules

---

## Phase 1: Project Setup & Environment Configuration

### 1.1 Development Environment Setup

**Timeline**: Week 1, Days 1-2

**Objectives**:
- Set up Python development environment
- Install required libraries
- Configure version control

**Steps**:

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# STUDENT-BUDGET VERSION: CPU-only, lightweight (~2GB)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric
pip install numpy pandas matplotlib seaborn
pip install scikit-learn scipy
pip install requests beautifulsoup4 lxml
pip install networkx
pip install tensorboard  # Optional for visualization
pip install pytest pytest-cov  # Testing

# Skip these to save space/time:
# - scrapy (use requests instead)
# - torchvision, torchaudio (not needed)

# Total size: ~2GB (vs 10GB+ with GPU support)
# Installation time: ~10 minutes
```

**Budget Reality Check**:
- ✅ Works on ANY laptop from last 5 years
- ✅ No GPU needed (CPU-only is fine)
- ✅ No cloud costs ($0/month)
- ✅ Can use Google Colab free tier if laptop struggles

**Key Libraries**:
- `torch` + `torch-geometric`: For GNN implementation
- `scrapy`: Web crawling framework
- `networkx`: Graph manipulation
- `beautifulsoup4`: HTML parsing
- `scikit-learn`: Baseline classifiers

**Deliverable**: Working Python environment with all dependencies

### 1.2 Project Structure

```
adaptive-qlearning-web-crawler/
├── src/
│   ├── __init__.py
│   ├── crawler/
│   │   ├── __init__.py
│   │   ├── base_crawler.py
│   │   ├── adaptive_crawler.py
│   │   └── heuristic_crawler.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── gnn_encoder.py
│   │   ├── qlearning_agent.py
│   │   ├── contextual_bandit.py
│   │   └── feature_extractor.py
│   ├── environment/
│   │   ├── __init__.py
│   │   ├── crawl_environment.py
│   │   └── reward_function.py
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── web_graph.py
│   │   └── graph_builder.py
│   └── utils/
│       ├── __init__.py
│       ├── url_utils.py
│       ├── text_processing.py
│       └── metrics.py
├── experiments/
│   ├── train_agent.py
│   ├── evaluate_baseline.py
│   └── compare_methods.py
├── tests/
│   ├── test_crawler.py
│   ├── test_models.py
│   └── test_environment.py
├── data/
│   ├── seeds/
│   ├── target_domains/
│   └── results/
├── configs/
│   ├── crawler_config.yaml
│   ├── model_config.yaml
│   └── experiment_config.yaml
├── notebooks/
│   ├── data_exploration.ipynb
│   └── results_analysis.ipynb
├── docs/
│   ├── DESIGN.md
│   └── API.md
├── requirements.txt
├── setup.py
├── README.md
└── WALKTHROUGH.md
```

---

## Phase 2: Data Collection & Preprocessing

### 2.1 Seed URL Collection & Graph Bootstrapping

**Timeline**: Week 1, Days 3-5

**Objectives**:
- Identify target domains
- Collect seed URLs
- Define relevance criteria
- **Build initial graph structure (bootstrap)**

**Critical Addition: Graph Bootstrapping**

Before training the GNN, we need an initial graph. This solves the chicken-egg problem:

```python
# src/graph/graph_builder.py
def bootstrap_initial_graph(seed_urls, max_pages=500):
    """
    Build starter graph with simple BFS crawl
    Budget-friendly: 500 pages in ~10 minutes
    """
    graph = WebGraph()
    queue = deque(seed_urls)
    visited = set()
    
    while queue and len(visited) < max_pages:
        url = queue.popleft()
        if url in visited:
            continue
        
        try:
            # Fetch page
            response = requests.get(url, timeout=5)
            html = response.text
            
            # Add to graph with features
            features = extract_features(url, html)
            graph.add_page(url, html, features)
            
            # Extract outgoing links
            links = extract_links(html, url)
            queue.extend(links[:10])  # Limit breadth
            
            visited.add(url)
            
        except Exception as e:
            logger.warning(f"Failed to fetch {url}: {e}")
            continue
    
    return graph
```

**Why This Matters**:
- GNN needs graph structure to compute embeddings
- Can't compute embeddings for unseen pages initially
- Bootstrap provides "seed" graph for cold-start
- Initialize unseen pages with zero embeddings (GNN learns over time)

**Process**:

1. **Define Target Domains**: Choose 3-5 target topics (e.g., "machine learning", "climate science", "blockchain")

2. **Collect Seeds**:
   - Manually curate 50-100 highly relevant URLs per topic
   - Use existing datasets (Common Crawl, DMOZ)
   - Leverage domain-specific portals

3. **Create Ground Truth**:
   - Label 500-1000 URLs as relevant/irrelevant per topic
   - Use for training relevance classifiers
   - Split: 70% train, 15% validation, 15% test

**Deliverable**: 
- `data/seeds/topic_seeds.json`
- `data/target_domains/labeled_urls.csv`

### 2.2 Feature Extraction Pipeline

**Timeline**: Week 2, Days 1-3

**Components**:

1. **URL Features**:
   - Domain authority
   - URL depth
   - Subdomain type
   - Path structure

2. **Content Features**:
   - TF-IDF vectors (top 5000 terms)
   - Topic model embeddings (LDA)
   - Named entity recognition
   - Language detection

3. **Anchor Text Features**:
   - Anchor text embedding (Word2Vec/BERT)
   - Link position on page
   - Surrounding context

4. **Graph Features** (extracted via GNN):
   - Node degree (in/out)
   - PageRank score
   - Clustering coefficient
   - Community membership

**Implementation**:
```python
# src/models/feature_extractor.py
class FeatureExtractor:
    def extract_url_features(self, url): ...
    def extract_content_features(self, html): ...
    def extract_anchor_features(self, anchor_text, context): ...
    def extract_graph_features(self, node_id, graph): ...
```

**Deliverable**: Feature extraction pipeline with unit tests

---

## Phase 3: Core Model Development

### 3.1 Graph Neural Network Implementation

**Timeline**: Week 2-3, Days 4-7

**Architecture**: GraphSAGE-based encoder (lightweight for CPU)

**Student-Budget Optimization**:

```python
# src/models/gnn_encoder.py
import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv

class WebGraphEncoder(nn.Module):
    """
    Lightweight GNN encoder (CPU-friendly, ~200K parameters)
    """
    def __init__(self, input_dim=174, hidden_dim=128, output_dim=64, num_layers=2):
        super().__init__()
        # Reduced from 3 to 2 layers for speed
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        self.convs.append(SAGEConv(hidden_dim, output_dim))
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.3)  # Reduced dropout
        
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.activation(x)
            x = self.dropout(x)
        x = self.convs[-1](x, edge_index)
        return x
```

**Training Strategy** (Pre-train Once, Then Freeze):

```python
# ONE-TIME PRE-TRAINING (not online)
def pretrain_gnn(bootstrap_graph, labeled_nodes, epochs=50):
    """
    Pre-train on bootstrap data (~30 min on CPU)
    Then FREEZE during crawling (save compute!)
    """
    model = WebGraphEncoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    
    for epoch in range(epochs):
        model.train()
        
        # Full-batch training (small graph)
        embeddings = model(bootstrap_graph.x, bootstrap_graph.edge_index)
        predictions = torch.sigmoid(embeddings[labeled_nodes])
        labels = torch.tensor([bootstrap_graph.get_label(n) for n in labeled_nodes])
        
        loss = criterion(predictions, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    # FREEZE for deployment
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
    return model
```

**Key Budget Decision**: 
- ❌ NO online GNN updates during crawling (too expensive)
- ✅ Pre-train once on bootstrap graph (30 minutes)
- ✅ Use frozen embeddings during training (fast)
- ✅ Optional: Fine-tune every 2000 pages if time allows

**Deliverable**: Pre-trained frozen GNN encoder (~30 min training)

### 3.2 Contextual Bandit Implementation

**Timeline**: Week 3, Days 1-4

**Algorithm**: LinUCB (Linear Upper Confidence Bound)

**Key Idea**: 
- Each link is an "arm"
- Context = [GNN embedding, URL features, content features, anchor features]
- Reward = relevance score of destination page
- UCB balances exploitation (high expected reward) with exploration (high uncertainty)

**Implementation**:

```python
# src/models/contextual_bandit.py
import numpy as np

class LinUCBBandit:
    """
    Contextual Bandit using Linear UCB algorithm
    """
    def __init__(self, context_dim, alpha=1.0):
        self.context_dim = context_dim
        self.alpha = alpha  # Exploration parameter
        
        # Initialize A and b for each arm (dynamically)
        self.arms = {}  # arm_id -> (A, b)
        
    def select_link(self, candidate_links, contexts):
        """
        Select best link using UCB criterion
        contexts: list of context vectors for each candidate link
        """
        ucb_scores = []
        for i, (link, context) in enumerate(zip(candidate_links, contexts)):
            if link not in self.arms:
                self.arms[link] = (
                    np.identity(self.context_dim),  # A
                    np.zeros(self.context_dim)       # b
                )
            
            A, b = self.arms[link]
            A_inv = np.linalg.inv(A)
            theta = A_inv @ b
            
            # UCB score: expected reward + exploration bonus
            expected_reward = context @ theta
            uncertainty = np.sqrt(context @ A_inv @ context)
            ucb_score = expected_reward + self.alpha * uncertainty
            
            ucb_scores.append(ucb_score)
        
        best_idx = np.argmax(ucb_scores)
        return candidate_links[best_idx], best_idx
    
    def update(self, link, context, reward):
        """Update parameters after observing reward"""
        A, b = self.arms[link]
        A += np.outer(context, context)
        b += reward * context
        self.arms[link] = (A, b)
```

**Deliverable**: Working contextual bandit with UCB

### 3.3 Q-Learning Agent

**Timeline**: Week 3-4, Days 5-7

**State Space** (Compact Representation):
- Current page embedding (from GNN): 64 dim
- Crawl budget remaining: 1 scalar
- Number of relevant pages found: 1 scalar
- Current depth: 1 scalar
- Average reward: 1 scalar
- Exploration rate: 1 scalar
- **Total: 69 dimensions** (compact!)

**Action Space**:
- Select next link to crawl (chosen by bandit): 0
- Stop crawling: 1
- **Total: 2 actions** (simplified from 4)

**Reward Function**:
```python
# src/environment/reward_function.py
def compute_reward(page, is_relevant, crawl_cost):
    """
    Reward = relevance_score - cost_penalty
    """
    relevance_reward = 10.0 if is_relevant else -5.0
    cost_penalty = 0.1 * crawl_cost  # penalize page load time
    depth_penalty = 0.05 * depth  # penalize deep crawling
    
    return relevance_reward - cost_penalty - depth_penalty
```

**Q-Learning Implementation** (Lightweight Function Approximation):

```python
# src/models/qlearning_agent.py
import torch
import torch.nn as nn

class QLearningAgent(nn.Module):
    """Small Q-network: only ~10K parameters (CPU-friendly)"""
    def __init__(self, state_dim=69, action_dim=2, learning_rate=0.001):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.gamma = 0.95
        self.epsilon = 0.1
        
    def get_action(self, state, valid_actions):
        """Epsilon-greedy action selection"""
        if np.random.rand() < self.epsilon:
            return np.random.choice(valid_actions)
        
        state_tensor = torch.FloatTensor(state)
        with torch.no_grad():
            q_values = self.network(state_tensor)
        
        valid_q = [q_values[a].item() for a in valid_actions]
        best_idx = np.argmax(valid_q)
        return valid_actions[best_idx]
    
    def update(self, state, action, reward, next_state, done):
        """Q-learning update with gradient descent"""
        state_tensor = torch.FloatTensor(state)
        next_state_tensor = torch.FloatTensor(next_state)
        
        # Current Q-value
        current_q = self.network(state_tensor)[action]
        
        # Target Q-value
        with torch.no_grad():
            if done:
                target_q = reward
            else:
                next_q = self.network(next_state_tensor)
                target_q = reward + self.gamma * torch.max(next_q)
        
        # MSE loss
        loss = (current_q - target_q) ** 2
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()
```

**Why Function Approximation?**
- ❌ Tabular Q-learning with 64-dim embeddings = infinite state space
- ✅ Neural network generalizes across similar states
- ✅ Only 10K parameters (trains in seconds per update)
- ✅ Works great on CPU

**Deliverable**: Q-learning agent with epsilon-greedy exploration

### 3.4 Hybrid Architecture Integration

**Timeline**: Week 4, Days 1-3

**Integration Strategy**:

```
┌─────────────────────────────────────────────────┐
│              Adaptive Crawler                    │
└─────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│           Q-Learning Agent                       │
│  - High-level navigation strategy                │
│  - Decides: continue/stop, domain switching      │
└─────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│           GNN Web Graph Encoder                  │
│  - Encodes graph structure                       │
│  - Produces node embeddings                      │
└─────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│       Contextual Bandit (LinUCB)                 │
│  - Selects specific link from candidates         │
│  - Uses context: GNN embedding + features        │
└─────────────────────────────────────────────────┘
```

**Workflow**:
1. **Bootstrap**: Build initial graph (500 pages, one-time)
2. **Current State**: Q-agent observes state (GNN embedding + crawl stats)
3. **High-Level Decision**: Q-agent decides to continue or stop crawling
4. **Link Candidates**: Extract all outgoing links from current page
5. **Filter Candidates**: **Reduce to max 50 links** (critical for efficiency!)
   ```python
   filtered_links = filter_candidates(
       raw_links, 
       current_url, 
       max_candidates=50  # Keeps LinUCB fast
   )
   ```
6. **Context Extraction**: For each filtered link, create context vector:
   - GNN embedding (pre-computed or zero if unseen)
   - URL/content/anchor features
7. **Link Selection**: Bandit selects best link using UCB (fast: <10ms for 50 candidates)
8. **Crawl**: Fetch selected page
9. **Reward**: Compute reward based on relevance
10. **Update**: 
   - Update bandit parameters (O(d²) per link)
   - Update Q-network (single gradient step)
   - NO GNN update (frozen)

**Deliverable**: Integrated hybrid crawler

---

## Phase 4: Training & Optimization

### 4.1 GNN Pre-training

**Timeline**: Week 4, Days 4-7

**Process**:
1. Build initial web graph from seed crawl (breadth-first, 10K pages)
2. Extract features for all nodes
3. Train GNN on labeled data (supervised)
4. Validate on held-out set
5. Save trained model

**Metrics**:
- Classification accuracy on relevance prediction
- ROC-AUC score
- Embedding quality (t-SNE visualization)

### 4.2 Hybrid Agent Training

**Timeline**: Week 5-6 (can be done overnight for 1 week)

**Student-Budget Training Algorithm**:

```
REDUCED BUDGET VERSION (practical for students):
- Episodes: 500 (instead of 5000)
- Pages per episode: 200 (instead of 1000)
- Total pages: 100K (manageable!)
- Training time: ~5 days running overnight
- Cost: $0 (local machine)

For each episode (e = 1 to 500):
    1. Sample seed URL from seed set
    2. Initialize crawl environment
    3. While budget < 200:  # Reduced from 1000
        a. Get current state from GNN (frozen embeddings)
        b. Q-agent decides: continue or stop
        c. If continue:
            - Extract candidate links
            - FILTER to max 50 candidates
            - Create contexts for each link
            - Bandit selects best link
            - Crawl selected link
            - Compute reward
            - Update bandit (fast)
            - Update Q-network (single gradient step)
        d. If stop: end episode
    4. Compute episode metrics
    5. Checkpoint every 50 episodes
    
    # NO GNN updates (frozen for efficiency)
```

**Hyperparameters to Tune** (Start with these):
- Q-learning: α=0.001 (network LR), γ=0.95, ε=0.1
- LinUCB: α=1.0 (exploration)
- GNN: No tuning needed (pre-trained and frozen)
- Crawl budget: 200 pages per episode

**Training Duration**: 
- 500 episodes × ~7 min/episode = **~58 hours** (2.5 days)
- Run overnight for 3-4 nights
- **Practical for students!**

**Deliverable**: Trained hybrid crawler

### 4.3 Baseline Implementations

**Timeline**: Week 7, Days 1-3

**Baselines to Implement**:

1. **Random Crawler**: Random link selection
2. **Best-First Crawler**: Greedy relevance classifier
3. **PageRank-Based**: Follow highest PageRank links
4. **Pure Q-Learning**: Q-learning without bandits/GNN
5. **Pure Bandit**: Contextual bandit without Q-learning

**Deliverable**: All baseline crawlers implemented

---

## Phase 4.5: Computational Efficiency & Costs

### 4.5.1 Time & Cost Analysis

**Per-Episode Breakdown** (200 pages):
```
Component          Time/Page    Total/Episode
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Network I/O        2.0s         400s (6.7 min)
HTML Parsing       0.05s        10s
Feature Extract    0.02s        4s
Link Filtering     0.005s       1s
Bandit Selection   0.01s        2s
Q-Network Update   0.001s       0.2s
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total per Episode               ~7 minutes
```

**Full Training (500 episodes)**:
- Time: 500 × 7 min = **3500 min = 58 hours**
- Spread over: **3-4 days** (run overnight)
- Electricity cost: ~10W × 58h = 0.58 kWh = **~$0.07**
- Cloud cost: **$0** (run locally)
- **Total cost: $0.07** 🎉

**Memory Usage**:
- Graph: ~550 KB
- Models: ~13 MB
- Runtime: ~100 MB
- **Peak RAM: <500 MB** (works on 4GB RAM laptop!)

### 4.5.2 Optimization Tips

```python
# TIP 1: Use multiprocessing for I/O
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(fetch_page, url) for url in batch]
    results = [f.result() for f in futures]

# TIP 2: Cache embeddings
embedding_cache = {}

def get_embedding(url):
    if url not in embedding_cache:
        embedding_cache[url] = gnn.encode(url)
    return embedding_cache[url]

# TIP 3: Limit bandit arms (memory)
if len(bandit.arms) > 1000:
    # Keep only frequently-used arms
    bandit.prune_arms(keep_top=1000)

# TIP 4: Use generators to save memory
def crawl_generator(seeds):
    for url in seeds:
        yield crawl_page(url)
```

---

## Phase 5: Evaluation & Analysis

### 5.1 Evaluation Metrics

**Timeline**: Week 7, Days 4-7

**Metrics**:

1. **Harvest Rate**: `relevant_pages / total_pages_crawled`
2. **Target Recall**: `relevant_pages_found / total_relevant_pages_available`
3. **Crawl Efficiency**: `relevant_pages / crawl_time`
4. **Precision at K**: Precision in first K pages
5. **Average Reward**: Mean episode reward
6. **Convergence Speed**: Episodes to reach 90% optimal performance

**Evaluation Protocol**:
- 3 target domains
- 5 seed URLs per domain
- Budget: 1000 pages per run
- 10 runs per configuration
- Report mean ± std

### 5.2 Experimental Evaluation

**Timeline**: Week 8

**Experiments**:

1. **Main Comparison**: Hybrid vs. all baselines
2. **Ablation Study**:
   - Hybrid vs. Q-learning only
   - Hybrid vs. Bandit only
   - Hybrid vs. No GNN
3. **Hyperparameter Sensitivity**:
   - Vary α (LinUCB exploration)
   - Vary ε (Q-learning exploration)
   - Vary GNN depth
4. **Scalability Analysis**:
   - Performance vs. graph size
   - Runtime vs. number of candidates
5. **Transfer Learning**:
   - Train on domain A, test on domain B

**Deliverable**: 
- Results tables
- Performance plots
- Statistical significance tests

### 5.3 Results Analysis

**Timeline**: Week 9

**Analysis Tasks**:

1. **Quantitative Analysis**:
   - Statistical tests (t-test, ANOVA)
   - Effect size calculations
   - Confidence intervals

2. **Qualitative Analysis**:
   - Visualize learned Q-values
   - Analyze bandit arm selections
   - Inspect GNN embeddings (t-SNE)
   - Case studies of crawl paths

3. **Failure Analysis**:
   - Identify failure modes
   - Analyze suboptimal decisions
   - Error categorization

**Deliverable**: Comprehensive analysis report

---

## Phase 6: Documentation & Deployment

### 6.1 Code Documentation

**Timeline**: Week 10, Days 1-3

**Tasks**:
- Add docstrings to all functions/classes
- Create API documentation
- Write usage tutorials
- Add inline comments

### 6.2 Deployment Package

**Timeline**: Week 10, Days 4-5

**Components**:
- Docker containerization
- REST API for crawler
- Configuration management
- Logging and monitoring

### 6.3 Research Paper Preparation

**Timeline**: Week 10-12

**Sections**:
1. Abstract
2. Introduction & Motivation
3. Related Work
4. Methodology (Hybrid Architecture)
5. Experimental Setup
6. Results & Discussion
7. Conclusion & Future Work
8. Appendices

---

## Phase 7: Advanced Extensions (Optional)

### 7.1 Multi-Task Learning
- Train GNN to predict multiple objectives (relevance, crawl cost, link quality)

### 7.2 Meta-Learning
- Use MAML to enable fast adaptation to new domains

### 7.3 Distributed Crawling
- Extend to multi-agent cooperative crawling

### 7.4 Adversarial Robustness
- Train against adversarial pages (SEO spam)

---

## Expected Timeline Summary (Student-Budget Version)

| Phase | Duration | Compute Time | Key Deliverables |
|-------|----------|--------------|------------------|
| 1. Setup | 1 week | ~1 hour | Environment, project structure |
| 2. Data | 1.5 weeks | ~2 hours | Seeds, bootstrap graph (500 pages) |
| 3. Models | 3 weeks | ~3 hours | GNN pre-training (30 min), Bandit, Q-net |
| 4. Training | 1 week* | ~60 hours | Trained agents (run overnight) |
| 5. Evaluation | 1.5 weeks | ~5 hours | Metrics, experiments, analysis |
| 6. Documentation | 1.5 weeks | ~2 hours | Docs, paper |
| **Total** | **9.5 weeks** | **~73 hours compute** | Complete research project |

*Can overlap with other work (runs in background)

**Budget Breakdown**:
- Hardware: $0 (use existing laptop)
- Cloud: $0 (run locally)
- Electricity: ~$0.10 total
- **Total Cost: ~$0.10** 💰

**Comparison to "Full" Version**:
| Metric | Full Version | Student Version | Difference |
|--------|--------------|-----------------|------------|
| Episodes | 5000 | 500 | -90% |
| Pages/Episode | 1000 | 200 | -80% |
| Training Time | 3-7 days GPU | 3-4 days CPU | Similar! |
| Cost | $50-200 cloud | $0.10 | -99.95% |
| Performance | 70-75% HR | 65-70% HR | -5-10% (acceptable!) |

---

## Key Success Factors

1. **Start Simple**: Implement basic versions first, then add complexity
2. **Test Incrementally**: Unit test each component before integration
3. **Log Everything**: Extensive logging for debugging and analysis
4. **Visualize Early**: Create visualizations to understand model behavior
5. **Iterate Based on Results**: Be prepared to adjust approach based on initial results

---

## Potential Challenges & Mitigation

| Challenge | Mitigation Strategy |
|-----------|---------------------|
| GNN training instability | Use batch normalization, gradient clipping |
| Bandit cold-start problem | Warm-start with supervised pre-training |
| Large action space | Prune candidates using simple heuristics |
| Reward sparsity | Shape rewards with intermediate signals |
| Graph construction overhead | Incremental graph updates, caching |
| Reproducibility | Set random seeds, version dependencies |

---

## Resources Required

- **Compute**: 1 GPU (V100 or better), 32GB RAM
- **Storage**: 100GB for crawled data
- **Time**: ~3 months full-time or 6 months part-time
- **External APIs**: None required (self-contained)

---

## Next Steps

After reading this walkthrough:
1. Review the DESIGN.md document for technical specifications
2. Set up the development environment (Phase 1)
3. Begin with data collection (Phase 2)
4. Follow the phased approach systematically

**Good luck with the implementation!**
