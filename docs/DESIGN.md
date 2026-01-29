# Design Document: Adaptive Q-Learning Web Crawler with Contextual Bandits and GNNs

## 1. System Overview

### 1.1 Architecture Philosophy

This system implements a **hierarchical reinforcement learning** approach to focused web crawling, combining three complementary techniques:

1. **Q-Learning**: Provides high-level navigation strategy and long-term planning
2. **Contextual Bandits**: Handles immediate link selection with efficient exploration
3. **Graph Neural Networks**: Captures structural information of the web graph for informed decision-making

### 1.2 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Web Crawler System                        │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Crawl Coordinator                           │
│  - Episode management                                            │
│  - Budget tracking                                               │
│  - Metrics collection                                            │
└─────────────────────────────────────────────────────────────────┘
                                 │
                ┌────────────────┴────────────────┐
                ▼                                  ▼
┌──────────────────────────────┐   ┌──────────────────────────────┐
│    Decision Layer            │   │    Knowledge Layer           │
│  ┌────────────────────────┐  │   │  ┌────────────────────────┐ │
│  │  Q-Learning Agent      │  │   │  │  Web Graph Manager     │ │
│  │  - State evaluation    │  │   │  │  - Graph construction  │ │
│  │  - Action selection    │  │   │  │  - Graph updates       │ │
│  │  - Policy learning     │  │   │  │  - Query interface     │ │
│  └────────────────────────┘  │   │  └────────────────────────┘ │
│              │                │   │              │              │
│              ▼                │   │              ▼              │
│  ┌────────────────────────┐  │   │  ┌────────────────────────┐ │
│  │  Contextual Bandit     │  │   │  │  GNN Encoder           │ │
│  │  - Link scoring        │  │   │  │  - Node embeddings     │ │
│  │  - UCB computation     │  │   │  │  - Feature learning    │ │
│  │  - Parameter updates   │  │   │  │  - Representation      │ │
│  └────────────────────────┘  │   │  └────────────────────────┘ │
└──────────────────────────────┘   └──────────────────────────────┘
                │                                  │
                └────────────────┬─────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Feature Extraction Layer                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ URL Features │  │Content Feats │  │ Anchor Text Features │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Web Interaction Layer                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │  HTTP Client │  │ HTML Parser  │  │  URL Normalizer      │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Component Specifications

### 2.1 Web Graph Manager

**Purpose**: Maintain and query the evolving web graph structure

**Responsibilities**:
- Incrementally build graph as pages are crawled
- Store node features and edge attributes
- Provide query interface for GNN
- Persist graph to disk for resumption
- Bootstrap initial graph structure
- Filter candidate links to manageable size

**Graph Bootstrapping Strategy** (Cold-Start Solution):

```python
def bootstrap_graph(self, seed_urls, max_pages=500):
    """
    Build initial graph with simple BFS crawl
    Budget-friendly: ~500 pages in 5-10 minutes
    """
    queue = deque(seed_urls)
    visited = set()
    pages_crawled = 0
    
    while queue and pages_crawled < max_pages:
        url = queue.popleft()
        if url in visited:
            continue
        
        # Fetch and add to graph
        content = self._fetch(url)
        self.add_page(url, content)
        visited.add(url)
        pages_crawled += 1
        
        # Extract links (breadth-first)
        links = self._extract_links(content)
        queue.extend(links[:10])  # Limit breadth
    
    # Initialize all nodes with zero embeddings initially
    # GNN will learn meaningful embeddings during training
    for node in self.G.nodes():
        if node not in self.node_features:
            self.node_features[node] = {'embedding': np.zeros(64)}
```

**Link Candidate Filtering** (Computational Efficiency):

```python
def filter_candidate_links(self, links, current_url, max_candidates=50):
    """
    Reduce candidates to manageable size
    Critical for LinUCB computational efficiency
    """
    filtered = []
    current_domain = extract_domain(current_url)
    
    # Priority 1: Same domain (more likely relevant)
    same_domain = [l for l in links if extract_domain(l) == current_domain]
    filtered.extend(same_domain[:max_candidates//2])
    
    # Priority 2: Known domains in graph
    known_domains = [l for l in links if l in self.url_to_node]
    filtered.extend(known_domains[:max_candidates//4])
    
    # Priority 3: New domains (exploration)
    new_domains = [l for l in links if l not in filtered]
    filtered.extend(new_domains[:max_candidates//4])
    
    return filtered[:max_candidates]
```

**Data Structures**:

```python
class WebGraph:
    """
    Represents the web as a directed graph
    """
    def __init__(self):
        self.G = nx.DiGraph()  # NetworkX graph
        self.node_features = {}  # node_id -> feature_dict
        self.edge_features = {}  # (src, dst) -> feature_dict
        self.url_to_node = {}    # url -> node_id
        self.node_to_url = {}    # node_id -> url
        
    # Core operations
    def add_page(self, url, html_content, features):
        """Add a new page to the graph"""
        
    def add_link(self, src_url, dst_url, anchor_text, context):
        """Add an edge between two pages"""
        
    def get_neighbors(self, url, direction='out'):
        """Get outgoing or incoming neighbors"""
        
    def get_node_features(self, url):
        """Retrieve features for a node"""
        
    def get_subgraph(self, node_list):
        """Extract subgraph for GNN processing"""
        
    def compute_graph_statistics(self):
        """Compute PageRank, degree centrality, etc."""
```

**Storage Format**: 
- Nodes: JSON with features
- Edges: CSV with source, target, attributes
- Use HDF5 for large-scale deployments

**Performance Considerations**:
- Incremental updates: O(1) node/edge addition
- Neighborhood queries: O(degree)
- Use adjacency list representation
- Cache frequently accessed subgraphs

### 2.2 GNN Encoder

**Purpose**: Learn node embeddings that capture graph structure and content

**Architecture**: GraphSAGE with 3 layers

**Input**:
- Node feature matrix: `X ∈ ℝ^{N × d}` where N = nodes, d = feature dim
- Adjacency information: Edge index tensor
- Mini-batch of target nodes

**Output**:
- Node embeddings: `Z ∈ ℝ^{N × h}` where h = embedding dim

**Layer Specifications**:

```
Layer 1: SAGEConv(input_dim=512, output_dim=256)
         + ReLU + Dropout(0.5)

Layer 2: SAGEConv(input_dim=256, output_dim=128)
         + ReLU + Dropout(0.5)

Layer 3: SAGEConv(input_dim=128, output_dim=64)
```

**Forward Pass**:

```python
def forward(self, x, edge_index):
    """
    Args:
        x: Node feature matrix [num_nodes, input_dim]
        edge_index: Graph connectivity [2, num_edges]
    
    Returns:
        Node embeddings [num_nodes, output_dim]
    """
    h = x
    for layer in self.sage_layers[:-1]:
        h = layer(h, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=0.5, training=self.training)
    
    h = self.sage_layers[-1](h, edge_index)
    return h
```

**Training Objective**:

$$
\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\sigma(z_i)) + (1-y_i) \log(1-\sigma(z_i)) \right] + \lambda \|\theta\|^2
$$

Where:
- $y_i$ = relevance label (1 = relevant, 0 = irrelevant)
- $z_i$ = node embedding
- $\sigma$ = sigmoid function
- $\lambda$ = L2 regularization coefficient (0.001)
- $\theta$ = model parameters

**Mini-batch Sampling**:
- Use neighbor sampling to scale to large graphs
- Sample k=10 neighbors per layer
- Batch size: 256 nodes

**Update Frequency** (Resource-Constrained Strategy):

```python
# STUDENT-BUDGET APPROACH:
# - Pre-train once on bootstrap data (500 pages)
# - NO online updates during crawling (too expensive)
# - Periodic batch updates if resources allow

# Phase 1: Pre-training (one-time, ~30 minutes on CPU)
gnn_model = pretrain_gnn(
    bootstrap_data,
    epochs=50,
    batch_size=64,
    device='cpu'  # No GPU needed!
)

# Phase 2: Freeze during crawling (save compute)
gnn_model.eval()
for param in gnn_model.parameters():
    param.requires_grad = False

# Phase 3: Optional fine-tuning (every 2000 pages)
if pages_crawled % 2000 == 0 and resources_available:
    fine_tune_gnn(gnn_model, new_labeled_pages, epochs=10)
```

**Rationale**: Online GNN training is computationally expensive and unnecessary. Pre-trained embeddings provide sufficient signal for initial exploration. Fine-tune only when significant new data accumulates.

**Memory Optimization**:
- Store only computed embeddings (not full graph)
- Use sparse adjacency matrices
- Batch process new nodes every 100 pages

### 2.3 Contextual Bandit (LinUCB)

**Purpose**: Select the best link from candidates using contextual information

**Algorithm**: Linear Upper Confidence Bound (LinUCB)

**State Variables**:

For each arm (link) $a$:
- $A_a \in \mathbb{R}^{d \times d}$: Design matrix (sum of outer products)
- $b_a \in \mathbb{R}^d$: Reward vector
- Initial: $A_a = I_d$, $b_a = 0$

**Context Vector** (per link):

```
context = [
    gnn_embedding (64 dim),      # From GNN encoder
    url_features (20 dim),        # Domain, depth, etc.
    content_features (50 dim),    # TF-IDF, topics
    anchor_features (30 dim),     # Anchor text embedding
    graph_features (10 dim)       # Degree, PageRank
]
Total: 174 dimensions
```

**Link Selection** (UCB):

$$
\text{UCB}(a, t) = \theta_a^T x_t + \alpha \sqrt{x_t^T A_a^{-1} x_t}
$$

Where:
- $\theta_a = A_a^{-1} b_a$: Estimated parameters
- $x_t$: Context vector at time $t$
- $\alpha$: Exploration parameter (tunable, default=1.0)
- First term: Exploitation (expected reward)
- Second term: Exploration (uncertainty bonus)

**Selection Process** (Resource-Efficient with Ridge Regularization):

```python
def select_link(self, candidate_links, contexts):
    """
    Args:
        candidate_links: List of URLs (max 50 after filtering)
        contexts: List of context vectors (one per link)
    
    Returns:
        selected_link: Chosen URL
        selected_idx: Index of chosen link
    
    Complexity: O(k * d²) where k=candidates, d=context_dim
    For k=50, d=174: ~1.5M ops (< 10ms on CPU)
    """
    ucb_scores = []
    
    for link, context in zip(candidate_links, contexts):
        # Initialize if new arm
        if link not in self.arms:
            # Add ridge regularization for stability
            self.arms[link] = {
                'A': 0.1 * np.identity(self.context_dim),  # Ridge: λI
                'b': np.zeros(self.context_dim),
                'count': 0
            }
        
        A = self.arms[link]['A']
        b = self.arms[link]['b']
        
        # Compute θ with ridge-regularized inverse
        # Use solve instead of inv for numerical stability and speed
        theta = np.linalg.solve(A, b)
        
        # Compute UCB
        exploitation = context @ theta
        
        # Simplified uncertainty (avoid full matrix solve)
        # Approximation for speed: ||context|| / sqrt(count + 1)
        if self.arms[link]['count'] > 0:
            uncertainty = np.linalg.norm(context) / np.sqrt(self.arms[link]['count'])
        else:
            uncertainty = np.linalg.norm(context)  # High uncertainty for new arms
        
        exploration = self.alpha * uncertainty
        ucb = exploitation + exploration
        
        ucb_scores.append(ucb)
    
    # Select link with highest UCB
    best_idx = np.argmax(ucb_scores)
    return candidate_links[best_idx], best_idx
```

**Parameter Update**:

After observing reward $r$ for link $a$ with context $x$:

$$
A_a \leftarrow A_a + x x^T
$$

$$
b_a \leftarrow b_a + r x
$$

```python
def update(self, link, context, reward):
    """Update arm parameters"""
    A = self.arms[link]['A']
    b = self.arms[link]['b']
    
    # Update
    A += np.outer(context, context)
    b += reward * context
    
    self.arms[link]['A'] = A
    self.arms[link]['b'] = b
    self.arms[link]['count'] += 1
```

**Hyperparameters**:
- $\alpha$ (exploration): Start with 1.0, decay to 0.1 over training
- Context dimension: 174
- Ridge regularization in A: 0.1 * I (for numerical stability)

### 2.4 Q-Learning Agent

**Purpose**: High-level navigation strategy and long-term optimization

**State Space** ($s \in \mathcal{S}$):

```python
state = {
    'page_embedding': np.array(64),      # From GNN
    'budget_remaining': float,           # Pages left to crawl
    'relevant_found': int,               # Relevant pages discovered
    'current_depth': int,                # Depth in crawl tree
    'domain_hash': int,                  # Current domain identifier
    'avg_reward': float,                 # Running average reward
    'exploration_rate': float,           # Current ε value
}
```

**State Representation**:
- Discretize continuous values into bins
- Use state hashing for Q-table lookup
- Alternative: Function approximation with neural network

**Action Space** ($a \in \mathcal{A}$):

```python
actions = {
    'CONTINUE': 0,      # Continue crawling from current page
    'BACKTRACK': 1,     # Return to previous page
    'SWITCH_DOMAIN': 2, # Jump to different domain
    'STOP': 3          # Terminate crawl
}
```

**Q-Value Function**:

$$
Q(s, a) \approx \mathbb{E}\left[ \sum_{t=0}^{\infty} \gamma^t r_t | s_0 = s, a_0 = a \right]
$$

**Update Rule** (Temporal Difference):

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_t + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t) \right]
$$

**Implementation** (Lightweight Function Approximation for Limited Resources):

```python
import torch
import torch.nn as nn

class QLearningAgent(nn.Module):
    """Lightweight Q-network for resource-constrained environments"""
    def __init__(self, state_dim=69, action_dim=4, learning_rate=0.001, 
                 discount=0.95, epsilon=0.1):
        super().__init__()
        # Small network: only ~10K parameters (CPU-friendly)
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.gamma = discount
        self.epsilon = epsilon
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
    def _state_to_tensor(self, state):
        """Convert state dict to tensor"""
        return torch.FloatTensor([
            *state['page_embedding'],  # 64 dim
            state['budget_remaining'] / 1000.0,  # normalize
            state['relevant_found'] / 100.0,
            state['current_depth'] / 10.0,
            state['avg_reward'],
            state['exploration_rate']
        ])  # Total: 69 dimensions
    
    def get_q_values(self, state):
        """Get Q-values for all actions"""
        state_tensor = self._state_to_tensor(state)
        with torch.no_grad():
            q_values = self.network(state_tensor)
        return q_values.numpy()
    
    def get_action(self, state, valid_actions):
        """ε-greedy action selection"""
        if np.random.rand() < self.epsilon:
            return np.random.choice(valid_actions)
        
        q_values = self.get_q_values(state)
        valid_q = [q_values[a] for a in valid_actions]
        best_idx = np.argmax(valid_q)
        return valid_actions[best_idx]
    
    def update(self, state, action, reward, next_state, done):
        """Q-learning update with gradient descent"""
        state_tensor = self._state_to_tensor(state)
        next_state_tensor = self._state_to_tensor(next_state)
        
        # Current Q-value
        current_q_values = self.network(state_tensor)
        current_q = current_q_values[action]
        
        # Target Q-value
        with torch.no_grad():
            if done:
                target_q = reward
            else:
                next_q_values = self.network(next_state_tensor)
                valid_next_actions = self._get_valid_actions(next_state)
                max_next_q = max([next_q_values[a] for a in valid_next_actions])
                target_q = reward + self.gamma * max_next_q
        
        # MSE loss
        loss = (current_q - target_q) ** 2
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)  # Gradient clipping
        self.optimizer.step()
        
        # Decay epsilon
        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
```

**Hyperparameters**:
- Learning rate $\alpha$: 0.1 (can use adaptive)
- Discount factor $\gamma$: 0.95
- Exploration rate $\epsilon$: 0.1 → 0.01 (decay)
- Decay rate: 0.995 per episode

### 2.5 Reward Function

**Purpose**: Provide learning signal for both Q-agent and Bandit

**Components**:

$$
R(s, a, s') = R_{\text{relevance}} + R_{\text{novelty}} - C_{\text{cost}} - P_{\text{depth}}
$$

**Relevance Reward**:

$$
R_{\text{relevance}} = 
\begin{cases}
+10 & \text{if page is highly relevant (score > 0.8)} \\
+5 & \text{if page is moderately relevant (0.5 < score ≤ 0.8)} \\
0 & \text{if page is marginally relevant (0.3 < score ≤ 0.5)} \\
-2 & \text{if page is irrelevant (score ≤ 0.3)}
\end{cases}
$$

**Novelty Reward**:

$$
R_{\text{novelty}} = +2 \cdot \mathbb{1}[\text{new domain}] + 1 \cdot \mathbb{1}[\text{new subdomain}]
$$

**Cost Penalty**:

$$
C_{\text{cost}} = 0.1 \cdot \frac{t_{\text{fetch}}}{t_{\text{avg}}} + 0.05 \cdot \frac{s_{\text{page}}}{s_{\text{avg}}}
$$

Where:
- $t_{\text{fetch}}$: Time to fetch page
- $t_{\text{avg}}$: Average fetch time
- $s_{\text{page}}$: Page size in KB
- $s_{\text{avg}}$: Average page size

**Depth Penalty**:

$$
P_{\text{depth}} = 0.1 \cdot \min(\text{depth}, 10)
$$

**Special Cases**:
- Early termination (budget exhausted): $R = -50$
- Finding seed domain: $R = +20$
- Duplicate page: $R = -5$

**Implementation**:

```python
class RewardFunction:
    def __init__(self, relevance_threshold=0.5):
        self.threshold = relevance_threshold
        self.avg_fetch_time = 2.0  # seconds
        self.avg_page_size = 100.0  # KB
        self.visited_domains = set()
        
    def compute_reward(self, page_info):
        """
        Args:
            page_info: {
                'relevance_score': float,
                'fetch_time': float,
                'page_size': float,
                'depth': int,
                'domain': str,
                'is_duplicate': bool
            }
        
        Returns:
            reward: float
        """
        # Handle special cases
        if page_info['is_duplicate']:
            return -5.0
        
        # Relevance reward
        score = page_info['relevance_score']
        if score > 0.8:
            r_rel = 10.0
        elif score > 0.5:
            r_rel = 5.0
        elif score > 0.3:
            r_rel = 0.0
        else:
            r_rel = -2.0
        
        # Novelty reward
        domain = page_info['domain']
        r_nov = 0.0
        if domain not in self.visited_domains:
            r_nov = 2.0
            self.visited_domains.add(domain)
        
        # Cost penalty
        c_cost = (0.1 * page_info['fetch_time'] / self.avg_fetch_time +
                  0.05 * page_info['page_size'] / self.avg_page_size)
        
        # Depth penalty
        p_depth = 0.1 * min(page_info['depth'], 10)
        
        # Total reward
        reward = r_rel + r_nov - c_cost - p_depth
        
        return reward
```

### 2.6 Feature Extractor

**Purpose**: Extract and encode features from URLs, content, and anchors

**URL Features** (20 dimensions):

```python
def extract_url_features(self, url):
    """
    Returns:
        features: {
            'domain_authority': float (0-1),     # Pre-computed or API
            'domain_length': int,                # Character count
            'path_depth': int,                   # Number of / in path
            'has_www': bool,                     # www subdomain
            'has_https': bool,                   # HTTPS protocol
            'query_params': int,                 # Number of params
            'extension': str (one-hot),          # .html, .pdf, etc.
            'subdomain_type': str (one-hot),     # www, blog, wiki, etc.
            'tld': str (one-hot),                # .com, .edu, .org, etc.
        }
    """
```

**Content Features** (50 dimensions):

```python
def extract_content_features(self, html_content):
    """
    Returns:
        features: {
            'tfidf_vector': np.array(30),        # Top-30 TF-IDF terms
            'topic_distribution': np.array(10),  # LDA topics
            'word_count': int,                   # Total words
            'unique_words': int,                 # Vocabulary size
            'avg_sentence_length': float,        # Words per sentence
            'has_images': bool,                  # Contains images
            'has_videos': bool,                  # Contains videos
            'language': str (one-hot),           # Detected language
            'readability_score': float,          # Flesch score
        }
    """
```

**Anchor Text Features** (30 dimensions):

```python
def extract_anchor_features(self, anchor_text, context):
    """
    Returns:
        features: {
            'anchor_embedding': np.array(20),    # Word2Vec average
            'anchor_length': int,                # Character count
            'position_on_page': float (0-1),     # Relative position
            'is_navigation': bool,               # Nav menu link
            'surrounding_text': np.array(8),     # Context embedding
            'link_density': float,               # Links in vicinity
        }
    """
```

**Graph Features** (10 dimensions):

```python
def extract_graph_features(self, node_id, graph):
    """
    Returns:
        features: {
            'in_degree': int,                    # Incoming links
            'out_degree': int,                   # Outgoing links
            'pagerank': float,                   # PageRank score
            'clustering_coef': float,            # Local clustering
            'betweenness': float,                # Betweenness centrality
            'closeness': float,                  # Closeness centrality
            'community_id': int,                 # Louvain community
            'hub_score': float,                  # HITS hub
            'authority_score': float,            # HITS authority
        }
    """
```

**Feature Normalization**:
- Standardization (zero mean, unit variance) for continuous features
- One-hot encoding for categorical features
- Feature scaling to [0, 1] range for better NN training

### 2.7 Crawl Coordinator

**Purpose**: Orchestrate the crawling process and manage resources

**Responsibilities**:
1. Initialize episodes with seed URLs
2. Manage crawl budget (page limit, time limit)
3. Coordinate interaction between components
4. Collect and log metrics
5. Handle errors and exceptions

**Episode Workflow**:

```python
class CrawlCoordinator:
    def __init__(self, config):
        self.config = config
        self.q_agent = QLearningAgent()
        self.bandit = LinUCBBandit()
        self.gnn_encoder = WebGraphEncoder()
        self.web_graph = WebGraph()
        self.reward_fn = RewardFunction()
        self.feature_extractor = FeatureExtractor()
        
    def run_episode(self, seed_url, budget):
        """
        Execute one crawling episode
        
        Args:
            seed_url: Starting URL
            budget: Max pages to crawl
        
        Returns:
            episode_stats: Dict with metrics
        """
        # Initialize
        current_url = seed_url
        pages_crawled = 0
        relevant_found = 0
        total_reward = 0
        crawl_path = []
        
        # Add seed to graph
        self.web_graph.add_page(seed_url, self._fetch(seed_url))
        
        while pages_crawled < budget:
            # 1. Get current state
            state = self._get_state(current_url, pages_crawled, 
                                   relevant_found, budget)
            
            # 2. Q-agent decides high-level action
            q_action = self.q_agent.get_action(state, 
                                              self._valid_actions(state))
            
            if q_action == 'STOP':
                break
            
            if q_action == 'CONTINUE':
                # 3. Extract candidate links
                candidates = self._get_candidate_links(current_url)
                
                if not candidates:
                    break
                
                # 4. Create contexts for each candidate
                contexts = []
                for link in candidates:
                    # Get GNN embedding
                    gnn_emb = self._get_gnn_embedding(link)
                    
                    # Get other features
                    url_feats = self.feature_extractor.extract_url_features(link)
                    # ... extract other features
                    
                    # Concatenate
                    context = np.concatenate([gnn_emb, url_feats, ...])
                    contexts.append(context)
                
                # 5. Bandit selects best link
                selected_link, idx = self.bandit.select_link(candidates, contexts)
                
                # 6. Crawl selected link
                page_content = self._fetch(selected_link)
                self.web_graph.add_page(selected_link, page_content)
                
                # 7. Compute reward
                relevance_score = self._classify_relevance(page_content)
                page_info = {
                    'relevance_score': relevance_score,
                    'fetch_time': fetch_time,
                    'page_size': page_size,
                    'depth': depth,
                    'domain': extract_domain(selected_link),
                    'is_duplicate': self._is_duplicate(selected_link)
                }
                reward = self.reward_fn.compute_reward(page_info)
                
                # 8. Update bandit
                self.bandit.update(selected_link, contexts[idx], reward)
                
                # 9. Update state and Q-values
                next_state = self._get_state(selected_link, pages_crawled + 1,
                                            relevant_found + (relevance_score > 0.5),
                                            budget)
                self.q_agent.update(state, q_action, reward, next_state, False)
                
                # 10. Update counters
                pages_crawled += 1
                if relevance_score > 0.5:
                    relevant_found += 1
                total_reward += reward
                current_url = selected_link
                crawl_path.append(selected_link)
            
            # Handle other actions (BACKTRACK, SWITCH_DOMAIN)
            # ...
        
        # Episode complete
        episode_stats = {
            'pages_crawled': pages_crawled,
            'relevant_found': relevant_found,
            'total_reward': total_reward,
            'harvest_rate': relevant_found / pages_crawled if pages_crawled > 0 else 0,
            'crawl_path': crawl_path
        }
        
        return episode_stats
```

---

## 3. Data Flow & Interactions

### 3.1 Training Loop

```
1. Initialize components (GNN, Q-agent, Bandit, Graph)

2. Pre-train GNN on initial labeled data

3. For each episode (e = 1 to max_episodes):
    a. Sample seed URL from seed set
    b. Run episode (see Crawl Coordinator)
    c. Collect episode statistics
    d. Log metrics
    
    e. Every N episodes:
        - Retrain GNN with newly crawled pages
        - Checkpoint models
        - Evaluate on validation set
        - Adjust hyperparameters if needed

4. Final evaluation on test set

5. Save trained models
```

### 3.2 Inference (Deployment)

```
1. Load pre-trained models (GNN, Q-table, Bandit parameters)

2. Initialize web graph from checkpoint (if resuming)

3. For each crawl request:
    a. Receive target topic and seed URLs
    b. Set crawl budget
    c. Run episode with trained agents
    d. Return crawled pages and statistics

4. Periodically update GNN with new data (online learning)
```

---

## 4. Training Procedures

### 4.1 GNN Training

**Objective**: Learn node embeddings that predict relevance

**Data Preparation**:
- Collect 1000+ labeled pages (relevant/irrelevant)
- Build initial graph with BFS crawl
- Extract node features

**Training Procedure**:

```python
def train_gnn(gnn_model, web_graph, labeled_nodes, epochs=100):
    optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    
    for epoch in range(epochs):
        gnn_model.train()
        
        # Sample mini-batch
        batch_nodes = sample_batch(labeled_nodes, batch_size=256)
        
        # Get subgraph
        subgraph = web_graph.get_subgraph(batch_nodes)
        
        # Forward pass
        embeddings = gnn_model(subgraph.x, subgraph.edge_index)
        predictions = torch.sigmoid(embeddings @ classifier_weights)
        
        # Compute loss
        labels = torch.tensor([web_graph.get_label(n) for n in batch_nodes])
        loss = criterion(predictions, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Logging
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    return gnn_model
```

**Validation**:
- Evaluate on held-out validation set every 10 epochs
- Early stopping if validation loss doesn't improve for 20 epochs
- Save best model checkpoint

### 4.2 Hybrid Agent Training

**Objective**: Learn optimal crawling policy

**Training Algorithm**:

```python
def train_hybrid_agent(config):
    coordinator = CrawlCoordinator(config)
    
    num_episodes = config['num_episodes']
    seeds = load_seed_urls(config['seed_file'])
    
    for episode in range(num_episodes):
        # Sample seed
        seed_url = np.random.choice(seeds)
        
        # Run episode
        stats = coordinator.run_episode(seed_url, config['budget'])
        
        # Log metrics
        log_episode_stats(episode, stats)
        
        # Periodic GNN retraining
        if episode % 100 == 0 and episode > 0:
            print(f"Retraining GNN at episode {episode}")
            coordinator.retrain_gnn()
        
        # Checkpointing
        if episode % 500 == 0 and episode > 0:
            save_checkpoint(coordinator, episode)
        
        # Evaluation
        if episode % 100 == 0:
            eval_stats = evaluate(coordinator, test_seeds)
            print(f"Episode {episode}: Harvest Rate = {eval_stats['harvest_rate']:.3f}")
```

**Convergence Criteria**:
- Average reward stabilizes (variance < threshold)
- Harvest rate plateaus on validation set
- Q-value updates become small (< 0.01)

### 4.3 Hyperparameter Tuning

**Search Space**:
```python
hyperparameters = {
    'q_learning': {
        'alpha': [0.05, 0.1, 0.2],
        'gamma': [0.9, 0.95, 0.99],
        'epsilon_start': [0.3, 0.5, 0.7],
        'epsilon_decay': [0.99, 0.995, 0.999]
    },
    'bandit': {
        'alpha': [0.5, 1.0, 2.0],
        'context_dim': [100, 174, 200]
    },
    'gnn': {
        'hidden_dim': [128, 256, 512],
        'num_layers': [2, 3, 4],
        'dropout': [0.3, 0.5, 0.7],
        'lr': [0.0001, 0.001, 0.01]
    }
}
```

**Tuning Strategy**:
1. Grid search for GNN (pre-training)
2. Random search for Q-learning + Bandit
3. Bayesian optimization for fine-tuning
4. Use validation set for selection
5. Evaluate best config on test set

---

## 5. Evaluation Protocol

### 5.1 Metrics

**Primary Metrics**:

1. **Harvest Rate**: 
   $$\text{HR} = \frac{\text{Relevant Pages Found}}{\text{Total Pages Crawled}}$$

2. **Target Recall**:
   $$\text{Recall} = \frac{\text{Relevant Pages Found}}{\text{Total Relevant Pages Available}}$$

3. **Crawl Efficiency**:
   $$\text{Efficiency} = \frac{\text{Relevant Pages Found}}{\text{Crawl Time (seconds)}}$$

**Secondary Metrics**:

4. **Precision@K**: Precision in first K pages
5. **F1-Score**: Harmonic mean of precision and recall
6. **Average Reward per Episode**
7. **Convergence Speed**: Episodes to reach 90% optimal
8. **Graph Coverage**: % of relevant subgraph discovered

### 5.2 Baseline Comparisons

**Baselines**:

1. **Random**: Random link selection
2. **Best-First**: Greedy classifier-based selection
3. **PageRank**: Follow highest PageRank links
4. **InfoSpider** (classic RL crawler)
5. **Pure Q-Learning**: Q-learning without bandits
6. **Pure LinUCB**: Bandit without Q-learning
7. **No-GNN**: Hybrid without graph features

**Statistical Testing**:
- Paired t-test for significance (p < 0.05)
- Effect size (Cohen's d)
- Confidence intervals (95%)

### 5.3 Experiment Setup

**Datasets**:
- 3 target domains (e.g., ML, Climate, Blockchain)
- 5 seed URLs per domain
- Budget: 1000 pages per run
- 10 runs per configuration
- Report mean ± std

**Reproducibility**:
- Fixed random seeds
- Version all dependencies
- Document hardware specs
- Share code and data

---

## 6. Computational Complexity Analysis

### 6.1 Time Complexity

**Per Crawl Step**:

| Component | Operation | Complexity | Typical Cost |
|-----------|-----------|------------|-------------|
| Link Extraction | Parse HTML | O(n) | ~50ms |
| Link Filtering | Filter candidates | O(m) | ~5ms |
| Feature Extraction | URL+Content+Anchor | O(k·f) | ~20ms |
| GNN Forward Pass | Embedding lookup | O(1)* | ~2ms |
| Bandit Selection | UCB computation | O(k·d²) | ~10ms |
| Q-Network Forward | Neural net | O(h²) | ~1ms |
| **Total per step** | | | **~90ms** |

*Amortized O(1) if embeddings pre-computed

Where:
- n = HTML size (~100KB)
- m = raw links per page (~100)
- k = filtered candidates (50)
- f = feature dimension (174)
- d = context dimension (174)
- h = hidden layer size (64)

**Full Episode (1000 pages)**:
- Time: 1000 × 90ms = 90 seconds (pure computation)
- With network I/O (2s/page): ~35 minutes
- **Budget-friendly**: ~$0 (no cloud costs)

### 6.2 Space Complexity

**Graph Storage**:
- Nodes: O(|V| · d) where d=feature_dim ≈ 500 nodes × 174 dim × 4 bytes = **340 KB**
- Edges: O(|E|) ≈ 5000 edges × 16 bytes = **80 KB**
- Embeddings: O(|V| · h) ≈ 500 × 64 × 4 = **128 KB**
- **Total graph**: ~550 KB (negligible!)

**Model Parameters**:
- GNN: 3 layers × (512→256→128→64) ≈ **200K parameters** → 800 KB
- Q-Network: (69→64→32→4) ≈ **10K parameters** → 40 KB
- Bandit: k arms × d² matrix ≈ 100 × 174² × 4 = **12 MB** (largest!)
- **Total models**: ~13 MB (easily fits in RAM)

**Peak Memory Usage**: < 100 MB (student laptop friendly!)

### 6.3 Training Time Estimates

**GNN Pre-training** (one-time):
- 500 pages, 50 epochs, batch_size=64
- CPU: ~30 minutes
- Cost: **$0** (local machine)

**Hybrid Agent Training** (1000 episodes):
- 1000 pages/episode, 1000 episodes
- ~35 min/episode × 1000 = **25 days** (ouch!)
- **Optimization**: Reduce to 500 pages/episode, 500 episodes = **6 days**
- Can run overnight for 1 week
- Cost: **$0** (local machine)

**Budget Alternative**: Use smaller budget (200 pages/episode)
- 200 × 1000 episodes = 200K pages
- ~7 min/episode × 1000 = **5 days**
- More practical for students!

### 6.4 Resource Optimization Tips

```python
# TIP 1: Cache embeddings
embedding_cache = {}

# TIP 2: Batch process features
features = batch_extract_features(links, batch_size=50)

# TIP 3: Use sparse matrices
from scipy.sparse import csr_matrix
A_sparse = csr_matrix(adjacency_matrix)

# TIP 4: Limit bandit arms (forget old arms)
if len(self.arms) > 1000:
    # Keep only top 1000 most-visited arms
    self.arms = dict(sorted(self.arms.items(), 
                           key=lambda x: x[1]['count'], 
                           reverse=True)[:1000])

# TIP 5: Use float32 instead of float64
torch.set_default_dtype(torch.float32)
```

---

## 7. System Requirements

### 7.1 Hardware

**Minimum (Student Budget - Works Fine!)**:
- CPU: 4 cores, 2.0 GHz (any modern laptop)
- RAM: 8 GB (sufficient for our optimizations)
- Storage: 20 GB free space
- GPU: **NOT REQUIRED** (CPU-only works great!)
- **Cost**: $0 (use existing laptop)

**Recommended (If Available)**:
- CPU: 6+ cores, 2.5+ GHz
- RAM: 16 GB
- Storage: 50 GB SSD
- GPU: Optional (GTX 1060+ or Google Colab free tier)
- **Cost**: $0 (still no cloud costs)

**Cloud Alternative (If Laptop Struggles)**:
- Google Colab Free: 12GB RAM, 100GB storage
- Kaggle Notebooks: 16GB RAM, 20GB storage  
- **Cost**: $0/month

### 7.2 Software (All Free & Open Source)

```bash
# Lightweight installation (~2GB)
pip install torch --index-url https://download.pytorch.org/whl/cpu  # CPU-only
pip install torch-geometric
pip install networkx numpy pandas scikit-learn
pip install beautifulsoup4 requests lxml
pip install matplotlib seaborn  # Visualization

# Optional (can skip for minimal setup)
pip install scrapy  # Or just use requests + BeautifulSoup
```

**Total Installation Size**: ~2 GB (vs. 10GB+ with CUDA)
**Installation Time**: ~10 minutes

### 6.3 Scalability Considerations

**Graph Size**:
- Up to 100K nodes: In-memory graph
- 100K - 1M nodes: Disk-backed graph with caching
- 1M+ nodes: Distributed graph database (Neo4j, JanusGraph)

**GNN Training**:
- Use neighbor sampling for large graphs
- Mini-batch training with DataLoader
- Gradient checkpointing for memory efficiency

**Crawling**:
- Parallel crawling with multiple workers
- Asynchronous HTTP requests
- Rate limiting and politeness delays

---

## 7. Deployment Architecture

### 7.1 Production System

```
┌─────────────────────────────────────────────┐
│            API Gateway (Flask/FastAPI)       │
│  - Crawl request handling                    │
│  - Authentication                            │
│  - Rate limiting                             │
└─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│            Task Queue (Celery + Redis)       │
│  - Asynchronous task processing              │
│  - Job scheduling                            │
└─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│         Crawler Workers (Multiple)           │
│  - Load trained models                       │
│  - Execute crawl episodes                    │
│  - Write results to storage                  │
└─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│         Storage Layer                        │
│  - PostgreSQL: Metadata                      │
│  - MongoDB: Crawled pages                    │
│  - S3/MinIO: Large files                     │
└─────────────────────────────────────────────┘
```

### 7.2 Monitoring & Logging

- **Metrics**: Prometheus + Grafana
- **Logging**: ELK Stack (Elasticsearch, Logstash, Kibana)
- **Tracing**: Jaeger for distributed tracing
- **Alerts**: PagerDuty or similar

---

## 8. Future Enhancements

### 8.1 Short-term (3-6 months)

1. **Multi-task GNN**: Predict relevance, crawl cost, link quality simultaneously
2. **Attention Mechanisms**: Add attention to GNN for interpretability
3. **Transfer Learning**: Pre-train on large corpus, fine-tune per domain
4. **A/B Testing Framework**: Compare policies in production

### 8.2 Long-term (6-12 months)

1. **Meta-Learning (MAML)**: Fast adaptation to new domains
2. **Multi-Agent Crawling**: Cooperative/competitive agents
3. **Adversarial Training**: Robustness against SEO spam
4. **Hierarchical RL**: Multi-level decision making
5. **Continual Learning**: Online adaptation without catastrophic forgetting

---

## 9. Failure Mode Analysis

### 9.1 Component Failures

| Failure Mode | Symptom | Detection | Recovery Strategy |
|--------------|---------|-----------|-------------------|
| **GNN Collapse** | All embeddings → 0 | Check embedding variance | Use hand-crafted features as fallback |
| **Bandit Over-Exploration** | Low harvest rate, random behavior | HR < 20% after 100 episodes | Reduce α, increase exploitation |
| **Q-Learning Local Optimum** | Stuck visiting same pages | Reward plateaus, low diversity | Inject random exploration episodes |
| **Network Failures** | HTTP timeouts, 404s | Exception handling | Retry with exponential backoff |
| **Memory Overflow** | OOM errors | Monitor RAM usage | Reduce batch size, clear caches |
| **Reward Hacking** | High reward but low relevance | Manual inspection | Adjust reward function |

### 9.2 Monitoring & Detection

```python
class HealthMonitor:
    def check_gnn_health(self, embeddings):
        """Detect GNN collapse"""
        variance = np.var(embeddings)
        if variance < 0.01:
            logger.warning("GNN collapse detected!")
            return 'DEGRADED'
        return 'HEALTHY'
    
    def check_exploration_health(self, episode_stats):
        """Detect exploration issues"""
        unique_domains = len(set(episode_stats['domains_visited']))
        if unique_domains < 5:
            logger.warning("Insufficient exploration!")
            return 'DEGRADED'
        return 'HEALTHY'
    
    def check_learning_progress(self, recent_rewards):
        """Detect learning stagnation"""
        if len(recent_rewards) > 100:
            trend = np.polyfit(range(len(recent_rewards)), recent_rewards, 1)[0]
            if trend < 0.001:  # Flat or declining
                logger.warning("Learning stagnation detected!")
                return 'STAGNANT'
        return 'PROGRESSING'
```

### 9.3 Fallback Strategies

**If GNN Fails**:
```python
# Use simple hand-crafted features instead
def get_embedding_fallback(url):
    return np.array([
        domain_authority(url),
        url_depth(url),
        tfidf_similarity(url, target_topic),
        # ... other simple features
    ])
```

**If Training Diverges**:
- Reduce learning rates by 10x
- Reload last checkpoint
- Increase exploration temporarily

**If Resource Constraints Hit**:
- Reduce graph size (keep only top-k visited nodes)
- Lower batch sizes
- Simplify GNN (2 layers instead of 3)

## 10. Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| GNN overfitting | High | Cross-validation, dropout, early stopping |
| Bandit cold-start | Medium | Warm-start with supervised pre-training |
| Q-learning divergence | Medium | Learning rate scheduling, target networks |
| Scalability issues | High | Incremental updates, distributed systems |
| Ethical concerns (politeness) | High | Rate limiting, robots.txt compliance |
| Data quality | Medium | Input validation, outlier detection |

---

## 10. Success Criteria

**Must Have**:
- Harvest rate > 60% on test domains
- Outperform best baseline by ≥10%
- Converge within 1000 episodes
- Handle 100K+ page graphs

**Should Have**:
- Harvest rate > 70%
- Outperform best baseline by ≥20%
- Transferable across domains (>50% performance)
- Real-time inference (<1s per decision)

**Nice to Have**:
- Harvest rate > 80%
- State-of-the-art performance
- Interpretable decisions
- Production-ready deployment

---

## 11. Conclusion

This design document provides a comprehensive blueprint for implementing an adaptive Q-learning web crawler with contextual bandits and GNNs. The hybrid architecture leverages the strengths of each component:

- **GNNs** capture graph structure
- **Contextual Bandits** enable efficient exploration
- **Q-Learning** provides long-term optimization

The system is designed to be modular, scalable, and extensible, with clear interfaces between components. Follow the walkthrough document for step-by-step implementation guidance.

**Next Steps**:
1. Review this design with team/advisor
2. Set up development environment
3. Begin Phase 1 implementation
4. Iterate based on experimental results

Good luck with the implementation!
