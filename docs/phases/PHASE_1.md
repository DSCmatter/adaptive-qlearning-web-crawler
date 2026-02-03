# Phase 1: Project Setup & Environment Configuration

**Timeline**: Week 1, Days 1-2  
**Status**: ✅ Complete  
**Date Completed**: February 2, 2026  
**Results**: [data/results/PHASE_1_RESULTS.md](../../data/results/PHASE_1_RESULTS.md)

---

## Overview

Phase 1 established the foundational development environment and project structure for the Adaptive Q-Learning Web Crawler. This phase focused on setting up dependencies, creating the codebase skeleton, and validating the setup works correctly.

---

## 1.1 Development Environment Setup

### Python Environment

**Version**: Python 3.14.2  
**Environment Manager**: venv (virtual environment)

#### Installation Steps

```bash
# Created virtual environment
python -m venv venv

# Activated environment (Windows)
venv\Scripts\activate

# Installed all dependencies
pip install -r requirements.txt
```

### Dependencies Installed

**Core Libraries**:
- `torch` (CPU-only) - PyTorch for neural networks
- `torch-geometric` - Graph neural network library
- `numpy`, `pandas` - Data manipulation
- `scipy`, `scikit-learn` - Scientific computing and ML utilities

**Web Crawling**:
- `requests` - HTTP requests
- `beautifulsoup4` - HTML parsing
- `lxml` - XML/HTML parser
- `networkx` - Graph data structure

**Visualization & Testing**:
- `matplotlib`, `seaborn` - Data visualization
- `pytest` - Unit testing
- `pyyaml` - Configuration files
- `tqdm` - Progress bars

### Budget Optimization

- **CPU-only PyTorch**: `--index-url https://download.pytorch.org/whl/cpu`
- **Total installation size**: ~2GB (vs 10GB+ with GPU support)
- **No cloud costs**: Runs on any laptop from last 5 years
- **Memory footprint**: <500MB RAM

---

## 1.2 Project Structure

Created complete directory structure following WALKTHROUGH.md specification:

```
adaptive-qlearning-web-crawler/
├── src/                          # Source code
│   ├── crawler/                  # Web crawling logic
│   │   ├── base_crawler.py      # Foundation crawler with politeness
│   │   └── adaptive_crawler.py  # Hybrid RL crawler (skeleton)
│   ├── models/                   # ML models
│   │   ├── gnn_encoder.py       # GraphSAGE network
│   │   ├── contextual_bandit.py # LinUCB implementation
│   │   ├── qlearning_agent.py   # Q-network
│   │   └── feature_extractor.py # 174-dim context vectors
│   ├── environment/              # RL environment
│   │   ├── crawl_environment.py # Gym-like interface
│   │   └── reward_function.py   # Reward computation
│   ├── graph/                    # Graph management
│   │   ├── web_graph.py         # NetworkX-based graph
│   │   └── graph_builder.py     # Bootstrap function
│   └── utils/                    # Utilities
│       ├── url_utils.py         # URL normalization
│       ├── text_processing.py   # Text utilities
│       └── metrics.py           # Evaluation metrics
├── experiments/                  # Training scripts
│   ├── train_agent.py           # Main training loop (skeleton)
│   ├── evaluate_baseline.py     # Baseline comparison
│   └── compare_methods.py       # Method comparison
├── tests/                        # Unit tests
│   └── test_crawler.py          # Crawler tests
├── data/                         # Data storage
│   ├── seeds/                   # Seed URL collections
│   ├── target_domains/          # Labeled data
│   ├── graphs/                  # Saved graph structures
│   └── results/                 # Experiment results
├── configs/                      # Configuration files
│   └── crawler_config.yaml      # Hyperparameters
├── notebooks/                    # Jupyter notebooks
└── docs/                         # Documentation
    └── phases/                  # Phase documentation
```

---

## 1.3 Core Component Implementations

### Base Crawler (`src/crawler/base_crawler.py`)

**Features**:
- Session management for connection pooling
- User-Agent header for politeness
- 1-second delay between requests
- URL validation and normalization
- Link extraction from HTML

**Key Methods**:
```python
fetch_page(url)      # Fetch HTML with error handling
extract_links(html)  # Parse links from HTML
is_valid_url(url)    # Filter invalid URLs
```

### Web Graph (`src/graph/web_graph.py`)

**Features**:
- NetworkX DiGraph for directed links
- Node feature storage
- Label storage for training
- PageRank computation
- Save/load functionality

**Key Methods**:
```python
add_page(url, html, features, label)  # Add node
add_link(from_url, to_url, anchor)    # Add edge
get_neighbors(url)                     # Get outlinks
get_pagerank(url)                      # Compute centrality
save(filepath) / load(filepath)        # Persistence
```

### GNN Encoder (`src/models/gnn_encoder.py`)

**Architecture**: GraphSAGE-based encoder
- Input dimension: 174 (feature vector)
- Hidden dimension: 128
- Output dimension: 64 (node embeddings)
- Layers: 2 SAGEConv layers
- Parameters: ~200K (CPU-friendly)

**Status**: Skeleton implemented, training pending Phase 3

### Contextual Bandit (`src/models/contextual_bandit.py`)

**Algorithm**: LinUCB (Linear Upper Confidence Bound)
- Context dimension: 174
- Confidence parameter: α = 1.0
- Ridge regularization: λ = 1.0

**Key Methods**:
```python
select_link(context, candidates)  # UCB selection
update(context, reward)           # Update arm parameters
```

### Q-Learning Agent (`src/models/qlearning_agent.py`)

**Architecture**: Feedforward network
- State dimension: 69
- Hidden layers: 64 → 32
- Actions: 2 (explore vs exploit)
- Parameters: ~10K (very lightweight)

**Hyperparameters**:
- Learning rate: 0.001
- Discount factor (γ): 0.95
- Epsilon decay: 0.995
- Min epsilon: 0.01

---

## 1.4 Configuration

### Hyperparameters (`configs/crawler_config.yaml`)

```yaml
crawler:
  max_pages_per_episode: 200
  max_depth: 10
  politeness_delay: 1.0

gnn:
  input_dim: 174
  hidden_dim: 128
  output_dim: 64
  num_layers: 2

bandit:
  alpha: 1.0
  lambda_reg: 1.0

qlearning:
  learning_rate: 0.001
  gamma: 0.95
  epsilon_start: 1.0
  epsilon_decay: 0.995
  epsilon_min: 0.01
```

---

## 1.5 Testing & Validation

### Test Crawler (`src/test_crawler.py`)

**Purpose**: Baseline minimal smart crawler for comparison

**Algorithm**: UCB bandit with keyword matching
- Relevance: 2+ keyword matches
- Policy: ε-greedy with UCB scoring
- Features: Keyword count, visited status

**Results**:
- Pages crawled: 34
- Relevant pages: 13
- **Harvest rate: 38%** ← Baseline to beat

### Skeleton Validation (`test_skeleton.py`)

**Tests**:
- ✅ All imports work
- ✅ BaseCrawler fetches pages
- ✅ WebGraph adds nodes/edges
- ✅ RewardFunction computes rewards
- ✅ LinUCB selects links
- ✅ Q-agent predicts actions

---

## 1.6 Documentation

### Files Created

1. **README.md**: Project overview, quick start, architecture
2. **requirements.txt**: All dependencies (no version pins)
3. **setup.py**: Package configuration
4. **This document**: Phase 1 summary

---

## Deliverables Summary

✅ **Environment**: Python 3.14.2 + venv + all dependencies  
✅ **Structure**: Complete project skeleton (20+ files)  
✅ **Core Components**: BaseCrawler, WebGraph, GNN, Bandit, Q-agent  
✅ **Configuration**: YAML-based hyperparameters  
✅ **Testing**: Test crawler (38% harvest rate baseline)  
✅ **Documentation**: README + setup instructions  

---

## Lessons Learned

### What Worked Well

1. **CPU-only approach**: No GPU needed, runs on any laptop
2. **Skeleton-first**: Created structure before complex implementations
3. **Test-driven**: Validated each component works independently
4. **Configuration files**: Easy to change hyperparameters without code edits

### Challenges

1. **Import issues**: Needed PYTHONPATH configuration for tests
2. **Virtual environment**: PowerShell activation required specific command
3. **Dependency size**: 2GB still significant for slow connections

### Design Decisions

1. **NetworkX vs custom graph**: Chose NetworkX for simplicity (good for 60-500 nodes)
2. **Feature dimensions**: 174 dims = 64 GNN + 110 handcrafted (not too large for CPU)
3. **Lightweight models**: ~200K params total (train in minutes, not hours)

---

## How to Run Phase 1

### Prerequisites
- Python 3.10+ installed
- Windows/Linux/macOS
- Internet connection for package downloads

### Step-by-Step Instructions

```bash
# 1. Navigate to project directory
cd \adaptive-qlearning-web-crawler

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Windows CMD:
venv\Scripts\activate.bat
# Linux/macOS:
source venv/bin/activate

# 4. Install all dependencies (~2GB, 5-10 minutes)
pip install -r requirements.txt

# 5. Verify installation
python -c "import torch; import torch_geometric; print('Success!')"

# 6. Test project structure
python test_skeleton.py

# Expected output:
# ============================================================
# Testing Project Skeleton
# ============================================================
# ✓ All imports successful
# ✓ BaseCrawler can fetch pages
# ✓ WebGraph can add nodes and edges
# ...
# ✓ All 6 tests passed!

# 7. (Optional) Run baseline test crawler
python src\test_crawler.py

# Expected output:
# Final Results:
# Pages crawled: 34
# Relevant pages found: 13
# Harvest rate: 38.24%
```

### Troubleshooting

**Problem**: `ModuleNotFoundError: No module named 'torch'`
- **Solution**: Virtual environment not activated. Run activation command again.

**Problem**: `pip` not recognized
- **Solution**: Use `python -m pip install -r requirements.txt`

**Problem**: PyTorch installation fails
- **Solution**: Install CPU-only version explicitly:
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  pip install torch-geometric
  ```

**Problem**: Test skeleton fails on imports
- **Solution**: Set PYTHONPATH:
  ```bash
  # Windows PowerShell
  $env:PYTHONPATH="d:\Coding\adaptive-qlearning-web-crawler\src"
  
  # Linux/macOS
  export PYTHONPATH="/path/to/adaptive-qlearning-web-crawler/src"
  ```

### Expected Results

✅ Virtual environment created with all dependencies  
✅ All skeleton tests pass  
✅ Test crawler achieves ~38% harvest rate (baseline to beat)  
✅ Ready for Phase 2  

### Time Required
- Setup: 10-15 minutes
- Testing: 5 minutes
- Total: ~20 minutes

---

## Next Steps → Phase 2

- ✅ Collect seed URLs for multiple topics
- ✅ Bootstrap initial graph (500 pages)
- ✅ Create labeled training data
- ✅ Implement feature extraction pipeline

**See**: [Phase 2 Documentation](PHASE_2.md)
