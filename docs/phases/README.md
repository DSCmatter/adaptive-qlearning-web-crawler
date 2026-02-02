# Phase Documentation Index

This directory contains detailed documentation for each phase of the Adaptive Q-Learning Web Crawler project.

---

## Completed Phases

### ✅ [Phase 1: Project Setup & Environment Configuration](PHASE_1.md)
**Timeline**: Week 1, Days 1-2  
**Status**: Complete  

**Summary**:
- Set up Python 3.14.2 development environment
- Created complete project structure (20+ files)
- Implemented skeleton for all core components
- Validated setup with test crawler (38% harvest rate)

**Key Deliverables**:
- Virtual environment with all dependencies
- BaseCrawler, WebGraph, GNN, Bandit, Q-agent skeletons
- Configuration files and documentation

---

### ✅ [Phase 2: Data Collection & Preprocessing](PHASE_2.md)
**Timeline**: Week 1, Days 3-5  
**Status**: Complete  

**Summary**:
- Collected 21 seed URLs across 3 topics (ML, climate, blockchain)
- Bootstrapped web graph (60 nodes, 600 edges)
- Implemented 174-dim feature extraction pipeline
- Created labeled training data (42 train / 9 val / 9 test)

**Key Deliverables**:
- Bootstrap graph saved to `data/graphs/bootstrap_graph.pkl`
- Feature extractor with URL, content, anchor, graph features
- Labeled dataset split into train/val/test
- Scripts for bootstrapping, labeling, and testing

---

## In Progress

### 🔄 Phase 3: Core Model Development (GNN Pre-training)
**Timeline**: Week 2-3, Days 4-7  
**Status**: Starting  

**Objectives**:
- Implement GNN training loop
- Pre-train GraphSAGE on bootstrap graph
- Evaluate on validation set
- Save trained model for deployment

**Documentation**: Coming next

---

## Future Phases

### 📋 Phase 4: Q-Learning Agent Training
- Train Q-network for high-level decisions
- Integrate with contextual bandits
- Implement epsilon-greedy exploration

### 📋 Phase 5: Hybrid System Integration
- Combine GNN + Bandit + Q-learning
- Implement adaptive crawler
- End-to-end crawling tests

### 📋 Phase 6: Evaluation & Baselines
- Compare against heuristic crawlers
- Measure harvest rate, precision, coverage
- Ablation studies

### 📋 Phase 7: Optimization & Deployment
- Hyperparameter tuning
- Production-ready crawler
- Documentation and paper writing

---

## Quick Navigation

| Phase | Status | Documentation | Key Scripts |
|-------|--------|---------------|-------------|
| Phase 1 | ✅ Complete | [PHASE_1.md](PHASE_1.md) | `setup.py`, `test_skeleton.py` |
| Phase 2 | ✅ Complete | [PHASE_2.md](PHASE_2.md) | `bootstrap_graph.py`, `auto_label_urls.py`, `test_features.py` |
| Phase 3 | 🔄 In Progress | Coming next | `train_gnn.py` (TBD) |
| Phase 4 | 📋 Planned | - | `train_agent.py` |
| Phase 5 | 📋 Planned | - | `experiments/train_agent.py` |
| Phase 6 | 📋 Planned | - | `evaluate_baseline.py`, `compare_methods.py` |
| Phase 7 | 📋 Planned | - | Final deployment scripts |

---

## How to Run Each Phase

### Phase 1: Project Setup

```bash
# 1. Clone/navigate to project directory
cd \adaptive-qlearning-web-crawler

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# 4. Install dependencies
pip install -r requirements.txt

# 5. Test the setup
python test_skeleton.py

# 6. Run test crawler (optional baseline)
python src\test_crawler.py
```

**Expected Output**: All tests pass, test crawler achieves ~38% harvest rate

---

### Phase 2: Data Collection

```bash
# Ensure venv is activated
.\venv\Scripts\Activate.ps1

# 1. Bootstrap initial graph (crawl 60-500 pages)
python experiments\bootstrap_graph.py
# Output: data/graphs/bootstrap_graph.pkl

# 2. Create labeling template
python experiments\create_labeled_data.py
# Output: data/target_domains/urls_to_label.csv

# 3. Auto-label URLs (or manually label in Excel)
python experiments\auto_label_urls.py
# Output: data/target_domains/labeled_urls.csv

# 4. Split into train/val/test
python experiments\create_labeled_data.py --split
# Output: train_labeled.csv, val_labeled.csv, test_labeled.csv

# 5. Test feature extraction
python experiments\test_features.py
# Output: Validation that 174-dim features work correctly
```

**Expected Output**: 
- 60 nodes, 600 edges in graph
- 42/9/9 train/val/test split
- Feature extraction with no NaN/Inf values

---

### Phase 3: GNN Training (Coming Next)

```bash
# To be added when Phase 3 is implemented
python experiments\train_gnn.py
```

---

## How to Use This Documentation

### For Understanding the Project
1. Read [PHASE_1.md](PHASE_1.md) to understand the setup
2. Read [PHASE_2.md](PHASE_2.md) to understand data preparation
3. Follow along with future phase docs as they're created

### For Reproducing Results
Each phase document includes:
- **Scripts to run**: Exact commands with expected outputs
- **Files created**: What data/models are generated
- **Validation steps**: How to verify it worked correctly

### For Extending the Project
Each phase document includes:
- **Design decisions**: Why we made specific choices
- **Trade-offs**: Pros/cons of our approach
- **Alternatives**: Other approaches we considered
- **Lessons learned**: What worked well, what didn't

---

## Documentation Standards

Each phase document follows this structure:

1. **Overview**: High-level summary
2. **Objectives**: What we're trying to achieve
3. **Implementation**: Detailed technical content
4. **Results**: Metrics and outcomes
5. **Scripts**: How to run the code
6. **Lessons Learned**: Reflections and insights
7. **Next Steps**: What comes next

---

## Related Documentation

- [`../../README.md`](../../README.md) - Project overview and quick start
- [`../WALKTHROUGH.md`](../WALKTHROUGH.md) - Original 9.5-week project plan
- [`../DESIGN.md`](../DESIGN.md) - System architecture and design
- [`../STUDENT_BUDGET_GUIDE.md`](../STUDENT_BUDGET_GUIDE.md) - Cost optimization

---

## Contributing

When completing a new phase:
1. Create `PHASE_N.md` following the template structure
2. Include all scripts, commands, and outputs
3. Document design decisions and trade-offs
4. Add lessons learned section
5. Update this README with status and links
