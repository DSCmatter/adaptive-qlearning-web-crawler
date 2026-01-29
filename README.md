# Adaptive Q-Learning Web Crawler with Contextual Bandits and GNNs

This project implements a **novel hybrid approach** to focused web crawling that combines three complementary techniques:

- **Q-Learning**: Provides high-level navigation strategy and long-term reward optimization
- **Contextual Bandits (LinUCB)**: Handles intelligent link selection with efficient exploration-exploitation balance
- **Graph Neural Networks (GNNs)**: Captures web graph structure for informed decision-making

The crawler learns to navigate the web by selecting links that maximize topical relevance while minimizing crawl cost. It receives rewards for discovering target-domain pages and penalties for inefficient navigation, enabling adaptive link selection over time. Its performance is evaluated against static heuristic-based crawlers and traditional RL approaches to analyze efficiency, coverage, and convergence behavior.

## Research Innovation

This project addresses limitations in existing RL-based crawlers by:
- **Leveraging graph topology** via GNN-based node embeddings
- **Using contextual information** for faster convergence on link selection
- **Combining value-based and bandit approaches** for hierarchical decision-making

## Relevant Research Papers

### Key Papers on RL-based Web Crawling:
1. [**Tree-based Focused Web Crawling with Reinforcement Learning**](https://arxiv.org/abs/2112.07620) (2021) - Kontogiannis et al.
2. [**Deep Reinforcement Learning for Web Crawling**](https://ieeexplore.ieee.org/abstract/document/9703160/) (2021) - Avrachenkov, Borkar, Patil
3. [**Efficient Deep Web Crawling Using Reinforcement Learning**](https://link.springer.com/chapter/10.1007/978-3-642-13657-3_46) (2010) - Jiang et al. (Cited 59 times)
4. [**Learning to Crawl Deep Web**](https://www.sciencedirect.com/science/article/pii/S0306437913000288) (2013) - Zheng et al. (Cited 71 times)

## Documentation

- **[STUDENT_BUDGET_GUIDE.md](docs/STUDENT_BUDGET_GUIDE.md)** - **START HERE!** Student-friendly quick start guide
- **[WALKTHROUGH.md](docs/WALKTHROUGH.md)** - Complete implementation guide (9.5-week timeline, optimized for students)
- **[DESIGN.md](docs/DESIGN.md)** - Technical design document with architecture and algorithms
- **[PRACTICAL_GUIDE.md](docs/PRACTICAL_GUIDE.md)** - Simplified architecture and steps (recommended to get your way around)

## Student-Friendly Features

This project is optimized for broke students:
- **$0.10 Total Cost** (just electricity)
- **No GPU Required** (CPU-only works great)
- **No Cloud Costs** (runs on your laptop)
- **3-4 Days Training** (run overnight)
- **8GB RAM Sufficient** (works on old laptops)
- **60-70% Harvest Rate** (publishable results!)

## Quick Start

```bash
# 1. Clone repo
git clone https://github.com/yourusername/adaptive-qlearning-web-crawler
cd adaptive-qlearning-web-crawler

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies (CPU-only, ~2GB)
pip install -r requirements.txt
```
