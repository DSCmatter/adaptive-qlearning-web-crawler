# Phase 5: Hybrid Crawler Integration

**Timeline**: Week 6
**Status**: ✅ Complete
**Date Completed**: March 23, 2026

---

## Overview

Phase 5 represents the culmination of all previous modules. We implemented the `AdaptiveCrawler` which unites the high-level Q-learning stopping mechanism (Phase 4), the specific link exploitation of the Contextual Bandit (Phase 4), and the rich structural encodings from our CPU-frozen GraphSAGE model (Phase 3). 

This assembled "Hybrid" crawler interacts directly with live web target vectors rather than simulated nodes, constructing the `WebGraph` iteratively as it moves and generating active `FeatureExtractor` context states for real-world candidate links on the wire.

---

## 5.1 Architecture Integration

The `AdaptiveCrawler` acts as an orchestrator overriding standard search patterns:

1. **State Extraction**: At each depth `d`, it parses the `WebGraph` extracting the current node state via the **Frozen GNN** combining it seamlessly with 5 general session metrics (Remaining Budget, Relevant Targets found, Current Depth, Average Session Reward, Exploration Rate) -> 69-Dim state.
2. **High-level Q-Decision**: Computes the Bellman mappings via `QLearningAgent` to ascertain whether current vectors merit further exploration (`CONTINUE`) or represent an exhaustive loop (`STOP`).
3. **Candidate Pooling**: Limits computational horizons by extracting 50 unvisited candidate URLs extracted organically via HTTP Requests.
4. **Bandit Selection**: Bypasses typical Queue iterations instead feeding the 50 URL contexts directly into the `LinUCBBandit` extracting mathematically the max upper-confidence bound linkage mapping to topic relevance. 

## 5.2 Live Component Validation

Validation executions mapped successfully against live endpoints (`ml_seeds.json`) confirming:
- Live page HTML generation.
- Dynamic feature mappings across 174 contextual dimensions.
- End-to-end mapping updating weights safely within RAM configurations bounding max limits smoothly.

## 5.3 Crawl Metrics 

A restricted 20-page run initiated against a standard Wikipedia ML seed successfully traversed topical structures mapped efficiently by the RL algorithms avoiding extraneous sink-holes:

```text
Starting adaptive crawl from https://en.wikipedia.org/wiki/Machine_learning
Fetching: https://en.wikipedia.org/wiki/Machine_learning...
... (20 Pages extracted natively) ...

============================================================
PHASE 5 Live Run Results
============================================================
Total Pages Crawled: 20
Relevant Targets:    20
Harvest Rate:        1.000
Avg Session Reward:  216.75
```

---

## Deliverables Summary

✅ `src/crawler/adaptive_crawler.py`: Engineered to dynamically track real target requests leveraging previously trained RL/Graph parameters gracefully.
✅ `experiments/run_hybrid_crawler.py`: Verification logic invoking models synchronously handling active HTTP execution natively.

---

## How to Run Phase 5

Ensure previous stages (Phase 3 Frozen GNN / Phase 4 RL Weights) are fully cached inside `data/models`.

```bash
# Verify venv triggers actively
.\venv\Scripts\Activate.ps1

# Initiate the live Hybrid run 
python experiments\run_hybrid_crawler.py
```