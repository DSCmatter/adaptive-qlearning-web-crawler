# Phase 3: Core Model Development (GNN Pre-training)

**Timeline**: Week 2-3, Days 4-7  
**Status**:  Complete  
**Date Completed**: March 21, 2026  
**Results**: [data/results/PHASE_3_RESULTS.md](../../data/results/PHASE_3_RESULTS.md)

---

## Overview

Phase 3 focuses on pre-training the Graph Neural Network (GNN) using the bootstrap graph collected in Phase 2. Following the project's "student budget" constraint, this component relies on a minimal computational footprint. The strategy trains a lightweight GraphSAGE architecture **offline** purely on CPU. After achieving optimal validation accuracy, the trained GNN weights are explicitly frozen, guaranteeing 0 backpropagation overhead during live Crawling (Phases 4-5).

---

## 3.1 Data Preparation

We integrated src/models/feature_extractor.py to extract 174-dimensional node feature vectors.

### Process
- Graph data: loaded from data/graphs/bootstrap_graph.pkl
- Labels: loaded from data/target_domains/train_labeled.csv, val_labeled.csv, and test_labeled.csv.

### Extracted Graph Details:
- **Total Nodes:** 2,153
- **Edge Connectivity:** 12,399 directed edges
- **Node Feature Dimension:** 174
- **Training split:** 350 nodes
- **Validation split:** 75 nodes
- **Testing split:** 75 nodes

---

## 3.2 GNN Pre-training Configuration

- **Architecture:** WebGraphEncoder (using PyTorch Geometric SAGEConv layer)
- **Classification Head:** Wrapped with GraphClassifier (Linear projection from GNN's 64-dim output to Binary Cross Entropy targets)
- **Input Feature Dimension:** 174
- **Hidden Filters:** 128
- **Output Embedding Dimension:** 64
- **Num Layers:** 2
- **Optimizer:** Adam (LR: 0.01, Weight Decay: 5e-4)
- **Loss:** BCEWithLogitsLoss
- **Training Budget:** 50 Epochs

### Metrics Output log

Epoch 010/50 - Train Acc: 0.8771 - Val Acc: 0.9067
Epoch 030/50 - Train Acc: 0.9286 - Val Acc: 0.9333
Epoch 050/50 - Train Acc: 0.9314 - Val Acc: 0.9600

## 3.3 Final Model and Freezing Strategy

Upon completion of 50 epochs, the best validation state generated an accuracy of **96.00%**. 
Evaluating on the hold-out test_labeled dataset mapped identical validated evaluation classification scoring:
- **Test Accuracy: 96.00%**

In anticipation of intensive live execution:
- Best graph embeddings weights were explicitly frozen recursively: param.requires_grad = False
- Optimized graph parameter object saved cleanly to: data/models/gnn_encoder_frozen.pt

---

## Deliverables Summary
 Offline Pre-training pipeline (experiments/train_gnn.py) handling end-to-end dataset extractions.
 Model Testing successful with excellent Test Accuracy mapping against validation metrics.
 Safe CPU weights freezing implementations executed successfully prior to caching outputs.
 Persisted the base model instance parameter checkpoints.

---

## Lessons Learned

### What Worked Well
1. **Lightweight Network:** Capping parameter variables firmly down to ~200k reduced typical GNN training periods to sub-2-seconds on CPU natively removing necessity of GPU deployments context switches entirely. 
2. **PyTorch Geometric Loading Constraints:** The internal mask building allowed direct mapping of network-X components seamlessly onto Pytorch edge index models using boolean indexing over labels properly maintaining graph edge structure limits seamlessly.

### Limitations
1. Creating node subsets requires strict index-tracking. Handled via dictionary structures 
ode_to_idx dictionaries ensuring non-misaligned tensor structures across the NetworkX strings layout.

---

## How to Run Phase 3

### Prerequisites
- Python Environment configured + Phase 2 datasets bootstrapped

### Step-by-Step Instructions

1. **Activate virtual environment**
   Windows PowerShell:
   .\venv\Scripts\Activate.ps1

2. **Trigger Script**
   python experiments\train_gnn.py

3. **Verify Outcomes Directory Outputs Logs**
   Ensures \data\models\gnn_encoder_frozen.pt\ builds accordingly natively.

---

## Next Steps  Phase 4
Q-Learning integration alongside agent context trackers loops. 
**See**: [Phase 4 Documentation](../README.md)
