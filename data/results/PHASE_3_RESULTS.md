# Phase 3 Results: GNN Pre-training

**Date Captured**: March 21, 2026

## 1. Graph Data Preparation Check
```
Loading graph...
Loaded 350 training labels, 75 validation labels, and 75 test labels
Extracting node features...
Nodes: 100%|████████████████████████████| 2153/2153 [00:00<00:00, 22716.23it/s]
Preparing edge indices...

Data Summary:
X shape: torch.Size([2153, 174])
Edge Index shape: torch.Size([2, 12399])
Train nodes: 350
Val nodes: 75
Test nodes: 75
```

## 2. Model Training Log
Optimizer: Adam (LR: 0.01, Weight Decay: 5e-4)
Loss: Binary Cross Entropy with Logits
Total Parameters: ~200K (CPU Optimized)

```
Starting training...
Epoch 010/50 - Train Loss: 0.2244, Train Acc: 0.8771 - Val Loss: 0.1962, Val Acc: 0.9067
Epoch 020/50 - Train Loss: 0.1299, Train Acc: 0.9286 - Val Loss: 0.1297, Val Acc: 0.9200
Epoch 030/50 - Train Loss: 0.1287, Train Acc: 0.9286 - Val Loss: 0.1029, Val Acc: 0.9333
Epoch 040/50 - Train Loss: 0.1265, Train Acc: 0.9429 - Val Loss: 0.1089, Val Acc: 0.9467
Epoch 050/50 - Train Loss: 0.1175, Train Acc: 0.9314 - Val Loss: 0.0888, Val Acc: 0.9600

============================================================
Training Complete!
Best Validation Accuracy: 0.9600
Test Accuracy: 0.9600
```

## 3. Persistent Storage
- **Weights Frozen**: Yes. Handled via standard recursive `.requires_grad_(False)` implementation on GNN instances.
- **Save Path**: `data/models/gnn_encoder_frozen.pt` (Lightweight model parameter dump ready for active contextual processing inference in phase 4).