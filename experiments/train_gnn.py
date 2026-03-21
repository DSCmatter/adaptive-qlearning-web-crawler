"""Phase 3: Pre-train GNN on bootstrap graph"""

import sys
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import numpy as np
from tqdm import tqdm
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from graph.web_graph import WebGraph
from models.feature_extractor import FeatureExtractor
from models.gnn_encoder import WebGraphEncoder

def load_labels(csv_path: Path) -> dict:
    """Load labels from CSV file"""
    labels = {}
    if not csv_path.exists():
        return labels
        
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['label'].strip() != '':
                labels[row['url']] = int(row['label'])
    return labels

def load_config() -> dict:
    """Load configuration details"""
    config_path = Path(__file__).parent.parent / 'configs' / 'crawler_config.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def prepare_data(graph_file: Path, train_file: Path, val_file: Path, test_file: Path, extractor):
    """Prepare PyTorch Geometric Data format from WebGraph"""
    print("Loading graph...")
    web_graph = WebGraph.load(graph_file)
    nodes = list(web_graph.graph.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    
    # Load labels
    train_labels = load_labels(train_file)
    val_labels = load_labels(val_file)
    test_labels = load_labels(test_file)
    
    print(f"Loaded {len(train_labels)} training labels, {len(val_labels)} validation labels, and {len(test_labels)} test labels")
    
    # Create masks and label tensor
    y = torch.zeros(len(nodes), dtype=torch.float)
    train_mask = torch.zeros(len(nodes), dtype=torch.bool)
    val_mask = torch.zeros(len(nodes), dtype=torch.bool)
    test_mask = torch.zeros(len(nodes), dtype=torch.bool)
    
    print("Extracting node features...")
    x_features = []
    
    for idx, node in enumerate(tqdm(nodes, desc="Nodes")):
        # Build features for node
        feat = extractor.build_context_vector(
            url=node,
            html="",
            anchor_text="",
            gnn_embedding=np.zeros(64),
            graph=web_graph
        )
        x_features.append(feat)
        
        # Set labels and masks
        if node in train_labels:
            y[idx] = train_labels[node]
            train_mask[idx] = True
        elif node in val_labels:
            y[idx] = val_labels[node]
            val_mask[idx] = True
        elif node in test_labels:
            y[idx] = test_labels[node]
            test_mask[idx] = True

    x = torch.tensor(np.array(x_features), dtype=torch.float)
    
    print("Preparing edge indices...")
    edges = list(web_graph.graph.edges())
    # Create undirected-like or directed edge_index; SAGEConv can use directed edges
    if len(edges) > 0:
        edge_index = torch.tensor([[node_to_idx[u], node_to_idx[v]] for u, v in edges], dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        
    return x, edge_index, y, train_mask, val_mask, test_mask, web_graph

class GraphClassifier(nn.Module):
    """Wrapper to train GNN alongside a classification head"""
    def __init__(self, encoder, hidden_dim=64):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x, edge_index):
        embeddings = self.encoder(x, edge_index)
        logits = self.classifier(embeddings)
        return logits.squeeze(-1)

def train_gnn():
    print("=" * 60)
    print("Phase 3: GNN Pre-training")
    print("=" * 60)
    
    # Paths
    base_dir = Path(__file__).parent.parent
    graph_file = base_dir / 'data' / 'graphs' / 'bootstrap_graph.pkl'
    train_file = base_dir / 'data' / 'target_domains' / 'train_labeled.csv'
    val_file = base_dir / 'data' / 'target_domains' / 'val_labeled.csv'
    test_file = base_dir / 'data' / 'target_domains' / 'test_labeled.csv'
    models_dir = base_dir / 'data' / 'models'
    models_dir.mkdir(exist_ok=True, parents=True)
    
    config = load_config()
    gnn_config = config.get('gnn', {})
    
    extractor = FeatureExtractor()
    
    # Data Preparation
    x, edge_index, y, train_mask, val_mask, test_mask, web_graph = prepare_data(
        graph_file, train_file, val_file, test_file, extractor
    )
    
    print(f"\nData Summary:")
    print(f"X shape: {x.shape}")
    print(f"Edge Index shape: {edge_index.shape}")
    print(f"Train nodes: {train_mask.sum().item()}")
    print(f"Val nodes: {val_mask.sum().item()}")
    print(f"Test nodes: {test_mask.sum().item()}")
    
    # Model Setup
    encoder = WebGraphEncoder(
        input_dim=gnn_config.get('input_dim', 174),
        hidden_dim=gnn_config.get('hidden_dim', 128),
        output_dim=gnn_config.get('output_dim', 64),
        num_layers=gnn_config.get('num_layers', 2),
        dropout=gnn_config.get('dropout', 0.3)
    )
    model = GraphClassifier(encoder, hidden_dim=gnn_config.get('output_dim', 64))
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    epochs = 50
    best_val_loss = float('inf')
    best_val_acc = 0.0
    
    print("\nStarting training...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        logits = model(x, edge_index)
        loss = criterion(logits[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = criterion(logits[val_mask], y[val_mask]).item()
            predictions = (torch.sigmoid(logits) > 0.5).float()
            val_acc = (predictions[val_mask] == y[val_mask]).float().mean().item()
            train_acc = (predictions[train_mask] == y[train_mask]).float().mean().item()
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save the best encoder state
            torch.save(encoder.state_dict(), models_dir / 'gnn_encoder_best.pt')
            best_val_acc = val_acc
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:03d}/{epochs} - "
                  f"Train Loss: {loss.item():.4f}, Train Acc: {train_acc:.4f} - "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    
    # Load best weights before freezing
    encoder.load_state_dict(torch.load(models_dir / 'gnn_encoder_best.pt', weights_only=True))
    
    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        logits = model(x, edge_index)
        predictions = (torch.sigmoid(logits) > 0.5).float()
        test_acc = (predictions[test_mask] == y[test_mask]).float().mean().item()
        
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Freeze the encoder parameters
    print("\nFreezing GNN parameters for active crawling...")
    encoder.freeze()
    torch.save(encoder.state_dict(), models_dir / 'gnn_encoder_frozen.pt')
    
    print(f"Saved frozen GNN to {models_dir / 'gnn_encoder_frozen.pt'}")
    print("=" * 60)

if __name__ == "__main__":
    train_gnn()