"""Graph Neural Network encoder for web graph"""

import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv


class WebGraphEncoder(nn.Module):
    """
    Lightweight GNN encoder using GraphSAGE
    Optimized for CPU execution (~200K parameters)
    """
    
    def __init__(
        self, 
        input_dim: int = 174, 
        hidden_dim: int = 128, 
        output_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # Build convolutional layers
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        self.convs.append(SAGEConv(hidden_dim, output_dim))
        
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index):
        """
        Forward pass
        
        Args:
            x: Node feature matrix [num_nodes, input_dim]
            edge_index: Edge connectivity [2, num_edges]
        
        Returns:
            Node embeddings [num_nodes, output_dim]
        """
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.activation(x)
            x = self.dropout(x)
        
        x = self.convs[-1](x, edge_index)
        return x
    
    def freeze(self):
        """Freeze model parameters (for inference only)"""
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
