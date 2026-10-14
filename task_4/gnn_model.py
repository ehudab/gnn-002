"""
GNN Architecture Definition: GraphSAGE for Node Classification on Cora.
Reused and extended for inductive node representation and classification.
"""
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class GraphSAGEClassifier(nn.Module):
    """
    2-Layer GraphSAGE model for inductive node classification.
    Allows returning either class log-probabilities or node embeddings.
    """
    def __init__(
        self,
        in_dim: int = 1433,
        hidden_dim: int = 64,
        num_classes: int = 7,
        dropout: float = 0.5,
        aggr: str = "mean"
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout = dropout
        self.aggr = aggr

        self.conv1 = SAGEConv(in_dim, hidden_dim, aggr=aggr)
        self.conv2 = SAGEConv(hidden_dim, num_classes, aggr=aggr)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_embeddings: bool = False
    ) -> torch.Tensor:
        """
        x: [N, in_dim] node features
        edge_index: [2, E] edge list
        return_embeddings: if True, returns hidden layer representations [N, hidden_dim]
        """
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h_emb = h  # Hidden embedding before dropout/final projection
        h = F.dropout(h, p=self.dropout, training=self.training)
        out = self.conv2(h, edge_index)

        if return_embeddings:
            return out, h_emb
        return out

    def get_config(self) -> Dict[str, Any]:
        return {
            "in_dim": self.in_dim,
            "hidden_dim": self.hidden_dim,
            "num_classes": self.num_classes,
            "dropout": self.dropout,
            "aggr": self.aggr
        }

