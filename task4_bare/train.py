#!/usr/bin/env python3
"""
Bare Untracked Baseline Training Script for Cora Node Classification.

This script represents the typical ad-hoc, untracked training baseline:
- No experiment tracking (no MLflow, W&B, or structured logs)
- No dataset versioning, hash checking, or lineage
- Hardcoded hyperparameters
- Overwrites raw PyTorch state dicts with no metadata, schemas, or dependencies
- No validation gates or automated promotion
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import SAGEConv


class GraphSAGEClassifier(nn.Module):
    """Simple 2-layer GraphSAGE classifier."""
    def __init__(self, in_dim=1433, hidden_dim=64, num_classes=7, dropout=0.5):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        out = self.conv2(h, edge_index)
        return out


def train_bare_baseline():
    print("=================================================================")
    print(" Running Bare Untracked Baseline Training")
    print(" (No MLflow tracking, no schema, raw artifact overwriting)")
    print("=================================================================")
    start_time = time.time()

    # Ad-hoc hardcoded hyperparameters (no config file or tracking)
    lr = 0.01
    weight_decay = 5e-4
    hidden_dim = 64
    epochs = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset directly without versioning or hash checks
    dataset = Planetoid(root="/tmp/cora_untracked", name="Cora")
    data = dataset[0].to(device)

    model = GraphSAGEClassifier(
        in_dim=dataset.num_node_features,
        hidden_dim=hidden_dim,
        num_classes=dataset.num_classes,
        dropout=0.5
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_acc = 0.0
    checkpoint_path = os.path.join(os.path.dirname(__file__), "model.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # Validation check
        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            preds = logits.argmax(dim=-1)
            val_acc = (preds[data.val_mask] == data.y[data.val_mask]).sum().item() / int(data.val_mask.sum())
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                # Raw state dict overwrite: no versioning, no metadata
                torch.save(model.state_dict(), checkpoint_path)

        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d} | Train Loss: {loss.item():.4f} | Best Val Acc: {best_val_acc:.4f}")

    # Evaluate test accuracy
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    with torch.no_grad():
        test_logits = model(data.x, data.edge_index)
        test_preds = test_logits.argmax(dim=-1)
        test_acc = (test_preds[data.test_mask] == data.y[data.test_mask]).sum().item() / int(data.test_mask.sum())

    elapsed = time.time() - start_time
    print("-----------------------------------------------------------------")
    print(f"Training completed in {elapsed:.2f}s")
    print(f"Final Test Accuracy: {test_acc:.4f}")
    print(f"Saved raw checkpoint to: {checkpoint_path}")
    print("Warning: Artifact lacks model signature, lineage, and packaging!")
    print("=================================================================")


if __name__ == "__main__":
    train_bare_baseline()

