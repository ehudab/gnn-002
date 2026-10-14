#!/usr/bin/env python3
"""
Bare Untracked Baseline Training Script for Cora Node Classification.

Limitations & Problems of this Untracked Baseline:
1. No Experiment Tracking: Hyperparameters (lr, weight_decay, hidden_dim, seed),
   metrics (train/val/test accuracy, loss curves), and runtime environment are not
   logged systematically. Comparing experiments relies on terminal scrolling or manual copy-pasting.
2. Lack of Reproducibility & Lineage: No code commit hash, dataset version/checksum,
   or random seeds are tied to output model artifacts.
3. Brittle Artifact Packaging: Model weights are saved as an ad-hoc raw state_dict (.pt file)
   without configuration schemas, preprocessing rules, class mappings, or dependency locks.
   Loading it elsewhere requires exact copy-pasted model class definitions.
4. No Governance or Quality Gates: Model weights are directly overwritten (e.g. baseline_model.pt)
   without automated regression validation against currently deployed baselines.
5. Inability to Deploy Seamlessly: Lacks standard model serving interfaces (like MLflow pyfunc),
   making containerized serving or rollback manual, error-prone, and slow.
"""

import time
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from gnn_model import GraphSAGEClassifier

def train_untracked():
    print("=== Running Bare Untracked Baseline Training ===")
    start_time = time.time()

    # Ad-hoc hardcoded configurations
    lr = 0.01
    weight_decay = 5e-4
    hidden_dim = 64
    epochs = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset directly without versioning/checksum
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
                # Raw overwrite with no metadata or versioning
                torch.save(model.state_dict(), "task_4/baseline_model.pt")

        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d}: Loss = {loss.item():.4f}, Best Val Acc = {best_val_acc:.4f}")

    # Final test evaluation
    model.load_state_dict(torch.load("task_4/baseline_model.pt"))
    model.eval()
    with torch.no_grad():
        test_logits = model(data.x, data.edge_index)
        test_preds = test_logits.argmax(dim=-1)
        test_acc = (test_preds[data.test_mask] == data.y[data.test_mask]).sum().item() / int(data.test_mask.sum())

    elapsed = time.time() - start_time
    print(f"\nTraining Complete in {elapsed:.2f}s")
    print(f"Final Test Accuracy: {test_acc:.4f}")
    print("Saved raw checkpoint to: task_4/baseline_model.pt")
    print("Warning: This artifact has no schema, no metadata, no tracking, and no rollback safety!")

if __name__ == "__main__":
    train_untracked()

