#!/usr/bin/env python3
"""
Tracked Training Script for Cora Node Classification using MLflow.
Supports:
- Logging parameters, metrics across epochs, seeds, dataset checksum, code commit
- Artifact packaging: weights, hyperparameters, preprocessing schema, and label mappings
- Automated reloading verification (checking that reloading reproduces identical predictions)
- Registration into MLflow Model Registry
"""

import os
import sys
import json
import hashlib
import time
import subprocess
import argparse
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid

try:
    import mlflow
    import mlflow.pyfunc
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

from gnn_model import GraphSAGEClassifier
from mlflow_model.wrapper import GNNWrapper

CORA_LABEL_MAPPING = {
    "0": "Theory",
    "1": "Reinforcement_Learning",
    "2": "Genetic_Algorithms",
    "3": "Neural_Networks",
    "4": "Probabilistic_Methods",
    "5": "Case_Based",
    "6": "Rule_Learning"
}

def get_git_commit() -> str:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
        return commit
    except Exception:
        return "git-unavailable-local"

def compute_dataset_hash(data) -> str:
    hasher = hashlib.sha256()
    hasher.update(data.x.cpu().numpy().tobytes()[:10000])
    hasher.update(data.edge_index.cpu().numpy().tobytes()[:10000])
    return hasher.hexdigest()[:12]

def run_experiment(
    exp_name: str,
    run_name: str,
    hidden_dim: int = 64,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    dropout: float = 0.5,
    aggr: str = "mean",
    seed: int = 42,
    epochs: int = 150,
    register_model_name: str = "CoraGraphSAGE"
) -> Dict[str, Any]:
    print(f"\n=======================================================")
    print(f"Starting Tracked Experiment: {run_name} (Seed={seed}, Hidden={hidden_dim}, LR={lr})")
    print(f"=======================================================")

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Cora dataset
    dataset = Planetoid(root="/tmp/cora_tracked", name="Cora")
    data = dataset[0].to(device)
    data_hash = compute_dataset_hash(data)
    git_commit = get_git_commit()

    model = GraphSAGEClassifier(
        in_dim=dataset.num_node_features,
        hidden_dim=hidden_dim,
        num_classes=dataset.num_classes,
        dropout=dropout,
        aggr=aggr
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if HAS_MLFLOW:
        mlflow.set_experiment(exp_name)

    artifacts_dir = f"task_4/artifacts/{run_name}"
    os.makedirs(artifacts_dir, exist_ok=True)

    weights_path = os.path.join(artifacts_dir, "weights.pt")
    config_path = os.path.join(artifacts_dir, "config.json")
    schema_path = os.path.join(artifacts_dir, "schema.json")

    config_data = {
        "in_dim": dataset.num_node_features,
        "hidden_dim": hidden_dim,
        "num_classes": dataset.num_classes,
        "dropout": dropout,
        "aggr": aggr,
        "lr": lr,
        "weight_decay": weight_decay,
        "seed": seed,
        "epochs": epochs
    }
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2)

    schema_data = {
        "feature_dim": dataset.num_node_features,
        "num_classes": dataset.num_classes,
        "label_mapping": CORA_LABEL_MAPPING,
        "dataset_name": "Cora",
        "dataset_hash": data_hash
    }
    with open(schema_path, "w") as f:
        json.dump(schema_data, f, indent=2)

    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_epoch = 0

    if HAS_MLFLOW:
        run_context = mlflow.start_run(run_name=run_name)
    else:
        class DummyContext:
            def __enter__(self): return self
            def __exit__(self, *args): pass
        run_context = DummyContext()

    with run_context:
        if HAS_MLFLOW:
            mlflow.log_params({
                "hidden_dim": hidden_dim,
                "lr": lr,
                "weight_decay": weight_decay,
                "dropout": dropout,
                "aggr": aggr,
                "seed": seed,
                "epochs": epochs,
                "git_commit": git_commit,
                "dataset_checksum": data_hash
            })

        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

            # Validation step
            model.eval()
            with torch.no_grad():
                val_out = model(data.x, data.edge_index)
                val_loss = F.cross_entropy(val_out[data.val_mask], data.y[data.val_mask]).item()
                val_preds = val_out.argmax(dim=-1)
                val_acc = (val_preds[data.val_mask] == data.y[data.val_mask]).sum().item() / int(data.val_mask.sum())

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_val_loss = val_loss
                    best_epoch = epoch
                    torch.save(model.state_dict(), weights_path)

            if HAS_MLFLOW and epoch % 10 == 0:
                mlflow.log_metrics({
                    "train_loss": loss.item(),
                    "val_loss": val_loss,
                    "val_acc": val_acc
                }, step=epoch)

        # Test Evaluation with best model
        model.load_state_dict(torch.load(weights_path))
        model.eval()
        with torch.no_grad():
            test_out = model(data.x, data.edge_index)
            test_preds = test_out.argmax(dim=-1)
            test_acc = (test_preds[data.test_mask] == data.y[data.test_mask]).sum().item() / int(data.test_mask.sum())
            test_loss = F.cross_entropy(test_out[data.test_mask], data.y[data.test_mask]).item()

        print(f"[{run_name}] Best Val Acc: {best_val_acc:.4f} (Epoch {best_epoch}) | Test Acc: {test_acc:.4f}")

        if HAS_MLFLOW:
            mlflow.log_metrics({
                "best_val_acc": best_val_acc,
                "best_val_loss": best_val_loss,
                "test_acc": test_acc,
                "test_loss": test_loss,
                "best_epoch": best_epoch
            })
            mlflow.log_artifacts(artifacts_dir, artifact_path="model_packaging")

            # Packaging as MLflow custom pyfunc
            artifacts_map = {
                "weights": weights_path,
                "config": config_path,
                "schema": schema_path
            }
            mlflow.pyfunc.log_model(
                artifact_path="pyfunc_model",
                python_model=GNNWrapper(),
                artifacts=artifacts_map
            )

        # Verification: Reload model and check predictions match identical inputs
        print("--> Verifying Model Packaging & Prediction Reproducibility...")
        wrapper = GNNWrapper()
        class MockCtx:
            def __init__(self, amap):
                self.artifacts = amap
        wrapper.load_context(MockCtx(artifacts_map))

        # Sample test node verification
        sample_node_ids = data.test_mask.nonzero(as_tuple=True)[0][:10].cpu().tolist()
        input_payload = {
            "features": data.x.cpu().numpy().tolist(),
            "edge_index": data.edge_index.cpu().numpy().tolist(),
            "target_node_ids": sample_node_ids
        }
        reloaded_res = wrapper.predict_dict(input_payload)
        reloaded_preds = [p["predicted_class"] for p in reloaded_res["predictions"]]
        original_preds = test_preds[sample_node_ids].cpu().tolist()

        assert reloaded_preds == original_preds, "FATAL: Reloaded model produced mismatched predictions!"
        print(f"✓ Reproducibility Verification Passed: {reloaded_preds} == {original_preds}")

    # Save summary metadata locally for offline access / gates
    summary_path = os.path.join(artifacts_dir, "run_summary.json")
    summary = {
        "run_name": run_name,
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "test_acc": test_acc,
        "test_loss": test_loss,
        "seed": seed,
        "hidden_dim": hidden_dim,
        "lr": lr,
        "artifacts_dir": os.path.abspath(artifacts_dir),
        "weights_path": os.path.abspath(weights_path),
        "config_path": os.path.abspath(config_path),
        "schema_path": os.path.abspath(schema_path)
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary

def run_all_three_experiments():
    """Runs minimum 3 experiments with varied hyperparameters and seeds."""
    os.makedirs("task_4/artifacts", exist_ok=True)
    exp_name = "Cora_Node_Classification_Pipeline"

    # Exp 1: Standard GraphSAGE (Baseline candidate)
    e1 = run_experiment(
        exp_name=exp_name,
        run_name="exp_1_graphsage_baseline",
        hidden_dim=64,
        lr=0.01,
        weight_decay=5e-4,
        aggr="mean",
        seed=42,
        epochs=120
    )

    # Exp 2: Higher Capacity + Regularization (Strong candidate)
    e2 = run_experiment(
        exp_name=exp_name,
        run_name="exp_2_graphsage_highcap",
        hidden_dim=128,
        lr=0.005,
        weight_decay=1e-4,
        dropout=0.4,
        aggr="mean",
        seed=101,
        epochs=140
    )

    # Exp 3: Max Aggregation / High Learning Rate (Regression test candidate)
    e3 = run_experiment(
        exp_name=exp_name,
        run_name="exp_3_graphsage_max_aggr",
        hidden_dim=32,
        lr=0.03,
        weight_decay=1e-3,
        dropout=0.6,
        aggr="max",
        seed=2024,
        epochs=100
    )

    print("\n=======================================================")
    print("ALL THREE EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print(f"Exp 1: Val Acc = {e1['best_val_acc']:.4f}, Test Acc = {e1['test_acc']:.4f}")
    print(f"Exp 2: Val Acc = {e2['best_val_acc']:.4f}, Test Acc = {e2['test_acc']:.4f}")
    print(f"Exp 3: Val Acc = {e3['best_val_acc']:.4f}, Test Acc = {e3['test_acc']:.4f}")
    print("=======================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-all", action="store_true", help="Run 3 benchmark experiments")
    args = parser.parse_args()

    if args.run_all or len(sys.argv) == 1:
        run_all_three_experiments()

