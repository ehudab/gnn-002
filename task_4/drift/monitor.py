"""
Graph Structure & Embedding Drift Detection Module.

Monitors:
1. Graph Structure Statistic: Average Node Degree & Degree Distribution Shift.
2. Embedding Representation Statistic: Mean Embedding Vector Norm & Feature Space Centroid Shift.

Important Conceptual Rationale:
- Why raw embedding coordinates from different model versions are NOT directly comparable:
  Neural network training is non-convex and exhibits rotational, reflective, and permutation
  symmetries in the latent feature space. Two identically architected models trained with different
  random seeds or updated checkpoints can achieve identical accuracy while orienting their basis vectors
  differently in R^d (orthogonal rotation / latent subspace drift). Therefore, drift monitoring MUST
  use a FIXED reference model version to compute representations across time snapshots, or compare
  invariant geometric metrics (such as pairwise cosine similarity distributions, degree stats, or vector norm distributions).
"""

import json
from typing import Dict, Any, Tuple
import numpy as np
import torch
from scipy import stats

from gnn_model import GraphSAGEClassifier

def compute_graph_statistics(features: np.ndarray, edge_index: np.ndarray) -> Dict[str, float]:
    """
    Computes graph structural and feature-level statistics.
    features: [N, D]
    edge_index: [2, E]
    """
    num_nodes = features.shape[0]
    num_edges = edge_index.shape[1] if edge_index.ndim == 2 else 0

    if num_edges > 0 and num_nodes > 0:
        degrees = np.bincount(edge_index[0], minlength=num_nodes)
        avg_degree = float(np.mean(degrees))
        std_degree = float(np.std(degrees))
        max_degree = int(np.max(degrees))
    else:
        avg_degree, std_degree, max_degree = 0.0, 0.0, 0

    feat_sparsity = float(np.mean(features == 0))
    feat_mean = float(np.mean(features))

    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "avg_degree": avg_degree,
        "std_degree": std_degree,
        "max_degree": max_degree,
        "feat_sparsity": feat_sparsity,
        "feat_mean": feat_mean
    }

def compute_embedding_statistics(
    model: GraphSAGEClassifier,
    features: np.ndarray,
    edge_index: np.ndarray
) -> Dict[str, float]:
    """
    Computes latent representation statistics using a FIXED model instance.
    """
    model.eval()
    x = torch.tensor(features, dtype=torch.float32)
    edges = torch.tensor(edge_index, dtype=torch.long)

    with torch.no_grad():
        _, embeddings = model(x, edges, return_embeddings=True)
        emb_np = embeddings.cpu().numpy()

    # Embedding norms ||h_i||_2
    norms = np.linalg.norm(emb_np, axis=1)
    mean_norm = float(np.mean(norms))
    std_norm = float(np.std(norms))

    # Centroid coordinate mean
    centroid = np.mean(emb_np, axis=0)
    centroid_norm = float(np.linalg.norm(centroid))

    return {
        "mean_embedding_norm": mean_norm,
        "std_embedding_norm": std_norm,
        "centroid_norm": centroid_norm,
        "embedding_dim": emb_np.shape[1]
    }

def detect_drift(
    baseline_stats: Dict[str, Any],
    current_stats: Dict[str, Any],
    degree_tolerance: float = 0.20,      # 20% shift triggers alert
    norm_tolerance: float = 0.15          # 15% shift triggers alert
) -> Tuple[bool, Dict[str, Any]]:
    """
    Compares baseline vs current statistics and flags alerts.
    """
    alerts = []
    details = {}

    # 1. Structural Drift Check (Average Degree)
    base_deg = baseline_stats["avg_degree"]
    curr_deg = current_stats["avg_degree"]
    deg_rel_change = abs(curr_deg - base_deg) / (base_deg + 1e-6)
    details["degree_rel_change"] = float(deg_rel_change)

    if deg_rel_change > degree_tolerance:
        alerts.append(
            f"STRUCTURAL DRIFT ALERT: Average node degree shifted by {deg_rel_change*100:.1f}% "
            f"(Baseline: {base_deg:.2f}, Current: {curr_deg:.2f}, Threshold: {degree_tolerance*100:.0f}%)"
        )

    # 2. Representation Drift Check (Mean Embedding Norm)
    base_norm = baseline_stats["mean_embedding_norm"]
    curr_norm = current_stats["mean_embedding_norm"]
    norm_rel_change = abs(curr_norm - base_norm) / (base_norm + 1e-6)
    details["embedding_norm_rel_change"] = float(norm_rel_change)

    if norm_rel_change > norm_tolerance:
        alerts.append(
            f"EMBEDDING DRIFT ALERT: Mean latent norm shifted by {norm_rel_change*100:.1f}% "
            f"(Baseline: {base_norm:.2f}, Current: {curr_norm:.2f}, Threshold: {norm_tolerance*100:.0f}%)"
        )

    drift_detected = len(alerts) > 0
    return drift_detected, {
        "drift_detected": drift_detected,
        "alerts": alerts,
        "details": details,
        "baseline_snapshot": baseline_stats,
        "current_snapshot": current_stats
    }

