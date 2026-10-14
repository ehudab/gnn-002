"""
Custom MLflow PyFunc Wrapper for Inductive GraphSAGE Serving.
Packages:
- PyTorch model state_dict
- Model architecture hyperparameters
- Feature preprocessing schema (expected in_dim, normalization rules)
- Target label mapping
- Input validation logic
"""

import json
import os
from typing import Dict, Any, List, Optional
import numpy as np
import torch
import torch.nn.functional as F

try:
    import mlflow.pyfunc
except ImportError:
    # Allow fallback if mlflow not directly imported
    class MockPyFunc:
        class PythonModel:
            pass
    mlflow = MockPyFunc()

from gnn_model import GraphSAGEClassifier


class GNNWrapper(mlflow.pyfunc.PythonModel):
    """
    MLflow pyfunc custom model interface for GraphSAGE on Cora.
    Handles graph context (features and edges) and predicts on target node IDs.
    """
    def __init__(self):
        self.model = None
        self.config = {}
        self.label_mapping = {}
        self.in_dim = 1433

    def load_context(self, context):
        """
        Loads model weights, config, and schema metadata from MLflow artifact context.
        """
        weights_path = context.artifacts.get("weights")
        config_path = context.artifacts.get("config")
        schema_path = context.artifacts.get("schema")

        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                self.config = json.load(f)

        if schema_path and os.path.exists(schema_path):
            with open(schema_path, "r") as f:
                schema_meta = json.load(f)
                self.label_mapping = schema_meta.get("label_mapping", {})
                self.in_dim = schema_meta.get("feature_dim", 1433)

        self.model = GraphSAGEClassifier(
            in_dim=self.config.get("in_dim", self.in_dim),
            hidden_dim=self.config.get("hidden_dim", 64),
            num_classes=self.config.get("num_classes", 7),
            dropout=self.config.get("dropout", 0.5),
            aggr=self.config.get("aggr", "mean")
        )

        if weights_path and os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location="cpu")
            self.model.load_state_dict(state_dict)

        self.model.eval()

    def predict(self, context, model_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        model_input dictionary:
        {
            "features": List[List[float]] shape [N, in_dim],
            "edge_index": List[List[int]] shape [2, E],
            "target_node_ids": List[int],
            "return_embeddings": bool (optional)
        }
        """
        return self.predict_dict(model_input)

    def predict_dict(self, model_input: Dict[str, Any]) -> Dict[str, Any]:
        """Direct dictionary inference method."""
        features = model_input.get("features")
        edge_index = model_input.get("edge_index")
        target_ids = model_input.get("target_node_ids")
        return_emb = model_input.get("return_embeddings", False)

        if features is None or edge_index is None or target_ids is None:
            raise ValueError("Input must contain 'features', 'edge_index', and 'target_node_ids'.")

        x = torch.tensor(features, dtype=torch.float32)
        if x.dim() != 2:
            raise ValueError(f"Features must be 2D matrix [num_nodes, in_dim], got shape {list(x.shape)}")
        if x.shape[1] != self.in_dim:
            raise ValueError(f"Incorrect feature dimensions: expected {self.in_dim}, received {x.shape[1]}")

        edges = torch.tensor(edge_index, dtype=torch.long)
        if edges.dim() != 2 or edges.shape[0] != 2:
            raise ValueError(f"edge_index must have shape [2, E], received shape {list(edges.shape)}")

        num_nodes = x.shape[0]

        # Validate edge endpoints
        if edges.shape[1] > 0:
            if (edges < 0).any() or (edges >= num_nodes).any():
                invalid_min = edges.min().item()
                invalid_max = edges.max().item()
                raise ValueError(
                    f"Invalid edge endpoints: node index out of range [0, {num_nodes-1}]. "
                    f"Found range [{invalid_min}, {invalid_max}]."
                )

        # Validate target node IDs
        if not target_ids:
            raise ValueError("target_node_ids cannot be empty.")

        for tid in target_ids:
            if tid < 0 or tid >= num_nodes:
                raise ValueError(f"Unknown target ID {tid}: must be within range [0, {num_nodes-1}]")

        self.model.eval()
        with torch.no_grad():
            if return_emb:
                logits, embeddings = self.model(x, edges, return_embeddings=True)
                emb_subset = embeddings[target_ids].cpu().numpy().tolist()
            else:
                logits = self.model(x, edges, return_embeddings=False)
                emb_subset = None

            probs = F.softmax(logits, dim=-1)
            target_probs = probs[target_ids].cpu().numpy()
            predicted_classes = np.argmax(target_probs, axis=-1).tolist()

        predictions = []
        for i, tid in enumerate(target_ids):
            c_idx = predicted_classes[i]
            c_name = self.label_mapping.get(str(c_idx), f"Class_{c_idx}")
            rec = {
                "node_id": tid,
                "predicted_class": int(c_idx),
                "predicted_label": str(c_name),
                "confidence": float(target_probs[i, c_idx]),
                "probabilities": [float(p) for p in target_probs[i]]
            }
            if emb_subset is not None:
                rec["embedding"] = emb_subset[i]
            predictions.append(rec)

        return {
            "predictions": predictions,
            "num_targets": len(target_ids),
            "num_nodes_in_context": num_nodes,
            "num_edges_in_context": edges.shape[1]
        }

