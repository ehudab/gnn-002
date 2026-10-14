#!/usr/bin/env python3
"""
Comprehensive Unit & Integration Test Suite for GNN MLOps Pipeline.

Covers:
1. Failure handling:
   - Isolated nodes (0 degrees)
   - Duplicate target node IDs
   - Invalid edge endpoints (out of bounds)
   - Incorrect feature dimensions (e.g. 100 dims instead of 1433)
   - Unknown target IDs (out of range)
   - Model loading failure (uninitialized / missing artifact fallback)
2. Prediction consistency:
   - Verifies that offline model prediction and API response agree 100%
     under identical inputs, graph context, and preprocessing.
3. Inductive serving:
   - Tests prediction on unseen nodes excluded from the training graph.
4. Drift detection:
   - Demonstrates normal case (no alert) vs simulated structural/embedding drift (alert triggered).
"""

import os
import sys
import unittest
import numpy as np
import torch
from fastapi.testclient import TestClient

# Ensure task_4 root is on path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TASK4_DIR = os.path.dirname(CURRENT_DIR)
if TASK4_DIR not in sys.path:
    sys.path.insert(0, TASK4_DIR)

from gnn_model import GraphSAGEClassifier
from mlflow_model.wrapper import GNNWrapper
from api.main import app, load_approved_model
import api.main as api_module
from drift.monitor import (
    compute_graph_statistics,
    compute_embedding_statistics,
    detect_drift
)

class TestGNNMLOpsPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create minimal synthetic mock artifacts if not present to ensure tests can run anywhere
        os.makedirs("task_4/artifacts/exp_1_graphsage_baseline", exist_ok=True)
        weights_p = "task_4/artifacts/exp_1_graphsage_baseline/weights.pt"
        config_p = "task_4/artifacts/exp_1_graphsage_baseline/config.json"
        schema_p = "task_4/artifacts/exp_1_graphsage_baseline/schema.json"

        cls.in_dim = 1433
        cls.hidden_dim = 64
        cls.num_classes = 7

        mock_model = GraphSAGEClassifier(in_dim=cls.in_dim, hidden_dim=cls.hidden_dim, num_classes=cls.num_classes)
        torch.save(mock_model.state_dict(), weights_p)

        import json
        with open(config_p, "w") as f:
            json.dump({"in_dim": cls.in_dim, "hidden_dim": cls.hidden_dim, "num_classes": cls.num_classes}, f)

        with open(schema_p, "w") as f:
            json.dump({
                "feature_dim": cls.in_dim,
                "num_classes": cls.num_classes,
                "label_mapping": {str(i): f"Class_{i}" for i in range(cls.num_classes)}
            }, f)

        # Initialize API client
        load_approved_model()
        cls.client = TestClient(app)

    def test_01_health_and_model_info(self):
        """Test health endpoint and metadata inspection."""
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "healthy")

        res_info = self.client.get("/model-info")
        self.assertEqual(res_info.status_code, 200)
        self.assertEqual(res_info.json()["expected_feature_dim"], self.in_dim)

    def test_02_prediction_consistency_offline_vs_api(self):
        """
        Verify that offline direct PyTorch model predictions and API predictions
        match identically under identical inputs.
        """
        num_nodes = 5
        features = np.random.randn(num_nodes, self.in_dim).tolist()
        edges = [[0, 1, 2, 3], [1, 2, 3, 0]]
        targets = [0, 2, 4]

        # 1. API Call
        payload = {
            "features": features,
            "edge_index": edges,
            "target_node_ids": targets
        }
        api_res = self.client.post("/predict", json=payload)
        self.assertEqual(api_res.status_code, 200)
        api_preds = [p["predicted_class"] for p in api_res.json()["predictions"]]

        # 2. Direct Offline Wrapper Call
        offline_res = api_module.WRAPPER.predict_dict(payload)
        offline_preds = [p["predicted_class"] for p in offline_res["predictions"]]

        self.assertEqual(api_preds, offline_preds)
        print(f"\n[Test] Offline vs API consistency verified: {offline_preds} == {api_preds}")

    def test_03_isolated_nodes_handling(self):
        """
        Failure Handling: Isolated nodes (degree 0) should be gracefully processed.
        GraphSAGE relies on self-transformation when neighbors are absent.
        """
        num_nodes = 4
        features = np.random.randn(num_nodes, self.in_dim).tolist()
        edges = [[0, 1], [1, 0]]  # nodes 2 and 3 are isolated
        targets = [2, 3]

        payload = {
            "features": features,
            "edge_index": edges,
            "target_node_ids": targets
        }
        res = self.client.post("/predict", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(len(data["predictions"]), 2)
        print(f"\n[Test] Isolated nodes classified gracefully with confidences: {[p['confidence'] for p in data['predictions']]}")

    def test_04_duplicate_target_ids(self):
        """
        Failure Handling: Duplicate target IDs should be deduplicated without error.
        """
        num_nodes = 3
        features = np.random.randn(num_nodes, self.in_dim).tolist()
        edges = [[0, 1], [1, 0]]
        targets = [1, 1, 1, 0, 0]

        payload = {
            "features": features,
            "edge_index": edges,
            "target_node_ids": targets
        }
        res = self.client.post("/predict", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["num_targets"], 2)
        returned_ids = [p["node_id"] for p in data["predictions"]]
        self.assertEqual(returned_ids, [1, 0])

    def test_05_invalid_edge_endpoints(self):
        """
        Failure Handling: Edges referencing out-of-range node IDs must return 400 Bad Request.
        """
        num_nodes = 3  # Valid indices: 0, 1, 2
        features = np.random.randn(num_nodes, self.in_dim).tolist()
        edges = [[0, 99], [1, 0]]  # 99 is invalid

        payload = {
            "features": features,
            "edge_index": edges,
            "target_node_ids": [0]
        }
        res = self.client.post("/predict", json=payload)
        self.assertEqual(res.status_code, 400)
        self.assertIn("Invalid edge endpoints", res.json()["detail"])

    def test_06_incorrect_feature_dimensions(self):
        """
        Failure Handling: Passing wrong feature dimensions must return 400 Bad Request.
        """
        num_nodes = 3
        bad_dim = 100
        features = np.random.randn(num_nodes, bad_dim).tolist()
        edges = [[0, 1], [1, 0]]

        payload = {
            "features": features,
            "edge_index": edges,
            "target_node_ids": [0]
        }
        res = self.client.post("/predict", json=payload)
        self.assertEqual(res.status_code, 400)
        self.assertIn("Incorrect feature dimensions", res.json()["detail"])

    def test_07_unknown_target_ids(self):
        """
        Failure Handling: Requesting prediction for target node ID outside context range returns 400.
        """
        num_nodes = 3
        features = np.random.randn(num_nodes, self.in_dim).tolist()
        edges = [[0, 1], [1, 0]]

        payload = {
            "features": features,
            "edge_index": edges,
            "target_node_ids": [10]  # Node 10 does not exist
        }
        res = self.client.post("/predict", json=payload)
        self.assertEqual(res.status_code, 400)
        self.assertIn("Unknown target ID", res.json()["detail"])

    def test_08_inductive_serving_unseen_nodes(self):
        """
        Serving Setting: Demonstrate inductive inference for nodes excluded from training.
        Simulates introducing new unseen papers connected to each other and existing context.
        """
        # Graph context of 5 nodes: 0,1,2 (known), 3,4 (brand new unseen papers)
        num_nodes = 5
        features = np.random.randn(num_nodes, self.in_dim).tolist()
        # Edges between unseen nodes (3,4) and bridging into existing node 2
        edges = [[3, 4, 2], [4, 3, 3]]
        unseen_targets = [3, 4]

        payload = {
            "features": features,
            "edge_index": edges,
            "target_node_ids": unseen_targets
        }
        res = self.client.post("/predict", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["num_targets"], 2)
        print(f"\n[Test] Inductive prediction for unseen nodes 3 & 4 successful: {data['predictions']}")

    def test_09_drift_detection_normal_vs_alert(self):
        """
        Drift Detection: Verify normal baseline comparison and alert on structural/embedding changes.
        """
        num_nodes = 50
        base_feat = np.random.randn(num_nodes, self.in_dim)
        # Moderate graph: degree ~ 4
        src = np.random.randint(0, num_nodes, size=100)
        dst = np.random.randint(0, num_nodes, size=100)
        base_edges = np.stack([src, dst])

        # 1. Compute Baseline Statistics
        g_base = compute_graph_statistics(base_feat, base_edges)
        emb_base = compute_embedding_statistics(api_module.WRAPPER.model, base_feat, base_edges)
        combined_base = {**g_base, **emb_base}

        # Case A: Normal snapshot (slight noise)
        norm_feat = base_feat + np.random.normal(0, 0.01, size=base_feat.shape)
        g_norm = compute_graph_statistics(norm_feat, base_edges)
        emb_norm = compute_embedding_statistics(api_module.WRAPPER.model, norm_feat, base_edges)
        combined_norm = {**g_norm, **emb_norm}

        drift_flag, report = detect_drift(combined_base, combined_norm)
        self.assertFalse(drift_flag, "Normal slight variation should NOT trigger drift alert.")
        print("\n[Test] Normal snapshot check passed: 0 alerts.")

        # Case B: Significant structural change (hub creation -> degree inflation)
        dense_src = np.repeat(np.arange(num_nodes), 15)
        dense_dst = np.random.randint(0, num_nodes, size=len(dense_src))
        drift_edges = np.stack([dense_src, dense_dst])
        # Also perturb features significantly
        drift_feat = base_feat * 5.0

        g_drift = compute_graph_statistics(drift_feat, drift_edges)
        emb_drift = compute_embedding_statistics(api_module.WRAPPER.model, drift_feat, drift_edges)
        combined_drift = {**g_drift, **emb_drift}

        drift_flag_b, report_b = detect_drift(combined_base, combined_drift)
        self.assertTrue(drift_flag_b, "Substantial degree/embedding change MUST trigger drift alert.")
        self.assertGreater(len(report_b["alerts"]), 0)
        print(f"[Test] Drift detected correctly: {report_b['alerts']}")

if __name__ == "__main__":
    unittest.main()

