#!/usr/bin/env python3
"""
Unit Tests for Model Registry Promotion Validation Gate & Rollback.
Tests:
- First-model auto-promotion behavior
- Successful promotion when exceeding threshold (min_delta)
- Blocked regression when candidate is inferior
- Tie behavior (retaining champion when candidate delta < min_delta)
- Missing-metric failure handling (fail-closed)
- Rollback execution and state restoration
"""

import os
import sys
import json
import unittest

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TASK4_DIR = os.path.dirname(CURRENT_DIR)
if TASK4_DIR not in sys.path:
    sys.path.insert(0, TASK4_DIR)

from validation_gate import evaluate_and_promote, rollback_to_previous, load_registry, REGISTRY_DB_PATH

class TestValidationGate(unittest.TestCase):

    def setUp(self):
        # Reset registry database before each test
        if os.path.exists(REGISTRY_DB_PATH):
            os.remove(REGISTRY_DB_PATH)
        os.makedirs("task_4/artifacts/test_gate", exist_ok=True)

    def tearDown(self):
        if os.path.exists(REGISTRY_DB_PATH):
            os.remove(REGISTRY_DB_PATH)

    def _create_mock_summary(self, filename: str, run_name: str, val_acc: float = None):
        p = os.path.join("task_4/artifacts/test_gate", filename)
        data = {
            "run_name": run_name,
            "artifacts_dir": f"task_4/artifacts/{run_name}"
        }
        if val_acc is not None:
            data["best_val_acc"] = val_acc
        with open(p, "w") as f:
            json.dump(data, f)
        return p

    def test_01_first_model_behavior(self):
        """First valid model meeting baseline floor should be promoted to Version 1."""
        cand_p = self._create_mock_summary("cand_v1.json", "cand_baseline", val_acc=0.81)
        promoted, msg = evaluate_and_promote(cand_p, baseline_floor=0.70)
        self.assertTrue(promoted)
        self.assertIn("Initial model", msg)

        reg = load_registry()
        self.assertEqual(reg["current_approved"]["version"], 1)
        self.assertEqual(reg["current_approved"]["run_name"], "cand_baseline")

    def test_02_blocked_regression(self):
        """Candidate with inferior validation accuracy must be blocked."""
        # Setup champion
        c1 = self._create_mock_summary("c1.json", "champ_v1", val_acc=0.85)
        evaluate_and_promote(c1)

        # Inferior candidate
        c2 = self._create_mock_summary("c2.json", "inferior_cand", val_acc=0.82)
        promoted, msg = evaluate_and_promote(c2, min_delta=0.005)
        self.assertFalse(promoted)
        self.assertIn("BLOCKED REGRESSION", msg)

        # Verify champion is retained
        reg = load_registry()
        self.assertEqual(reg["current_approved"]["run_name"], "champ_v1")

    def test_03_tie_or_insufficient_delta_behavior(self):
        """Candidate that does not exceed champion by min_delta should be rejected."""
        c1 = self._create_mock_summary("c1.json", "champ_v1", val_acc=0.850)
        evaluate_and_promote(c1)

        # Delta = +0.002, which is < min_delta (0.005)
        c2 = self._create_mock_summary("c2.json", "marginal_cand", val_acc=0.852)
        promoted, msg = evaluate_and_promote(c2, min_delta=0.005)
        self.assertFalse(promoted)
        self.assertIn("INSUFFICIENT DELTA", msg)

        reg = load_registry()
        self.assertEqual(reg["current_approved"]["run_name"], "champ_v1")

    def test_04_successful_promotion_and_version_bump(self):
        """Candidate beating champion by >= min_delta is promoted to Version 2."""
        c1 = self._create_mock_summary("c1.json", "champ_v1", val_acc=0.850)
        evaluate_and_promote(c1)

        # Delta = +0.020 >= 0.005
        c2 = self._create_mock_summary("c2.json", "superior_cand", val_acc=0.870)
        promoted, msg = evaluate_and_promote(c2, min_delta=0.005)
        self.assertTrue(promoted)
        self.assertIn("promoted to Version 2", msg)

        reg = load_registry()
        self.assertEqual(reg["current_approved"]["version"], 2)
        self.assertEqual(reg["current_approved"]["run_name"], "superior_cand")

    def test_05_missing_metric_fail_closed(self):
        """Missing or malformed metrics must result in immediate rejection."""
        c_bad = self._create_mock_summary("bad.json", "corrupt_run", val_acc=None)
        promoted, msg = evaluate_and_promote(c_bad)
        self.assertFalse(promoted)
        self.assertIn("missing 'best_val_acc'", msg)

    def test_06_rollback_restoration(self):
        """Demonstrate emergency rollback to previously approved version."""
        c1 = self._create_mock_summary("c1.json", "stable_v1", val_acc=0.85)
        evaluate_and_promote(c1)

        c2 = self._create_mock_summary("c2.json", "new_v2", val_acc=0.88)
        evaluate_and_promote(c2)

        reg = load_registry()
        self.assertEqual(reg["current_approved"]["run_name"], "new_v2")

        # Trigger rollback
        success, msg = rollback_to_previous()
        self.assertTrue(success)
        self.assertIn("ROLLBACK SUCCESSFUL", msg)

        reg_post = load_registry()
        # Should restore stable_v1 model under bumped version 3
        self.assertEqual(reg_post["current_approved"]["run_name"], "stable_v1")
        self.assertEqual(reg_post["current_approved"]["version"], 3)

if __name__ == "__main__":
    unittest.main()

