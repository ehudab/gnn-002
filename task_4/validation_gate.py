#!/usr/bin/env python3
"""
Model Registry & Promotion Validation Gate.

Implements strict governance rules:
- Candidate comparison against Current Approved Champion on identical validation set.
- Target Metric: Validation Accuracy (best_val_acc).
- Threshold: Candidate must exceed Approved Champion by at least `min_delta` (default: 0.005 / +0.5%).
- Tie Behavior: If within min_delta, Champion is retained (conservative bias prevents churn).
- Missing-Metric Behavior: REJECT candidate automatically (fail-closed security).
- First-Model Behavior: Auto-promoted as Initial Approved Model if it meets baseline floor (val_acc >= 0.70).
"""

import os
import json
import argparse
from typing import Dict, Any, Tuple

REGISTRY_DB_PATH = "task_4/artifacts/model_registry.json"

def load_registry() -> Dict[str, Any]:
    if os.path.exists(REGISTRY_DB_PATH):
        with open(REGISTRY_DB_PATH, "r") as f:
            return json.load(f)
    return {
        "model_name": "CoraGraphSAGE",
        "current_approved": None,
        "history": []
    }

def save_registry(registry: Dict[str, Any]):
    os.makedirs(os.path.dirname(REGISTRY_DB_PATH), exist_ok=True)
    with open(REGISTRY_DB_PATH, "w") as f:
        json.dump(registry, f, indent=2)

def evaluate_and_promote(
    candidate_summary_path: str,
    min_delta: float = 0.005,
    baseline_floor: float = 0.70
) -> Tuple[bool, str]:
    """
    Evaluates candidate against promotion rules.
    Returns (promoted: bool, reason: str).
    """
    registry = load_registry()

    # Rule 1: Missing-metric validation
    if not os.path.exists(candidate_summary_path):
        return False, f"REJECTED: Candidate summary file not found at {candidate_summary_path}"

    with open(candidate_summary_path, "r") as f:
        candidate = json.load(f)

    cand_val_acc = candidate.get("best_val_acc")
    if cand_val_acc is None or not isinstance(cand_val_acc, (int, float)):
        return False, "REJECTED: Candidate is missing 'best_val_acc' metric (Fail-Closed Behavior)."

    cand_name = candidate.get("run_name", "unknown_candidate")
    current_approved = registry.get("current_approved")

    # Rule 2: First-Model Behavior
    if current_approved is None:
        if cand_val_acc >= baseline_floor:
            candidate["version"] = 1
            candidate["status"] = "APPROVED"
            candidate["promotion_reason"] = f"Initial model met baseline floor ({cand_val_acc:.4f} >= {baseline_floor})"
            registry["current_approved"] = candidate
            registry["history"].append(candidate)
            save_registry(registry)
            return True, f"PROMOTED: Initial model '{cand_name}' approved (Val Acc: {cand_val_acc:.4f} >= {baseline_floor})"
        else:
            return False, f"REJECTED: First model '{cand_name}' failed minimum floor ({cand_val_acc:.4f} < {baseline_floor})"

    curr_val_acc = current_approved.get("best_val_acc", 0.0)
    delta = cand_val_acc - curr_val_acc

    # Rule 3 & 4: Threshold & Tie Behavior
    # If delta <= min_delta: Tie or regression -> retain current champion
    if delta < 0:
        reason = (
            f"BLOCKED REGRESSION: Candidate '{cand_name}' val_acc={cand_val_acc:.4f} is inferior to "
            f"approved champion '{current_approved.get('run_name')}' val_acc={curr_val_acc:.4f} (Δ={delta:+.4f})"
        )
        registry["history"].append({
            "run_name": cand_name,
            "status": "REJECTED_REGRESSION",
            "val_acc": cand_val_acc,
            "reason": reason
        })
        save_registry(registry)
        return False, reason

    if delta < min_delta:
        reason = (
            f"RETAINED CURRENT (TIE / INSUFFICIENT DELTA): Candidate '{cand_name}' (val_acc={cand_val_acc:.4f}) "
            f"did not beat champion by required threshold min_delta={min_delta} (Δ={delta:+.4f} < {min_delta})"
        )
        registry["history"].append({
            "run_name": cand_name,
            "status": "REJECTED_TIE",
            "val_acc": cand_val_acc,
            "reason": reason
        })
        save_registry(registry)
        return False, reason

    # Rule 5: Successful Promotion
    new_version = current_approved.get("version", 1) + 1
    candidate["version"] = new_version
    candidate["status"] = "APPROVED"
    candidate["previous_champion"] = current_approved.get("run_name")
    candidate["promotion_reason"] = f"Exceeded champion val_acc by Δ={delta:+.4f} (min_delta={min_delta})"

    # Archive previous
    registry["history"].append({
        "run_name": current_approved.get("run_name"),
        "status": "ARCHIVED",
        "demoted_at_version": new_version
    })

    registry["current_approved"] = candidate
    registry["history"].append(candidate)
    save_registry(registry)

    return True, f"PROMOTED: Candidate '{cand_name}' promoted to Version {new_version} (Val Acc: {cand_val_acc:.4f}, Δ={delta:+.4f})"

def rollback_to_previous() -> Tuple[bool, str]:
    """
    Rolls back current approved model to previously archived approved version.
    """
    registry = load_registry()
    current = registry.get("current_approved")
    if not current:
        return False, "Rollback failed: No current approved model found."

    # Find last approved version before current
    approved_history = [
        h for h in registry.get("history", [])
        if h.get("status") == "APPROVED" and h.get("run_name") != current.get("run_name")
    ]

    if not approved_history:
        return False, "Rollback failed: No previous approved model found in history."

    previous = approved_history[-1]
    rollback_version = current.get("version", 1) + 1

    restored = dict(previous)
    restored["version"] = rollback_version
    restored["status"] = "APPROVED"
    restored["promotion_reason"] = f"ROLLBACK from regressed version '{current.get('run_name')}'"

    registry["history"].append({
        "run_name": current.get("run_name"),
        "status": "ROLLED_BACK",
        "reason": "Emergency rollback triggered"
    })
    registry["current_approved"] = restored
    registry["history"].append(restored)
    save_registry(registry)

    return True, f"ROLLBACK SUCCESSFUL: Restored '{restored.get('run_name')}' as Version {rollback_version}."

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Registry Promotion Gate")
    parser.add_argument("--candidate", type=str, help="Path to candidate run_summary.json")
    parser.add_argument("--min-delta", type=float, default=0.005)
    parser.add_argument("--rollback", action="store_true", help="Trigger rollback to previous model")
    args = parser.parse_args()

    if args.rollback:
        success, msg = rollback_to_previous()
        print(msg)
        exit(0 if success else 1)

    if args.candidate:
        success, msg = evaluate_and_promote(args.candidate, min_delta=args.min_delta)
        print(msg)
        exit(0 if success else 1)

