# Bare Untracked Baseline vs. Production MLOps Pipeline

This directory contains the **untracked baseline training script** ([`train.py`](file:///home/ehudab/Naod/iCog/Training/GNN/task4_bare/train.py)) and an analysis of its critical limitations compared to the enterprise MLOps pipeline developed in [`task_4`](file:///home/ehudab/Naod/iCog/Training/GNN/task_4).

---

## 1. Quick Start

Run the bare untracked script:
```bash
python task4_bare/train.py
```
This trains a GraphSAGE model on Cora and saves a raw PyTorch state dict file (`model.pt`) with standard terminal outputs.

---

## 2. Architectural Comparison

```
┌──────────────────────────────────────────────┐       ┌─────────────────────────────────────────────────────────────┐
│          Bare Untracked Baseline             │       │               Improved MLOps Pipeline (task_4)              │
├──────────────────────────────────────────────┤       ├─────────────────────────────────────────────────────────────┤
│ 1. Hardcoded parameters in Python script     │       │ 1. Centralized configs & full MLflow param logging          │
│ 2. Unversioned dataset downloaded to /tmp    │  vs.  │ 2. Data hashing, lineage tracking, and split verification   │
│ 3. Ad-hoc terminal printouts                 │       │ 3. Step-by-step metric logging (loss curves, ROC-AUC, F1)   │
│ 4. Overwrites raw state_dict (.pt file)      │       │ 4. Self-contained MLflow PyFunc packaging with conda env    │
│ 5. Manual, blind deployment                  │       │ 5. Automated Validation Gate (canary checks & thresholds)   │
│ 6. No concept of graph drift or degradation  │       │ 6. Real-time drift detection (structural & embedding norm)  │
└──────────────────────────────────────────────┘       └─────────────────────────────────────────────────────────────┘
```

---

## 3. Detailed Limitations of the Bare Baseline

### A. Zero Experiment Tracking & Loss of History
* **Bare Baseline**: Hyperparameters (`lr`, `hidden_dim`, `dropout`, `weight_decay`), epoch losses, and final validation/test metrics exist only as ephemeral terminal text.
* **The Risk**: When comparing multiple runs or sweeps, engineers must manually copy numbers into spreadsheets. If a terminal window closes or script restarts, hyperparameter settings and metric histories are permanently lost.
* **MLOps Fix**: `task_4/train_tracked.py` uses **MLflow Tracking** to record every run with unique IDs, exact hyperparameter dictionaries, learning curves, and evaluation tables.

### B. Lack of Artifact Lineage & Reproducibility
* **Bare Baseline**: The output artifact is a bare `model.pt` file. There is no record of which Git commit produced it, what dataset version was used, or what random seed was initialized.
* **The Risk**: If the model exhibits an anomaly in staging or production, it is impossible to audit or reproduce the exact training conditions.
* **MLOps Fix**: MLflow logs the active Git commit hash, environment packages, and dataset checksums with every run artifact.

### C. Brittle Artifact Packaging (The Dependency Nightmare)
* **Bare Baseline**: `torch.save(model.state_dict(), "model.pt")` saves only raw tensor weight dictionaries.
* **The Risk**: 
  1. To load `model.pt` in a serving service or web API, the consumer must import the exact same Python class definition (`GraphSAGEClassifier`).
  2. If the serving environment has a slightly different PyTorch, CUDA, or `torch_geometric` version, loading crashes silently or fails with serialization errors.
  3. The file contains no input/output schema defining required tensor shapes or data types.
* **MLOps Fix**: The improved pipeline wraps the model into a standard **MLflow PyFunc** with:
  - Input/output signature definitions.
  - A bundled `conda.yaml` and `requirements.txt` capturing exact dependencies.
  - A standardized `.predict()` interface that handles graph conversion automatically.

### D. No Governance or Automated Quality Gates
* **Bare Baseline**: The script blindly saves `model.pt`, directly overwriting the previous model checkpoint even if the new run underperformed or diverged.
* **The Risk**: A buggy run or bad data can overwrite a healthy production model with a degraded checkpoint without human or automated review.
* **MLOps Fix**: `task_4/validation_gate.py` introduces a strict **Promotion Gate**:
  - Automatically compares the candidate against the current production baseline.
  - Evaluates minimum accuracy thresholds and degradation tolerances ($\le 2\%$).
  - Runs topology sanity checks (isolated node handling, self-loops).
  - Only models that pass all automated criteria are promoted in the Model Registry.

### E. Graph-Specific Risks & No Drift Monitoring
* **Bare Baseline**: Completely blind to post-deployment data distribution shifts.
* **The Risk**: In Graph Neural Networks, data changes not only in node features but also in graph connectivity (degree spikes, edge density shifts) and latent embedding rotations. The bare baseline has no tooling to diagnose when or why a model stops working.
* **MLOps Fix**: `task_4/drift/monitor.py` implements dedicated **Structural Drift** (average degree, density shift) and **Representation Drift** ($L_2$ norm distribution shifts) monitoring.

---

## 4. Summary Table

| Capability | Untracked Baseline (`task4_bare/train.py`) | Production MLOps (`task_4/`) |
| :--- | :--- | :--- |
| **Tracking** | Ephemeral `print()` statements | MLflow experiments, parameters & metric curves |
| **Artifact Format** | Raw PyTorch `.pt` file | Self-contained MLflow PyFunc model bundle |
| **Environment** | Assumes ambient local environment | Locked `requirements.txt` & Docker container |
| **Serving Interface** | Custom manual code | Containerized FastAPI REST API |
| **Model Governance** | Manual file overwriting | Automated Validation Gate & Registry stages |
| **Post-Deploy Monitoring**| None | Structural & Embedding Latent Drift monitoring |

