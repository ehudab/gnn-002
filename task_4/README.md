# Task 4: Production MLOps Pipeline for Graph Neural Networks (GraphSAGE)

An end-to-end, reproducible, tracked, governed, and containerized deployment system extending the Cora citation network node classifier into an enterprise-grade MLOps platform.

---

## 1. System Architecture Overview

```
                        ┌──────────────────────────────────────────────┐
                        │              GNN MLOps Pipeline              │
                        └──────────────────────┬───────────────────────┘
                                               │
               ┌───────────────────────────────┼───────────────────────────────┐
               ▼                               ▼                               ▼
     [ Experiment Tracking ]         [ Model Packaging ]            [ Validation Gate ]
     • MLflow Run Logger             • Custom PyFunc Wrapper        • Champion vs Candidate
     • Seeds, Params, Metrics        • weights.pt, config.json      • Min-delta (+0.5%) rule
     • Dataset checksum & Git SHA    • schema.json (1433-dim, labels• Tie & Fail-closed handling
               │                               │                               │
               └───────────────────────────────┼───────────────────────────────┘
                                               │
               ┌───────────────────────────────┼───────────────────────────────┐
               ▼                               ▼                               ▼
    [ Containerized Serving ]         [ Inductive Inference ]        [ Drift & Monitoring ]
    • FastAPI REST API               • Unseen test nodes            • Degree distribution shift
    • Docker & Docker Compose        • Graph context validation     • Mean embedding norm shift
    • /predict, /health, /model-info • Isolated node robustness     • Subspace rotation audit
```

---

## 2. Directory Structure

```
task_4/
├── train.py                     # Bare untracked baseline script (demonstrating anti-patterns)
├── train_tracked.py             # Reproducible training with MLflow tracking (3+ runs, seeds, hashes)
├── gnn_model.py                 # 2-Layer GraphSAGE architecture with embedding extraction
├── validation_gate.py           # Model registry, promotion threshold gate & automated rollback
├── Dockerfile                   # Production container definition for FastAPI service
├── docker-compose.yml           # Multi-container stack (MLflow Tracker + FastAPI inference engine)
├── requirements.txt             # Pinned dependency manifest
├── api/
│   └── main.py                  # FastAPI service with strict boundary, dimension & context checks
├── mlflow_model/
│   └── wrapper.py               # Custom MLflow PyFunc wrapper with schema enforcement
├── drift/
│   └── monitor.py               # Graph structural and representation space drift detection
├── tests/
│   ├── test_api.py              # Failure modes, offline-API consistency, inductive serving, drift tests
│   └── test_gate.py             # Promotion gate rules, tie-breaking, fail-closed & rollback tests
└── .github/workflows/
    └── ci.yml                   # CI/CD pipeline: lint, test, gate verification & container build
```

---

## 3. Requirements & Implementation Map

| Requirement | Implementation Component | Verification / Command |
|---|---|---|
| **Untracked Baseline** | [`train.py`](file:///home/ehudab/Naod/iCog/Training/GNN/task_4/train.py) | Highlights lack of lineage, manual overwrites, missing schemas |
| **Experiment Tracking** | [`train_tracked.py`](file:///home/ehudab/Naod/iCog/Training/GNN/task_4/train_tracked.py) | Logs 3 distinct experiment runs, seeds (42, 101, 2024), Git commit, data hash |
| **Model Packaging** | [`wrapper.py`](file:///home/ehudab/Naod/iCog/Training/GNN/task_4/mlflow_model/wrapper.py) | PyFunc wrapper with weights, config, and schema; verifies reload consistency |
| **API & Docker** | [`main.py`](file:///home/ehudab/Naod/iCog/Training/GNN/task_4/api/main.py) & [`Dockerfile`](file:///home/ehudab/Naod/iCog/Training/GNN/task_4/Dockerfile) | `/predict` accepting features, edge_index, target_node_ids |
| **Serving Settings** | [`test_api.py#test_08`](file:///home/ehudab/Naod/iCog/Training/GNN/task_4/tests/test_api.py) | Inductive inference for unseen nodes excluded from training graph |
| **Registry & Validation Gate**| [`validation_gate.py`](file:///home/ehudab/Naod/iCog/Training/GNN/task_4/validation_gate.py) | Defines `min_delta=+0.005`, tie retention, fail-closed missing metrics, rollback |
| **CI & Rollback** | [`.github/workflows/ci.yml`](file:///home/ehudab/Naod/iCog/Training/GNN/task_4/.github/workflows/ci.yml) | Blocks regressions on push; `validation_gate.py --rollback` restores champion |
| **Drift Detection** | [`monitor.py`](file:///home/ehudab/Naod/iCog/Training/GNN/task_4/drift/monitor.py) | Average degree shift & embedding norm shift; explains coordinate rotation |
| **Failure Handling** | [`test_api.py`](file:///home/ehudab/Naod/iCog/Training/GNN/task_4/tests/test_api.py) | Tests isolated nodes, duplicates, invalid endpoints, wrong dims, unknown targets |
| **Prediction Consistency** | [`test_api.py#test_02`](file:///home/ehudab/Naod/iCog/Training/GNN/task_4/tests/test_api.py) | Confirms 100% agreement between offline model and API response |

---

## 4. Live Demonstration & Execution Guide

### Step 1: Run the Bare Untracked Baseline
Observe the limitations (overwriting `.pt` weights, missing experiment history, no schemas):
```bash
python task_4/train.py
```

### Step 2: Run Reproducible Tracked Experiments (3 Runs)
Executes 3 hyperparameter experiments with varied seeds, logs parameters, metrics, artifacts, and verifies reloading consistency:
```bash
python task_4/train_tracked.py --run-all
```

### Step 3: Test Model Promotion Gate & Regression Blocking
1. **Initial Model Promotion** (First-model auto-approval floor check):
   ```bash
   python task_4/validation_gate.py --candidate task_4/artifacts/exp_1_graphsage_baseline/run_summary.json
   ```
2. **Superior Model Promotion** (Exceeds current champion by `min_delta >= 0.005`):
   ```bash
   python task_4/validation_gate.py --candidate task_4/artifacts/exp_2_graphsage_highcap/run_summary.json
   ```
3. **Blocked Regression Demonstration** (Inferior candidate rejected):
   ```bash
   python task_4/validation_gate.py --candidate task_4/artifacts/exp_3_graphsage_max_aggr/run_summary.json
   ```
4. **Emergency Rollback Demonstration**:
   ```bash
   python task_4/validation_gate.py --rollback
   ```

### Step 4: Run the Complete Test Suite
Executes all failure mode checks, offline vs API consistency, inductive serving, and drift detection:
```bash
python -m unittest task_4/tests/test_api.py
python -m unittest task_4/tests/test_gate.py
```

### Step 5: Start Local Microservices via Docker Compose
Launches the MLflow Tracking Server (`:5000`) and the GNN Inference FastAPI service (`:8000`):
```bash
cd task_4
docker compose up --build -d
```

- **MLflow Tracking UI**: [http://localhost:5000](http://localhost:5000)
- **FastAPI OpenAPI Swagger Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **API Health Check**:
  ```bash
  curl http://localhost:8000/health
  ```

---

## 5. API Usage Examples

### Example: Inductive Inference on Existing & Unseen Nodes
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [
      [0.0, 1.0, ... 1433 floats ...],
      [1.0, 0.0, ... 1433 floats ...]
    ],
    "edge_index": [
      [0, 1],
      [1, 0]
    ],
    "target_node_ids": [0, 1],
    "return_embeddings": false
  }'
```

**Response Format**:
```json
{
  "model_version": 2,
  "model_run_name": "exp_2_graphsage_highcap",
  "num_targets": 2,
  "num_nodes_in_context": 2,
  "num_edges_in_context": 2,
  "predictions": [
    {
      "node_id": 0,
      "predicted_class": 3,
      "predicted_label": "Neural_Networks",
      "confidence": 0.8841,
      "probabilities": [0.01, 0.02, 0.01, 0.8841, 0.04, 0.02, 0.0159]
    }
  ]
}
```

---

## 6. Key Conceptual Rationales

### Why Graph-Context is Required for GNN Inference
Unlike standard tabular or image models where sample $x_i$ is evaluated in isolation, a Graph Neural Network computes:
$$h_u^{(l)} = \text{COMBINE}\left(h_u^{(l-1)}, \text{AGGREGATE}\left(\{h_v^{(l-1)} : v \in \mathcal{N}(u)\}\right)\right)$$
To classify node $u$ at layer $L$, the serving system must receive the $L$-hop computational subgraph surrounding $u$. Providing only target node $u$'s features collapses message passing into an isolated node transformation.

### Why Raw Embedding Coordinates Drift Between Versions
Neural network loss surfaces have permutation, reflection, and continuous rotation symmetries in latent representation space $\mathbb{R}^d$. Retraining with a new random seed or different hyperparameter set $W'$ rotates the latent coordinate system:
$$z' = R \cdot z, \quad \text{where } R^\top R = I$$
Even if classification accuracy is identical, raw Euclidean coordinate distances $\|z_i^{(A)} - z_i^{(B)}\|$ will register false drift alerts. Therefore, monitoring must:
1. Use a **fixed reference model checkpoint** to project subsequent data snapshots.
2. Monitor **coordinate-invariant statistics** (e.g. vector $L_2$ norm distributions, pairwise cosine similarity distributions, and graph topology metrics).

