# Graph Neural Networks (GNN): Foundations to Production MLOps

[![Python](https://img.shields.io/badge/Python-3.14%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-torch__geometric-red.svg)](https://pyg.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.143%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-3.17%2B-0194E2.svg)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED.svg)](https://www.docker.com/)
[![Tests](https://img.shields.io/badge/Tests-15%20Passing-brightgreen.svg)](#7-testing--verification)

> **Author:** Naod Abebe ([naodabeben@gmail.com](mailto:naodabeben@gmail.com))  
> **Organization:** iCog Labs / Training  
> **Topic:** Advanced Graph Representation Learning, Empirical Benchmarking, Rigorous Leakage Audits, and Production MLOps Engineering

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Architecture](#2-repository-architecture)
3. [Summary of Tasks](#3-summary-of-tasks)
   - [Task 1: Synthetic Fraud Detection with GCN from Scratch](#task-1-synthetic-fraud-detection-with-gcn-from-scratch)
   - [Task 2: GNN Architecture Comparison on Cora](#task-2-gnn-architecture-comparison-on-cora)
   - [Task 3: Link Prediction & Rigorous Data Leakage Audit](#task-3-link-prediction--rigorous-data-leakage-audit)
   - [Task 4: Enterprise GNN MLOps Platform](#task-4-enterprise-gnn-mlops-platform)
4. [Master Presentation & Defense Documentation](#4-master-presentation--defense-documentation)
5. [Installation & Environment Setup](#5-installation--environment-setup)
6. [Quickstart & Execution Guide](#6-quickstart--execution-guide)
7. [Testing & Verification](#7-testing--verification)
8. [Docker & Microservices Deployment](#8-docker--microservices-deployment)
9. [Mathematical & Theoretical Formulations](#9-mathematical--theoretical-formulations)

---

## 1. Project Overview

Real-world relational systems—financial transaction networks, academic citation webs, and knowledge graphs—are inherently **non-Euclidean, irregular, and permutation-invariant**. Standard deep learning methods (MLPs, CNNs, Transformers) rely on Euclidean grid assumptions and independent and identically distributed (i.i.d.) observations, failing to leverage structural dependencies.

This repository implements a complete, rigorous curriculum in **Graph Representation Learning**:
- **First-Principles Implementation:** Building Graph Convolutional Networks (GCNs) directly in pure PyTorch with custom symmetric degree normalization and permutation equivariance proofs.
- **Empirical Rigor & Benchmarking:** Controlled multi-seed architecture bake-offs (MLP vs. GCN vs. GraphSAGE vs. GAT) with depth sweeps quantifying over-smoothing and topological ablation sweeps.
- **Methodological Soundness:** Auditing catastrophic data leakage in link prediction pipelines and engineering provably symmetric decoders.
- **Production-Grade MLOps Engineering:** Transitioning from brittle, untracked baseline scripts to a fully tracked, governed, tested, containerized, and monitored inductive inference serving platform.

---

## 2. Repository Architecture

```
.
├── README.md                                          # Master repository documentation (this file)
├── pyproject.toml                                     # Python project configuration and dependencies
├── uv.lock                                            # Locked dependency graph
│
├── Presentation_and_Evaluation_Guide_All_Tasks.md     # 680+ line comprehensive defense & presentation guide
├── generate_master_presentation_pdf.py                # Pure-stdlib PDF generator compiling master guide
├── Presentation notes — All Tasks Master Evaluation Guide.pdf # Executive defense PDF
│
├── task_1/                                            # Task 1: GCN From Scratch for Fraud Detection
│   ├── data/
│   │   ├── generate_dataset.py                        # Stochastic Block Model (SBM) graph generator
│   │   ├── nodes.csv, edges.csv, dataset.npz          # Synthetic transaction graph (N=250, E=636)
│   │   ├── summary.txt                                # Dataset statistics & homophily breakdown
│   │   └── Synthetic Fraud Dataset — Design Rationale.md # Detailed domain design rationale
│   ├── gcn_fraud_notebook.ipynb                       # Clean PyTorch implementation, checks & ablations
│   └── Presentation notes — GCN fraud detection.pdf   # Task 1 presentation deck
│
├── task_2/                                            # Task 2: Architecture Comparison on Cora
│   ├── gnn_cora_comparison.ipynb                      # MLP, GCN, GraphSAGE, GAT comparison & depth sweeps
│   └── Presentation notes — GNN architecture comparison on Cora.pdf # Task 2 presentation deck
│
├── task_3/                                            # Task 3: Link Prediction & Data Leakage Audit
│   ├── link_prediction_cora.ipynb                     # Clean vs. leaky pipelines, decoders, candidate scoring
│   ├── link_prediction_knowledge.md                   # Theoretical link prediction knowledge document
│   ├── generate_notes_pdf.py                          # Task 3 PDF compiler
│   └── Presentation notes — Link prediction on Cora.pdf # Task 3 presentation deck
│
├── task_4/                                            # Task 4: Production GNN MLOps Pipeline
│   ├── train.py                                       # Baseline untracked training (anti-pattern demonstration)
│   ├── train_tracked.py                               # MLflow-tracked training (3 runs, seeds, schemas, hashes)
│   ├── gnn_model.py                                   # 2-layer GraphSAGE classifier with embedding extraction
│   ├── validation_gate.py                             # Automated registry validation gate & rollback engine
│   ├── Dockerfile                                     # Production container specification for inference API
│   ├── docker-compose.yml                             # Multi-service stack (FastAPI API + MLflow UI)
│   ├── requirements.txt                               # Pinned container requirements
│   ├── README.md                                      # Task 4 dedicated technical documentation
│   ├── Presentation_Guide_Task_4.md                   # Task 4 live demonstration guide
│   ├── api/
│   │   └── main.py                                    # FastAPI REST engine with graph-context validation
│   ├── drift/
│   │   └── monitor.py                                 # Graph structural & latent embedding drift detector
│   ├── mlflow_model/
│   │   └── wrapper.py                                 # Custom MLflow PyFunc wrapper with reload verification
│   └── tests/
│       ├── test_api.py                                # API boundary, inductive inference, & failure tests
│       └── test_gate.py                               # Validation gate promotion, tie-break, & rollback tests
│
└── task4_bare/                                        # Untracked Baseline Reference
    ├── train.py                                       # Naive untracked script
    ├── README.md                                      # Critical comparative critique of naive workflows
    └── model.pt                                       # Bare state dict checkpoint
```

---

## 3. Summary of Tasks

### Task 1: Synthetic Fraud Detection with GCN from Scratch
- **Problem Formulation:** Financial fraudsters coordinate via shared infrastructure (IPs, devices, mule rings), creating high-homophily relational graphs. Individual transaction features (amount, account age, velocity) overlap significantly between legitimate users and fraudsters.
- **Dataset:** Stochastic Block Model (SBM) with $N=250$ nodes (75 fraud, 175 legitimate), $E=636$ edges, average degree $=5.09$, and **Edge Homophily $=0.887$**.
- **Implementation:** Implemented completely from scratch in PyTorch without `torch_geometric` abstractions:
  - Symmetric degree normalization: $\hat{A} = \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}$, where $\tilde{A} = A + I_N$.
  - Numerical sanity checks: Single-node arithmetic manual trace matched forward pass within $< 10^{-6}$.
  - Permutation equivariance verified across random permutation matrices: $\| \text{GCN}(PX, PAP^\top) - P \cdot \text{GCN}(X, A) \|_\infty < 10^{-7}$.
- **Empirical Results (5 Random Seeds, Mean ± Std):**
  - **MLP (Features Only):** Accuracy $76.8\% \pm 2.1\%$, Fraud Recall $61.3\% \pm 4.2\%$, Macro F1 $0.720$.
  - **GCN (Features + Graph):** Accuracy **$91.2\% \pm 1.4\%$**, Fraud Recall **$88.0\% \pm 3.1\%$**, Macro F1 **$0.895$**.
  - **Structural Proof:** When $A$ is replaced with identity $I_N$, GCN accuracy collapses to $76.4\%$ (matching MLP). Under edge corruption ($0\% \to 100\%$), accuracy drops monotonically from $91.2\%$ to $72.1\%$.

---

### Task 2: GNN Architecture Comparison on Cora
- **Problem Formulation:** Systematic multi-model benchmark on the Cora citation graph ($N=2,708$, $E=5,429$, $D=1,433$ bag-of-words features, $C=7$ academic subject categories).
- **Architectures Compared:**
  - **MLP:** Structure-blind control baseline ($59.0\% \pm 1.1\%$).
  - **GCN:** Isotropic fixed degree normalization ($81.5\% \pm 0.7\%$).
  - **GraphSAGE:** Decoupled self- and neighborhood feature projection ($81.0\% \pm 0.5\%$).
  - **GAT:** Anisotropic learned multi-head self-attention (**$83.0\% \pm 0.6\%$**).
- **Over-Smoothing Analysis:**
  - Depth sweep across layers $L \in \{1, 2, 3, 4\}$.
  - Monitored pairwise cosine similarity: at 2 layers, similarity is $0.24$ with peak accuracy ($81.5\%$); at 4 layers, representations collapse to a single point (cosine similarity surges to **$0.94$**), dropping accuracy to $58.1\%$.
- **Error Analysis:** Misclassifications isolate into:
  1. *Heterophilous Boundary Papers:* Interdisciplinary works spanning distinct domains (e.g., genetic algorithms applied to neural nets).
  2. *Peripheral Low-Degree Papers:* Nodes with degree $\le 1$ with sparse text features.

---

### Task 3: Link Prediction & Rigorous Data Leakage Audit
- **Problem Formulation:** Predict citation relationships $(u, v)$ without node category labels ($5,278$ unique canonical undirected links; $70/15/15$ train/val/test splits).
- **Automated Leakage Audit:** Implemented 6 programmatic assert checks ensuring:
  - Both directions of undirected edges remain in the same split.
  - Held-out positive validation and test edges are strictly purged from the message-passing graph.
  - Negative samples never overlap with observed edges or across splits.
- **Decoder Comparison:**
  - **Dot-Product Decoder:** Parameter-free, inherently symmetric.
  - **Symmetric MLP Decoder:** Uses commutative feature representations $f(u, v) = [|z_u - z_v| \;;\; z_u \odot z_v]$ to guarantee identical scores regardless of input node ordering ($s(u, v) \equiv s(v, u)$).
- **The Catastrophic Leakage Demonstration:**
  - *Clean GNN Pipeline:* Test ROC-AUC = **$0.8979$**, Average Precision = **$0.9050$**.
  - *Leaky Pipeline (Test Edges Retained in MP Graph):* Test ROC-AUC = **$0.9828$**, AP = **$0.9815$**.
  - **Metric Inflation:** Artificially inflates performance by **$+8.5\%$ AUC** and **$+7.7\%$ AP**, testing trivial memorization rather than generalization.

---

### Task 4: Enterprise GNN MLOps Platform
- **Problem Formulation:** Addressing the critical failure modes of naive ML workflows (`task4_bare/`) by deploying an enterprise-grade, tracked, containerized, and monitored GNN serving platform.
- **Key Components:**
  1. **Experiment Tracking (`train_tracked.py`):** Logs seeds ($42, 101, 2024$), learning rates, hidden dimensions, aggregation operators, Git commit SHA, dataset MD5 checksums, and evaluation curves to MLflow.
  2. **Model Packaging (`mlflow_model/wrapper.py`):** Bundles PyTorch weights, hyperparameter configuration, and I/O signature into an MLflow PyFunc model with reload-consistency assertions.
  3. **Automated Validation Gate (`validation_gate.py`):**
     - Minimum test accuracy floor ($70\%$).
     - Candidate promotion requires $\Delta \text{Acc} \ge +0.5\%$ over current champion.
     - Strict tie-retention (champion retained on ties) and fail-closed safety on corrupted metadata.
     - One-command automated rollback restoring previous champion.
  4. **Production Inference Service (`api/main.py`):** FastAPI application with Pydantic validation, graph-context verification, inductive inference on unseen test nodes, and isolated node robustness.
  5. **Drift Detection & Monitoring (`drift/monitor.py`):**
     - *Graph Structural Drift:* Tracks average node degree and graph density shifts.
     - *Representation Space Drift:* Tracks latent $L_2$ norm distribution shifts, accounting for latent subspace rotation ($z' = R \cdot z$).

---

## 4. Master Presentation & Defense Documentation

For technical defenses, academic examinations, or viva reviews, comprehensive guides are provided:

- **[Master Presentation & Defense Guide (`Presentation_and_Evaluation_Guide_All_Tasks.md`)](Presentation_and_Evaluation_Guide_All_Tasks.md):**
  - Section 0: Foundations of Graph Representation Learning
  - Section 1: Task 1 Defense (Single-node trace, SBM design, structural proofs)
  - Section 2: Task 2 Defense (Cora benchmark, depth sweep, over-smoothing, error analysis)
  - Section 3: Task 3 Defense (Symmetric decoder derivation, leakage audit, missing link analysis)
  - Section 4: Task 4 Defense (Live terminal commands, validation gate, drift rationale)
  - Section 5: Examiner Master Q&A Cheat Sheet
- **PDF Deck Generation:**
  Generate the formatted multi-page PDF presentation using pure Python standard library:
  ```bash
  python generate_master_presentation_pdf.py
  ```
  Produces `Presentation notes — All Tasks Master Evaluation Guide.pdf`.

---

## 5. Installation & Environment Setup

### Prerequisites
- Python $\ge 3.10$ (tested on Python 3.14)
- Docker & Docker Compose (optional, for containerized microservices)

### Clone & Virtual Environment Setup
```bash
# Clone the repository
git clone https://github.com/<your-username>/GNN.git
cd GNN

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install required packages
pip install --upgrade pip
pip install fastapi httpx matplotlib mlflow pydantic pytest scikit-learn torch-geometric uvicorn
```

*(Note: If using `uv`, you can synchronize the environment directly via `uv sync`.)*

---

## 6. Quickstart & Execution Guide

### Task 1: Synthetic Fraud Detection
```bash
# Optional: Regenerate the synthetic SBM dataset
python task_1/data/generate_dataset.py

# Launch Jupyter to explore the pure PyTorch GCN implementation
jupyter notebook task_1/gcn_fraud_notebook.ipynb
```

### Task 2: GNN Architecture Comparison on Cora
```bash
# Launch Jupyter to run the multi-seed comparison and depth sweeps
jupyter notebook task_2/gnn_cora_comparison.ipynb
```

### Task 3: Link Prediction & Data Leakage Audit
```bash
# Launch Jupyter to run clean vs. leaky pipelines and leakage checks
jupyter notebook task_3/link_prediction_cora.ipynb
```

### Task 4: Production GNN MLOps Pipeline

#### Step 1: Run the Bare Baseline (Anti-Pattern Demonstration)
```bash
python task_4/train.py
```

#### Step 2: Run Reproducible Tracked Experiments
Executes 3 distinct runs (Baseline, High-Cap Champion, and Max-Aggr Regression) with pinned seeds, logs parameters, metrics, artifacts, and verifies reloading consistency:
```bash
python task_4/train_tracked.py --run-all
```

#### Step 3: Test Model Promotion Gate & Regression Blocking
```bash
# 1. Promote initial baseline candidate (Approved as Version 1)
python task_4/validation_gate.py --candidate task_4/artifacts/exp_1_graphsage_baseline/run_summary.json

# 2. Promote superior candidate (Val Acc 84.4%, Delta = +3.2% >= +0.5% -> Promoted to Version 2)
python task_4/validation_gate.py --candidate task_4/artifacts/exp_2_graphsage_highcap/run_summary.json

# 3. Block regression candidate (Val Acc 77.2% vs 84.4% -> Blocked)
python task_4/validation_gate.py --candidate task_4/artifacts/exp_3_graphsage_max_aggr/run_summary.json

# 4. Demonstrate emergency rollback (Restores Version 1)
python task_4/validation_gate.py --rollback
```

---

## 7. Testing & Verification

The repository contains an automated unit and integration test suite covering API boundaries, inductive inference on unseen nodes, graph-context validation, drift detection, and gate logic:

```bash
# Run the entire test suite
python -m unittest discover task_4/tests
```

### Test Coverage Highlights
- `test_01_predict_endpoint_success`: Validates standard multi-node predictions.
- `test_02_prediction_consistency_offline_vs_api`: Confirms 100% agreement between offline model inferences and API responses.
- `test_03_isolated_nodes_graceful_handling`: Verifies isolated nodes ($d=0$) are scored without numerical errors.
- `test_08_inductive_prediction_for_unseen_nodes`: Validates inductive inference for new nodes excluded from training graph.
- `test_09_drift_detection_average_degree_and_latent_norm`: Tests detection of structural shifts and embedding norm drift.
- `test_gate.py`: Verifies promotion threshold enforcement ($\ge +0.5\%$), tie-break champion retention, missing metric fail-closed rejection, and automated rollbacks.

---

## 8. Docker & Microservices Deployment

Task 4 includes a complete multi-container stack orchestrated via Docker Compose:

### 1. Build and Launch Services
```bash
cd task_4
docker compose up --build -d
```

### 2. Available Endpoints
- **FastAPI REST Service:** [http://localhost:8000](http://localhost:8000)
- **Interactive Swagger Documentation:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **MLflow Tracking Server UI:** [http://localhost:5000](http://localhost:5000)

### 3. Test Inference Endpoint
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [
      [0.0, 1.0, 0.0, ... 1433 float values ...],
      [1.0, 0.0, 0.0, ... 1433 float values ...]
    ],
    "edge_index": [
      [0, 1],
      [1, 0]
    ],
    "target_node_ids": [0, 1],
    "return_embeddings": false
  }'
```

---

## 9. Mathematical & Theoretical Formulations

### Message Passing Neural Networks (MPNN)
At layer $l$, the embedding $h_u^{(l)}$ of node $u$ is updated via:
$$m_u^{(l)} = \text{AGGREGATE}^{(l)}\left( \left\{ \text{MESSAGE}^{(l)}\left(h_u^{(l-1)}, h_v^{(l-1)}, e_{uv}\right) : v \in \mathcal{N}(u) \right\} \right)$$
$$h_u^{(l)} = \text{COMBINE}^{(l)}\left( h_u^{(l-1)}, m_u^{(l)} \right)$$

### GCN Symmetric Degree Normalization (Task 1 & Task 2)
$$H^{(l+1)} = \sigma\left( \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(l)} W^{(l)} \right)$$
Where $\tilde{A} = A + I_N$ and $\tilde{D}_{ii} = \sum_j \tilde{A}_{ij}$. Adding self-loops prevents isolated node features from collapsing to zero, while symmetric normalization dampens high-degree hub dominance.

### Symmetric MLP Decoder (Task 3)
For undirected link prediction between nodes $u$ and $v$, link existence is invariant under node swap ($s(u, v) = s(v, u)$). The symmetric feature representation is defined as:
$$f(u, v) = \left[ |z_u - z_v| \;;\; z_u \odot z_v \right]$$
Because $|z_u - z_v| = |z_v - z_u|$ and $z_u \odot z_v = z_v \odot z_u$, symmetry holds unconditionally:
$$s(u, v) = \sigma(\text{MLP}(f(u, v))) \equiv s(v, u)$$

### Latent Representation Space Rotation & Drift Invariance (Task 4)
Neural network loss surfaces exhibit continuous orthogonal rotation symmetries. Two retraining runs with different initializations satisfy:
$$z' = R \cdot z \quad \text{where} \quad R^\top R = I$$
Because raw Euclidean distances $\|z_i - z_i'\|$ yield spurious drift alerts under coordinate rotation, monitoring relies on **rotation-invariant metrics**:
- Vector $L_2$ norm distributions: $\|z_i\|_2 = \|R z_i\|_2$
- Pairwise cosine similarities: $\frac{(R z_u)^\top (R z_v)}{\|R z_u\| \|R z_v\|} = \frac{z_u^\top R^\top R z_v}{\|z_u\| \|z_v\|} = \frac{z_u^\top z_v}{\|z_u\| \|z_v\|}$
- Topological degree distribution shifts.

---

## License

This project is created for research and educational purposes under iCog Labs.
