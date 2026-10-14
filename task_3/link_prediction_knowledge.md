# Link Prediction on Graph Neural Networks — Knowledge Document

> **Task 3 · Cora Citation Network · GNN Encoder + Decoder Comparison + Data Leakage Audit**

---

## Table of Contents

1. [What Is Link Prediction?](#1-what-is-link-prediction)
2. [The Cora Dataset](#2-the-cora-dataset)
3. [Undirected Edge Handling](#3-undirected-edge-handling)
4. [Train / Validation / Test Split Strategy](#4-train--validation--test-split-strategy)
5. [Negative Sampling](#5-negative-sampling)
6. [GNN Encoder: GraphSAGE](#6-gnn-encoder-graphsage)
7. [Decoders: Dot-Product vs. MLP](#7-decoders-dot-product-vs-mlp)
8. [Symmetry Constraint on the MLP Decoder](#8-symmetry-constraint-on-the-mlp-decoder)
9. [Common-Neighbor Baseline](#9-common-neighbor-baseline)
10. [Data Leakage in Link Prediction](#10-data-leakage-in-link-prediction)
11. [Automated Leakage Checks](#11-automated-leakage-checks)
12. [Evaluation Metrics: ROC-AUC & Average Precision](#12-evaluation-metrics-roc-auc--average-precision)
13. [Decoder Comparison & Failure Cases](#13-decoder-comparison--failure-cases)
14. [Top-10 Unobserved Link Candidates](#14-top-10-unobserved-link-candidates)
15. [Conclusions](#15-conclusions)
16. [Glossary](#16-glossary)

---

## 1. What Is Link Prediction?

**Link prediction** is the task of inferring missing or future edges in a graph. Given a graph $G = (V, E)$, we want to estimate the probability that an edge $(u, v) \notin E$ should exist.

### Applications

| Domain | Task |
|---|---|
| Social networks | Friend recommendation |
| Knowledge graphs | Entity relation completion |
| Biology | Protein-protein interaction discovery |
| Citation networks | Paper recommendation |
| E-commerce | Product co-purchase prediction |

### Problem Formulation

Given observed edges $E_{\text{obs}}$, predict a score $s(u, v) \in [0, 1]$ for every candidate unobserved pair $(u, v)$. Higher score = higher confidence that an edge exists.

A **binary classifier** is trained:
- **Positive examples:** known edges from $E_{\text{obs}}$ held out for evaluation
- **Negative examples:** sampled non-edges

---

## 2. The Cora Dataset

Cora is a standard benchmark citation-network dataset:

| Property | Value |
|---|---|
| Nodes | 2,708 (academic papers) |
| Directed edges | 10,556 |
| Unique undirected edges | ~5,278 |
| Node features | 1,433 bag-of-words features |
| Classes | 7 (not used in this task) |

Each node is a paper; each edge represents a citation. The network is treated as **undirected** because citation relationships are symmetric for the purpose of topological similarity.

> [!NOTE]
> Although Cora is often used for node classification, here we strip away all label information and treat it as a pure link prediction problem. The node-feature matrix is used as GNN input but never for supervision.

---

## 3. Undirected Edge Handling

Cora's edge list stores both directions: $(u, v)$ and $(v, u)$. Treating it naively as directed would:
- Artificially double the number of positive examples
- Risk placing $(u, v)$ in train and $(v, u)$ in test — leaking structural information

**Deduplication strategy:**

```python
# Canonical form: always store the smaller node ID first
canonical = (min(u, v), max(u, v))
```

After deduplication, we work with **unique canonical pairs** only. When building the message-passing graph, we add both directions back:

```python
mp_edge_index = cat([train_pos, train_pos[[1,0]]], dim=1)
```

This ensures:
1. Exactly one copy of each edge in the positive set
2. Message passing remains undirected (information flows both ways)

---

## 4. Train / Validation / Test Split Strategy

### Split Proportions

```
Unique positive pairs: U
  Training  : 70% → U_train  (message-passing + supervision)
  Validation: 15% → U_val    (model selection)
  Test      : 15% → U_test   (final evaluation, touched ONCE)
```

### The Critical Rule

> **Both directions of a canonical pair must reside in the same split.**

If $(u, v)$ is in the test set, then both $(u, v)$ and $(v, u)$ are test edges. Placing them in different splits would:
- Leak $(v, u)$ into the message-passing graph (via training)
- Give the model direct access to the test edge's neighbourhood

### Message-Passing Graph

The GNN only uses **training edges** for neighbourhood aggregation — at all stages:

| Stage | MP Graph |
|---|---|
| Training | Train edges only |
| Validation | Train edges only |
| Test | Train edges only |

This simulates the real-world scenario where held-out edges are genuinely unknown.

---

## 5. Negative Sampling

### Why Negative Sampling?

Real graphs are sparse — the fraction of existing edges is tiny. We cannot evaluate on all $\binom{N}{2}$ pairs. Instead we sample a balanced set of "non-edges" as negatives.

### Requirements

1. **No self-loops:** $(u, u)$ is never a negative
2. **No positive contamination:** a sampled negative cannot be any observed positive edge (in any split)
3. **No split overlap:** val negatives and test negatives are disjoint
4. **Fixed evaluation negatives:** once sampled, val and test negatives never change — this ensures reproducible, comparable metrics

### Algorithm

```
1. Build set of ALL observed positive pairs (both directions)
2. Sample val negatives:  reject if in positives OR already in train negatives
3. Sample test negatives: reject if in positives OR in val negatives OR in train negatives
4. Train negatives:       re-sampled fresh each epoch (data augmentation)
```

### Why Re-sample Train Negatives?

Sampling a new set of negatives each epoch exposes the model to a larger variety of non-edge configurations, improving generalisation and reducing overfitting to a fixed negative set.

---

## 6. GNN Encoder: GraphSAGE

### Architecture

We use a **2-layer GraphSAGE** encoder:

```
Input features x ∈ ℝ^{N × 1433}
    ↓  SAGEConv(1433 → 128) + ReLU + Dropout(0.3)
    ↓  SAGEConv(128 → 64)
Output embeddings z ∈ ℝ^{N × 64}
```

### Why GraphSAGE?

| Property | Reason |
|---|---|
| Inductive | Can generate embeddings for unseen nodes |
| Scalable | Mini-batch sampling possible (we use full-graph here) |
| No label supervision | Encoder is trained end-to-end with link prediction objective |
| Feature-aware | Incorporates node features unlike purely structural methods |

### The Encoder's Role

The encoder maps every node to a dense embedding $z_u \in \mathbb{R}^{64}$ by aggregating information from its **training-edge neighbours**. Two nodes with similar neighbourhoods and features will have similar embeddings, which is what the decoder exploits.

### Training Objective

Binary cross-entropy over positive and negative pairs:

$$\mathcal{L} = -\frac{1}{|B|} \sum_{(u,v) \in B} \left[ y_{uv} \log s(u,v) + (1-y_{uv}) \log(1-s(u,v)) \right]$$

where $y_{uv} = 1$ for positive pairs and $0$ for negatives.

---

## 7. Decoders: Dot-Product vs. MLP

### 7.1 Dot-Product Decoder

$$s(u, v) = \sigma(z_u^\top z_v)$$

- **Pros:** Zero extra parameters, computationally cheap, directly measures cosine-like similarity in embedding space
- **Cons:** Assumes similarity is all that matters; cannot model asymmetric or complex interaction patterns (though we enforce undirected)
- **Symmetry:** $z_u^\top z_v = z_v^\top z_u$ — inherently symmetric ✓

### 7.2 MLP Decoder

$$s(u, v) = \sigma\!\left(\text{MLP}\!\left([|z_u - z_v|;\; z_u \odot z_v]\right)\right)$$

The input to the MLP is the concatenation of:
- $|z_u - z_v|$ — absolute elementwise difference
- $z_u \odot z_v$ — elementwise product

Both are **permutation-invariant** under swapping $u \leftrightarrow v$, which guarantees $s(u, v) = s(v, u)$ exactly.

- **Pros:** More expressive — can learn non-linear combinations; can capture complex interaction patterns
- **Cons:** More parameters; risk of overfitting; slightly slower
- **Architecture:** Linear(128→64) → ReLU → Dropout(0.2) → Linear(64→1)

### Why Not Concatenate $[z_u; z_v]$?

Concatenation $(z_u; z_v)$ is **asymmetric**: a different input to the MLP when the pair order is reversed. This would violate the requirement that undirected links produce the same score in both directions.

---

## 8. Symmetry Constraint on the MLP Decoder

**Requirement:** For an undirected graph, $\text{Score}(u, v) = \text{Score}(v, u)$.

**Our solution — symmetric feature construction:**

| Feature | Expression | $u \leftrightarrow v$? |
|---|---|---|
| Absolute difference | $|z_u - z_v|$ | $|z_v - z_u| = |z_u - z_v|$ ✓ |
| Elementwise product | $z_u \odot z_v$ | $z_v \odot z_u = z_u \odot z_v$ ✓ |

Since both input features are permutation-invariant, the MLP output is identical regardless of node ordering.

**Automated verification:**
```python
def verify_mlp_symmetry(decoder, z, n_checks=100):
    idx = torch.randint(0, N, (2, n_checks))
    fwd = decoder(z, idx)
    rev = decoder(z, torch.stack([idx[1], idx[0]]))
    max_diff = (fwd - rev).abs().max().item()
    assert max_diff < 1e-6  # machine-precision equality
```

---

## 9. Common-Neighbor Baseline

The **common-neighbor** (CN) heuristic scores each pair by the number of shared neighbours **in the training graph only**:

$$\text{CN}(u, v) = |N_{\text{train}}(u) \cap N_{\text{train}}(v)|$$

- No learning required
- Purely structural — ignores node features
- Serves as a lower bound: a GNN that cannot beat CN is not learning useful embeddings

**Why training graph only?** Using val/test edges to compute CN would leak information about held-out connectivity — the same issue as with the GNN.

**Intuition:** Papers with many common co-citations are likely related and may themselves be connected.

---

## 10. Data Leakage in Link Prediction

### What Is Data Leakage?

Data leakage occurs when information about the test set contaminated the training or evaluation process, producing **optimistically biased metrics** that do not reflect real-world performance.

### The Specific Leakage in Link Prediction

When GNN message passing uses **held-out edges** as part of the input graph:

```
Leaky scenario:
  MP graph = train edges + val edges + test edges
                                ↑
                         THESE SHOULD NOT BE HERE
```

**What happens mechanically:**

1. The GNN aggregates features from node $v$'s neighbours, including nodes connected via test edges
2. Node $u$'s embedding gets "contaminated" with signal from its test-edge neighbour $v$
3. When the decoder scores the pair $(u, v)$, it has already incorporated $v$'s influence into $u$'s representation
4. The model effectively "recognizes" the edge it was supposed to predict as unknown

### Analogy

> It's like studying for an exam with the answer key. The student performs perfectly on that exam but fails to generalise to new problems.

### Quantitative Impact

In our experiment:

| Metric | Valid | Leaky | Inflation |
|---|---|---|---|
| ROC-AUC | (see notebook) | (see notebook) | +Δ |
| Avg Precision | (see notebook) | (see notebook) | +Δ |

Leakage can easily inflate AUC by 5–15 percentage points on Cora-scale graphs.

### Why Leaky Metrics Are Dangerous

1. **Wrong model selection:** A weaker model with more leakage may rank above a better clean model
2. **False confidence:** Researchers report 0.97 AUC but deployment gives 0.82
3. **Incorrect conclusions:** "Our GNN is state-of-the-art" — but only because of the evaluation bug

---

## 11. Automated Leakage Checks

We formally prove the absence of leakage with assert-based checks:

```python
# Check 1: Val positive pairs absent from train-edge set
leaked_val = val_pos_set & train_edge_set
assert len(leaked_val) == 0

# Check 2: Test positive pairs absent from train-edge set
leaked_test = test_pos_set & train_edge_set
assert len(leaked_test) == 0

# Check 3: Val and test positive sets are disjoint
assert len(val_pos_set & test_pos_set) == 0

# Negative checks: no negative pair hits any observed positive
assert len(val_neg_set  & pos_canonical_set) == 0
assert len(test_neg_set & pos_canonical_set) == 0
```

**Why canonical pairs are sufficient for "both directions":** We store each edge as $(u, v)$ with $u < v$. This single canonical representation captures both $(u, v)$ and $(v, u)$ simultaneously. If the canonical pair is absent from the training set, both directed versions are absent.

---

## 12. Evaluation Metrics: ROC-AUC & Average Precision

### ROC-AUC (Area Under the ROC Curve)

**Interpretation:** The probability that a randomly chosen positive pair is ranked higher than a randomly chosen negative pair.

$$\text{AUC} = P(s(u^+, v^+) > s(u^-, v^-))$$

- **Range:** [0, 1]; random = 0.5; perfect = 1.0
- **Threshold-free:** evaluates ranking quality across all decision boundaries
- **Use case:** Good general summary metric

### Average Precision (AP)

**Interpretation:** Weighted mean of precisions at each threshold, with weights equal to the recall increase at that threshold.

$$\text{AP} = \sum_n (R_n - R_{n-1}) \cdot P_n$$

- Emphasises **high-precision at low recall** (top of the ranked list)
- More informative than AUC when positives are rare
- **Use case:** Applications where we only act on the very top predictions

### Why Both?

ROC-AUC treats all ranking positions equally. AP penalises models that bury positives behind negatives at the top of the list. Reporting both gives a more complete picture.

---

## 13. Decoder Comparison & Failure Cases

### When Does the Dot Decoder Fail?

The dot decoder assumes edge likelihood is a simple inner product in embedding space. It fails when:
- Two nodes have similar overall structure but an edge between them would be contextually wrong
- The optimal scoring function is non-linear in the embedding difference

### When Does the MLP Decoder Fail?

The MLP decoder can overfit when:
- Embeddings are low quality (undertrained encoder)
- Training set is small relative to decoder parameters
- The expressive power of the MLP is not beneficial at the given embedding dimension

### Failure-Case Taxonomy

| Error Type | Description | Cause |
|---|---|---|
| **False Negative (FN)** | True edge scored low | Nodes embedded far apart despite a real connection |
| **False Positive (FP)** | Non-edge scored high | Nodes embedded close together but no real connection |

**FN root causes:**
- Node pair connected by a "bridge" in the graph — structurally dissimilar neighbourhoods
- Features of the two papers differ substantially (different subfields)

**FP root causes:**
- Two papers heavily co-cited but never citing each other directly
- Structural equivalence without semantic relationship

---

## 14. Top-10 Unobserved Link Candidates

We score 50,000 random unobserved pairs and present the highest-scoring ones.

### Why "Unobserved" ≠ "Confirmed Negative"

The Cora dataset is a **snapshot** collected at a specific moment. An unobserved pair may be:

| Situation | Interpretation |
|---|---|
| Future citation | Will exist but hasn't been published yet |
| Existing but unlabelled | The dataset is known to be incomplete |
| Model hallucination | The model assigns high similarity to unrelated nodes |
| True negative | Genuinely no relationship |

### Why Predictions Require Validation

1. **Ground truth is incomplete:** Citation databases are never exhaustive
2. **Model uncertainty:** No uncertainty estimate is provided alongside scores
3. **Domain knowledge gap:** The model has no access to paper abstracts, author networks, or temporal ordering
4. **Threshold sensitivity:** A score of 0.9 is high, but what decision boundary is appropriate in this domain?

**Recommended validation workflow:**
1. Look up both papers in a full citation database (e.g., Semantic Scholar, Google Scholar)
2. Check if a citation exists in either direction
3. Have domain experts assess whether the papers are topically related enough to warrant a connection
4. Consider temporal ordering — can paper A have cited paper B given publication dates?

---

## 15. Conclusions

### Pipeline Summary

```
Raw Cora data
    ↓ Remove self-loops, deduplicate to canonical undirected pairs
    ↓ 70/15/15 split (both directions of each pair in same split)
    ↓ Fixed negative sampling (no contamination, no overlap)
    ↓ Automated leakage assertions
    ↓ GraphSAGE encoder (train-edges MP only)
    ↓ Two decoders: Dot-Product | MLP (symmetric)
    ↓ Common-Neighbor baseline
    ↓ Valid test evaluation (ROC-AUC, AP)
    ↓ Leaky experiment (for comparison only)
    ↓ Top-10 candidate predictions
```

### Key Takeaways

| Finding | Implication |
|---|---|
| GNN encoders beat CN baseline | Node features contain structural prediction signal beyond pure topology |
| MLP decoder parity with Dot | In high-quality embedding spaces, simpler decoders can match more complex ones |
| Leakage inflates metrics significantly | Always audit the MP graph; report clean-experiment numbers only |
| Symmetry by construction is reliable | Design choices (symmetric features) more robust than regularisation |
| Top candidates need validation | Model scores are hypotheses, not facts |

### Broader Lessons

1. **Evaluation protocol is as important as the model.** A leaky evaluation can make any model look state-of-the-art.
2. **Both directions of undirected edges must be treated consistently** throughout splitting, negative sampling, and the message-passing graph.
3. **Structural baselines (CN) are strong.** Always compare against them before claiming GNN superiority.
4. **Negative samples are not ground-truth negatives.** Cora (and most link prediction benchmarks) use sampled negatives, not exhaustively verified negatives. This introduces noise into the evaluation.

---

## 16. Glossary

| Term | Definition |
|---|---|
| **Link prediction** | Task of inferring missing/future edges in a graph |
| **Canonical pair** | Edge $(u, v)$ stored with $u < v$ to avoid duplication |
| **Message passing** | GNN operation where each node aggregates features from its neighbours |
| **MP graph** | The graph used for message passing (should exclude held-out edges) |
| **Data leakage** | When test-set information is inadvertently included in training/inference |
| **ROC-AUC** | Area under the ROC curve; probability that positives rank above negatives |
| **Average Precision** | Area under the Precision-Recall curve |
| **Common Neighbor** | Structural baseline: $|\mathcal{N}(u) \cap \mathcal{N}(v)|$ |
| **GraphSAGE** | Inductive GNN that samples and aggregates from neighbours |
| **Dot-product decoder** | $s(u,v) = \sigma(z_u^\top z_v)$ — inherently symmetric |
| **MLP decoder** | Learned scoring function on symmetric pair features |
| **Negative sampling** | Sampling non-edges to serve as negative training/evaluation examples |
| **Early stopping** | Halting training when validation metric stops improving |
| **Symmetry** | $s(u, v) = s(v, u)$ — required for undirected link prediction |
| **False Negative (FN)** | True edge scored below threshold |
| **False Positive (FP)** | Non-edge scored above threshold |

