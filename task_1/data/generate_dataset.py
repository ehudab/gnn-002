"""
Synthetic fraud-detection graph dataset generator.

Design:
- 250 nodes, 75 fraud (30%) / 175 legit (70%)
- 3 numerical features per node: transaction_amount, account_age_days, velocity_per_hour
  Drawn from class-conditional distributions with deliberate overlap.
- Edges generated via a stochastic block model: same-class nodes connect with
  probability p_in, different-class nodes connect with probability p_out.
  This models fraud rings sharing devices/IPs/mule accounts.
- Undirected graph, no self-loops in the raw edge list (self-loops added later
  inside the GCN layer itself, not in the stored graph).

Output files (saved to /tmp/fraud_dataset/):
  nodes.csv   - node_id, transaction_amount, account_age_days, velocity_per_hour, label
  edges.csv   - source, target   (each undirected edge listed once)
  dataset.npz - same data as numpy arrays, for direct loading into torch tensors
  summary.txt - basic stats (degree distribution, edge homophily, etc.)
"""

import numpy as np
import os

SEED = 42
rng = np.random.default_rng(SEED)

N_NODES = 250
N_FRAUD = 75
N_LEGIT = N_NODES - N_FRAUD  # 175

P_IN = 0.03   # edge prob between two same-class nodes
P_OUT = 0.005 # edge prob between two different-class nodes

OUT_DIR = "/tmp/fraud_dataset"
os.makedirs(OUT_DIR, exist_ok=True)

# ---- 1. Labels ----
# 0 = legit, 1 = fraud
labels = np.array([1] * N_FRAUD + [0] * N_LEGIT)
# shuffle node order so fraud/legit aren't trivially grouped by index
perm = rng.permutation(N_NODES)
labels = labels[perm]

# ---- 2. Features (class-conditional, overlapping) ----
# transaction_amount: log-normal, fraud shifted higher
amount = np.where(
    labels == 1,
    rng.lognormal(mean=np.log(150), sigma=0.6, size=N_NODES),
    rng.lognormal(mean=np.log(80), sigma=0.6, size=N_NODES),
)

# account_age_days: fraud accounts skew newer
age = np.where(
    labels == 1,
    rng.gamma(shape=2.0, scale=30, size=N_NODES),       # mean ~60
    rng.gamma(shape=4.0, scale=100, size=N_NODES),      # mean ~400
)

# velocity_per_hour: fraud bursts transactions
velocity = np.where(
    labels == 1,
    rng.gamma(shape=3.0, scale=1.3, size=N_NODES),      # mean ~4
    rng.gamma(shape=2.0, scale=0.5, size=N_NODES),      # mean ~1
)

features = np.stack([amount, age, velocity], axis=1)

# ---- 3. Edges (stochastic block model) ----
edges = []
for i in range(N_NODES):
    for j in range(i + 1, N_NODES):
        p = P_IN if labels[i] == labels[j] else P_OUT
        if rng.random() < p:
            edges.append((i, j))
edges = np.array(edges)

# ---- 4. Save CSVs ----
import csv

with open(os.path.join(OUT_DIR, "nodes.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["node_id", "transaction_amount", "account_age_days", "velocity_per_hour", "label"])
    for i in range(N_NODES):
        w.writerow([i, round(float(amount[i]), 2), round(float(age[i]), 2), round(float(velocity[i]), 2), int(labels[i])])

with open(os.path.join(OUT_DIR, "edges.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["source", "target"])
    for s, t in edges:
        w.writerow([int(s), int(t)])

np.savez(
    os.path.join(OUT_DIR, "dataset.npz"),
    features=features,
    labels=labels,
    edges=edges,
)

# ---- 5. Summary stats ----
degree = np.zeros(N_NODES, dtype=int)
for s, t in edges:
    degree[s] += 1
    degree[t] += 1

same_class_edges = sum(1 for s, t in edges if labels[s] == labels[t])
total_edges = len(edges)
homophily = same_class_edges / total_edges if total_edges > 0 else 0

summary = f"""Synthetic fraud dataset summary
================================
Nodes: {N_NODES}  (fraud={N_FRAUD}, legit={N_LEGIT})
Edges: {total_edges}
Average degree: {degree.mean():.2f}  (min={degree.min()}, max={degree.max()})
Isolated nodes (degree 0): {(degree == 0).sum()}
Edge homophily (fraction of edges connecting same-class nodes): {homophily:.3f}

Feature means by class:
  Fraud  -> amount={amount[labels==1].mean():.1f}, age={age[labels==1].mean():.1f}, velocity={velocity[labels==1].mean():.2f}
  Legit  -> amount={amount[labels==0].mean():.1f}, age={age[labels==0].mean():.1f}, velocity={velocity[labels==0].mean():.2f}
"""
print(summary)
with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
    f.write(summary)

print(f"Saved dataset to {OUT_DIR}/")
