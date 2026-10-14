# Synthetic fraud detection dataset — design rationale

## Why this dataset

Fraud detection is a natural fit for a GCN assignment: a single transaction's raw numbers (amount, account age, velocity) are often ambiguous on their own, but fraud rings share devices, IPs, and mule accounts, so fraudulent nodes cluster together in the graph. That gives the model two complementary signals — noisy features and informative structure — which is exactly what's needed to show a GCN beating an MLP and to prove the model is actually using the graph, not just the features.

Rather than hand-building a literal "shared device" simulator, the graph is generated with a stochastic block model (SBM): same-class nodes connect with higher probability than different-class nodes. This keeps the generator simple and fully controllable while still reading as a believable fraud-ring structure.

## Dataset description

250 nodes split 75 fraud (30%) / 175 legit (70%). Each node has 3 numerical features drawn from class-conditional, overlapping distributions:

| Feature | Fraud mean | Legit mean |
| --- | --- | --- |
| Transaction amount | 165.1 | 94.1 |
| Account age (days) | 59.6 | 400.2 |
| Velocity (txns/hr) | 3.96 | 1.04 |

Edges are generated via a stochastic block model: p\_in = 0.03 (same-class), p\_out = 0.005 (cross-class), undirected, no self-loops stored (added inside the GCN layer instead). The realized graph has 636 edges, average degree 5.09, and edge homophily 0.887 (89% of edges connect same-class nodes) — confirming the graph carries real class signal. 5 nodes ended up isolated (degree 0), which is expected at this density and useful later for discussing how a GCN handles disconnected nodes.

Generated with numpy, seed 42, for reproducibility. Files: `nodes.csv`, `edges.csv`, `dataset.npz`, `summary.txt`.
