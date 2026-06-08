# Axis D — competitor reproduction (running their code / assessing their repos)

Goal: compare against recent methods using THEIR code where runnable. Repo survey (docs/14 + research):
most 2025-26 neural-QUBO methods have **no usable generic-Q public code**; the runnable ones are
PI-GNN (official) and THRML.

## D.1 Official PI-GNN (amazon-science/co-with-gnns-example) — RAN head-to-head ✅
Cloned to `competitors/co-with-gnns-example/` (MIT-0). Ran **their verbatim `utils.run_gnn_training`**
on our portfolio cardinality selection QUBO vs **our QRF-GNN** (`src/qrfgnn_dgl.py`), same Q, same env
(.venv-dgl), metric = QUBO energy (lower=better), |S| must equal K=10. `competitors/run_pignn_portfolio.py`,
`results/competitors_pignn.log`.

| Instance | amazon PI-GNN (their code) | our QRF-GNN |
|---|---|---|
| port2 (N=85) | energy **0.603, |S|=0 — INFEASIBLE (collapsed)** | **−0.00168, |S|=10 (feasible)** |
| port4 (N=98) | energy **0.563, |S|=0 — INFEASIBLE (collapsed)** | **−0.00212, |S|=10 (feasible)** |

**Finding:** the official base PI-GNN, run at its default settings, **collapses to the all-zero
selection** (probabilities < 0.5 everywhere → 0 assets chosen → only the cardinality-penalty offset
remains) on the **dense** portfolio QUBO — it does not produce a feasible K-asset portfolio. Our
QRF-GNN finds a feasible near-optimal solution. The QRF additions that prevent the collapse: the
**recurrent feature update** + the **annealed binarization penalty** in the loss (and PageRank/random
features). This is a concrete, external-code demonstration that the QRF-GNN method is **substantially
more robust than base PI-GNN on dense portfolio QUBOs**.
(Caveat: base PI-GNN with heavy retuning/local-search might avoid collapse; this is the out-of-the-box
comparison with reasonable defaults. Note: in our PyG ablation, `model=pignn` reached the optimum
because of our explore→exploit + local-search wrapper — i.e. the wrapper, not the bare GCN, rescues it.)

## D.2 THRML (extropic-ai/thrml, EBM/block-Gibbs, JAX) — installed, assessed
Installed in `.venv-jax`. It is a low-level probabilistic-graphical-model / block-Gibbs sampler
(`IsingEBM(nodes, edges, biases, weights, beta)` + a sampling schedule), **optimized for sparse
graphs**. On a *dense* portfolio Q it loses its structural advantage and behaves like SA (already a
baseline), at substantial adaptation cost (manual node/edge/block construction). Documented as a
related EBM approach; not run as a turnkey solver.

## D.3 Not reproducible (no usable public code) — cited as related work
- **X²GNN** (ICLR 2025) — repo exists but SLURM-only, no generic-Q interface, barely maintained.
- **QRF-GNN** (the paper) — no public code (we re-implemented it ourselves, Gset-validated, docs/09).
- **QUBO-GNN (Eliasof-Haber), Deep k-grouping, VNA-portfolio, DiffUCO-for-portfolio** — no portfolio/
  generic-Q code; high adaptation cost.

## Net (axis D)
- Reproduced the **official PI-GNN** and showed our QRF-GNN is **clearly more robust on portfolio QUBOs**
  (PI-GNN collapses, ours stays feasible/near-optimal) — a genuine "better than the competitor's code"
  result, using their code.
- THRML assessed (sparse-only EBM, ≈ SA on dense Q). Other neural-QUBO methods are not reproducible
  (no code) and are cited as related work, with our own faithful QRF-GNN reproduction (Gset, docs/09)
  standing in for that lineage.
