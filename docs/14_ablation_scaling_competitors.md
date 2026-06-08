# Ablation, scaling, and competitor reproduction (QUBO-GNN solver study)

Goal: via systematic hyperparameter / penalty / formulation ablation + scaling + reproducing recent
methods, find where the **QUBO-GNN is genuinely best** and compare consistently. `src/ablation.py`,
`src/exp_scaling_qubo.py`, `results/ablation/`, `results/scaling_qubo/`.

## A. Ablation (mean optimality gap vs SCIP-exact over port2, port4, synth-N200; lower=better)
λ=0.5, cardinality selection QUBO + convex re-weight. Baseline GNN = qrf, recurrent+pagerank+refine+LS,
lr1e-3, 1200 ep, hidden128, 3 layers, embed32, penalty×10.

| config | mean gap % | note |
|---|---|---|
| Tabu(50) | −0.005 | matches exact |
| **GNN base** and **every variant** (pignn, no_recurrent, no_pagerank, no_refine, lr 3e-3/5e-4, epochs 2500, hidden 64/256, layers 2/4, embed 16/64, anneal 2e-4, penalty 2×/5×/20×, rounds 24) | **−0.005** | all match exact |
| no_localsearch | **10.508** | ← the one essential component |
| SA(100) | 7.652 | classical heuristic lags |

(−0.005% = numerically optimal; tiny negative is solver tolerance.)

**Findings (honest):**
1. At N≤200 these portfolio cardinality QUBOs are **combinatorially easy**: GNN reaches the exact
   optimum under *every* hyperparameter/penalty/formulation setting tried.
2. **The only ingredient that matters is the 1-flip local-search polish** (removing it → 10.5% gap).
   Recurrent features, PageRank, the QRF-vs-PI-GNN architecture, penalty weight (2×–20×), and the
   binarization-anneal penalty make **no difference** at this scale.
3. The GNN's real edge here is over **SA (7.65% worse)** and tabu is also strong (≈exact).
4. Implication: to expose a *meaningful* GNN advantage (where its learned structure matters and it beats
   tabu/exact), we must go to **scale**, where local-search-from-random and exact B&B both struggle.

## B. Scaling QUBO benchmark (dense synthetic, K=N/20; same QUBO for all solvers)
Compares GNN vs SA vs tabu vs SCIP(global, time-limited) on **QUBO energy** (gap to best; lower better).
`results/scaling_qubo/`.

| N | SCIP (global QUBO) | SA(100) | Tabu(50) | **GNN** |
|---|---|---|---|---|
| 500 | **47685% / timeout 90s** (fails) | 58.7% / 4.6s | **0.000% / 1.2s** | **0.000% / 16.9s** |
| 1000 | fails (timeout) | 58.5% / 20s | **0.000% / 2.1s** | **0.002% / 42s** |
| 1500 | fails | 64.5% / 56s | **0.000% / 3.7s** | **0.000% / 77s** |
| 2000 | fails | 62.5% / 99s | **0.000% / 5.3s** | **0.000% / 141s** |

Even at **N=2000, Tabu still reaches the optimum (gap 0, 5.3 s)** and the GNN only ties it (0%, 141 s,
slower). Tabu does **not** degrade on these factor-structured QUBOs at any tested scale → the
per-instance GNN never beats tabu; it ties at best and is slower. SA and exact-global fail at scale.

(SCIP here solves the *raw nonconvex QUBO* globally and fails at scale; the *structured convex MIQP*
solver scip_cardinality/Gurobi is strong — different solver. Point: among generic QUBO solvers, only
tabu and the GNN reach the optimum; SA and exact-global fail.)

**Findings (honest, decisive):**
- **GNN ties Tabu** at N=500 and N=1000 (both reach the best energy, gap ≈ 0) — but **Tabu is ~20×
  faster** (2 s vs 42 s).
- **SA fails badly at scale** (58% gap) — the GNN's one clear, consistent win is over SA.
- The exact global solver (SCIP) **does not finish** on these dense 500–1000-var nonconvex QUBOs in the
  budget — so "exact" is unavailable at scale, but **Tabu fills that role and matches the GNN cheaply**.
- **Conclusion: the QUBO-GNN does NOT beat the strong classical heuristic (tabu).** It matches the best
  achievable energy but offers no speed or quality edge over tabu; it only dominates SA.

## C. Competitor reproduction status (consistent head-to-head)
From the repo survey (`see conversation`): most 2025-26 neural-QUBO methods have **no usable public code**.
- **PI-GNN** (Schuetz et al.; amazon-science/co-with-gnns-example, DGL, generic xᵀQx) — **reproduced**:
  it is our method's direct ancestor; covered by the ablation `model=pignn` (PyG) and the verbatim DGL
  `qrfgnn_dgl.py`. At N≤200 it ties our GNN (both = exact); the differentiator is scale (Part B).
- **THRML** (Mancilla et al. 2026, EBM/block-Gibbs, JAX; extropic-ai/thrml, 1k★) — **installed &
  assessed** (`.venv-jax`). It is a low-level probabilistic-graphical-model / block-Gibbs sampler
  (build nodes/edges/blocks + `IsingEBM(biases, weights, beta)` + sampling schedule), **optimized for
  sparse graphs**. On a *dense* portfolio Q it loses its structural advantage and behaves like SA
  (which is already a baseline), at substantial adaptation cost. Not pursued as a turnkey solver;
  documented as a related EBM approach.
- **QRF-GNN, QUBO-GNN (Eliasof-Haber), Deep k-grouping, VNA-portfolio** — **no public code** → cannot
  reproduce; cited as related work only.
- **X²GNN** (ICLR 2025) — repo exists but barely maintained, SLURM-only, **no generic-Q interface**;
  high effort, deferred.

## D. Verdict (after the full study)
After a systematic sweep of **hyperparameters, penalty functions, formulations, scale (N≤1000), and
competitor reproduction**, the QUBO-GNN:
- **matches the exact optimum / best-found energy** everywhere (small and large N);
- **consistently beats simulated annealing** (7.6% at small N; 58% at N=500–1000);
- **ties — but does not beat — strong local search (tabu) and greedy**, and is **slower** than them;
- **cannot beat the exact optimum** (impossible).

**So, on portfolio-style QUBOs, "QUBO-GNN strictly better than all others" is NOT achievable**: these
QUBOs (factor-model covariance + cardinality) are too combinatorially benign — tabu/greedy reach the
optimum cheaply. This robustly confirms the Angelini & Ricci-Tersenghi critique in the portfolio domain
(a clean, publishable characterization), but it is a *negative* on the "beat everyone" goal.

**The only remaining path to a genuine GNN win is AMORTIZATION (a different method):** train ONE model
across many instances so inference is a single fast forward pass, then beat *per-instance* tabu/exact on
a STREAM of related QUBOs (e.g. daily rebalancing) on **amortized time at matched quality**. Our current
solver optimizes *per instance* (so it is slower than tabu); an amortized/meta-learned model is a
separate build (supervised or meta-learning across a distribution of portfolio QUBOs). This is the
honest, scoped next step if a "win" is required.
