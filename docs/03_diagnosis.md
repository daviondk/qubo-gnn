# Diagnosis: why the current `qubo_portofolio_opt.ipynb` experiment is fundamentally flawed

_Date: 2026-06-04. Author: analysis of the existing notebook + literature._

## TL;DR
The notebook applies the **QRF-GNN** unsupervised QUBO solver (arXiv:2407.16468 — note: the
notebook calls it "qrf-gnn"; the `ResSAGE` + PageRank + recurrent-feature architecture is QRF-GNN,
**not** the original PI-GNN despite the markdown header) to a **continuous Markowitz mean-variance
portfolio** encoded as a QUBO with 16 bits/asset (132 assets → 2112 binary variables).

This experiment **cannot succeed by construction**, for two independent reasons:

1. **The problem is convex.** Continuous mean-variance is a convex QP; the classical solver
   (PyPortfolioOpt/ECOS) already returns the *global* optimum. A QUBO discretization can only
   *approach* that optimum from below and adds discretization error — beating it on the same
   objective is mathematically impossible.
2. **The "good" reported result is an artifact of post-hoc normalization,** not of the GNN
   solving the QUBO. The raw QUBO solution was massively infeasible (weights summed to **0.37**,
   not 1.0; budget violation 0.63), and the code rescues it with `weights_norm = weights_raw/raw_sum`.
   After projecting back onto the simplex, almost any support looks near the frontier.

Net: the headline "GNN-QUBO Sharpe 0.1461 vs Max-Sharpe 0.1463" is **meaningless**.

## Detailed findings

### F1 — Convex objective (the killer)
Markowitz `min wᵀΣw − q·μᵀw  s.t. Σw=1, w≥0` with `Σ⪰0` is a convex QP. Solved to global
optimality in polynomial time. Sources: Fraunhofer benchmark (arXiv:2509.17876) shows Gurobi
solves all such instances to proven optimality "in seconds"; Lozano audit (arXiv:2605.17623)
shows a CPU tabu sampler matches D-Wave hybrid to 1e-3. **You can only tie, never beat.**

### F2 — Normalization hides infeasibility
`results cell 38`: `Budget residual: -0.629511`, `Total weight sum: 0.370489`.
The optimizer did **not** satisfy the budget constraint. `cell 36` then normalizes the weights,
so the comparison in `cell 37` measures the *normalized* portfolio, not what the QUBO produced.
The "0.13% volatility gap from the efficient frontier" is therefore not evidence the solver works.

### F3 — Penalty weights are miscalibrated
`risk_aversion=2000`, `return_penalty=1200`, `budget_penalty=100`. The risk term dwarfs the budget
penalty by 20×, so the optimizer rationally ignores the budget constraint → sum=0.37. Penalty
coefficients must be on comparable scales (and large enough that violating any constraint costs
more than any objective gain).

### F4 — `n_bits=16` is absurd
Discretization step `1/(2^16−1) ≈ 1.5e-5` — far finer than any portfolio needs. It inflates the
QUBO to 2112 vars, makes the 16 bits per asset highly redundant/degenerate, and most bits stay 0
(see `cell 40`: nearly all assets at level 0). 4–6 bits is plenty; for the interesting problem
(asset *selection*) you want 1 bit per asset.

### F5 — Wrong regime for the method
QRF-GNN's published wins are on **sparse** graphs (Gset MaxCut, 800–10k nodes). A portfolio
covariance graph is **dense / fully connected** — exactly the regime where PI-GNN-style relaxations
are reported to degrade (arXiv:2507.13703 "quality plummets with graph density").

### F6 — Wrong baseline / circular target
The target return was set to the **max-Sharpe point** (`target_vol_alpha=1`), then success was
declared for landing near max-Sharpe. Circular. And the comparison is against the *continuous*
convex frontier (unbeatable), not against the right combinatorial baselines (Gurobi MIQP,
simulated annealing, tabu) on the **same** QUBO.

### F7 — Implementation gaps
`decode_bitstring`, `build_qubo_graph`, `evaluate_qubo` are *used* but their definitions are only
referenced as a commented-out `import portfolio_markowitz_qubo` — the notebook relies on a module
that isn't in the repo, so it is not self-contained/reproducible.

### F8 — Method never validated
The QRF-GNN implementation was never checked against the paper's own MaxCut Gset numbers
(G14≈3058, G70≈9559). If it doesn't reproduce those, portfolio results are untrustworthy regardless.

## What "beating SOTA" can and cannot mean here
- **Hopeless:** beating a convex solver on continuous mean-variance. Don't try.
- **Realistic:** the **cardinality-constrained** portfolio (pick exactly K of N assets) is NP-hard
  (MIQP) and genuinely combinatorial. There the bar is exact MIQP (Gurobi/CPLEX) for optimality
  and fast metaheuristics (tabu/GA/SA) for the speed–quality frontier. See `04_experimental_plan.md`.
- **Mandatory honesty checks** (every literature audit insists on these): report (a) optimality gap
  vs Gurobi, (b) wall-clock at *matched* quality, (c) feasibility rate, (d) a *trivial* greedy/random
  baseline — Angelini & Ricci-Tersenghi (2206.13211) warn a near-linear-time greedy may already beat the GNN.
