# Experimental plan: testing QRF-GNN/PI-GNN on portfolio optimization, the right way

_Goal: a defensible result where the GNN-QUBO solver matches or beats SOTA on a problem where that is
actually possible. Hardware: local RTX 3090 (24 GB). See `03_diagnosis.md` for why the old setup can't._

## Guiding principles (from the literature audit)
1. **Pick a problem that is genuinely combinatorial.** Cardinality-constrained mean-variance
   (select K of N) is NP-hard (MIQP). Continuous mean-variance is convex → unbeatable; abandon it.
2. **Compare on the SAME objective against the RIGHT baselines:** exact MIQP (Gurobi) for the
   optimality gap; simulated annealing (`dwave-neal`) + tabu (`dwave-tabu`) for the QUBO-heuristic
   class; greedy + random for the sanity floor.
3. **Never normalize away infeasibility.** Feasibility rate is a reported metric, not a fix-up step.
4. **Validate the solver first** on its own home turf (Gset MaxCut) before trusting portfolio numbers.

## Stage 0 — Environment (blocker; see README)
Local Python is 3.14 with no ML stack. PyTorch + DGL have **no 3.14 wheels**. Create a conda env with
**Python 3.11** and install: `torch` (cu121), `dgl` (cu121) OR migrate to **PyTorch-Geometric** (easier
on Windows than DGL), plus `numpy<2`, `pandas`, `networkx`, `cvxpy`, `PyPortfolioOpt`, `dimod`,
`dwave-neal`, `dwave-tabu`, `gurobipy` (free academic license), `yfinance`, `scikit-learn`, `matplotlib`.
Decision needed: **DGL vs PyG** (PyG recommended on Windows). All code goes in `src/`.

## Stage 1 — Reproduce the solver on MaxCut Gset (validation gate)
Run the existing ResSAGE/QRF-GNN on the `Gset/` graphs we already have (G14, G15, G22, G49, G50, G55).
Target the QRF-GNN paper numbers (G14≈3058, G22≈13359, G55≈10294, …). 20 seeds each, report best & mean.
**Gate:** if we can't get within ~1–2% of the paper's cut values, fix the implementation before
touching portfolio. Artifact: `results/gset_maxcut/`.

## Stage 2 — Build a clean, correct QUBO portfolio pipeline (`src/`)
Modules (self-contained, unit-tested against a brute-force/Gurobi check on tiny instances):
- `data.py` — load OR-Library port1–port5 (download from Beasley site to `data/orlib/`) and an
  S&P 500 panel via yfinance (cache to `data/`). Compute μ, Σ.
- `qubo_cardinality.py` — QUBO for **cardinality-constrained** mean-variance:
  - selection vars x_i ∈ {0,1} (pick asset), optional K bits per selected weight;
  - risk `λ wᵀΣw`, return `−μᵀw`, budget penalty `A(Σw−1)²`, cardinality penalty `B(Σx_i−K)²`,
    linking penalty so w_i>0 ⇒ x_i=1;
  - **auto-calibrated penalty weights** (A,B set from the spectral scale of the objective so any
    constraint violation costs more than any objective gain). Return Q (dict + dense), offset, decoder.
- `baselines.py` — exact MIQP (Gurobi `min wᵀΣw s.t. μᵀw≥r, Σw=1, Σz=K, εz≤w≤δz, z∈{0,1}`),
  simulated annealing & tabu on the QUBO, greedy, random, and the continuous convex frontier.
- `solve_gnn.py` — QRF-GNN solver wrapping the ResSAGE net + recurrent feature update + the QUBO loss,
  with annealed penalty schedule (per Sun et al. 2207.11542) and multi-restart + local-search polish.
- `metrics.py` — QUBO energy, optimality gap vs Gurobi, feasibility (budget & cardinality), realized
  return/vol/Sharpe, wall-clock, and (for OR-Library) % deviation from the unconstrained frontier.

## Stage 3 — The honest experiment
- **Primary:** OR-Library port1–port5, K=10, trace the cardinality-constrained efficient frontier
  (sweep target returns). Compare GNN-QUBO vs SA / tabu / greedy on QUBO energy AND vs Gurobi MIQP for
  optimality gap and the frontier-deviation metric used since Chang et al. (2000).
- **Secondary (where a win is plausible):** large-N S&P 500 (N=300–500), K∈{10,20,30}, where exact MIQP
  starts to slow — measure wall-clock at matched quality. **Hybrid variant:** GNN/QUBO selects the
  K-asset support, then a convex QP (cvxpy) sets the weights → compare its Sharpe/return to Gurobi MIQP
  and to SA-selected support. This is our best shot at a clean "match SOTA, faster / scale further".
- Optional stretch: non-convex extension (discrete lots or nonlinear transaction costs) where MIQP
  scales badly and a learned heuristic has the most room.

## Stage 4 — Reporting
For every method × instance: QUBO energy, optimality gap %, feasibility %, Sharpe/return/vol,
wall-clock; frontier plots; tables mirroring the OR-Library convention. Save to `results/` + a final
`docs/05_results.md`. Be explicit about where we match vs beat vs lose — no normalization rescues.

## Success criteria (realistic, tiered)
- **Minimum (validation):** reproduce Gset MaxCut within ~1–2% of QRF-GNN paper.
- **Solid:** on OR-Library, GNN-QUBO matches/beats SA & tabu on QUBO energy and gets within a small
  optimality gap of Gurobi, with full feasibility (no normalization).
- **Strong (the "beat SOTA" goal):** on large-N S&P 500, the hybrid GNN-select + convex-weight matches
  Gurobi MIQP Sharpe while being faster, or scales to N where MIQP times out — beating SA/tabu/greedy
  at matched wall-clock.
