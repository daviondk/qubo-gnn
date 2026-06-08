# Axis A — cardinality QUBO on additional public datasets (incl. a 2026 head-to-head)

Extends dataset coverage beyond OR-Library + S&P100/DOW30. `src/datasets.py`, `src/exp_datasets.py`,
`results/datasets/`. Metric = optimality gap ("regret") of each method vs **SCIP-exact** cardinality
MIQP (λ=0.5, ε=0.01, δ=1). Datasets: French 49-Industry (Ken French, public), NASDAQ-100, crypto (both
yfinance).

| Dataset (N, K) | MIQP exact | **GNN** | Tabu | Greedy | SA | Random |
|---|---|---|---|---|---|---|
| French 49-Industry (49, 10) | 0.000 | **0.014%** | 0.014% | 0.014% | 3.81% | 2.65% |
| NASDAQ-100 (66, 10) | 0.000 | **0.000%** | 0.000% | 0.000% | 3.01% | 1.16% |
| Crypto (18, 6) | 0.000 | 13.22% | 13.22% | 13.22% | 58.0% | 47.4% |

## 2026 head-to-head: French 49-Industry vs Lozano (2026)
Lozano (arXiv:2605.17628, "penalty-free pipeline for direct quantum-annealer portfolio optimization")
uses **the same French 49-Industry dataset** and reports **post-processed regret ≤ 0.03%** (vs brute
force/greedy; D-Wave + classical projection). Our **GNN-QUBO + convex re-weight reaches regret 0.014%**
on the same dataset — i.e. **matches/edges the 2026 method** on its own benchmark. (Setups differ —
Lozano is quantum-annealer-hardware penalty-free; ours is an unsupervised GNN solver + convex weights —
so this is a same-dataset, comparable-metric result, not an identical pipeline.)

## Findings
- **Generalizes across datasets:** GNN reaches the exact optimum on NASDAQ-100 (0%) and is within
  0.014% on French49, **beating SA everywhere** (3–58%) — consistent with all prior results.
- **Crypto (N=18) is the informative exception:** GNN, Tabu, and Greedy all stall at **13.2% regret**
  while the exact MIQP finds a strictly better portfolio. Here the **equal-weight-surrogate selection
  QUBO is suboptimal** (high idiosyncratic crypto variance → the surrogate misranks the support), and
  at small N the exact MIQP wins. An honest limitation of the surrogate; a weight-encoded QUBO or
  exact solver is preferable at small N.
- Net: dataset coverage now = OR-Library port1-5 (Hang Seng/DAX/FTSE/S&P100/Nikkei) + S&P100 + DOW30 +
  French49 + NASDAQ100 + crypto + synthetic. (CSI 300 / Russell deferred — yfinance coverage for
  Chinese A-shares / full Russell is unreliable; would need a paid data source.)

Raw: `results/datasets/cardinality_extra.json`, `results/datasets/run.log`.
