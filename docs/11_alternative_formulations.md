# Alternative portfolio formulations — can the GNN beat greedy / SOTA? (honest results)

_Goal: find a portfolio QUBO where the GNN genuinely beats greedy and the exact solver (a *new*
result), per the user's "maybe something works". Tested 2026-06-04. `src/exp_sectors.py`,
`results/sectors/`._

## Formulation tested: sector-capped cardinality (diversification)
Choose exactly K assets, **at most `cap` per sector** (Σ caps > K, so the solver must decide how many
per sector AND which). This is genuinely non-modular — forward-greedy could fill good sectors early and
get stuck. QUBO uses slack bits to turn each cap into an equality penalty
(`selection_qubo_sector_caps`). Synthetic factor market with strong within-sector correlation and
sector return tilts (so diversification actually bites). Objective: `(ra/K²) z'Σz − (rw/K) μ'z`.

### Results (gap vs best feasible; lower = better)
| N (S,K,cap) | SCIP exact | Greedy(caps) | Tabu | SA | **GNN** |
|---|---|---|---|---|---|
| 90 (9,18,3) | 0.000 (0.6 s, *optimal*) | **0.000** (0.01 s) | 0.000 | 28.2% | **21.0%** |
| 300 (15,45,4) | 0.000 (8.2 s, *optimal*) | **0.002%** (0.7 s) | 8.7% | 35.1% | **14.4%** |

### Honest reading
- **The sector caps did NOT break greedy.** A cap-aware forward-greedy stays essentially optimal
  (0–0.002% gap) and is the fastest method.
- **The exact solver (SCIP) does NOT time out** even at N=300 with caps (8 s, proven optimal).
- **The GNN lags (14–21%)** on the slack-augmented QUBO — it beats SA but loses to greedy, Tabu, exact.
  Even a *perfect* QUBO solver could only **tie** here (greedy/exact already hit the optimum), so better
  GNN tuning would not produce a *win*.

## Why portfolio selection is hard to "beat SOTA" on
Across every variant we tried (plain cardinality, frontier sweep, scaling to N=400, sector caps), the
pattern is identical and matches the **Angelini & Ricci-Tersenghi (2206.13211)** critique, now
confirmed in the *portfolio* domain:
1. The convex-risk + cardinality structure makes **greedy near-optimal** and **branch-and-bound prune
   efficiently** — the problems are formally NP-hard but **combinatorially benign** at N ≤ ~300.
2. So the achievable ceiling is the exact optimum, which cheap classical methods already reach. A
   learned QUBO solver can **match** it (we showed MED = exact floor) but has **nothing to beat**.

## What the GNN genuinely delivers (the defensible contribution)
- **Matches the exact optimum / best published MED** on the standard benchmark (first GNN-QUBO placed
  there) — `08_*`.
- **Beats simulated annealing and (sometimes) tabu** on dense portfolio QUBOs — a real solver-quality
  win in that class.
- The **method is faithfully reproduced** (Gset within 0.3%) — `09_*`.

## Where a genuine win likely still hides (not yet exhausted)
These are the regimes where incumbents (greedy AND exact) actually fail — the only places "beat SOTA"
is realistic. Each needs more compute and is scoped as next work:
1. **Fixed-charge / non-convex transaction costs** (fixed cost per held position, or concave market
   impact). Fixed charges weaken B&B and make greedy genuinely myopic — the classic hard case.
2. **Discrete-lot / integer holdings** at scale (integer QP), where exact scaling degrades fastest.
3. **Amortized multi-period / rolling rebalancing**: train the GNN once, infer near-optimal selections
   in ms across thousands of rebalance dates — compete on *amortized time*, not single-instance
   optimality (the niche from Stopfer & Wagner 2509.17876 and the hybrid line).
4. **Very large N (≥1000) with dense Σ**, where even SCIP B&B finally times out and greedy's O(K·N²)
   becomes costly — a speed–quality frontier argument.

## Formulation 2 tested: fixed-charge transaction costs (non-convex, user-chosen)
`min λ w'Σw − (1−λ)μ'w + c·Σ z_i`, `Σw=1`, `0≤w_i≤z_i`, z binary — fixed cost c per held position,
**endogenous** number of positions. The classic non-convex "fixed-charge" portfolio (Rosenberg/Mugel
lineage). Methods: SCIP exact MIQP; forward-greedy re-solving the convex QP each step; GNN via a
cardinality sweep (best K'-subset + c·K'). `src/exp_fixed_charge.py`, `results/fixed_charge/`.

| N=100 | SCIP exact | Greedy(QP) | GNN(K-sweep) |
|---|---|---|---|
| c=2e-4 | **−0.004711** (opt, |S|=2, 1.1s) | **−0.004711** (|S|=2, 0.9s) | −0.004549 (3.4% gap, 41s) |
| c=3e-5 | **−0.005063** (opt, |S|=3, 1.0s) | **−0.005063** (|S|=3, 1.1s) | −0.005035 (0.5% gap, 41s) |

Same outcome: **SCIP and greedy reach the optimum in ~1 s; the GNN lags and is ~40× slower.** The
fixed cost just controls sparsity (high c → 2 positions, low c → a few more); the problem stays easy
for greedy/exact at every c.

## Verdict (definitive, across all tested formulations)
Tested: plain cardinality (port1–5 + synthetic), efficient-frontier sweep, scaling to N=400, **sector
caps** (N=90,300), **fixed-charge transaction costs** (N=100, two cost levels). In **every** case the
**convex-risk** structure keeps the problem combinatorially benign: **greedy and exact (SCIP) reach the
optimum cheaply, and the GNN-QUBO can only match it — never beat it** (and on harder QUBO encodings it
lags). This robustly confirms the Angelini & Ricci-Tersenghi (2206.13211) critique in the *portfolio*
domain — a clean, publishable negative result in itself.

**The GNN's genuine, defensible contributions remain:** (1) faithful reproduction of the method
(Gset within 0.3%); (2) matches the exact optimum / best published MED on the standard benchmark — the
first learned GNN-QUBO placed there; (3) beats simulated annealing / tabu on dense portfolio QUBOs.

**The only avenue left for a *new* "beat SOTA" result is to change the OBJECTIVE, not the constraints:**
non-convex *risk measures* where greedy is genuinely myopic and exact solvers blow up —
- **CVaR / scenario-based / drawdown** portfolio optimization (large scenario MIPs);
- **higher-moment** (skewness/kurtosis) objectives;
- **robust / distributionally-robust** portfolios.
These make the *value of a subset* non-modular and the exact problem genuinely large/hard — the regime
where a learned QUBO heuristic could finally win. This is a different research direction (new
objective + new QUBO encoding), scoped as future work.
