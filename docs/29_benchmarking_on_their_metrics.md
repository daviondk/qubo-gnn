# Benchmarking our QUBO solver on the papers' OWN tasks + metrics (user directive)

Goal: stop measuring gap-to-our-in-sample-optimum; instead adopt each top paper's exact objective + dataset
+ protocol + metrics, slot OUR solver in, and compare on their numbers. Tests whether our solver handles
real financial tasks.

## #1 QAOA-XY (Mancilla 2026, 2602.14827) -- cardinality K=5 of N=10, q=0.3 MV, monthly 2025, 5bps (E56)
- Data VALIDATED (my HRP 1.08 ≈ their 0.98; EW-all-10 0.97).
- Our solver solves the QUBO correctly (greedy=tabu=GNN agree). CAUSAL Sharpe: equal-weight 0.32-0.48,
  max-Sharpe-reweight <=0.63 -- ALL below equal-weight-all-10 (0.97).
- Their QAOA 1.81 ≈ HINDSIGHT ORACLE best-5 (1.83) -> likely LOOKAHEAD bias; not causally reproducible.
- LESSON: in a tiny 10-megacap universe, causal cardinality selection has no alpha (selecting 5 < holding 10).
  A fair test of our solver needs (a) a larger universe and (b) a causal, public-number baseline -> DSL next.

## Methodology principle (added to paper)
When a paper's headline ~ the hindsight oracle, suspect lookahead. Report causal vs oracle side-by-side.

## #2 DSL max-Sharpe (Kim 2025) on S&P100 -- causal (E57)
EqualWeight 0.90 > PlugIn-MaxSharpe-ALL 0.71 >= Ours-tabu/GNN-K20 0.69. The EXACT classical max-Sharpe
optimum overfits and loses to 1/N (DeMiguel-2009); our solver solves it correctly but inherits the
overfit -> objective, not solver, is the issue. DSL S&P500-rolling (0.47) < our 1/N (0.90); DSL edge only
on curated S&P-Top30 (1.10). CONCLUSION: real-world objective != classical in-sample optimum; need robust /
decision-focused objectives. Solver value = hard-combinatorial regimes + amortization, on a robustified objective.

## #3 Stopfer&Wagner 2025 (2509.17876) -- OUR solver on THEIR exact instances (E60)
Theta=achieved-vol/continuous-opt on their nasdaq minvola instances:
n=10: GNN 1.28/SA 1.16/tabu 1.12; n=20: 1.44/1.47/1.33; n=50: 2.69/3.05/2.47; n=100: 5.09/7.41/6.66.
=> reproduces their result with our solver: small-n our QUBO methods ~= their heuristic (1.2-1.5) and beat
their generic SA/tabu (>=2.0); large-n all QUBO methods blow up, native MIP=1.0. Root = convexity pitfall
(plain MinVola is a convex QP). For classical Markowitz, QUBO/GNN is dominated by direct QP/MIQP; our value
is non-convex/discrete variants + amortization. Their data+code: competitors/portfolio_opt_benchmark.

## HARD PORTFOLIO VARIANTS (user direction) — where exact solvers genuinely struggle + ready benchmarks
Plain MV = convex QP (trivial exact); plain cardinality = NP-hard but Gurobi solves to 1000 assets in sec
(Stopfer). GENUINELY HARD variants (exact times out / can't close gap) + public instances:
1. QOBLIB 06-portfolio (multi-period + cardinality + tx-cost + short, ALREADY binary/QUBO; best-known table
   + checker). git.zib.de/qopt/qoblib + github twobombs/QOBLIB. BARRIER: QUBO gen needs ZIMPL toolchain
   (their bundled binary = Linux + not in sparse-checkout; cloned-repo exec blocked; .zpl reconstruction
   complex: 4-index x [asset,lot,short/long,period] + penalty7 + slack encodings). Best-known objectives
   ARE available (solutions/uqo/README.md) for future ZIMPL-generated runs.
2. Cardinality DRO (MISDO, Kobayashi 2112.12454): SCIP-SDP FAILS to close gap on 85 assets/3600s. OR-Lib port2/5.
3. Cardinality mean-CVaR many scenarios (Kobayashi 2005.12797): MILP explodes w/ scenarios. OR-Lib (have).
4. Cardinality + min-buy-in MIQP (Frangioni-Gentile/Bertsimas): CPLEX doesn't scale >200. di.unipi.it (geo-block).
5. VNA MINLP cardinality+turnover+tx-cost (2507.07159): Mosek can't close 0.01% gap on Russell3000 in DAYS. No public instances.
=> TESTING hypothesis directly (E68): cardinality + min-buy-in MIQP via SCIP at increasing N.
