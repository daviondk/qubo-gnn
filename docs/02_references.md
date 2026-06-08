# References & sources (with local copies in `../papers/`)

All arXiv PDFs downloaded 2026-06-04 into `papers/`. Filenames embed a short tag + the arXiv id.

## A. The method under test
- **PI-GNN** — Schuetz, Brubaker, Katzgraber, *Combinatorial Optimization with Physics-Inspired
  Graph Neural Networks*, arXiv:2107.01188 (2021); Nature Mach. Intell. 4:367 (2022).
  `papers/PI-GNN_Schuetz2021_2107.01188.pdf` (also the original local `2107.01188v2.pdf`).
- **QRF-GNN** (the architecture actually in the notebook) — Pugacheva, Ermakov, Lyskov, Makarov,
  Zotov, *Enhancing GNNs Performance on Combinatorial Optimization by Recurrent Feature Update*,
  arXiv:2407.16468 (2024). `papers/QRF-GNN_Pugacheva2024_recurrent-feature_2407.16468.pdf`

## B. The critique / "does it actually beat classical?" debate
- Angelini & Ricci-Tersenghi, *Modern GNNs do worse than classical greedy … (MIS)*, arXiv:2206.13211
  (2022); NMI 5:29 (2023). `papers/Critique_Angelini-RicciTersenghi2022_greedy-beats-GNN_2206.13211.pdf`
- Boettcher, *Inability of a GNN heuristic to outperform greedy … (MaxCut)*, arXiv:2210.00623 (2022);
  NMI 5:24 (2023). `papers/Critique_Boettcher2022_maxcut_2210.00623.pdf`
- Schuetz et al. reply to Angelini, arXiv:2302.03602. `papers/Reply_Schuetz2023_to-Angelini_2302.03602.pdf`
- Schuetz et al. reply to Boettcher, arXiv:2303.12096. `papers/Reply_Schuetz2023_to-Boettcher_2303.12096.pdf`
- Gamarnik, *Barriers for the performance of GNN in discrete random structures*, arXiv:2306.02555
  (2023); PNAS 120(46). `papers/Gamarnik2023_barriers-GNN-PNAS_2306.02555.pdf`

## C. Better unsupervised QUBO/CO neural solvers (PI-GNN successors)
- Sun, Guha, Dai, *Annealed Training for CO on Graphs*, arXiv:2207.11542 (2022).
  `papers/AnnealedTraining_Sun2022_2207.11542.pdf`
- Wang & Li, *Unsupervised Learning for CO Needs Meta-Learning* (Meta-EGN), arXiv:2301.03116 (ICLR
  2023). `papers/MetaEGN_Wang-Li2023_2301.03116.pdf`
- Sanokowski et al., *Variational Annealing on Graphs*, arXiv:2311.14156 (NeurIPS 2023).
  `papers/VariationalAnnealingGraphs_Sanokowski2023_2311.14156.pdf`
- Eliasof & Haber, *Quadratic Binary Optimization with GNNs (QUBO-GNN)*, arXiv:2404.04874 (2024).
  `papers/QUBO-GNN_Eliasof-Haber2024_2404.04874.pdf`
- *Binarizing Physics-Inspired GNNs for CO*, arXiv:2507.13703 (2025) — addresses the dense-graph
  degradation directly relevant to portfolio QUBOs. `papers/BinarizingPI-GNN2025_2507.13703.pdf`

## D. QUBO portfolio formulations (incl. the ones in the user's slide deck)
- Rosenberg et al., *Solving the Optimal Trading Trajectory Problem Using a Quantum Annealer*,
  arXiv:1508.06182 (2016). `papers/Rosenberg2016_trading-trajectory-DWave_1508.06182.pdf`
- Mugel et al., *Dynamic Portfolio Optimization with Real Datasets …*, arXiv:2007.00017 (2020).
  `papers/Mugel2020_dynamic-portfolio-quantum_2007.00017.pdf`
- *Quantum Portfolio Optimization with Investment Bands and Target Volatility*, arXiv:2106.06735
  (2021) — closest to the notebook's encoding. `papers/InvestmentBands-TargetVol_2021_2106.06735.pdf`
- *Portfolio Optimisation Using the D-Wave Quantum Annealer*, arXiv:2012.01121 (2021).
  `papers/DWave-annealer-portfolio_2021_2012.01121.pdf`
- *Quantum-Inspired Portfolio Optimization in the QUBO Framework*, arXiv:2410.05932 (2024).
  `papers/QuantumInspired-QUBO-portfolio_2024_2410.05932.pdf`
- *A real world test of Portfolio Optimization with Quantum Annealing*, arXiv:2303.12601 (2023).
  `papers/RealWorldTest-QA-portfolio_2023_2303.12601.pdf`
- *Scaling the Variational Quantum Eigensolver for Dynamic Portfolio Optimization*, arXiv:2412.19150
  (2024). `papers/ScalingVQE-dynamic-portfolio_2024_2412.19150.pdf`
- *Best practices for portfolio optimization by quantum computing*, Nature Sci. Rep.
  s41598-023-45392-w (2023) — in user deck; not on arXiv (paywalled HTML).

## E. Benchmarks, SOTA baselines, skeptical audits
- Chang, Meade, Beasley, Sharaiha, *Heuristics for cardinality constrained portfolio optimisation*,
  Comput. & Oper. Res. 27(13):1271 (2000) — **OR-Library port1–port5** benchmark. Data:
  http://people.brunel.ac.uk/~mastjjb/jeb/orlib/portinfo.html
- *Time-limited Metaheuristics for Cardinality-constrained Portfolio Optimisation*, arXiv:2307.04045
  (2023). `papers/TimeLimitedMetaheuristics-ORlib_2023_2307.04045.pdf`
- Stopfer & Wagner (Fraunhofer), *Quantum Portfolio Optimization: An Extensive Benchmark*,
  arXiv:2509.17876 (2025) — Gurobi/SCIP/SA/tabu/QAOA/D-Wave on 250 instances n≤1000.
  `papers/Fraunhofer2025_quantum-portfolio-benchmark_2509.17876.pdf`
- Morapakula et al., *End-to-End Portfolio Optimization with Quantum Annealing*, arXiv:2504.08843
  (2025). `papers/EndToEnd_QA-portfolio2025_2504.08843.pdf`
- Lozano, *Where the Quantum Lives in D-Wave Hybrid Portfolio Optimization* (operational audit),
  arXiv:2605.17623 (2026) — strongest skeptical source. (2026 id; not re-downloaded — see note.)

## E2. Additional OR-Library benchmark methods (2nd sweep, see `10_extended_published_results.md`)
- Tuo, Geng, Zhou (2016), *Econ. Comp. & Econ. Cybernetics Studies & Research* 50(1):311 — HSDS
  harmony search (⚠ MED on a different ~10× scale; do not merge).
- Mansouri & Sadeghi-Moghadam (2021), arXiv:2101.03312 — ARO (Chang %-error metric).
- Deng, Lin, Lo (2012), *ESWA* 39(4):4558 — improved PSO (%-error).
- Nikiporenko (2023), arXiv:2307.04045 — time-limited metaheuristics (%-error).
- **Xu, Tang, Yiu, Peng (2024)**, *INFORMS J. Computing* 36(2):690, DOI 10.1287/ijoc.2022.0344 —
  exact global-optimum branch-and-bound for CCPO (the optimal-floor reference).
- Exist but paywalled/unverified numbers: Zheng et al. 2023 (Mayfly, ESWA 230:120656); Kalayci 2017
  & Chen 2012 (ABC); Baykasoğlu 2015 (GRASP); Swarm&Evol.Comp 2020 hybrid; Ann.Oper.Res 2026 clustering.

## F. Tools / libraries
- OR-Library (Beasley): http://people.brunel.ac.uk/~mastjjb/jeb/orlib/portinfo.html
- PyPortfolioOpt: continuous convex baselines.
- `dimod` / `dwave-neal` (simulated annealing) / `dwave-tabu`: classical QUBO heuristics SOTA.
- Gurobi (`gurobipy`): exact MIQP — the optimality-gap reference (free academic license).
- DGL / PyTorch-Geometric: GNN implementation.

> Note: a few 2026-dated ids surfaced by search (2601.13465 Min&Gomes, 2602.23976, 2603.16904,
> 2605.17623 Lozano, 2603.09301) are recent/late and were not all re-fetched; verify before citing.
