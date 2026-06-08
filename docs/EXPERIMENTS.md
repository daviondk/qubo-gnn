# Experiment index (reproducibility map)

Every experiment is a script in `src/` (not a notebook — for reproducibility/versioning); results are
in `results/`; analysis in `docs/`. Three isolated envs (see README). Always set
`KMP_DUPLICATE_LIB_OK=TRUE` and `UV_CACHE_DIR=D:/uv-cache`.

| # | What | Script / command | Env | Output | Doc |
|---|---|---|---|---|---|
| 0 | Diagnosis of original notebook | (analysis) | — | — | docs/03 |
| 1 | Cardinality QUBO vs Gurobi/SA/tabu/greedy | `python src/exp_cardinality.py port4` | .venv | results/cardinality/*.json | docs/05 |
| 2 | Efficient-frontier sweep + plot | `python src/exp_frontier.py port2 10 12` | .venv | results/frontier/* | docs/05 |
| 3 | Synthetic scaling (N=400) | `python src/exp_scaling.py 400 20` | .venv | results/scaling/*.json | docs/05 |
| 4 | OR-Library MED vs published (Cura metric) | `python src/exp_orlib_med.py port1..5` | .venv | results/orlib_med/med_results.json | docs/07,08 |
| 5 | SCIP exact MED floor (incl N=225) | `python src/exp_scip_exact.py port1..5` | .venv | results/orlib_med/scip_exact.json | docs/08,10 |
| 6 | **Gset reproduction (original DGL QRF-GNN)** | `python src/qrfgnn_dgl.py "Gset/G14*" 20 6000` | .venv-dgl | results/maxcut/repro_*.log | docs/09 |
| 7 | Portfolio via ORIGINAL method (stage 1 select) | `python src/qrfgnn_select.py port1..5` | .venv-dgl | results/qrfgnn_portfolio/*_supports.json | docs/08 |
| 8 | Portfolio ORIGINAL method (stage 2 MED eval) | `python src/qrfgnn_eval.py port1..5` | .venv | results/qrfgnn_portfolio/med_eval.json | docs/08 |
| 9 | Alt formulation: sector caps | `python src/exp_sectors.py 300 15 45 4` | .venv | results/sectors/*.json | docs/11 |
| 10 | Alt formulation: fixed-charge tx-cost | `python src/exp_fixed_charge.py 100 2e-4` | .venv | results/fixed_charge/*.json | docs/11 |
| 11 | Live backtest (DOW/S&P100, Sharpe etc.) | `python src/backtest.py sp100 15 63 2005-01-01 hist` | .venv | results/backtest/*.json | docs/13 |
| 12 | Backtest + ML-μ forecast | `python src/backtest.py sp100 15 63 2005-01-01 ridge` (or `hgb`,`mom`) | .venv | results/backtest/*_{ridge,hgb,mom}.json | docs/13 |
| 13 | **Ablation (hyperparams/penalties/formulations)** | `python src/ablation.py` | .venv | results/ablation/ablation.json | docs/14 |
| 14 | Scaling QUBO solver benchmark (N=500,1000) | `python src/exp_scaling_qubo.py 500,1000` | .venv | results/scaling_qubo/scaling.json | docs/14 |
| 15 | **Amortized GNN (the win)** | `MODE=sup K=15 python src/amortized.py` | .venv | results/amortized/*.json | docs/15 |
| 16 | Cardinality on new datasets (French49/NASDAQ100/crypto) | `python src/exp_datasets.py` | .venv | results/datasets/cardinality_extra.json | docs/16 |
| 17 | MPE metric vs Chang/Deng/ARO | `python src/orlib_mpe.py` | .venv | results/orlib_med/mpe*.* | docs/18 |
| 18 | Index tracking (cardinality) | `python src/exp_index_tracking.py` | .venv | results/index_tracking/ | docs/17 |
| 19 | Downside-risk (semivariance) | inline (see docs/17) | .venv | results/datasets/semivariance.json | docs/17 |
| 20 | **Reproduce PI-GNN (their code) vs ours** | `python competitors/run_pignn_portfolio.py` | .venv-dgl | results/competitors_pignn.log | docs/19 |
| 21 | **Amortized OOD transfer** (train S&P100, test NASDAQ100/French49) | `python src/amortized_transfer.py` | .venv | results/amortized/transfer.json | docs/15 |
| 22 | **CVaR (scenario) cardinality** vs exact MILP | `python src/exp_cvar.py` | .venv | results/cvar/ | docs/17 |
| 23 | Per-instance scaling N=1500,2000 | `TL=60 python src/exp_scaling_qubo.py 1500,2000` | .venv | results/scaling_qubo/run_large.log | docs/14 |

## Core library modules (`src/`)
- `qubo.py` — QUBO container, energy, 1-flip local search (verified delta formula).
- `qubo_portfolio.py` — selection QUBO (cardinality), sector-cap QUBO, weight-encoded QUBO, penalties.
- `gnn_model.py`, `gnn_solver.py` — PyG QRF/PI-GNN + explore→exploit + seeded-SA refine solver.
- `qrfgnn_dgl.py` — **verbatim original DGL QRF-GNN** (faithful paper reproduction).
- `baselines.py` — Gurobi MIQP, SCIP MIQP, SCIP global QUBO, SA, tabu, greedy, convex reweight, frontier.
- `orlib_metrics.py` — exact Cura (2009) MED/VRE/MRE.
- `portfolio_data.py` — OR-Library loader + yfinance; `backtest.py` — walk-forward engine + strategies.
- `mu_forecast.py` — ML return forecaster (ridge/HGB) + momentum + Ledoit-Wolf.
- `amortized.py` — amortized (supervised/unsupervised) GNN across instance streams.

## Environments
- `.venv` — torch 2.6 cu124 + PyG + cvxpy + scip + neal + tabu + gurobipy + yfinance (main).
- `.venv-dgl` — torch 2.3 cu121 + dgl 2.2.1 (CPU) — the verbatim original method.
- `.venv-jax` — thrml (assessed competitor). `requirements.txt` freezes `.venv`.

## Reproducibility notes
- Random seeds fixed in each script (seed=0/1). GNN on GPU (RTX 3090).
- DGL on CPU (Windows has no CUDA DGL wheels) — cut quality device-independent.
- Free Gurobi license caps MIQP ~N<225; SCIP (free, no limit) used beyond that.
- All raw numbers are in `results/**` (JSON) with `.log` transcripts; figures in `results/frontier/*.png`.
| 24 | Amortized robustness (5 seeds) | `python src/amortized_seeds.py` | .venv | results/amortized/seeds.json | docs/15 |
| 25 | **Big real S&P500 (N=461) cardinality vs exact** | `python src/exp_large_real.py 30 750` | .venv | results/large_real/ | docs/21 |
| 26 | Bigger CVaR (real S&P500, 2514 scen) | `python src/exp_cvar.py` (+ inline sp500) | .venv | results/cvar/sp500_big.log | docs/21 |
| -- | **Paper figures** | `python src/make_figures.py` | .venv | results/figures/*.png | paper/ |
| -- | **LaTeX paper (compile)** | `cd paper && ./tectonic.exe main.tex` | tectonic | paper/main.pdf | paper/ |

## Autonomous-loop experiments (E-series) — experiments/ + experiments/results/
| ID | What | Script | Result |
|---|---|---|---|
| E5 | CRA-annealing per-instance (neg) | experiments/arch_lab5_cra.py | breaks +LS; tabu still wins |
| E6 | penalty-free encoding (neg) | experiments/arch_lab6_penaltyfree.py | no ranking gain |
| E8 | amortized @scale N=461 +curve+ckpt | experiments/e8_amortized_scale.py | 0.13% gap, 1276x |
| E9 | wide Optuna SAGE/GAT/GraphConv | experiments/e9_optuna_wide.py | SAGE>GAT>GraphConv |
| E11 | hardness sweep | experiments/e11_hardness_sweep.py | tabu 0% all hardness |
| E12 | amortized on hard random (neg) | experiments/e12_amortized_hard.py | 62% (needs related) |
| E13/E14 | amortized CVaR + OOD | experiments/e13_amortized_cvar.py, e14_* | 0.17% / OOD 0.48% |
| E15 | amortized in live backtest | experiments/e15_amortized_backtest.py | Sharpe parity ~840x |
| E16/E17 | amortized index-tracking (neg) | experiments/e16_*, e17_* | 45-48% (idiosyncratic) |
| E18/E21 | warm-start hybrid (+scale) | experiments/e18_warmstart_tabu.py, e21_* | 0.025%@18x / 0.035%@10x |
| E19 | CRA label-free amortized | experiments/e19_cra_amortized.py | 13% (vs 71% plain) |
| E22 | sector-cap multi-constraint | src/exp_sectors.py 200 20 60 5 | greedy-native=exact |
| E23 | multi-seed backtest (error bars) | experiments/e23_amortized_seeds_backtest.py | Sharpe 0.862±0.002 |
| E24 | iterative-refine (neg) | experiments/e24_iterative_refine.py | flat/worse |
| E25 | weight-encoded/integer-lot QUBO | experiments/e25_weight_qubo.py | tabu/SA 0%, SCIP fails |
| C1-C4 | modern-ML competitors (DiffOpt/DRL/E2E-DRO/GNN-decision) | experiments/competitors/c1-c4 | optimizer-vs-investor |
| HAMD | our solvers on HAMD 2026 instance | experiments/hard_instance_compare.py | GNN 1.1%, greedy 31% |
| DSL | their 2025 code on our S&P100 | competitors/DSLwDE + eval_dsl.py | Sharpe 0.672 |
| qqa | PQQA (their pip pkg) on our QUBO | experiments/competitors/run_qqa.py | 17-34% (GNN+LS beats) |
Envs: .venv (main), .venv-dsl, .venv-qqa, .venv-jax. Lit: papers/lit_2025_2026/ (130 PDFs), docs/25.
