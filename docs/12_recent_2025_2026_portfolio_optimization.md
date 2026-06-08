# 2025–2026 portfolio OPTIMIZATION literature: datasets, metrics, comparable numbers

Two verified sweeps (2026-06-04), ML and non-ML, focused on the optimization task with concrete
datasets + numbers, so we know where to benchmark against the latest analogues. All arXiv IDs fetched
and confirmed. Paywalled/unparseable numbers flagged — none invented.

## ⚠ Key practical finding
**OR-Library port1–5 is NOT used in any 2025–2026 ML portfolio paper** — the ML field moved to live
**S&P 500 / S&P 100, DOW 30, CSI 300, Russell, NASDAQ, French 49-Industry** with backtest metrics
(Sharpe/Sortino/MaxDD/turnover). OR-Library + MED is still standard only for **classic metaheuristics/
exact** cardinality work. ⇒ To compare with the *latest* analogues we must add live-data backtests
(Sharpe etc.); to compare with the *classic cardinality* line we keep OR-Library MED (done, docs/08).

## A. ML / learning approaches to portfolio optimization (2025–2026)
Rigor: 🟢 exact/convex baseline · 🟡 classical heuristic baseline · 🔴 only beats equal-weight/index.

| Paper (arXiv) | Method | Dataset(s) | Headline numbers | Rigor |
|---|---|---|---|---|
| Ranabhat 2025 (2507.07159) | Variational **neural annealing**, MINLP (tx-cost, turnover, vol-cap) | S&P500(478), Russell1000(857), Russell3000(2008); 10y monthly | TTS 2580s vs **Mosek** 6709s (R3000); Sharpe within ~18% of Mosek | 🟢 Mosek |
| Hwang & Zohren 2025 (2510.03129, SIT) | Signature **Transformer**, max-Sharpe E2E | S&P100, DOW30, CSI300; 2000–16/17–19/20–24 | Sharpe **0.6717** (40-asset S&P100) vs EWP 0.5759; Sortino 0.823, MaxDD 0.361 | 🟢/🟡 GMV+CVaR |
| Lee 2024 (2409.09684)* | **Decision-focused** MV (diff. opt layer) | DOW30, S&P100; 2010–24 | Sharpe **1.302** (DOW30), 0.801 (S&P100) | 🟢 exact MV layer |
| Lozano 2026 (2605.17628) | **Penalty-free QUBO** (D-Wave) cardinality | French 49-Industry (N≤49) | regret 0.027% (N=49); *"no quantum advantage"* | 🟢 brute-force ≤N24 |
| Sood (JPMorgan) 2026 (2602.17098) | Model-free **DRL** vs MVO | not stated ⚠️ | DRL Sharpe 1.17 vs MVO 0.68 ⚠️unverified | 🟡 MVO |
| Gao 2025 (2509.22088, FactorDiff) | Conditional **diffusion**, MV/CVaR | CSI300 (113), 2017–25 | daily Sharpe 0.105 ⚠️un-annualized | 🟡 shrinkage |
| Fernandes 2026 (2605.28853) | E2E diff. Sharpe/Omega/CVaR/RP | 50 S&P500, 2007–23 | Sharpe 0.29 vs S&P −0.02 | 🔴 |
| Mancilla 2026 (2601.07792, THRML) | **Energy-based/Ising** index tracking | 100 S&P500, 2023–25 | tracking error 4.31% vs 5.66–6.30% | 🔴 |
| Kashif 2026 (2605.17307) | **SAC DRL** global equities | Nasdaq100, Nikkei225, EuroStoxx50 | **no significant alpha vs B&H** (honest null) | 🟢 null |
| Park 2026 (2603.19288) | Joint return/risk **DNN** | 10 US large-caps | Sharpe 0.91, ret 36.4% | 🔴 |
| Ozechi 2026 (2604.24486) | GNN/DRL/Transformer/AE comparison vs MVO | equities+ETFs+bonds 2015–23 | ⚠️ numbers in PDF only | 🟡 MVO |
| Wade 2026 (2605.19278) | **GraphSAGE** vol-forecast → portfolio | 465 S&P500, 2015–25 | "best forecast ≠ best Sharpe"; code on GitHub | 🟡 |
\* 2024, included as the canonical decision-focused MV anchor.

## B. Non-ML (exact / metaheuristic / quantum) (2025–2026)
| Paper (arXiv/DOI) | Method | Dataset & size | Exact solver? | Numbers |
|---|---|---|---|---|
| **Stopfer & Wagner 2025 (2509.17876)** ⭐ | Benchmark: QAOA/QA vs classical | **Nasdaq 1978 assets, 2020–23; 250 inst., n≤1000** | **Gurobi+SCIP** | Gurobi **all optimal in seconds**, >1000× SCIP; QA≈random; "very limited room for quantum advantage" |
| Decomp. pipeline 2025 (2409.10301, PRR 7:023142) | RMT+spectral clustering decomposition | large real PO | **Gurobi** | ~80% subproblem size reduction |
| Gomez Cadavid 2026 (2602.23976) | Trapped-ion BF-DCQO | S&P500 250 assets, K=125 | Gurobi (claimed) | 11–14 clusters; gap tables ⚠️not extractable |
| Mancilla 2026 (2602.14827) | QAOA XY-mixer direct indexing | 10 US eq., K=5, 2025 | **No** (SA,HRP) | Sharpe 1.81 vs SA 1.31 vs HRP 0.98; turnover 76.8% |
| Weinberg 2026 (2603.16904) | Walk-forward QUBO via QAOA | S&P500 10 stocks, test 2025 | **No** | Sharpe 0.588 vs GA 0.575; −44.5% tx cost |
| Wang 2025 (swevo 102162) | Improved **Dung Beetle** metaheuristic | **OR-Library port1-5** + 6 sets | not mentioned | MED/VRE/MRE ⚠️paywalled |
| Jiang 2025 (NRL 72:825) | Sparsity-driven **exact MIQP** | CCMV | B&B context | gap/runtime ⚠️paywalled |

## C. Standard public datasets (ranked) + metric conventions
**Datasets to benchmark on (by frequency in 2025–26):**
1. **S&P 500 / S&P 100** (dominant; sizes 10→500; Yahoo/WRDS). 2. **CSI 300** (Chinese A-share).
3. **DOW 30** (small, clean — best sanity check). 4. **Russell 1000/3000** (large-scale stress, N≤2008).
5. **French 49-Industry** (fully public, ideal for cardinality/QUBO). 6. **NASDAQ (1978 assets)** —
Fraunhofer large-scale generator. 7. **Nasdaq100/Nikkei225/EuroStoxx50** (multi-market robustness).
8. **OR-Library port1-5** — still standard for classic cardinality/exact (MED/VRE/MRE).

**Metrics to report:**
- Backtest/OOS: **Sharpe, Sortino, annualized return, volatility, MaxDD, turnover, tx-cost-adjusted
  return, final wealth / Calmar.**
- Cardinality (OR-Library): **MED / VRE / MRE** vs unconstrained frontier (51 pts), + runtime, +
  proven optimality gap when exact available.
- Quantum/QUBO (Fraunhofer convention): **approximation ratio Θ≥1, feasibility %, #feasible samples in
  fixed time budget, runtime**, ALWAYS with a Gurobi/SCIP exact baseline.

## D. Cleanest reproducible targets to compare against (exact numbers above)
1. **Ranabhat 2025 (2507.07159)** — solver-quality / optimality-gap vs **Mosek** on public
   S&P500/Russell. ⭐ Best match for our QUBO/physics-inspired-solver angle (TTS + Sharpe-gap).
2. **Hwang & Zohren 2025 (SIT, 2510.03129)** — risk-adjusted allocation; 3 public indices, fixed
   splits, full Sharpe/Sortino/MaxDD/final-wealth tables vs convex optimizers.
3. **Lozano 2026 (2605.17628)** — honest cardinality-QUBO with brute-force exact + explicit regret %.
4. **Stopfer & Wagner 2025 (2509.17876)** — the quantum/QUBO benchmark standard (Gurobi baseline).
5. **Lee 2024 (2409.09684)** — decision-focused MV reference numbers (DOW30/S&P100).

## E. Recommendation for OUR work (to be comparable to the latest)
1. Keep the **OR-Library MED** result (docs/08) — comparable to the classic cardinality line.
2. **Add live-data backtests** on **S&P 100 + DOW 30 + CSI 300** with the SIT splits (2000–16/17–19/
   20–24) reporting **Sharpe/Sortino/MaxDD/turnover** → directly comparable to Hwang & Zohren 2025 and
   the DRL/diffusion papers.
3. For the solver-quality angle, replicate the **Ranabhat setup** (S&P500/Russell, tx-cost MINLP) and
   report **time-to-solution + Sharpe-gap vs Mosek/Gurobi/SCIP** — the cleanest "are we competitive
   with the exact solver" comparison.
4. Use **French 49-Industry** for the cardinality-QUBO comparison vs Lozano 2026 (penalty-free).

> Verification: ~30 source fetches across both sweeps; IDs 2502–2605 = Feb 2025–May 2026. Unverified
> (paywalled/PDF-locked): Sood Sharpe 1.17/0.68; Wang & Jiang per-dataset numbers; PCE (2511.21305);
> several "numbers in PDF only" rows. Full APA-7 lists in the conversation transcript.
