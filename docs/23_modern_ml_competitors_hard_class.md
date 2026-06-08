# Modern ML/DL/RL for the HARD, CONSTRAINED portfolio class — competitor map & honest positioning

Targeted survey (2023–2026) of learned methods for the *hard, constrained* portfolio class (cardinality
+ transaction costs + turnover + box/sector + CVaR/robust/multi-period), with datasets, metrics, code
status, and what we can actually compare against. Sources verified by fetching where possible.

## The honest strategic finding (most important)
The genuinely hard portfolio constraints — **CVaR/scenario, transaction costs, turnover, robust/DRO,
multi-period** — are **continuous and largely convex (or convex after reformulation)**. The field's
momentum AND all the reproducible code are in:
- **Differentiable convex-optimization layers** (cvxpylayers / OptNet) — end-to-end / decision-focused.
- **Constrained / safe deep RL** (PPO + safety layer / interior-point policy optimization).

**QUBO forces binarization that throws away this convex structure** → it is *not* the natural tool for
the continuous-constrained class. QUBO/GNN is natural only for the **discrete sub-structure**:
**cardinality selection and integer lots**. And there, exact MIQP (Gurobi) solves OR-Library-scale to
optimality in seconds, so a learned QUBO/GNN solver must justify itself on **scale / speed**, not
quality. ⇒ **Defensible positioning of our work:** a *scalable discrete-selection module* (cardinality /
integer lots) benchmarked on **optimality gap + runtime vs exact MIQP**, and (our real win) **amortized
throughput**. For the full hard class, the right comparators/tools are constrained-DRL + diff-opt, which
we should cite and (ideally) benchmark against — not claim QUBO beats them.

## Modern methods (verified)
### Constrained / safe RL
- **Li et al. 2024 (IJCNN)** — *Cardinality and Bounding Constrained Portfolio Optimization using Safe
  RL* (doi:10.1109/IJCNN60899.2024.10651491). PPO + interior-point (IPO) + safety-layer projection;
  cardinality(top-K) + box + transaction cost (0.05%); multi-period. **Data:** DAX30/HangSeng58/FTSE61/
  S&P100/Nikkei218, 2019-11→2023-10. **Metric:** annualized return, Sharpe. Beats 9 online-portfolio
  heuristics (SR gains +0.12…+1.33). **No exact-MIP baseline. NO public code.** ⇒ cite as SOTA safe-RL
  cardinality; not reproducible; uses the *same five index universes* as OR-Library (common ground).
- **FinRL / POE** (Liu et al.) — open-source constrained-DRL infra (A2C/DDPG/PPO/SAC/TD3),
  transaction-cost + turnover on DJIA-30. **Public code.** Solves the *easy* continuous class (no
  cardinality/CVaR/exact baseline). ⇒ usable reproducible DRL baseline.

### End-to-end / decision-focused / differentiable optimization
- **Anis & Kwon 2024/2025 (EJOR 322:273-288, doi:10.1016/j.ejor.2024.08.030)** — *End-to-end,
  decision-based, cardinality-constrained* MV; nonlinear MIP cardinality embedded via **three
  continuous relaxations as implicit differentiable layers**. The conceptually closest learned-
  cardinality method. **Paywalled; numbers unverified; NO public code found.** ⇒ key cite, not
  reproducible.
- **Costa & Iyengar 2023 (Quant. Finance 23:1465-1482, arXiv:2206.05134)** — *Distributionally-Robust
  End-to-End*; cvxpylayers; learns risk-tolerance + robustness radius. **Public code:
  github.com/Iyengar-Lab/E2E-DRO (PyTorch+cvxpylayers, runnable).** ⇒ **the best reproducible modern
  baseline** (robust sub-class).
- **Uysal, Li & Mulvey 2024 (Ann. OR, arXiv:2107.04636)** — e2e risk-budgeting implicit layer (no code
  confirmed). **Parra-Diaz & Castro-Iragorri 2025 (arXiv:2504.19980)** — deep declarative risk
  budgeting (7 ETFs; no code). **Hwang et al. 2025 (arXiv:2502.00828)** — decision-informed NN + LLM
  (S&P100/DOW30; no code). **Linghu et al. 2025 (arXiv:2512.11273)** — integrated prediction + multi-
  period w/ turnover/costs (no code).

### CVaR / scenario / robust with ML
- **Fernandes & Desell 2026 (arXiv:2605.28853)** — e2e NN with differentiable Sharpe/Omega + **CVaR** +
  risk-parity reg; 50 S&P500, OOS 2022-23; Sharpe 0.29 vs −0.02 (no code).
- **Iterative Problem-Driven Scenario Reduction for CVaR** (arXiv:2510.15251) — learned scenario
  reduction (adaptable). **Mean-CVaR via DNN** (EAAI 2025, paywalled).
- QUBO/quantum: **Morapakula et al. 2025 (arXiv:2504.08843)** — D-Wave CQM, cardinality+budget+sector+
  integer lots, tiny NSE/BSE 10–70 assets, self-disclaims advantage, no code.

## Standard datasets/metrics for the hard class (de-facto, no shared leaderboard)
- **Static cardinality MIQP:** OR-Library port1–5 (Hang Seng 31/DAX 85/FTSE 89/S&P 98/Nikkei 225);
  metric = deviation from unconstrained frontier + **optimality gap & runtime vs exact MIQP**.
  (Li 2024 uses the same five universes ⇒ natural common ground.)
- **Dynamic/DRL:** DJIA-30 (FinRL), S&P500/NASDAQ100, the five-index set; metrics = AR, Sharpe/Sortino,
  MaxDD, turnover, net-of-cost return, CVaR(5%); scenario counts ~250–2000.
- **e2e/DFL & robust:** Ken French factors (E2E-DRO), S&P100/DOW30, small ETF baskets; metric =
  decision regret / risk-adjusted net return vs two-stage.
- **Open gap we can fill:** very few learned-method papers report an **optimality gap vs exact MIP** —
  doing so honestly on OR-Library port1–5 (+ large real universes) is a defensible contribution.

## Comparison targets we can actually run
1. **E2E-DRO** (Costa & Iyengar) — public code; primary diff-opt/robust baseline.
2. **FinRL/POE** — public code; constrained-DRL baseline.
3. **Our own simplified modern baseline:** a neural net + **differentiable convex portfolio layer
   (cvxpylayers)** with a soft-cardinality/CVaR objective — a faithful-in-spirit DFL competitor we can
   put head-to-head with our GNN-QUBO+reweight hybrid on the same data/task.
4. Exact MIQP/MILP (Gurobi/SCIP) — the optimality-gap reference.

## Recommendation
Reframe the paper around the **defensible niche**: learned **discrete-selection** (cardinality/integer
lots) at scale, with honest exact-MIP gaps + amortized speed; and **benchmark against a modern diff-opt
(cvxpylayers) and/or constrained-DRL baseline** on a shared hard-constrained task — reporting where each
tool wins. Do NOT claim QUBO/GNN beats DRL/diff-opt on the continuous-constrained class.
