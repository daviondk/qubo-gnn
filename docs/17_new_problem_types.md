# Axis B — new problem types beyond mean-variance cardinality

## B.1 Index tracking (cardinality): pick K assets + weights to replicate a benchmark
Objective = tracking-error variance `(w−b)ᵀΣ(w−b)`, benchmark b = equal-weight of all N assets.
`src/qubo_portfolio.tracking_qubo`, `src/exp_index_tracking.py`. Metric = annualized tracking error %
and optimality gap vs SCIP-exact. 2026-relevant (Mancilla/THRML index tracking; Dhingra 2026 review).

| Instance (N,K) | SCIP exact ann.TE% | **GNN** gap | Tabu gap | SA gap |
|---|---|---|---|---|
| port2 (85,10) | 10.74 | **8.3%** | 17.8% | 38.2% |
| port4 (98,10) | 10.80 | 35.3% | 12.1% | 23.3% |
| french49 (49,10) | 3.24 | 160% | 133% | 125% |
| nasdaq100 (66,10) | 5.28 | 128% | 81% | 92% |

**Honest finding:** index tracking is **harder for all QUBO heuristics** than mean-variance — the
**exact solver (SCIP) clearly dominates** (gaps 8–160%), and the **GNN is inconsistent** (beats tabu on
port2, worse on the other three). Root cause: the **equal-weight selection surrogate** `w=z/K` is a poor
proxy for the true tracking error (which needs optimally-fitted weights per candidate support), so
GNN/SA/tabu all pick suboptimal supports. **Not a GNN win;** for tracking, prefer exact MIQP or a
weight-encoded QUBO. A clean, documented negative.

## B.2 Downside-risk (semivariance) cardinality — same pipeline, downside objective
Swap the covariance Σ for the **semicovariance** (covariance of below-mean returns) in the selection
QUBO → targets downside risk (post-modern portfolio theory) instead of total variance. Reuses
`exp_cardinality.run_point` with Σ = semicov. `results/datasets/semivariance.json`.

| Dataset (N,K) | MIQP exact | **GNN** | Tabu | Greedy | SA |
|---|---|---|---|---|---|
| french49 (49,10) | 0.000 | **0.000%** | 0.000% | 0.000% | 2.34% |
| nasdaq100 (66,10) | 0.000 | **0.000%** | 0.000% | 0.000% | 0.000% |

**Finding:** on downside-risk (semivariance) cardinality the GNN **matches the exact optimum** (= Tabu =
Greedy), beating SA — same benign pattern as mean-variance, because the objective is still a quadratic
form. So switching the *risk measure* (variance → semivariance) does not change the verdict; switching
to a *non-quadratic / poor-surrogate* objective (index tracking) does (B.1).

## B.3 CVaR (scenario-based, Rockafellar–Uryasev) cardinality — a win regime at scale
min CVaR₉₅%(loss) s.t. Σw=1, cardinality=K, w≥0. `src/exp_cvar.py`. Methods: exact mean-CVaR **MILP**
(SCIP: binary z + w + η + per-scenario slacks), hybrid (downside-risk/semivariance **selection QUBO**
by GNN or tabu → **CVaR-LP** weights on the support, cvxpy), equal-weight. Metric = achieved CVaR.

| Instance | exact MILP (SCIP) | EqualWeight | Tabu+CVaR-LP | **GNN+CVaR-LP** |
|---|---|---|---|---|
| french49 (N49, 400 scen) | **0.01629** (opt, 0.9 s) | 0.02683 (+65%) | 0.01764 (+8.3%) | 0.01764 (+8.3%) |
| nasdaq100 (N66, 400 scen) | **0.01100** (opt, 30.7 s) | 0.01993 (+81%) | 0.01242 (+12.9%) | 0.01238 (+12.6%) |
| **synthetic (N200, 3000 scen)** | 0.00543 (**TIMEOUT 120 s, gap 4.15**) | 0.01411 (+174%) | **0.00516 (best)** | **0.00538 (beats exact)** |

**Findings:**
- At small scale (N≤66, 400 scenarios) the exact MILP solves to optimality and wins; the hybrid is
  ~8–13% off (the semivariance selection surrogate is an imperfect proxy for CVaR), GNN ≈ tabu.
- **At scale (N=200, 3000 scenarios) the exact MILP TIMES OUT** (120 s, 4.15 gap) and **both hybrids
  beat its incumbent in ~0 s** — Tabu+CVaR-LP is best (0.00516), GNN+CVaR-LP (0.00538) also beats the
  timed-out exact (0.00543). **This is a genuine win regime for the QUBO-select + CVaR-LP hybrid over
  the exact solver.** GNN is a competitive selector (beats exact; marginally behind tabu).
- Combined with amortization (docs/15): an amortized GNN selector would produce these CVaR selections
  in **ms** across a stream of scenario sets — compounding the advantage over the timing-out MILP.

## B.4 Discrete lots — scoped
- **Discrete lots** (integer holdings) ≈ the `weight_qubo` formulation (binary weight bits + budget
  penalty) already in `qubo_portfolio.py` — a discrete-weight QUBO; at small N exact MIQP dominates
  (cf. crypto in docs/16), so deferred as not a GNN-win regime.
- **CVaR / scenario** (Rockafellar–Uryasev) is an LP/MILP with per-scenario slack variables + a VaR
  level η; mapping to QUBO needs slack/η binarization and many scenarios → large QUBO where exact MIP
  genuinely slows (the most promising "hard" regime). Scoped as a larger build; the **semivariance**
  variant (B.2) is the tractable downside-risk proxy delivered here.

## Net (axis B)
Across index tracking (B.1) and downside risk (B.2): the GNN-QUBO **remains competitive on quadratic-
risk selection but does not beat exact/tabu on the harder tracking objective** — the equal-weight
surrogate is the bottleneck. Consistent with the overall verdict: GNN matches/【beats SA】 on
mean-variance-type QUBOs; exact/tabu win where the surrogate is a poor proxy.

## CVaR amortization — realized backtest nuance (E32)
The amortized CVaR selector matches the per-instance hybrid on the CVaR SELECTION objective (0.17% gap,
E13) and on realized OOS TAIL RISK (CVaR5 -0.0210≈-0.0209, MaxDD -0.279≈-0.283) at ~470x speedup, but its
realized OOS SHARPE lags ~14% (0.666 vs 0.774) — unlike mean-variance where amortized = per-instance
exactly (E15). Honest: CVaR amortization is strong on selection + tail risk, imperfect on realized Sharpe.

## B.5 Min-variance regime (high lambda) is EXACT-HARD (E46/E47/E48)
At high risk-aversion (lambda>=0.9, risk-dominated, dense-covariance selection) the cardinality QUBO is
genuinely hard: SCIP global-QUBO TIMES OUT 8/8 @60s (64% gap, E48); tabu can fail at lambda=0.9 (17.7%,
E47, verified non-artifact) though it recovers at 0.99 (0.33%). GREEDY and GNN+LS are robust (0.12-2%).
GNN-amortization beats linear-amortization here by 3-6 pts (E46) -- the regime where graph message-passing
earns its keep (pairwise z'Sigma z is graph-structured). Practical: for min-variance cardinality, prefer
robust heuristics (greedy / GNN+LS / well-tuned tabu) over exact MIP; the GNN's graph structure adds value.

## B.6 Practical recommendation: greedy+tabu warm-start (E49)
The most regime-robust per-instance solver is GREEDY-WARM-STARTED TABU (greedy init + ~8 tabu reads):
worst-case gap 0.5% across lambda in {0.5,0.9,0.99}, vs cold-tabu 20.8% (fails @0.9), greedy 2.0%, GNN+LS
2.0%. Cheap + robust. The GNN remains competitive-but-not-necessary per-instance (its value: amortization
+ risk-dominated amortization edge over linear, E46).
