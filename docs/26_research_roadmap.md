# Research roadmap & playbook — learned GNN-QUBO solvers for cardinality-constrained portfolios

The single go-to document for continuing this research. Synthesizes everything: our results (docs/01–24,
experiments/LOG.md) + the 2025–2026 literature sweep (docs/25, `papers/lit_2025_2026/`). Tells you **how
to compare, what to measure, on what, against whom, what to build, what to run, and in what order.**

---
## 0. The one-paragraph thesis (what the whole project is about)
Cardinality-constrained portfolio optimization has two distinct sub-problems that the field routinely
conflates: **(A) the optimization problem** — given (μ, Σ), find the best K-asset portfolio under a stated
objective; and **(B) the investment problem** — make money out-of-sample under estimation error. Our
learned **GNN-QUBO solver wins (A)** (matches exact MIP, beats SA/PI-GNN, + amortized ms inference), while
**decision-focused ML/RL wins (B)** (higher OOS Sharpe). The defensible contribution is: a **scalable,
amortized, honestly-benchmarked discrete-selection solver** for (A), plus the **clean separation of (A)
from (B)** via an optimality-gap-vs-MIP column that nobody else reports. Do **not** market "GNN beats
everyone".

---
## 1. How to compare (evaluation protocol — do this every time)
1. **Always report BOTH axes** for any method:
   - **Optimization quality:** optimality gap vs exact (Gurobi/SCIP MIQP/MILP) on the *stated* objective,
     %; plus wall-clock and per-instance vs amortized inference time.
   - **Investment quality:** out-of-sample walk-forward backtest metrics (§2) net of transaction costs.
2. **Walk-forward, never in-sample only.** Lookback 252d, rebalance every 21–63d, train/test split by
   time (no leakage). Charge transaction cost on turnover at each rebalance.
3. **Multi-seed everything stochastic** (≥5 seeds, report mean ± std). GPU is nondeterministic — a single
   seed misleads (we learned this: C4 single-seed 1.16 → validated 1.01 ± 0.06).
4. **Same instances across all methods** (same μ̂, Σ̂, K, constraints, cost). Use the metric-complete
   harness `experiments/competitors/c1_diffopt.py` as the template.
5. **Hard instances, not easy ones.** Mean-variance cardinality is easy (exact solves N≤1000 in seconds,
   Stopfer&Wagner 2509.17876). To test solver quality, use genuinely hard regimes (§4) or report honestly
   that it's easy. Follow MaxCut-Bench-style rigor (2406.11897) for any QUBO sanity check.
6. **Statistical significance:** Diebold-Mariano / paired tests on returns; report p-values (THRML
   2601.07792 and Fernandes 2605.28853 do this).
7. **Honesty rules (IRON):** relaxed loss ≠ binarized quality (Bu&Shin 2506.16732) — always evaluate the
   *decoded* solution; never renormalize an infeasible QUBO solution (the original-notebook artifact);
   report negative results; state when a baseline beats us.

---
## 2. What to measure (the metric set — report ALL that the field reports)
**Optimization metrics (our distinctive axis):**
- Optimality gap vs exact MIP/MIQP (%) — the column Li 2024 / FinRL / DFL / quantum papers omit.
- Energy/objective gap to best-found; feasibility (|S| = K?); time (per-instance & amortized).
- For OR-Library: **MED** (mean Euclidean distance to unconstrained frontier, Cura 2009) + **MPE %**
  (Chang 2000) — the standard cardinality-portfolio metrics.

**Investment metrics (the field's standard set — match these for comparability):**
- **Annualized return** (gross & **net-of-cost**), **Sharpe**, **Sortino**, **Max Drawdown**, **turnover**,
  **OOS CVaR(95%)/Expected Shortfall**, final wealth. (Li 2024, FinRL, E2E-DRO, Fernandes 2605.28853.)
- For index tracking: **tracking error** (THRML 2601.07792: 4.31% TE benchmark).
- Increasingly expected: EVaR / DRO-robust metrics (DeePM 2601.05975), regime/stress tests (DARL).

**What "better for MY task" means (decision rules):**
- A solver is a **better optimizer** iff lower optimality gap at equal-or-less time (or much faster at
  equal gap → the amortization win).
- A method is a **better investor** iff higher net-of-cost Sharpe/Sortino with ≤ turnover and ≤ tail risk
  (CVaR/MaxDD), multi-seed, statistically significant.
- Our headline claims must be one of: (i) match exact at far lower amortized cost; (ii) win where exact
  times out (scenario CVaR at scale); (iii) honest parity/loss elsewhere.

---
## 3. Benchmarks & datasets to use
**Cardinality-portfolio (optimization) — standard:**
- **OR-Library port1–5** (Hang Seng 31 / DAX 85 / FTSE 89 / S&P100 98 / Nikkei 225) — THE standard; port5
  = Krylova's Nikkei. Use MED + MPE + exact gap. (We have these.)
- Extended sizes: **French 49-Industry** (Lozano 2026 2605.17628 dataset — direct head-to-head), large real
  universes **S&P 500 (~460)**, **NASDAQ-100**, **Russell 1000/3000** (VNA 2507.07159 scale). (S&P500/NASDAQ
  downloaded; Russell needs data access.)
**Investment (backtest) — standard:**
- **DJIA-30** (FinRL canonical), **S&P 100 / S&P 500**, **NASDAQ-100**, the **5-index set** (DAX/HSI/FTSE/
  S&P/Nikkei, Li 2024 — same as OR-Library universes), **CSI 300** (China), crypto. Date ranges 2005–2025.
**Hard / scenario:**
- Scenario CVaR: real returns + bootstrap/Monte-Carlo scenarios (we use S&P500 + up to 8000 scenarios).
- Multi-period + tx-cost + integer lots: Chen&Koch 2502.05226 setup (499 S&P, integer lots, short-sale).
**CO sanity (method validation):** Gset MaxCut (best-known, optimum unknown), MaxCut-Bench (2406.11897),
hard-CSP benchmark (Angelini 2026 2602.18419).

---
## 4. Where the problem is actually HARD (where a learned solver can win)
Mean-variance cardinality is easy (exact wins). Genuinely hard regimes — focus here:
1. **Scenario CVaR / Expected Shortfall at scale** (many scenarios × many assets) — exact MILP times out
   (we showed N=200/3000 scen: hybrid beats timed-out MILP). Push scenarios↑, add cardinality.
2. **Integer lots + transaction costs + turnover + sector caps simultaneously** (non-convex MINLP;
   Chen&Koch 2502.05226). LP/MIQP relaxations stop being tight.
3. **Robust / distributionally-robust / multi-period** (EVaR, worst-case scenarios; DeePM, E2E-DRO).
4. **Very large N** (2000+; Russell 3000) where even MIP/Mosek slow (VNA 2507.07159 territory).

---
## 5. Who to compare against (baseline ladder)
**Exact (optimality reference):** Gurobi MIQP (free ≤~225), **SCIP** (free, any N), Mosek (conic, large).
**Classical heuristics:** simulated annealing (dwave-neal), **tabu** (dwave-tabu) — the tough per-instance
baseline, greedy forward selection, convex reweight, 1/N (DeMiguel 2009 — often beats MV OOS!).
**Published cardinality metaheuristics:** GA/TS/SA/PSO (Cura 2009), IPSO-SA (Mozafari 2011), Firefly
(Bacanin 2014), ARO — compare on MED/MPE.
**Modern ML/DL (the ones the field uses — we built/should add):**
- Decision-focused / differentiable-opt: **DiffOpt** (built), **E2E-DRO** (Costa&Iyengar, code), IPMO
  (2512.11273), SPO (2601.04062), DSL (2503.13544, code), Anis&Kwon cardinality DFL (closed).
- Constrained / safe RL: **our DRL** (built), FinRL/POE (code), Li 2024 safe-RL (closed).
- Generative: diffusion factor/scenario models (2504.06566 code, 2509.22088).
**Learned QUBO/Ising solvers (closest competitors — ADD these):**
- ⭐ **VNA** (Variational Neural Annealing, 2507.07159) — learned annealer, 2000+ assets, the most direct
  learned-solver-at-scale comparator.
- ⭐ **THRML EBM** (2601.07792) — cardinality index-tracking as Ising sampling; near-identical task.
- ⭐ **Lozano penalty-free** (2605.17628) + **Gurobi audit** (2605.17623) — French49 head-to-head, reports gap.
- **QRF-GNN** (2407.16468, our core), **PI-GNN** (official, collapses), X²GNN, GCON, DiffUCO.
**Quantum (context, not our fight):** D-Wave hybrid, QAOA/XY-mixer/Dicke (2602.14827), VQE-CVaR (2508.18625).

---
## 6. Architectures & improvements to try (to beat our current solver)
Current solver = QRF-GNN (SAGE + recurrent + PageRank) + explore→exploit + 1-flip LS + seeded-SA;
amortized = supervised SAGE (Optuna-tuned: dropout + kNN-12 + basic feats → 0.39% test / ~0 OOD).
Known weakness: **bare unsupervised GNN collapses on dense covariance QUBO (p=0 saddle, Exp1)** — LS carries it.

Priority upgrades (each maps to a 2025-26 paper; test in `experiments/`):
1. ⭐ **Fix the relaxation→binary collapse** (highest value, directly our Exp1):
   - **CRA — continuous-relaxation-annealing** (Ichikawa 2309.16965, code) — annealed penalty, rounding-free.
   - **Binarizing PI-GNN** (Krutský 2507.13703) — fuzzy-logic/BNN binarization for dense graphs.
   - **Gini-coefficient annealing** (Deep k-grouping 2505.20972).
   - **Learned rounding** (2505.13405) instead of threshold/top-K.
2. ⭐ **Dense-graph-aware backbone:**
   - Heterophily framing (QUBO-GNN 2404.04874) + **multi-filter spectral** (GCON 2405.20543, code).
   - **State-space / parameter-free anti-oversmoothing** (2502.10818); Mamba global context (2511.06756);
     linear-cost global nodes (RANGE 2502.13797).
   - **Degree-based positional encodings** (benchmark says best on dense graphs, 2411.12732).
3. **Cardinality-aware training** done right (our Exp2 knorm failed): use principled UCO cardinality
   handling (Bu et al. 2405.08424) + derandomization-in-training (Bu&Shin 2506.16732).
4. **Explore/exploit refinement in the loop** (X²GNN) — fold LS-like refinement into the neural step.
5. **Amortization upgrades** (our real win): stronger/exact teacher labels, larger N (where tabu is
   costly → bigger speedup), multi-period rolling, OOD across more universes; wider Optuna (GAT/GIN,
   edge-feature convs, ensembles, more trials).
6. **Decision-focused GNN** (our C4) — promising but currently ≈ MLP; try richer reward (Sharpe/CVaR
   net-of-cost), GNN over correlation graph, multi-seed at larger N.
7. **Penalty-free encoding** (Lozano 2605.17628) — drop cardinality penalty (keeps graph sparse), enforce
   K by projection → directly attacks the dense-graph cause of the collapse.

---
## 7. Experiments to run (prioritized backlog, each with a hypothesis)
**Tier 1 — fix the solver / strongest scientific payoff:**
- E5: CRA-annealing vs Binarizing vs learned-rounding on dense portfolio QUBO (port4/port5/synth300,
  N up to 2000) — H: closes the bare-GNN gap from ~100% to single digits *without* LS. Metric: GNN-alone
  gap. (Extends `experiments/arch_lab.py`.)
- E6: penalty-free encoding + projection — H: bare GNN no longer collapses (sparse graph). Metric: bare |S|, gap.
- E7: heterophily/multi-filter backbone + degree-PE — H: improves GNN-alone ranking on dense graphs.
**Tier 2 — strengthen the amortization win:**
- E8: amortized at large N (S&P500/Russell-sample) with exact-quality teacher — H: speedup compounds,
  gap stays ~0; report ms inference vs tabu seconds.
- E9: wider Optuna (GAT/GIN/edge-conv/ensembles, 100+ trials, multi-objective test+OOD+robustness).
**Tier 3 — hard-regime wins (where exact fails):**
- E10: cardinality CVaR at scale (more scenarios, +tx-cost) vs VNA + THRML + timed-out MILP — H: amortized
  hybrid beats timed-out exact and ≈ VNA at lower cost. Metric: CVaR + gap + time + OOS TE.
- E11: integer-lot + tx-cost + sector MINLP (Chen&Koch setup) — H: exact times out, learned hybrid competitive.
**Tier 4 — comparison completeness:**
- E12: add VNA (2507.07159) + THRML (2601.07792, code) as baselines on our tasks; multi-seed all
  competitors (incl. NASDAQ) for full error bars in docs/24 table.
- E13: Gset MaxCut + MaxCut-Bench + hard-CSP (2602.18419) sanity — H: our GNN+LS ≈ best-known but tabu
  competitive (honest CO positioning).

---
## 8. Research plan (phased)
**Phase A (now) — fix & strengthen the core (Tier 1+2):** E5–E9. Goal: a GNN-QUBO solver whose *own*
output is good on dense graphs (not just via LS), + a hardened amortized model. Deliver: updated docs/14–15,
new figures, LOG entries.
**Phase B — hard-regime wins (Tier 3):** E10–E11 + add VNA/THRML baselines. Goal: a defensible "learned
solver wins where exact fails" result on a real hard portfolio class. Deliver: docs update + figures.
**Phase C — completeness & honesty (Tier 4):** E12–E13, multi-seed all, statistical tests. Goal: airtight
comparison table.
**Phase D — paper finalization:** fold A–C into `paper/main.tex`; reframe around (1) amortized discrete-
selection solver, (2) optimizer-vs-investor separation w/ optimality-gap column, (3) hard-regime win, (4)
honest CO positioning citing the 2026 debate papers. Target venue: a quant-finance/ML workshop or journal.

---
## 9. Positioning & novelty (what makes this publishable)
- **Gap filled:** no published *learned GNN-QUBO* solver for *cardinality* portfolios (only Krylova 2024
  which fails the constraint, + quantum hybrids). We make it work (selection-QUBO + LS + reweight) and
  amortize it.
- **Methodological contribution:** the **optimality-gap-vs-MIP** column + the **optimizer/investor
  separation** — exposes the field's "better solver ⇒ better portfolio" error (Li 2024/FinRL/DFL omit it).
- **Honesty as strength:** independent 2026 evidence (Angelini 2602.18419, TMLR 2502.03669, MaxCut-Bench,
  Bu&Shin) corroborates our negative findings — we report what's real.

---
## 10. Pitfalls / do-not-repeat
- Don't optimize the noisy in-sample MV objective and expect OOS gains (decision-focused wins there).
- Don't trust single-seed GPU runs. Don't renormalize infeasible QUBO solutions. Don't claim wins on easy
  mean-variance. Don't install cvxpylayers/diffcp into `.venv` (breaks numpy→2; use pure-torch diff-opt).
- Don't compare on the wrong objective (equal-weight surrogate vs financial objective — the N=461 mix-up).

Pointers: results `results/**`, `experiments/results/**`; lit `papers/lit_2025_2026/` + `docs/25`;
experiment index `docs/EXPERIMENTS.md`; competitor results `docs/24`; paper `paper/main.tex`.
