# Autonomous-loop findings (tasks.txt phase) — consolidated record

Running record of the continuous research loop (experiments E11–E19 + lit sweeps + direct-SOTA-code).
Full detail in `experiments/LOG.md`; this is the navigable summary.

## Per-instance QUBO solving — SETTLED (tabu dominates; GNN no niche)
- E11 hardness sweep (smooth factor-cov → frustrated): **tabu reaches best (0%) across the whole
  spectrum**; GNN ties greedy (20–37% behind tabu) on frustrated instances. Architecture is not the
  bottleneck — problem-easiness / local-search strength is.
- HAMD 2026 instance (n200,K40, harder cubic-derived structure): GNN 1.1% ≈ SA/tabu, **greedy fails
  (31%)**, SCIP-global times out — the GNN's SA-refine nears tabu but doesn't beat it.
- Confirms (with E5/E6/Exp1–3): no per-instance win over tabu on cardinality portfolio QUBOs.

## Amortization — the genuine win, now FULLY MAPPED
WORKS (related instance streams, risk-return selection):
- Mean-variance: 0.13% gap at scale N=461 (1276× speedup, E8); **live-backtest OOS Sharpe parity 0.861
  vs 0.863 at ~840×** (E15); OOD across universes (docs/15).
- CVaR(95%): 0.17% in-dist + OOD NASDAQ 0.48%/median 0.00% (E13/E14).
- **Warm-start hybrid (E18):** GNN-init + 4 short tabu reads → **0.025% gap at ~18× faster than cold
  tabu** — best quality/speed frontier point.
FAILS (boundaries):
- Unrelated random frustrated QUBOs: 62% (E12) — no shared structure to amortize.
- Index-tracking: 45–48% (E16/E17) — intrinsic idiosyncrasy; index-aware features don't fix it.
- Label-free (unsupervised) amortization: CRA-annealing helps a lot (71%→~13%, E19) but supervision
  (~1%) remains the recommended recipe.
Scope statement: *amortized learned selection works when the optimal K-set correlates with per-asset
risk/return features on a related instance stream; it fails for idiosyncratic (index-matching) or
unrelated instances; supervision beats unsupervised.*

## Architecture — settled
- E9 wide Optuna (60 trials): **SAGE > GAT > GraphConv** (0.042 vs 0.110 vs 0.230); best = SAGE +
  dropout 0.34 + kNN-12 + 4 layers. Attention/edge-weighting do not help. Curves+checkpoints saved.

## Direct SOTA-code comparison + reproducibility
- **DSL (2025, their public code) on our S&P100**: ensemble OOS Sharpe 0.672 (2019–24, 10bps,
  full-allocation) — provenance-honest direct run; below our cardinality solvers (different period/task).
- **Reproducibility gap:** the closest learned-solver SOTA (THRML 2026, VNA 2025) publish NO portfolio
  code (only generic field-free solver libs) — documented (docs/27).
- HAMD 2026 code execution blocked by classifier → used its instance with our solvers.

## Literature (docs/25, 128 PDFs in papers/lit_2025_2026/)
- 5 axes incl. QUBO↔GNN connections (Axis 5). Key actionable: Carathéodory exact-k projection
  (2510.24039), QQA4CO runnable solver, DyCO-GNN temporal warm-start, Free-Energy Machine.
- Tracked: ⭐ QRF-GNN successor (ICLR 2026, iterative refinement) — our core method's successor.

## Paper status
`paper/main.tex` (+ refs.bib + compiled main.pdf ~460KB) updated with: amortization across scale/
objectives/backtest + warm-start, modern-ML comparison + optimality-gap separation, direct-SOTA-code +
reproducibility gap, per-instance-tabu-dominance. Markdown twin docs/PAPER.md.

## Net contribution (defensible, honest)
1. A correctly-engineered amortized GNN-QUBO portfolio selector: train once, ms inference, ~tabu/exact
   quality, generalizes across scale/objectives/universes on related streams; warm-start accelerates the
   exact-quality solver ~18×.
2. The optimizer-vs-investor separation (optimality-gap-vs-MIP) exposing "better solver ≠ better
   portfolio".
3. Honest negatives: no per-instance win over tabu; amortization boundaries; SOTA reproducibility gap.

## UPDATE — per-instance fully explored; amortization rigor; new baselines (E22–E25)
- Per-instance GNN never beats best metaheuristic, across 4+ QUBO structures: binary cardinality (easy,
  all=exact), frustrated (E11 tabu 0%), sector-capped (E22 greedy-native=exact, QUBO-solvers handicapped
  by slack encoding), weight-encoded/integer-lot (E25 tabu/SA 0%, SCIP-global fails 20-31%, GNN ~1-2.5%).
- Architecture/formulation tricks all negative for per-instance: Rprop, kNN-sparsify, CRA (E5),
  penalty-free (E6), cardinality-aware (E2), GAT/edge-conv (E9), iterative-refinement (E24).
- Amortization rigor: backtest Sharpe 0.862±0.002 (5 seeds, E23) = per-instance 0.863; warm-start
  hybrid 0.025%@18x (E18) / 0.035%@10x at N=461 (E21); CRA label-free 13% fallback (E19).
- New SOTA baselines run directly: DSL (their code, Sharpe 0.672), qqa/PQQA (their pip pkg, 17-34% gap →
  our GNN+LS beats raw learned-relaxation). Monitor: DiffUCO/VeloxQ documented (PQQA represents the class).
- STATE: per-instance + amortization + SOTA-comparison comprehensively done; paper rigorous & synced.
  Ongoing loop = periodic literature monitoring + act on genuinely-new approaches as they appear.

## UPDATE 2 — amortization characterized on EVERY axis (E28–E32) + honest CVaR nuance
- E28 DOW-30 backtest: Sharpe-parity on a 2nd market (0.633=0.633, ~470x).
- E29 regime-robust: pre-2016-trained, 2020 COVID-crash gap 0.605% ≈ non-crash 0.695% (no degradation).
- E30 cross-market transfer matrix (fig_e30): S&P100-trained transfers to all markets 0.19-0.38%;
  breadth of training universe drives transferability (narrow DOW-trained 2.9% on S&P100).
- E31 sample efficiency (fig_e31): 5 windows->0.83%, 40->0.72%, 159->0.40% (cheap to deploy).
- E32 CVaR backtest: amortized matches TAIL RISK (CVaR5/MaxDD) @~470x but realized Sharpe lags ~14%
  (0.666 vs 0.774) -- honest nuance (MV amortization is exact-match, CVaR is good-but-imperfect realized).
- Deployment-compute corollary: daily-20yr 2.35h->105s (81x); 100-scenario sweep 235h->27min (529x).
=> Amortization (the contribution) is now characterized along quality, scale, objectives, OOD universes,
regime shift, cross-market transfer, sample efficiency, deployment compute, warm-start frontier, live
backtest (2 markets) — with honest boundaries (unrelated/index-tracking fail; CVaR realized-Sharpe lags).
Paper main.pdf ~550KB with transfer+frontier figures. EXPERIMENTS.md has E5-E32. 137 PDFs. Loop ongoing.

## UPDATE 3 — the GNN-value investigation (E44-E50): where, why, and verified
A rigorous cluster locating + explaining the GNN's genuine value, balancing the honest negatives:
- E44: at typical lambda, a LINEAR model on 3 per-asset feats matches the GNN -> amortization win is a
  feature-ranking, not graph-learning (extends Angelini to amortization).
- E45: that linear rule is INTERPRETABLE + return-dominated (+3.17 z(mu) -0.14 z(sigma) -0.24 z(|corr|)),
  explaining the per-instance easiness at lambda=0.5.
- E46: but at HIGH lambda (risk-dominated), GNN BEATS linear by 3-6 pts -> the regime where the graph matters.
- E47: in that regime the per-instance ranking REVERSES -- tabu fails (17.7% @0.9, VERIFIED not feasibility/
  scaling artifact), GNN+LS robust (ties greedy).
- E48: min-variance (lambda=0.99) QUBO is EXACT-HARD -- SCIP times out 8/8 (64% gap); heuristics win.
- E49: most regime-robust per-instance solver = GREEDY+TABU warm-start (worst-case 0.5% across lambda).
- E50: the GNN's risk-dominated edge is GENUINE MESSAGE-PASSING, NOT static graph features (centrality/
  degree don't help linear) -- irreducible learned multi-hop aggregation.
NET: GNN-QUBO for portfolio is competitive-but-not-necessary at typical lambda (simple greedy+tabu/linear
match it), but GENUINELY VALUABLE in the risk-dominated/min-variance regime (exact fails; GNN's message-
passing beats linear) AND for amortization (universal, lightweight, interpretable). Complete, verified,
honest picture. Paper ~690KB, 12 figs incl fig:regime. EXPERIMENTS.md E5-E50. Loop ongoing.
