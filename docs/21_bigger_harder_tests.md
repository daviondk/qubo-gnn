# Bigger & harder tests — answering "exact solutions for hard problems shouldn't exist"

Motivation (reviewer/user intuition): *it is suspicious that everything reaches the "exact" optimum —
that does not happen for genuinely hard problems.* Correct. We test bigger and harder to show **where
the problem really is easy (and exact provably solves it) vs where it is hard (and the exact solver
fails, so a learned/heuristic solver wins).**

## A. Big real portfolio — S&P 500, N=461, empirical covariance (mean-variance cardinality)
`src/exp_large_real.py`, `results/large_real/`. Financial objective `λ wᵀΣw − (1−λ)μᵀw`, K=30, 750-day
empirical Σ. Compared to the **exact continuous MIQP (SCIP)**.

| method | objective | gap to exact | time | note |
|---|---|---|---|---|
| SCIP-MIQP (exact) | −0.001108 | 0.00% | 39.2 s | **proved optimal (gap 0)** |
| **GNN + reweight** | −0.001108 | **0.000%** | 6.7 s | = exact |
| Tabu + reweight | −0.001108 | 0.000% | 1.3 s | = exact |
| Greedy + reweight | −0.001108 | 0.000% | 1.0 s | = exact |
| SA + reweight | −0.001078 | 2.74% | 4.0 s | |

**Finding (definitive):** even at **N=461 real S&P 500**, the exact MIQP **provably solves to optimality
in 39 s**, and GNN/Tabu/Greedy + convex re-weight **all match it (gap 0)**. So mean-variance cardinality
is **genuinely easy** (formally NP-hard, practically tractable: smooth PSD covariance ⇒ local search
reaches the global optimum, branch-and-bound prunes well). The "exact everywhere" pattern is **real,
not a bug** — it tells us this objective is the *wrong* place to look for a learned-solver win. (Earlier
the SCIP row looked 68% off only because it was mistakenly compared on the equal-weight surrogate; on
the true financial objective it is the optimum and the GNN matches it.)

## B. The genuinely hard problem — CVaR at scale (where exact FAILS)
CVaR(95%) is non-smooth and scenario-based; the exact mean-CVaR **MILP** grows with #scenarios and
**does not solve** at scale. `src/exp_cvar.py`.

| instance | exact MILP | EqualW | Tabu+CVaR-LP | GNN+CVaR-LP |
|---|---|---|---|---|
| synthetic N200, 3000 scen | 0.00543 (**timeout, gap 4.15**) | 0.01411 | **0.00516 (beats exact)** | 0.00538 (beats exact) |
| S&P500 N461, 2514 scen | **0.01655** (timeout 119s, but gap 0.2% — near-opt) | 0.02732 | 0.01718 (+3.85%) | 0.01771 (+7.07%) |

**Honest nuance:** the "hybrid beats exact" CVaR win is **regime-specific.** On the hard *synthetic*
case (N200/3000, dense) the MILP's timeout incumbent is poor (gap 4.15) and **both hybrids beat it**. On
the *real* S&P500 case (N461/2514) the CVaR-MILP, though it hits the time limit, returns a
**near-optimal incumbent (proved gap 0.2%)** that **beats the hybrids** (Tabu +3.85%, GNN +7.07%) — the
LP relaxation of real-data CVaR is tight, so even a timed-out MILP is strong. So CVaR is *sometimes* a
win and *sometimes* not; GNN+LP $\approx$ Tabu+LP (here slightly worse). We report both, no cherry-pick.

## Takeaway (for the paper's framing)
- **Don't oversell mean-variance:** it is easy; exact solves it even at N≈460; the GNN ties, it does not
  win. Report this honestly — it is itself a useful negative (confirms Angelini–Ricci-Tersenghi for
  portfolios and corrects naive "quantum/GNN advantage" claims on mean-variance).
- **The real opportunity is hard objectives:** CVaR/scenario (and, by extension, robust, multi-period,
  multi-constraint) where exact MILP/MIQP blows up. There the hybrid + amortized GNN are genuinely
  valuable. This is the honest, defensible contribution.

## E10 — hard CVaR at 8000 scenarios (S&P500 N=461) — honest speed-not-quality win
`experiments/hard_portfolio.py 30 8000`. SCIP-MILP times out (239 s, proved gap 3% → optimum unknown),
incumbent CVaR 0.01603. GNN+CVaR-LP 0.01655 (+3.23%, 7.2 s); Tabu+LP 0.01657 (+3.37%, 3 s); EqualW +59%.
Even at 8000 scenarios on REAL data the timed-out exact incumbent still edges the hybrid (tight real-data
CVaR LP relaxation). So the hard-CVaR win is **speed (~35–80×) at ~3% quality cost**, GNN ≈ tabu — NOT a
quality win (that only occurs in the synthetic-bad-incumbent regime, docs/17). The amortized speed story
(E8) is the stronger, more honest headline.
