# Arena coverage — where the unsupervised GNN-QUBO method stands, and where we looked

Systematic record of the benchmark "perebor": every problem arena considered, the verdict, and why.
Rule throughout: compare **only against other papers' published numbers on the same public benchmark and the
same metric** (no fabricated comparisons; no comparison to our own tabu/SA).

## Verdict summary

| arena | benchmark / competitor | our standing | verdict |
| :-- | :-- | :-- | :-- |
| **Weighted Max-3-SAT** | SATLIB uf/uuf, **HyperSAT 2025** | beat 4/6 datasets | 🏆 **WIN** |
| **Max-3-SAT (random)** | N=100 r∈{4.0,4.15,4.3}, **OptGNN** Tab.2 | beat OptGNN+ErdosGNN, all ratios | 🏆 **WIN** |
| Min Dominating Set | BA, DiffUCO/SDDS | ~2–3% behind diffusion SOTA; beats EGN/LTFT | mid-pack |
| Min Vertex Cover | RB, DiffUCO | ~2.4% behind | mid-pack |
| Maximum Clique | RB, DiffUCO/X2GNN | behind diffusion; beats EGN | mid-pack |
| SK spin glass | dim128, tabu/SA | +1.1% | competitive (no paper#) |
| NAE-3SAT | random, tabu | 97% clauses | competitive (no paper#) |
| Modularity / community | SBM, Louvain | Q 0.03 vs 0.51 | ❌ fail (needs clustering arch) |
| LABS | N=20, best-known | +61% | ❌ fail (specialized) |
| Constrained portfolio | QOBLIB | can't reach quality | ❌ boundary (penalty ill-conditions QUBO) |

## Excluded (already covered by the QIGNN paper — not "new")
Max-Cut, Maximum Independent Set, Graph Coloring. (Huge literature here, but off-limits for a *new* result.)

## Arenas investigated and **discarded as not validly reproducible** (per the methodology)
- **SplitGNN** (weighted MaxSAT, arXiv:2511.19544, Nov 2025): instances use unreleased custom generators
  (WUF/WPL/WPS/WDP). Distribution-matched reproduction gave a 137× scale mismatch → invalid. (LOG E107.)
- **RUN-CSP / LightSolver Max-2-SAT** (1909.08387 / 2302.06926): RUN-CSP numbers are published only as plots
  (Fig. 2, % satisfied vs ratio); the LightSolver benchmark is *time-to-optimal* (all methods reach the
  optimum) → no quality differentiation. Released `2SAT_N_Eval` instances exist but the paper gives no
  extractable per-instance number.
- **SAT-decision GNNs** (NeuroSAT, SAT-GATv2 2502.*, MILP-SAT-GNN 2507.01825, G4SATBench): metric is
  satisfiability-prediction **accuracy**, not number of unsatisfied clauses → different task, not comparable.

## Arenas where the published SOTA is **not a learned solver** (out of our reach by design)
- Generic QUBO near-optimality (VCM, NeurIPS'24, gap 0.034%); spin glass / Ising (VeloxQ'25, QIS3) — these
  are specialized/physics solvers operating near the optimum.
- TSP / CVRP / routing — autoregressive/construction paradigm; QUBO-TSP is O(n²)+constraints, not our fit.

## Why this is the converged picture
Two independent 2025 meta-studies corroborate the landscape: **FrontierCO** (arXiv:2505.16952) and
**"Time to Rethink AI for Combinatorial Optimization"** (TMLR, arXiv:2502.03669) both report a *substantial,
persistent gap between ML-based solvers and human-designed classical SOTA across all problem types*. So a
realistic target for any learned solver is to be **best among learned solvers** on a problem family — which is
exactly what we demonstrate on Max-SAT.

## Structural criterion — *why* SAT is the niche (and where to stop looking)

A learned solver can only **win** on a problem that is simultaneously:
1. **greedy-weak** — simple greedy / local construction is *far* from optimal (otherwise greedy already wins);
2. **learned-SOTA-imperfect** — the published learned baselines leave room (otherwise they win);
3. **not in QIGNN** — Max-Cut / MIS / coloring are off-limits as a *new* result;
4. a clean low-order energy our relaxation can optimize.

Mapping the candidates against this:

| problem | greedy-weak? | learned-SOTA imperfect? | not in QIGNN? | result |
| :-- | :-- | :-- | :-- | :-- |
| **Max-SAT (phase transition)** | ✅ (needs WalkSAT) | ✅ (OptGNN/HyperSAT imperfect) | ✅ | 🏆 **win** |
| Max-Cut / MIS / coloring | ✅ | — | ❌ excluded | — |
| Maximum Clique | ✅ | ❌ (diffusion SOTA strong) | ✅ | mid-pack |
| Spin glass / Ising | ✅ | ❌ (physics solvers near-opt) | ✅ | competitive, not better |
| **Max coverage / facility location / set cover** | ❌ **greedy ≈ 0.99–1.00 ≈ Gurobi** | — | ✅ | not winnable (greedy-dominated) |
| Constrained portfolio | ❌ (greedy/exact strong) | — | ✅ | boundary (also ill-conditioned) |

Concrete evidence for the "greedy-dominated" row: on **maximum coverage** (Bu et al. 2024, arXiv:2405.08424,
Table 2) plain **greedy scores 0.99** of the Gurobi-optimal objective across rand500/rand1000/twitch/railway —
i.e. essentially optimal — so *no* learned method (theirs, EGN, or ours) can beat it. Submodular-coverage and
facility-location problems are "false-hard" for learning, exactly like cardinality portfolio.

**Implication:** the intersection of all four conditions is essentially the **SAT/CSP-at-phase-transition**
family — which we already win. The remaining list problems fail condition (1) (greedy-dominated), (2) (strong
diffusion/physics SOTA), or (3) (in QIGNN). This is why the search converges.

## SAT-optimization arena — exhaustively checked (only OptGNN + HyperSAT give clean comparable tables)
Every learned-MaxSAT paper found was examined; only two report avg-(weighted-)unsatisfied-clause tables on
standard/distribution-matched instances — and we beat both. The rest are non-comparable by metric or data:
- **DeepSP** (Marino, arXiv:2012.06344): metric = accuracy of reproducing Survey-Propagation marginals,
  tested at `N=10⁴`, `α=4.23`; not an avg-unsat comparison.
- **Understanding GNNs for SAT via Approximation** (arXiv:2408.15418, 2024): metric = number of *satisfiable*
  problems solved (decision SAT + decimation), not MaxSAT objective.
- **SAT-decision GNNs** (NeuroSAT, SAT-GATv2, MILP-SAT-GNN, G4SATBench): satisfiability-prediction accuracy.
- **SplitGNN / RUN-CSP**: custom unreleased generators / plot-only numbers (see above).
=> The cleanly-comparable, winnable learned-MaxSAT arena is exhausted by the two wins.

## Conclusion of the perebor
The method's genuine, defensible niche is **(weighted) Max-SAT, best-in-class among learned/unsupervised
solvers** (two independent same-data/same-metric wins: HyperSAT, OptGNN — see `33_maxsat_writeup.md`). On
graph CO it is mid-pack (behind the 2024-25 diffusion SOTA, ahead of older learned methods); on clustering
and hard-constrained QUBO it fails. Cleanly-reproducible *winnable* learned arenas appear exhausted; remaining
list problems are niche quantum-annealing demos, non-reproducible custom benchmarks, or classical-dominated.
Ready to test any specific public benchmark on request.
