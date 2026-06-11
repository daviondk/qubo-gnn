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
| Maximum Clique | RB, DiffUCO/X2GNN | behind diffusion; beats EGN | mid-pack ⚠️=MIS-complement (paper-adjacent) |
| Max-3-Cut (k=3) | Gset, **ROS** 2024 Tab.7 | ~ROS; behind ANYCSP+MOH | mid-pack (E108) ⚠️ cut-family (close to paper's Max-Cut) |
| Min-Bisection | Gset, KL+spectral | beat KL on G14; fail toroidal | mixed (E112) ⚠️ cut-family (close to paper's Max-Cut) |
| **Quadratic Knapsack** (packing+capacity) | random, exact(SCIP)+greedy | within ~1% of optimum, beats greedy 2/3 | competitive, NOT in paper (E113) |
| SK spin glass | dim128, tabu/SA | +1.1% | competitive (no paper#) |
| NAE-3SAT | random, tabu | 97% clauses | competitive (no paper#) |
| **Densest-k-Subgraph** | SNAP Facebook, greedy-peel | **OPTIMAL at k≤20** (ties greedy-peel), behind k=30 | competitive (E110) |
| **Number Partitioning** | random, Karmarkar-Karp | 8×–228× worse than KK | ❌ clear loss (E111) |
| **QAP** (assignment) | QAPLIB, DL-GNN (Two-Stage GPN 9–30%) | gap 0.1–6.7% (looks like beats DL-GNN) | ❌ **NOT a method win** — ablation: 2-opt LS does it, GNN adds nothing (E116; frustrated like NPP) |
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

### The deeper mechanism: relaxation vs. search (validated on 2 live runs)
Condition (2) sharpens into one rule that predicts every result. Learned/heuristic solvers split into:
- **relaxation / single-shot** — emit one (soft) assignment + decode: OptGNN, HyperSAT, ErdosGNN, ROS, EGN.
- **iterative search** — anytime stochastic search over assignments: classical (WalkSAT, MOH, Survey
  Propagation, tabu) and learned (ANYCSP [IJCAI'23, 2208.10227], X2GNN, diffusion samplers DiffUCO/SDDS).

Our method (relaxation + light 1-flip) **beats relaxation/single-shot solvers** but **loses to iterative
search**. Every observation follows:
- Max-SAT: published SOTA = relaxation (OptGNN/HyperSAT) → **we win** (E105/E106).
- Max-3-Cut: ROS (relaxation) we ≈/beat, but ANYCSP + MOH (search) lead → **mid-pack** (E108).
- Graph CO (MDS/MVC/MaxClique): diffusion *samplers* (search) lead → mid-pack.
So the honest, bulletproof claim is **best among relaxation-based learned solvers**; iterative-search methods
(learned or classical) are a stronger, separate league we do not contest. Note ANYCSP benchmarks SAT only as
decision (#solved, Table 3) and at N=10⁴ (Table 4) — not on the N=100 avg-unsat regime — so it is not in our
comparison table; it is cited as the canonical learned-*search* method that defines the stronger league.

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

### Three tiers by energy landscape (the full live-run picture)
Combining the relaxation-vs-search mechanism with the energy structure gives a clean 3-tier map, each tier
confirmed by live runs:
1. **WIN** — clean low-order energy **and** published SOTA is a *relaxation* method: Max-SAT (E105/E106).
2. **MID-PACK** — clean low-order energy but SOTA is a *search* method: Max-3-Cut (E108/E109), MDS/MVC/
   MaxClique, Densest-k-Subgraph (E110, optimal at small k, ties strong greedy).
3. **CLEAR LOSS** — *frustrated / dense* energy our relaxation cannot navigate: Number Partitioning (E111,
   8–228× worse than Karmarkar-Karp), modularity, LABS, constrained portfolio. Here even our local search
   can't rescue the relaxation; a purpose-built heuristic (KK, Louvain) dominates.
The method's home is Tier-1; it is respectable in Tier-2 and should not be used in Tier-3.

## Conclusion of the perebor
The method's genuine, defensible niche is **(weighted) Max-SAT, best-in-class among learned/unsupervised
solvers** (two independent same-data/same-metric wins: HyperSAT, OptGNN — see `33_maxsat_writeup.md`). On
graph CO it is mid-pack (behind the 2024-25 diffusion SOTA, ahead of older learned methods); on clustering
and hard-constrained QUBO it fails. Cleanly-reproducible *winnable* learned arenas appear exhausted; remaining
list problems are niche quantum-annealing demos, non-reproducible custom benchmarks, or classical-dominated.
Ready to test any specific public benchmark on request.
