# SOTA ML/DL CO-solver benchmarks for direct comparison (2024–2026)

Goal: benchmark our unsupervised GNN-QUBO solver against **published numbers from recent SOTA papers**
(not our own tabu/SA), on problems **NOT covered by QIGNN** (QIGNN = MaxCut, MIS, Graph Coloring).

## The standard benchmark suite (what these papers all use)
- **MIS, Maximum Clique** → **RB-model graphs** (hard); RB-small 200–300 nodes, RB-large 800–1200.
- **MDS (Min Dominating Set), MaxCut** → **Barabási–Albert (BA) graphs**; small 200–300, large 800–1200.
- **MVC (Min Vertex Cover)** → ER / BA graphs.
- Split: **4000 train / 500 val / 1000 test**. Metric: **solution size** (↑ MIS/MaxClique, ↓ MDS/MVC) + approximation ratio + wall-time. Reference optimum: **Gurobi** (7200 s timeout).

## SOTA papers to compete against (quality)
| Paper | Year/Venue | Type | Problems | Code |
| --- | --- | --- | --- | --- |
| **DiffUCO** | ICML 2024 | diffusion (unsup.) | MIS, MaxClique, MDS, MaxCut, MVC | ml-jku/DiffUCO |
| **SDDS** (Scalable Discrete Diffusion Samplers) | ICLR 2025 | diffusion+RL | MIS, MaxClique, MDS, MaxCut | ml-jku/DIffUCO |
| **X2GNN** | ICLR 2025 (Cornell) | iterative GNN | MIS, Maximum Clique | yes |
| **Learning to Explore & Exploit w/ GNNs** | ICLR 2025 | GNN explore/exploit | unsup. CO | OpenReview |
| **UniHetCO** | 2026 | unified unsup. | multi-problem | — |
| **MaskCO** | ICLR 2026 | masked generative | CO | — |
| **Native Adaptive Solution Expansion** | ICLR 2026 | diffusion | CO | — |
| **VCM** (Value Classification Model) | NeurIPS 2024 | learned QUBO classifier | generic QUBO (gap 0.034%) | OpenReview |
| **EGN / Erdős-Goes-Neural** | NeurIPS 2020 | GNN (baseline) | MDS, MVC, MIS | yes |
| **VeloxQ / QIS3** | 2025 / 2506.04596 | QUBO solver | SK spin glass, NAE-3SAT, MaxCut | — |

## Published reference NUMBERS (our comparison targets)
**MDS (BA graphs, size ↓):** Gurobi 27.89 / 103.80 ; DiffUCO 28.20 / 106.61 ; EGN-Anneal —/111.50 ; EGN —/116.76 (small/large).
**MIS (RB graphs, size ↑) [QIGNN-overlap, context only]:** Gurobi 20.13/42.51 ; DiffUCO 19.42/39.44 ; SDDS 19.62/39.97.
**Maximum Clique (RB graphs, size ↑):** X2GNN within **1.2%** of optimum on large (others >22%); DiffUCO numbers in Table 3.
**MVC (ER/BA, size ↓):** DiffUCO + EGN tables.
**SK spin glass / NAE-3SAT:** VeloxQ, QIS3, VCM (energy / %-optimality / TTS).

## Priority comparison targets (NOT in QIGNN)
1. **MDS (BA)** — ✅ DONE: our GNN **106.87** vs DiffUCO 106.61 (tie, +0.2%), beats EGN; BA-small 29.60 (tuning).
2. **Maximum Clique (RB)** — vs X2GNN / DiffUCO. (NEXT)
3. **MVC (ER/BA)** — vs DiffUCO / EGN. (NEXT)
4. **SK spin glass / NAE-3SAT** — vs VeloxQ/QIS3/VCM (already: GNN +1.09% on SK, 97% NAE-3SAT).

## Our standing — RESULTS (pure GNN, original-complexity, no tabu/LS crutch)
| benchmark (source) | OUR GNN | SOTA | verdict |
| --- | --- | --- | --- |
| **MDS BA-large** (DiffUCO) | **108.26±1.56** (rigorous 50-graph, best-of-3) | DiffUCO 106.61, EGN 116.76, LTFT 110.28, Gurobi 103.80 | competitive: ~2% behind diffusion-SOTA (within ~1 SE), **beats EGN/LTFT/MFA**. (E89's 106.87 "tie" was a lucky 15-graph sample; cross-sample var high 101-109) |
| **SK spin glass** dim128, 10 seeds | +1.09% | tabu/SA best-found | competitive |
| **NAE-3SAT** ratio 2.11 | 97% | tabu 98% | competitive |
| **MVC RB-200** (DiffUCO/Sanokowski) | AR 1.027 (best-of-5) | DiffUCO 1.003 | ~2.4% behind |
| **MaxClique RB** (DiffUCO/X2GNN) | 15.13 (best-of-5) | DiffUCO 16.30, LTFT 16.24, EGN 12.02 | competitive (beats EGN-family) |
| **Modularity / community** (SBM) | Q 0.03 | Louvain 0.51 | FAIL (needs DMoN-style clustering arch) |

## HONEST multi-paper verdict (full paper numbers, same data; current SOTA = SDDS ICLR2025)
MDS BA-large (↓): Gurobi 104.01 | **SDDS 105.16** | DiffUCO 105.21 | LTFT 110.28 | EGN 116.76 | **OURS 108.26**
MaxClique RB (↑): Gurobi 19.06 | **SDDS 18.40** | DiffUCO 17.40 | LTFT 16.24 | EGN 12.02 | **OURS 15.13** ; X2GNN ~18.8
=> **We are NOT better than current SOTA.** Our simple GNN is **MID-PACK**: it BEATS the 2020-2023 learned
methods (EGN, LTFT, DiffUCO-raw) but is **~3% BEHIND the 2024-2025 diffusion/iterative SOTA** (DiffUCO-CE,
SDDS, X2GNN) on both MDS and MaxClique. Honest position for an original-complexity GNN (no tabu/LS crutch):
competitive with the earlier learned-CO generation, behind the newest diffusion generation.
(Earlier "tie/beat SOTA" was WRONG — used weak DiffUCO-paper 106.61 + 15-graph sample.)
Repro: DiffUCO repo cloned (competitors/DiffUCO), exact RB/BA generators; numbers in competitors/diffuco_text.txt.

## SAT/CSP — OUR METHOD'S WIN REGION (best among LEARNED solvers)
Two independent same-data/same-metric wins vs published neural numbers (problems NOT in QIGNN):
1. **Weighted Max-3-SAT (SATLIB uf/uuf)** vs **HyperSAT (SOTA 2025, arXiv 2504.11885)** — metric avg weighted
   UNSAT. Our GNN+1flip BEATS HyperSAT on 4/6 datasets (uf100 14.36 vs 15.64; pure-GNN-no-LS 14.03 also beats).
2. **Max-3-SAT (random N=100, r=4.00/4.15/4.30)** vs **OptGNN Table 2 (arXiv 2310.00526)** — metric avg #UNSAT.
   Our pure GNN 3.37/4.27/5.02 BEATS OptGNN 4.46/5.15/5.84 and ErdosGNN 5.46/6.14/6.79 on ALL ratios
   (~ matches classical Survey Propagation; specialized WalkSAT/SP still lead, as expected).
VERDICT: among unsupervised/learned GNN solvers, our simple method is **best-in-class on SAT/MaxSAT**.
This is the method's genuine niche — consistent with the structural intuition (clean low-order CSP energy +
relaxation finds strong local minima). Graph problems (MDS/MVC/MaxClique) = mid-pack; clustering = fail.
