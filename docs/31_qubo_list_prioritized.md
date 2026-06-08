# QUBO formulations list — prioritized for ML/DL benchmarking (2024–2026)

Sorted the `qubo_formulations.txt` list (Lucas-2014 / Glover-tutorial-1811.11538 + the xa0.de list) by:
(a) recency & popularity of ML/DL solver papers (2024–2026), (b) availability of standard benchmarks +
published numbers, (c) NOT already in QIGNN (MaxCut/MIS/Coloring excluded), (d) QUBO-GNN-amenability.

## TIER S — hot + benchmarks + numbers + not-in-QIGNN → TEST (and our status)
| Problem | Key 2024-26 papers (numbers) | Dataset | Metric | OUR status |
| --- | --- | --- | --- | --- |
| **Min Dominating Set (MDS)** | DiffUCO (ICML24), SDDS (ICLR25), EGN | BA graphs | size ↓ | ✅ **TIE SOTA** 106.87 vs 106.61 |
| **Min Vertex Cover (MVC)** | DiffUCO, Sanokowski'23, EGN | RB/ER | approx-ratio | ~3% behind (retuning penalty) |
| **Maximum Clique** | DiffUCO, X2GNN (ICLR25), LTFT | RB | size ↑ | weak (decode bottleneck; retuning) |
| **SK spin glass / Ising** | VeloxQ'25, QIS3'25, VCM (NeurIPS24) | random ±J, dim 128 | energy / %-opt / TTS | +1.09% vs tabu/SA |
| **NAE-3SAT / Max-SAT / CSP** | QIS3, "GNNs for Hard CSP" (2026) | random @ phase transition | %clauses / %-opt | NAE-3SAT 97% |

## TIER A — huge literature but EXCLUDED (in QIGNN)
MaxCut (MaxCutBench 2024, Gset/BA/RB — 16 algos), MIS (RB/ER), Graph Coloring. (context only; many SOTA here)

## TIER B — benchmarks exist, fewer ML/DL papers OR harder fit (optional)
| Problem | Why borderline |
| --- | --- |
| Quadratic Knapsack (finance) | Billionnet-Soutif + github benchmarks, but **classical-dominated**, few neural 2024-25 |
| TSP / CVRP | huge ML4CO (TSPLIB, ML4CO-Bench-101) BUT **autoregressive/construction paradigm**, QUBO-TSP=O(n²)+constraints → not QUBO-GNN's strength |
| Number Partitioning | annealer papers; **phase-transition** hardness (need right regime), few learned-solver tables |
| Graph Partitioning / Community Detection | modularity; some ML, weaker standard numbers |
| QAP | QAPLIB classic; few recent neural |

## TIER C — SKIP (niche single-application, mostly D-Wave 2018-2021, no learned-solver benchmark series)
airport gate, nurse scheduling, traffic flow, refinery/job-shop scheduling, paint shop, sensor placement,
nurse/social-worker, railway dispatch, cutting stock, molecular similarity, Ramsey, transaction settlement,
flight gate, container, EV-bus, robot path, etc. — single-paper QUBO demos, not popular benchmarked topics.

## Key context (from the lit review)
- Field trend: **classical local search often beats highly-cited learned methods** (MaxCutBench 2024:
  S2V-DQN/ECO-DQN beaten by LS). So our HONEST competitive results (MDS tie, SK +1%) are meaningful.
- Most-benchmarked: MaxCut, MIS, MVC (excluded trio dominates) → next non-excluded hot set = MDS, MaxClique,
  MVC, SK/Ising, SAT/CSP = exactly TIER S (already covered).

## VERDICT — what to test
1. **MDS** ✅ done (win). 2. **MVC / MaxClique** — finish penalty-aligned retune. 3. **SK/Ising** ✅.
4. **NEW worth adding:** **Max-SAT / 3-SAT / hard-CSP** ("GNNs for Hard CSP" 2026 benchmark) — hot, not-in-QIGNN,
   QUBO-amenable, has 2026 metrics. 5. Everything in TIER C: skip.
