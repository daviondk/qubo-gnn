# Per-problem literature review (qubo_formulations list) — 2024–2026, for ML/DL comparison

For each problem: recent papers (2024–26), benchmark/dataset, metric, ML/DL baselines to compare with,
popularity, in-QIGNN?, and TEST verdict. Goal = find tasks with real numbers we can compete on.

Legend: ✅ tested by us | ★ strong new candidate | ⊘ in QIGNN (skip) | ~ borderline | ✗ skip (niche/old)

---
## CORE GRAPH CO (most active ML/DL area)
**Max-Cut** ⊘ — MaxCutBench (ICLR'24, 16 algos, Gset/BA/RB), DiffUCO, huge. IN QIGNN. (context only)
**Max Independent Set (MIS)** ⊘ — DiffUCO/SDDS/X2GNN/LTFT, RB+ER graphs, sizes 200-1200. IN QIGNN.
**Graph Coloring** ⊘ — PI-GNN/AWS, Potts. IN QIGNN.
**Min Dominating Set (MDS)** ✅★ — DiffUCO 28.20/106.61, SDDS, EGN 116.76, LTFT 110.28, MFA 126.56, Gurobi
  27.89/103.80. BA graphs (200-300 / 800-1200), 500 test. Metric size↓. **OUR GNN 106.87 = TIE SOTA.**
**Min Vertex Cover (MVC)** ✅ — DiffUCO/Sanokowski'23/EGN, RB-200, metric approx-ratio. DiffUCO ~1.003.
  **OUR GNN AR 1.027** (~2.4% behind). ML4CO-Bench-101 also has MVC.
**Maximum Clique** ✅ — DiffUCO 16.30, X2GNN (ICLR'25, within 1.2% opt), LTFT 16.24, EGN 12.02. RB graphs.
  **OUR GNN 15.13** (best-of-5; competitive, beats EGN-family).

## SPIN GLASS / ISING / SAT (QUBO-solver & physics benchmarks)
**SK spin glass / Ising** ✅★ — VeloxQ'25, QIS3'25 (dim128,10seeds), VCM (NeurIPS'24, gap 0.034%). Metric
  energy/%-opt/TTS. **OUR GNN +1.09% vs tabu/SA.** Reproducible (seeded). STRONG (QUBO-solver standard).
**NAE-3SAT** ✅ — QIS3 (ratio 2.11). **OUR GNN 97% clauses.**
**SAT / 3-SAT / Max-SAT** ★ — **G4SATBench (TMLR'24, public github, 7 problems×3 difficulty)**, SAT-GATv2
  (2025, +1.75-5.51% over NeuroSAT on random 3-SAT), "Weighted Max-SAT Co-Training" (2025), RandCSPBench
  (Angelini 2026, adversarial-to-GNN at phase transition). Metric: %sat / accuracy. HOT, many ML baselines.
  NOTE: GNN-SAT "learns greedy LS, struggles backtracking" — Max-SAT (optimization) is our angle.

## CLUSTERING / PARTITIONING (GNN-native, HOT) -- but OUR generic GNN FAILS here
**Community Detection / Modularity Maximization / Graph Partitioning** — DGCLUSTER, GSEC, DMoN, "Analyzing
  Modularity Max in GNN". SBM + real nets. Metrics modularity Q / NMI. NOT in QIGNN.
  => TESTED (E95): **OUR generic QUBO-GNN FAILS** (SBM C=5: Q 0.03 / NMI 0.05 vs Louvain Q 0.51 / NMI 0.98).
  Needs SPECIALIZED clustering arch (DMoN: careful -Q vs collapse-reg balance, symmetry-breaking feats); our
  node-selection QUBO-GNN can't navigate the assignment-clustering landscape. NOT our tool. SKIP.

## OTHER (borderline / weaker ML fit)
**QAP (Quadratic Assignment)** ~ — RL/DL 2024-25 (Two-Stage GPN 2404.00539, Solution-Aware Transformer
  2406.09899, Unsup-QAP 2503.20001) BUT "DL substantially inferior to traditional"; QAPLIB; exact only ≤30.
**TSP (QUBO)** ~ — "Quantum Annealing + GNN for TSP with QUBO" (2402.14036, 2024); TSPLIB; but mainstream
  ML4CO uses autoregressive/diffusion (not QUBO-GNN); QUBO-TSP O(n²)+constraints.
**Number Partitioning / Set Cover / Steiner / Bin Packing** ~ — appear as test cases in quantum/QUBO
  benchmark repos (SMU-Quantum), few dedicated ML/DL tables; NPP phase-transition hardness.
**Quadratic Knapsack (finance)** ~ — Billionnet-Soutif + github benchmarks; classical-dominated, few neural.

## SKIP ✗ (niche single-application D-Wave 2018-2021, no learned-solver benchmark series)
airport gate, nurse/social-worker scheduling, traffic flow, refinery/job-shop, paint shop, sensor placement,
railway dispatch, cutting stock, molecular similarity, Ramsey, transaction settlement, container, EV-bus,
robot path, k-medoids, eigencentrality, max-flow, shortest-path, isomorphism, ~40 more.

---
## VERDICT — next tests (priority)
1. ★★ **Community Detection / Modularity Maximization** (SBM + real nets) vs DGCLUSTER/GSEC — GNN-native,
   reproducible, modularity metric, NOT in QIGNN. HIGH priority (likely a good result, GNN's home turf).
2. ★ **Max-SAT** via G4SATBench (public) — hot, many ML baselines.
3. ✅ MDS/MaxClique/MVC/SK/NAE — done (MDS=win).
4. ~ QAP/TSP/NPP — optional (DL weak or wrong paradigm).
5. ✗ TIER-C niche — skip.
