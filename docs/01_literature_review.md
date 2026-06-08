# Literature review (full research reports)

Two deep web/arXiv research passes, run 2026-06-04. Verbatim, with citations. See
`02_references.md` for local PDF paths and `03_diagnosis.md` for how this applies to the notebook.

---

## PART 1 — Critiques & successors of PI-GNN for QUBO/CO

**Anchor paper.** Schuetz, Brubaker, Katzgraber, *Combinatorial Optimization with Physics-Inspired
Graph Neural Networks*, arXiv:2107.01188 (2021); Nature Mach. Intell. 4:367–377 (2022). Unsupervised,
per-instance: relax the QUBO/Ising Hamiltonian into a differentiable loss, train a GNN to minimize
the continuous relaxation, then round to a binary solution. Claimed scaling to millions of nodes for
MaxCut/MIS. **Caveat that runs through the whole literature:** PI-GNN is *unsupervised & per-instance*,
while many "competitors" (DIFUSCO etc.) are *supervised/data-driven*, and the strongest baselines
(greedy, BLS, KaMIS, Gurobi) are *classical*. Don't compare across categories naively.

### 1. The published debate (Nature MI, Jan 2023)
- **Angelini & Ricci-Tersenghi** (MIS critique), arXiv:2206.13211; NMI 5:29 (2023). On MIS on random
  d-regular graphs a simple **greedy beats PI-GNN in quality and is ~10⁴× faster** at million-variable
  scale. Original superiority claims lacked adequate classical baselines.
- **Boettcher** (MaxCut critique), arXiv:2210.00623; NMI 5:24 (2023). On sparse MaxCut, PI-GNN is only
  marginally better than plain gradient descent and is **outperformed by greedy**.
- **Schuetz replies**, arXiv:2302.03602 (to Angelini) and arXiv:2303.12096 (to Boettcher). Argue the
  critiques use a non-representative sparse/random-regular regime where greedy is near-optimal; claim
  "sizable improvements" with better tuning; emphasize **GPU-parallel, scalable** inference vs.
  sequential greedy. They dispute *representativeness* more than provide head-to-head numbers.
- **Gamarnik** (theoretical tie-breaker), arXiv:2306.02555; PNAS 120(46) (2023). Proves
  **constant-depth GNNs cannot beat the overlap-gap-property (OGP) threshold** on random structures,
  which classical algorithms already reach. A complexity-theoretic ceiling explaining the empirics.

**Consensus:** On sparse/random MaxCut & MIS, PI-GNN does **not** beat simple greedy/local search —
essentially uncontested. PI-GNN's value is as a *general, flexible, GPU-parallel template*, useful on
very large/dense/structured instances and problems lacking good handcrafted heuristics.

### 2. QRF-GNN (the architecture in the notebook)
Pugacheva, Ermakov, Lyskov, Makarov, Zotov, *Enhancing GNNs Performance on Combinatorial Optimization
by Recurrent Feature Update*, arXiv:2407.16468 (2024).
- **Architecture:** unsupervised QUBO-loss GNN. Three parallel SAGEConv (GraphSAGE) layers with
  different aggregators (mean/pool/mean) — the "ResSAGE" residual design. Static features = random
  vector (dim 10) + shared dummy vector + PageRank. **Defining trick: recurrent feature update** — the
  previous iteration's output is concatenated back as a dynamic node feature for iterative refinement.
  Hidden dims 50 (MaxCut) / 140 (coloring).
- **Headline:** on **Gset MaxCut** (G14–G70, 800–10k nodes) QRF-GNN **beats all learning methods**
  (PI-GNN, RUN-CSP) and is **competitive with the best classical heuristics** (BLS, TSHEA) at far lower
  runtime. Examples: G14 → 3058 (QRF) vs 3026 (PI-GNN), 2943 (RUN-CSP), 3064 (BLS, best-known);
  G70 → 9559 (QRF) vs 9421 (PI-GNN), 9319 (RUN-CSP), 9541 (BLS), 9548 (TSHEA) — QRF slightly *exceeds*
  BLS/TSHEA on G70 at ~1000 s vs ~7–11k s. Also best learning method on d-regular MaxCut, graph
  coloring, and large MIS (375.44 vs KaMIS 374.57 on 9–11k-node ER, faster).
- **Verdict:** the most credible "GNN matches/edges classical Gset MaxCut" claim — but the margin over
  classical is **thin and instance-specific**. (Verify the full Gset table in the PDF before quoting.)

### 3. Principled PI-GNN successors
| Paper | id / year | Method | Result |
|---|---|---|---|
| Annealed Training for CO on Graphs (Sun et al.) | 2207.11542 / 2022 | Energy-based model + **annealed loss** (temperature schedule) to escape init traps | Beats other unsupervised neural methods on 4 CO types |
| Unsupervised Learning for CO Needs Meta-Learning (Wang & Li, Meta-EGN) | 2301.03116 / ICLR'23 | **Meta-learn an initialization** adapting in a few steps per instance | Beats EGN/PI-GNN-style per-instance training |
| Variational Annealing on Graphs (Sanokowski et al.) | 2311.14156 / NeurIPS'23 | **Autoregressive** model (drops mean-field independence) + annealed entropy | Superior to independent-variable solvers on hard CO |
| DIFUSCO (Sun & Yang) | 2302.08224 / NeurIPS'23 | **Supervised** discrete graph diffusion | SOTA on TSP/MIS — *not Gset MaxCut, not unsupervised* |
| Binarizing PI-GNNs | 2507.13703 / 2025 | Fixes rounding/binarization + **dense-graph degradation** | Directly relevant: portfolio QUBOs are dense |

**Bottom line (skeptical):** PI-GNN debunked on sparse MaxCut/MIS (Gamarnik gives the why);
QRF-GNN is the strongest "GNN competes on Gset" evidence; annealing/meta-learning are the best
principled fixes to PI-GNN's local-optima/per-instance weaknesses; diffusion solvers are a different
(supervised) category.

---

## PART 2 — QUBO portfolio optimization, benchmarks, SOTA, and where a win is realistic

### 1. Formulations
- **Convex Markowitz (the trap).** `min wᵀΣw − q·μᵀw  s.t. Σw=1, w≥0`, `Σ⪰0` → convex QP, solved to
  *global* optimum in poly time (Gurobi/CPLEX/MOSEK, or closed-form when only budget is active).
  Discretizing to QUBO throws away precision and manufactures an artificially hard problem; on the
  same objective a discretized solver can only **approach, never beat** the convex optimum. Any
  "quantum/QUBO advantage" claim on pure continuous mean-variance is comparing to a self-handicap.
- **Binary encoding + penalty methods.** Weights expanded in K bits (`w_i = Σ_k 2^k b_{ik}/scale`);
  equality constraints (budget `Σw=1`, cardinality `Σx=K`) folded in as quadratic penalties
  `λ(Σx−K)²`; inequalities need slack vars. Penalties make the graph **fully connected** — a problem
  for quantum hardware and for PI-GNN-style methods.
- **Cardinality-constrained selection (the real target).** "Hold exactly/at most K of N assets",
  `Σ 1[w_i>0]=K`, with floor/ceiling bounds → **MIQP, NP-hard.** The discrete which-assets decision
  is naturally binary → clean QUBO; the continuous how-much weights are a convex sub-problem on the
  chosen support.
- **Discrete lots / integer holdings / transaction costs / multi-period.** Rosenberg et al.
  arXiv:1508.06182 (multi-period, transaction costs, D-Wave, avoids inverting Σ); Mugel et al.
  arXiv:2007.00017 (8y daily, 52 assets, D-Wave Hybrid + tensor networks, up to 1272 qubits).

### 2. Standard benchmarks
- **OR-Library port1–port5** (Chang et al. 2000): Hang Seng **31**, DAX **85**, FTSE **89**,
  S&P100 **98**, Nikkei225 **225** assets. Standard setup: cardinality **K=10**, bounds ε=0.01, δ=1.0;
  trace the cardinality-constrained efficient frontier; quality = % deviation from the *unconstrained*
  frontier. The single most-reported cardinality benchmark.
- Larger: Cesarone/Tardella extensions up to ~1318 assets; modern quantum papers often generate their
  own (e.g. Fraunhofer: 1978 NASDAQ assets, n=3…1000) → fragmented, not directly comparable.

### 3. SOTA & the right baselines
- For cardinality-constrained MIQP the SOTA baseline is an **exact MIQP solver (Gurobi/CPLEX)**, not a
  heuristic. Fraunhofer (arXiv:2509.17876): Gurobi solves *all* instances to proven optimality "in
  seconds", >1000× faster than SCIP at n=1000; at fixed 60s a problem-specific classical heuristic
  reaches Θ≈1.2–1.5 for n≤100, while **D-Wave annealing and QAOA perform ≈ random sampling** →
  *"only very limited room for a potential quantum advantage."*
- On the OR-Library frontier, **tabu search and modern metaheuristics** dominate the heuristic class
  (Chang et al. 2000; arXiv:2307.04045 time-limited metaheuristics).
- Skeptical audit — Lozano arXiv:2605.17623 (2026): in D-Wave hybrid runs **QPU is ~0.7% of runtime,
  ~99% classical**; a CPU TabuSampler matches the hybrid objective to 1e-3 at equal wall-clock;
  cardinality penalties force fully-connected graphs (chain-break >93% at N=80). "Reported D-Wave
  hybrid wins are constraint-native classical pipelines, not quantum-sampling wins."

### 4. Learning/GNN approaches to portfolio QUBO
- PI-GNN (2107.01188) and the critique (2206.13211) — see Part 1.
- Eliasof & Haber, *QUBO with GNNs* (QUBO-GNN), arXiv:2404.04874 (2024): reframes QUBO as heterophilic
  node classification, self-supervised; **general QUBO, not portfolio-specific**; SOTA claim vs
  "exhaustive search and heuristics" but baselines under-specified — treat cautiously.
- **Gap found:** no published, peer-reviewed paper applies an unsupervised GNN / differentiable-QUBO
  relaxation specifically to **cardinality-constrained portfolio selection with a rigorous Gurobi /
  OR-Library comparison.** This is a genuine opportunity.

### 5. Recent quantum/QUBO portfolio papers (2024–2026)
- End-to-End QA portfolio, arXiv:2504.08843 (2025): D-Wave CQM selects assets, classical weights;
  10 NIFTY-50 stocks; Sharpe 2.55 vs 1.65 ETF — but **no Gurobi/optimization baseline** → weak evidence.
- Fraunhofer benchmark, arXiv:2509.17876 (2025): see §3 — the rigorous head-to-head.
- Lozano audit, arXiv:2605.17623 (2026): see §3 — strongest skeptical source.

### 6. Where a QUBO/GNN approach can realistically win
- **Hopeless:** continuous mean-variance (convex). Avoid.
- **Realistic (bar = exact MIQP):** cardinality-constrained selection. The honest targets are NOT
  "beat Gurobi on objective" (it's optimal in seconds) but:
  1. **Speed–quality frontier at very large N** (thousands of assets, dense Σ) where exact MIQP slows,
     or with non-convex regularizers / discrete lots / nonlinear transaction costs where MIQP scales
     worse — match a good metaheuristic (tabu/GA) faster.
  2. **Hybrid: GNN picks the K-asset support → convex QP sets the weights.** Sidesteps the
     discretization-error trap entirely. Best chance of a clean, defensible result.
- **Reporting discipline:** always report (a) optimality gap vs Gurobi, (b) wall-clock at matched
  quality, (c) feasibility rate, (d) a trivial greedy/random baseline. Heed Angelini & Ricci-Tersenghi:
  a near-linear-time greedy may already beat the GNN.
