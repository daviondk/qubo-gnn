# Current literature sweep (2024–2026) — verified, via academic-research-skills methodology

Two rigorous, citation-verified sweeps run 2026-06-04 following the installed `deep-research` skill's
bibliography → source-verification → synthesis pipeline (every arXiv ID fetched & confirmed; sources
graded; SOTA claims triangulated and stress-tested). Reports kept verbatim.

> **How this connects to the goal:** the user's main objective is to make a *portfolio optimization
> problem in QUBO form* perform well with the **QRF-GNN** algorithm (from
> `original_from_paper_gnn_example_Copy1-2.ipynb`). The two load-bearing takeaways below:
> (1) the strongest *verified* "neural beats classical" result is **X²GNN (ICLR 2025)** — beats Gurobi
> *under a time budget*, not at optimality — and its **explore+exploit + iterative refinement** is the
> design to emulate; (2) the **learned-GNN-QUBO × portfolio** cell is essentially **empty** — a clean,
> defensible niche, provided we benchmark on OR-Library + Gurobi/SA honestly.

---

## SWEEP A — Learning-based QUBO/Ising solvers (2024–2026)

### Skeptical headline
- Genuinely *beating* strong classical heuristics (BLS, TSHEA, KaMIS, Gurobi) remains rare. Credible
  neural results either beat *other neural methods* but only *approach* classical (QRF-GNN), or beat
  **Gurobi under a fixed/short time budget** (X²GNN).
- The actual new best-known MaxCut on Gset **G63** (2025) came from a **classical** GPU
  population-annealing heuristic (arXiv:2510.21105), **not** a neural method — reality check.
- No verified 2024–2026 learning method dethrones BLS/TSHEA across the full Gset MaxCut benchmark.

### Most relevant entries
- **X²GNN** — Acikalin, Ferber, Gomes, *Learning to Explore and Exploit with GNNs for Unsupervised
  CO*, **ICLR 2025** (OpenReview vaJ4FObpXN; code github.com/utkuumur/X2GNN). Unsupervised:
  **explore** (diverse multi-solution generation) + **exploit** (neural stochastic iterative
  refinement). Reports better solutions than **Gurobi under a time budget** on large MaxCut/MIS/
  MaxClique; MaxClique within 1.2% of optimal where other learners stall ~22%. ⭐ The architecture to
  emulate for "beat classical under budget".
- **QRF-GNN** — Pugacheva et al., arXiv:2407.16468 (2024; ICLR'24 acceptance unconfirmed →
  preprint-grade). Recurrent feature update + parallel SAGE; beats all neural baselines; on Gset it is
  **comparable to**, not decisively better than, BLS/TSHEA, winning mainly on **runtime at very large
  scale** (G70: cut 9559 @ ~1000s vs BLS ~11000s). **This is the method we are reproducing.**
- **GCON** — Wenkel et al., multi-filter GNN, **LoG 2024 (oral)**, arXiv:2405.20543. Self-supervised;
  "on par with Gurobi" on MaxCut (on par, not beating).
- **Binarizing PI-GNNs** — Krutský et al., **ECAI 2025**, arXiv:2507.13703. Finds a *phase transition*
  in PI-GNN training as graph density rises (degenerate solutions) and proposes fuzzy/binarized
  rounding — **directly relevant**: portfolio QUBOs are dense, the regime where PI-GNN degrades.
- **DiffUCO** — Sanokowski et al., **ICML 2024**, arXiv:2406.01661. Unsupervised diffusion CO; new
  SOTA across MIS/MaxCl/MDS/MaxCut/MVC vs neural methods. **DIFUSCO** (NeurIPS'23, 2302.08224,
  supervised) and **DISCO** (2406.19705, preprint) are the supervised diffusion line (TSP/MIS focus).
- **LLM-as-CO-solver** — Jiang et al., **NeurIPS 2025**, arXiv:2509.16865; SFT+RL on 7B, avg opt-gap
  1.03–8.20% across 7 NP-hard problems. **Dataless-RL Max-Cut** — Maliakal et al., arXiv:2505.13405
  (beats Goemans-Williamson only). **QIS3** — Yang et al., arXiv:2506.04596 (quantum-inspired
  *classical*, claims optimality on 94% MaxCut incl. vs Gurobi/Neal — single-source, unverified).
- **Benchmarks/surveys:** **ML4CO-Bench-101** (NeurIPS'25 D&B, OpenReview ye4ntB1Kzi; no arXiv) —
  neutral evaluation of neural CO solver families. **Min & Gomes, "GNNs are Heuristics"**,
  arXiv:2601.13465 (2026) — reframes the field; GNNs as unsupervised heuristics.

### Verification notes
All IDs resolved: 2107.01188, 2407.16468, 2507.13703, 2406.19705, 2406.01661, 2509.16865, 2505.13405,
2509.00099, 2506.04596, 2405.20543, 2501.05845, 2510.21105, 2601.13465, 2601.10583, 2302.08224,
2206.13211. No predatory venues. Unverified/flagged: QIS3 numbers; DiffUCO/DISCO exact margins vs
classical; QRF-GNN ICLR'24 acceptance; ML4CO-Bench-101 arXiv mirror (cite NeurIPS/OpenReview).

---

## SWEEP B — QUBO/quantum portfolio × ML/GNN/RL (2024–2026)

### Load-bearing findings
1. **The exact intersection (learned/GNN/differentiable-QUBO solver applied to portfolio) is nearly
   empty.** GNN-for-QUBO ignores portfolio; learning-for-portfolio ignores QUBO solving. The niche.
2. **Strongest recent result is skeptical:** Stopfer & Wagner, *Quantum Portfolio Optimization: An
   Extensive Benchmark*, arXiv:2509.17876 (2025) — **Gurobi solves up to 1000-asset instances to
   proven optimality in seconds; >1000× faster than SCIP; a tailored classical heuristic beats
   QAOA/annealing at fixed runtime.** *"Only very limited room for a potential quantum advantage."*
   **This is the bar.**
3. Most "quantum advantage" portfolio claims omit a Gurobi/exact head-to-head.

### Closest hits to the niche
- **Khan, Mirza Mohammed & Li (2025)**, *Portfolio Optimization: A Neurodynamic Approach Based on
  Spiking Neural Networks*, **Biomimetics 10(12):808** (MDPI, mild caution). Spiking-NN neurodynamic
  solver for **cardinality-constrained Markowitz**; on 50 assets reports 0.261% daily return,
  *claims to beat exact MIQP (ECOS_BB 0.225%)* — ⚠️ internally implausible (can't beat exact on its
  own objective unless feasible sets differ); scrutinize. Closest neural-solver-for-cardinality hit;
  NOT on OR-Library.
- **Li et al. (2024)**, *Cardinality and Bounding Constrained Portfolio Optimization using Safe RL*,
  **IJCNN 2024**. Safe DRL (IPO) with constraints; learning + cardinality but **no QUBO**, no exact
  baseline reported.
- **Eliasof & Haber**, *QUBO-GNN*, arXiv:2404.04874, **ECAI 2025** — GNN-for-QUBO, **no portfolio**;
  the most natural vehicle to carry *into* the niche.
- **Bai et al. (2025)**, *Deep k-grouping* (OH-PUBO), arXiv:2505.20972 — differentiable QUBO
  relaxation template; coloring/partitioning, not portfolio.

### Recent quantum-portfolio with (or without) classical baselines
- ⭐ **Stopfer & Wagner 2025**, arXiv:2509.17876 — the rigorous benchmark (Gurobi included). The bar.
- **Yuan et al. 2024**, arXiv:2410.16265 — QAOA discrete GMV; advantage only in shot-count scaling,
  killed by thermal noise; no Gurobi.
- **Mancilla et al. 2026**, arXiv:2602.14827 — QAOA XY-mixer, Sharpe 1.81 vs SA 1.31 on **10 equities**;
  no Gurobi, tiny universe.
- **Gomez Cadavid et al. 2026**, arXiv:2602.23976 — trapped-ion BF-DCQO, 250 assets; no Gurobi h2h.
- **Soloviev & Krompiec 2025**, arXiv:2511.21305 — Pauli correlation encoding, 250 vars; no Gurobi.

### Hybrid (discrete select + convex reweight) — our recommended pattern
- **Morapakula et al. 2025**, *End-to-End Portfolio Optimization with Quantum Annealing*,
  arXiv:2504.08843, **Adv. Quantum Tech. 2025**. Textbook hybrid: QUBO/CQM selection → convex
  reweight → rebalance. No Gurobi baseline.
- **Lozano 2026**, *A Penalty-Free Pipeline for Direct Quantum-Annealer Portfolio Optimization*,
  arXiv:2605.17628. **Drops the cardinality penalty**; samples objective-only QUBO, enforces
  cardinality by **deterministic classical projection** (chain-break 71–92% → ≤0.04%; regret ≤0.03%).
  Single-author preprint, greedy baseline only — but the *penalty-free + learned/classical projection*
  idea is methodologically important and worth adopting.
- **Palmer et al. 2022**, arXiv:2208.11380 — index tracking with cardinality on annealer (foundational).
- **Mohseni et al. 2026**, arXiv:2603.00260 — constrained quantum opt at utility scale (knapsack, not
  portfolio) but a rare paper *with* a genuine Gurobi head-to-head.

### Standard benchmark & SOTA
OR-Library **port1–5** (Chang et al. 2000): Hang Seng 31, DAX 85, FTSE 89, S&P 98, Nikkei 225;
metrics **MED / VRE / MRE** vs unconstrained frontier. **No single learning-based SOTA holder** — a
dense field of metaheuristics (ABC, mayfly, improved dung-beetle, 2025 local-search hybrids); and the
embarrassingly strong simple bar: **SA reaches near-optimal in ~5 s** (Nikiporenko, arXiv:2307.04045).
**No GNN/RL/QUBO method appears on the OR-Library MED leaderboard at all.**

### Conclusion — the niche and the bar to beat
The niche is real and clean: **no learned/GNN/unsupervised-differentiable-QUBO solver competes on the
standard cardinality-constrained portfolio benchmarks on the established metrics.** Must-beat
baselines: **(1) Gurobi/exact MIQP** (non-negotiable — report optimality gap + wall-clock);
**(2) best metaheuristics + the ~5 s SA bar**; **(3) heed the GNN-CO skeptic lesson** (a naive PI-GNN
often loses to greedy). Where a learned GNN-QUBO can plausibly win: **amortized speed at scale /
multi-period rolling rebalancing** (train once, infer in ms across thousands of rebalance dates);
**penalty-free select-then-convex-reweight hybrid** (extends Lozano 2605.17628); **transaction-cost /
discrete-lot / non-convex** variants where exact MIQP scaling degrades. Frame the contribution as
*amortized / multi-period / cost-aware*, not "beat Gurobi on one small static instance."

> Full APA-7 reference lists for both sweeps are preserved in the conversation transcript and the IDs
> above are all verified-resolvable. New PDFs to consider downloading next:
> X²GNN (ICLR'25), DiffUCO (2406.01661), Khan spiking-NN (Biomimetics), Lozano penalty-free
> (2605.17628), Stopfer&Wagner already in `papers/` (2509.17876).
