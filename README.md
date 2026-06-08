# qubo-gnn — Graph Neural Networks for Combinatorial Optimization

Unsupervised **GNN-QUBO** solver (PI-GNN / QRF-GNN / QIGNN lineage) applied across many combinatorial
optimization (CO) problems, with a strict, reproducible comparison to recent SOTA on the **same
benchmarks and metrics**. The goal of the project is to map **where this method is competitive — and
where it wins**.

## TL;DR results
- **Weighted Max-3-SAT (SATLIB):** our simple GNN **beats the 2025 SOTA HyperSAT on 4/6 datasets**
  (avg weighted UNSAT, lower=better). An ablation confirms the win comes from the network, not the
  local-search post-step (pure GNN, no LS, already beats HyperSAT).
- **On par with SOTA** (within ~1–3%, beating earlier learned methods EGN/LTFT/MFA) on Minimum Dominating
  Set, Minimum Vertex Cover, Sherrington–Kirkpatrick spin glass; competitive on Maximum Clique,
  Densest-k-Subgraph, Graph Partitioning, Set Cover, Quadratic Knapsack.
- **Portfolio optimization (boundary):** works well **without hard constraints** (matches exact MIQP;
  amortized inference ≈ 1000× faster), but on the genuinely **constrained** benchmark the penalty
  encoding ill-conditions the QUBO (coeffs blow up ~3·10⁴ → 2·10⁹) and quality drops — an honest
  applicability boundary that motivated the move to general CO.

| Weighted Max-3-SAT (SATLIB), avg weighted UNSAT ↓ | UF100 | UUF100 | UF200 | UUF200 | UF250 | UUF250 |
|---|---|---|---|---|---|---|
| **Our GNN** | **14.36** | **17.88** | **24.88** | **35.52** | 35.92 | 42.68 |
| HyperSAT (SOTA 2025) | 15.64 | 20.46 | 28.98 | 35.55 | 33.24 | 41.64 |
| Liu et al. 2023 (baseline) | 32.48 | 41.65 | 67.38 | 81.68 | 79.06 | 100.04 |

## Method
Unsupervised GNN over the problem graph (GraphSAGE, residual blocks) outputs a relaxation `p` that
minimizes the problem energy `pᵀQ p` (or a problem-specific differentiable energy) with annealed
binarization, followed by multi-threshold rounding and a light 1-flip local search (as in the original
PI-GNN). Optional QIGNN-style iterative refinement feeds the hidden state back as a dynamic node feature.
No training labels — the network optimizes directly on the instance.

## Layout
- `src/` — the GNN-QUBO solver, QUBO builders, baselines.
- `experiments/` — experiment scripts (`e*.py`) + `LOG.md` (full chronological experiment journal).
- `docs/` — research notes, literature reviews, SOTA comparison tables, per-problem analysis.

Benchmarks (SATLIB, DiffUCO RB/BA generators, OR-Library, QOBLIB, Gset) and third-party repos are
**downloaded separately** (not vendored); see [experiments/LOG.md](experiments/LOG.md) and `docs/` for
exact sources and how each instance set is produced.

## Documentation — what to read

**Start here**
- [experiments/LOG.md](experiments/LOG.md) — full chronological experiment journal (every run, result, honest correction).
- [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) — experiment index.
- [docs/26_research_roadmap.md](docs/26_research_roadmap.md) — research roadmap.

**General CO & SAT — SOTA comparison (the main results)**
- [docs/30_sota_benchmarks.md](docs/30_sota_benchmarks.md) — SOTA benchmark tables + our standing (MDS / MaxClique / MVC / SK / MaxSAT).
- [docs/29_benchmarking_on_their_metrics.md](docs/29_benchmarking_on_their_metrics.md) — comparing on competitors' own metrics.
- [docs/27_sota_methodology_and_direct_comparison.md](docs/27_sota_methodology_and_direct_comparison.md) — methodology for direct, same-data comparison.
- [docs/31_qubo_list_prioritized.md](docs/31_qubo_list_prioritized.md) — QUBO problems triaged by reproducibility/recency.
- [docs/32_per_problem_litreview.md](docs/32_per_problem_litreview.md) — per-problem literature + which to compete on.

**Literature reviews**
- [docs/01_literature_review.md](docs/01_literature_review.md) · [docs/06_current_literature_2024-2026.md](docs/06_current_literature_2024-2026.md) · [docs/25_literature_2025_2026.md](docs/25_literature_2025_2026.md) — surveys (2024–2026).
- [docs/02_references.md](docs/02_references.md) — references. · [docs/22_krylova_deep_read.md](docs/22_krylova_deep_read.md) — closest prior work (Krylova) deep-read.

**Phase 1 — portfolio optimization (where the method's boundary was mapped)**
- [docs/03_diagnosis.md](docs/03_diagnosis.md) — why the original notebook experiment was flawed.
- [docs/04_experimental_plan.md](docs/04_experimental_plan.md) · [docs/05_results.md](docs/05_results.md) — plan + results.
- [docs/07_benchmark_metrics_and_published_results.md](docs/07_benchmark_metrics_and_published_results.md) · [docs/08_comparison_results.md](docs/08_comparison_results.md) · [docs/10_extended_published_results.md](docs/10_extended_published_results.md) · [docs/18_master_comparison_tables.md](docs/18_master_comparison_tables.md) — metrics & published-number comparisons.
- [docs/09_reproduction_original_method.md](docs/09_reproduction_original_method.md) — reproducing the original method (Gset).
- [docs/11_alternative_formulations.md](docs/11_alternative_formulations.md) · [docs/16_new_datasets.md](docs/16_new_datasets.md) · [docs/17_new_problem_types.md](docs/17_new_problem_types.md) — formulations, datasets, problem variants.
- [docs/12_recent_2025_2026_portfolio_optimization.md](docs/12_recent_2025_2026_portfolio_optimization.md) · [docs/13_backtest_results.md](docs/13_backtest_results.md) — recent portfolio lit + live backtest.
- [docs/14_ablation_scaling_competitors.md](docs/14_ablation_scaling_competitors.md) · [docs/15_amortized_results.md](docs/15_amortized_results.md) · [docs/21_bigger_harder_tests.md](docs/21_bigger_harder_tests.md) — ablations, amortization (≈1000× speedup), scaling.
- [docs/19_competitor_reproduction.md](docs/19_competitor_reproduction.md) · [docs/23_modern_ml_competitors_hard_class.md](docs/23_modern_ml_competitors_hard_class.md) · [docs/24_competitor_comparison_results.md](docs/24_competitor_comparison_results.md) — competitor reproduction & comparison (optimizer-vs-investor).
- [docs/28_autonomous_loop_findings.md](docs/28_autonomous_loop_findings.md) — consolidated findings.

## Reproduce
```bash
uv venv --python 3.12 .venv
uv pip install -r requirements.txt   # torch from --index-url https://download.pytorch.org/whl/cu124
# set KMP_DUPLICATE_LIB_OK=TRUE (torch + dwave-neal OpenMP)
```
Each result in `experiments/LOG.md` is reproduced by the corresponding `experiments/e*.py` script;
SOTA numbers are taken from the competitors' papers and compared on identical instances/metrics.
