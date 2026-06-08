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
**downloaded separately** (not vendored); see `experiments/LOG.md` and `docs/` for exact sources and how
each instance set is produced.

## Reproduce
```bash
uv venv --python 3.12 .venv
uv pip install -r requirements.txt   # torch from --index-url https://download.pytorch.org/whl/cu124
# set KMP_DUPLICATE_LIB_OK=TRUE (torch + dwave-neal OpenMP)
```
Each result in `experiments/LOG.md` is reproduced by the corresponding `experiments/e*.py` script;
SOTA numbers are taken from the competitors' papers and compared on identical instances/metrics.
