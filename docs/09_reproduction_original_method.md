# Faithful reproduction of the original QRF-GNN method (DGL), and its portfolio application

_Addresses the requirement to use the EXACT method from `original_from_paper_gnn_example_Copy1-2.ipynb`
(not a library rewrite) and to reproduce the Gset numbers._

## Why the earlier PyG rewrite underperformed (GNN-alone ~95% vs paper 99.8%)
Running the **verbatim original DGL code** (`src/qrfgnn_dgl.py`) on G14 gives **cut 3012 in 1000 epochs**
and **3042 (99.3%) with 3 seeds** — vs the PyG rewrite's ~95%. The differences that mattered (all now
restored to the original):
1. **Recurrent channel = previous LOGITS** (`h0` = second output of `ResSAGE.forward`, unbounded),
   not the bounded probs I used in PyG.
2. **lr = 0.014** (I used 1e-3), **up to 1e5 epochs** with patience=100 (I used ~2500).
3. **Binarization penalty grows as λ=epoch/1e4** inside `loss_func` (I had a different schedule).
4. **BatchNorm1d** inside the SAGE residual blocks (I substituted LayerNorm for stability).
5. **20 random seeds, keep best** (the paper's protocol; I used 1–3).
6. **clip_grad_norm = 2.0**; fixed random+ones+PageRank inputs (the trainable embedding is created but
   unused); `aggregator_type='pool'` for the second SAGE branch.

Conclusion: it was the training details/protocol, not the GNN concept — and partly the library port.

## Environment
- `.venv-dgl`: **torch 2.3.0+cu121 + dgl 2.2.1** (the notebook's stack) + numpy<2, networkx.
- **DGL CUDA wheels are Linux-only**; on Windows only the CPU DGL build is available. We run on **CPU**
  — cut quality is device-independent (only speed differs). Large Gset graphs (G50/G55 5k, G70 10k
  nodes) need a Linux GPU DGL build and are out of scope on this Windows box.
- Portfolio deps (cvxpy/gurobi) conflict with torch 2.3 on numpy (cvxpy 1.7 needs numpy≥2). So the
  pipeline is **two-stage**: stage 1 (GNN selection) in `.venv-dgl`; stage 2 (convex re-weight + MED)
  in `.venv`.

## Gset reproduction results (original DGL code, CPU)
Protocol: 20 seeds (G14,G15) / 8 seeds (G22), 6000/4000 epochs, keep best cut.

| Instance | n | our best cut | best-known | ratio | paper QRF-GNN | gap to paper |
|---|---|---|---|---|---|---|
| G14 | 800 | **3050** | 3064 | 0.9954 | 3058 | −0.26% |
| G15 | 800 | **3041** | 3050 | 0.9970 | 3049 | −0.26% |
| G22 | 2000 | **13302** | 13359 | 0.9957 | 13344 | −0.31% |

**Reproduced within ~0.3% of the paper's QRF-GNN on all three graphs** (20 seeds for G14/G15, 8 for
G22, on CPU), versus the PyG rewrite's ~95%. The small residual gap is consistent with the paper using
more epochs/seeds (up to 1e5 epochs, 20 seeds) and the weighted Gset_neg variant. Raw:
`results/maxcut/repro_*.log`. **Verdict: the original DGL method is faithfully reproduced; the earlier
shortfall was the PyG port + training-protocol changes.**

## Portfolio application of the ORIGINAL method
`src/qrfgnn_portfolio.py` feeds portfolio QUBOs to the verbatim `run_gnn_training`:
- `qubo_to_dgl(QUBO)` builds the DGL graph from off-diagonal nonzeros and a **globally scaled** Q
  (scaling preserves the optimum but keeps the original loss's binarization penalty in the right
  regime, as in MaxCut where |off-diag| ~ O(1)).
- `solve_selection_original(mu, Σ, K, λ)` runs the original GNN on the cardinality selection QUBO and
  returns the top-K assets.
- Stage 2 (`qrfgnn_eval.py`) convex-reweights (ε=0.01, δ=1) and computes the Cura MED vs published.
Validated: returns exactly K=10 assets on port1. Full MED comparison: `08_comparison_results.md` /
`results/qrfgnn_portfolio/` (to fill).

## Different portfolio formulations to test with the original method (the user's "try many")
1. **Cardinality selection** (equal-weight surrogate) — done in PyG; re-running with original method.
2. **Full weight-encoded QUBO** (binary weight bits + budget + cardinality penalties) — pure end-to-end.
3. **Min-variance / risk-parity with cardinality.**
4. **Index tracking** (minimize tracking error to a benchmark, cardinality).
5. **Discrete-lot / integer holdings** (non-convex, where greedy is myopic — best chance to beat it).
