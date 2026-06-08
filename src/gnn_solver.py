"""Unsupervised GNN-QUBO solver.

Pipeline: trainable node embedding + GNN -> relaxed loss probs^T Q probs (annealed binarization) ->
EXPLORE (sample several roundings of the soft solution) -> EXPLOIT (1-flip local search) -> best.

Design choices that fix the original notebook's failures:
  * Trainable nn.Embedding input (PI-GNN's key ingredient) -- lets the model assign each node freely.
  * Loss uses the SAME symmetric Q as the discrete energy (exact relaxation).
  * Track best DISCRETE energy, not the continuous loss.
  * Recurrent channel is bounded probs (QRF-GNN), with grad clipping + NaN guard for stability.
  * Multi-threshold + stochastic rounding then local search (X2GNN-style explore/exploit).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from qubo import QUBO, local_search_1flip
from gnn_model import PIGNN, QRFGNN, pagerank_vector


@dataclass
class GNNHypers:
    model: str = "qrf"          # "qrf" or "pignn"
    dim_embedding: int = 16
    hidden: int = 64
    n_layers: int = 3
    dropout: float = 0.1
    lr: float = 5e-3
    epochs: int = 5000
    patience: int = 800
    tol: float = 1e-6
    anneal_rate: float = 2e-4   # binarization penalty grows as anneal_rate*epoch (gentle)
    recurrent: bool = True
    use_pagerank: bool = True
    eval_every: int = 25
    n_round_samples: int = 16   # stochastic roundings explored each eval
    local_search: bool = True
    ls_passes: int = 100
    refine_sa: bool = True      # X2GNN-style exploit: seed SA from the GNN solution
    refine_reads: int = 20


def _round_and_polish(qubo, p, rng, n_samples, do_ls, ls_passes):
    """Explore several roundings of soft p, polish each with local search, return (best_x, best_e)."""
    cands = [(p > 0.5).astype(np.int8)]
    for thr in (0.4, 0.6):
        cands.append((p > thr).astype(np.int8))
    for _ in range(n_samples):
        cands.append((rng.random(len(p)) < p).astype(np.int8))  # stochastic rounding
    best_x, best_e = None, np.inf
    for x in cands:
        e = qubo.energy(x)
        if do_ls:
            x, e = local_search_1flip(qubo, x, max_passes=ls_passes)
        if e < best_e:
            best_e, best_x = e, x
    return best_x, best_e


def solve_qubo_gnn(qubo: QUBO, hypers: GNNHypers, device: str = "cuda", seed: int = 0,
                   verbose: bool = False) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    n = qubo.n
    Q = torch.tensor(qubo.Q, dtype=torch.float32, device=device)

    ei_np, ew_np = qubo.edge_index_weight()
    if ei_np.shape[1] == 0:
        ei_np = np.vstack([np.arange(n), np.arange(n)]).astype(np.int64)
        ew_np = np.ones(n, dtype=np.float32)
    edge_index = torch.tensor(ei_np, dtype=torch.long, device=device)
    edge_weight = torch.tensor(ew_np, dtype=torch.float32, device=device)

    # trainable embedding (+ optional fixed pagerank feature)
    emb = nn.Embedding(n, hypers.dim_embedding).to(device)
    extra = []
    if hypers.use_pagerank:
        extra.append(pagerank_vector(edge_index, n, device))
    extra_feat = torch.cat(extra, dim=1) if extra else None
    in_dim = hypers.dim_embedding + (extra_feat.shape[1] if extra_feat is not None else 0)
    if hypers.recurrent:
        in_dim += 1

    Model = QRFGNN if hypers.model == "qrf" else PIGNN
    net = Model(in_dim, hidden=hypers.hidden, n_layers=hypers.n_layers, dropout=hypers.dropout).to(device)
    opt = torch.optim.Adam(list(net.parameters()) + list(emb.parameters()), lr=hypers.lr)

    h0 = torch.zeros((n, 1), device=device)
    best_x = np.zeros(n, dtype=np.int8)
    best_energy = qubo.energy(best_x)
    prev_loss = float("inf"); stall = 0
    t0 = time.time()
    idx = torch.arange(n, device=device)

    for epoch in range(hypers.epochs):
        net.train()
        feats = emb(idx)
        if extra_feat is not None:
            feats = torch.cat([feats, extra_feat], dim=1)
        probs, logits = net(feats, edge_index, edge_weight, h0=h0 if hypers.recurrent else None)
        p = probs.squeeze(-1)
        cost = p @ (Q @ p)
        lbd = hypers.anneal_rate * epoch
        loss = cost + lbd * (p * (1.0 - p)).sum()
        if not torch.isfinite(loss):
            h0 = torch.zeros((n, 1), device=device); continue
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(net.parameters()) + list(emb.parameters()), 5.0)
        opt.step()
        if hypers.recurrent:
            h0 = probs.detach()

        if epoch % hypers.eval_every == 0:
            pnp = p.detach().cpu().numpy()
            x, e = _round_and_polish(qubo, pnp, rng, hypers.n_round_samples,
                                     hypers.local_search, hypers.ls_passes)
            if e < best_energy:
                best_energy, best_x = e, x
            lv = float(loss.detach())
            if abs(prev_loss - lv) < hypers.tol:
                stall += hypers.eval_every
                if stall >= hypers.patience:
                    if verbose: print(f"  early stop @ {epoch}")
                    break
            else:
                stall = 0
            prev_loss = lv
            if verbose and epoch % (hypers.eval_every * 20) == 0:
                print(f"  epoch {epoch:5d} loss {lv:12.2f} best_energy {best_energy:12.2f}")

    energy_pre_refine = best_energy
    if hypers.refine_sa:
        import neal
        seed_state = {i: int(best_x[i]) for i in range(n)}
        res = neal.SimulatedAnnealingSampler().sample(
            qubo.to_dimod(), initial_states=seed_state, num_reads=hypers.refine_reads, seed=seed)
        xr = np.array([res.first.sample[i] for i in range(n)], dtype=np.int8)
        er = qubo.energy(xr)
        if er < best_energy:
            best_energy, best_x = er, xr

    return {"x": best_x, "energy": best_energy, "time": time.time() - t0,
            "energy_pre_refine": energy_pre_refine, "epochs_run": epoch + 1, "seed": seed,
            "p_continuous": p.detach().cpu().numpy()}


def solve_qubo_gnn_multi(qubo: QUBO, hypers: GNNHypers, device: str = "cuda",
                         n_restarts: int = 3, base_seed: int = 0, verbose: bool = False) -> dict:
    best = None; energies = []
    for r in range(n_restarts):
        res = solve_qubo_gnn(qubo, hypers, device, seed=base_seed + r * 101 + 7, verbose=verbose)
        energies.append(res["energy"])
        if best is None or res["energy"] < best["energy"]:
            best = res
    best["all_energies"] = energies; best["n_restarts"] = n_restarts
    return best
