"""Apply the EXACT original QRF-GNN method (qrfgnn_dgl.run_gnn_training) to portfolio QUBOs.

We reuse the verbatim original training loop and feed it portfolio QUBOs built by our formulations.
Global scaling of Q does not change the QUBO optimum but keeps the original loss's binarization
penalty (lbd = epoch/1e4) in a sensible regime (like MaxCut, where |off-diag| ~ O(1)).

Run in the .venv-dgl environment.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import networkx as nx
import torch
import dgl

from qrfgnn_dgl import (get_gnn, run_gnn_training, qubo_dict_to_torch, TORCH_DEVICE, TORCH_DTYPE)
from qubo import QUBO
from qubo_portfolio import selection_qubo
# NOTE: convex re-weighting + MED metric live in the eval stage (.venv with cvxpy) to keep this
# GNN environment (.venv-dgl: torch2.3 + dgl + numpy<2) free of the numpy2-requiring cvxpy stack.


def qubo_to_dgl(qubo: QUBO):
    """Build a DGL graph (from off-diagonal nonzeros) and a normalized q_torch for the original method."""
    Q = qubo.Q.copy()
    n = qubo.n
    off = Q - np.diag(np.diag(Q))
    scale = np.mean(np.abs(off[off != 0])) if np.any(off != 0) else 1.0
    Qn = Q / scale  # global scaling preserves the optimum; keeps loss penalty regime sane
    g = nx.Graph(); g.add_nodes_from(range(n))
    rows, cols = np.nonzero(np.triu(off, 1))
    g.add_edges_from(zip(rows.tolist(), cols.tolist()))
    dgl_graph = dgl.from_networkx(g).to(TORCH_DEVICE)
    q_torch = torch.tensor(Qn, dtype=TORCH_DTYPE, device=TORCH_DEVICE)
    return dgl_graph, q_torch


def solve_selection_original(mu, Sigma, K, lam, *, epochs=4000, lr=0.014, seeds=3):
    """Pick K assets via the ORIGINAL QRF-GNN on the cardinality selection QUBO. Returns support."""
    ra, rw = float(lam), float(1 - lam)
    q = selection_qubo(mu, Sigma, K, risk_aversion=ra, return_weight=rw)
    dgl_graph, q_torch = qubo_to_dgl(q)
    n = q.n
    best_support, best_obj = None, np.inf
    for s in range(1, seeds + 1):
        import random
        random.seed(s); np.random.seed(s); torch.manual_seed(s)
        hypers = {'dim_embedding': 10, 'hidden_dim': 51, 'dropout': 0.5, 'number_classes': 1,
                  'prob_threshold': 0.5, 'number_epochs': epochs, 'tolerance': 1e-4, 'patience': 100}
        net, embed, opt = get_gnn(n, hypers, {'lr': lr}, TORCH_DEVICE, TORCH_DTYPE)
        _, _, best_probs, best_bitstring, _ = run_gnn_training(
            q_torch, dgl_graph, net, embed, opt, epochs, 1e-4, 100, 0.5)
        probs = best_probs.detach().cpu().numpy()
        # enforce exactly K via top-K probability (the cardinality penalty already biases here)
        support = np.argsort(-probs)[:K]
        # score by the ORIGINAL selection-QUBO objective (equal weight) for picking best seed
        x = np.zeros(n); x[support] = 1
        obj = q.energy(x)
        if obj < best_obj:
            best_obj, best_support = obj, support
    return np.sort(best_support)
