"""MaxCut on Gset graphs -- the validation gate for the GNN-QUBO solver.

MaxCut QUBO (minimization):  energy(x) = x^T Q x = -cut(x)
  Q_ii = -sum_j w_ij,   Q_ij = Q_ji = w_ij     (so cut = -energy)
"""
from __future__ import annotations

import os

import numpy as np

from qubo import QUBO

# Best-known cut values for common Gset instances (from the QRF-GNN paper / BLS literature).
GSET_BEST_KNOWN = {
    "G14": 3064, "G15": 3050, "G22": 13359, "G49": 6000, "G50": 5880, "G55": 10294, "G70": 9541,
}
# QRF-GNN reported cuts (arXiv:2407.16468) -- our reproduction target.
QRF_GNN_REPORTED = {
    "G14": 3058, "G15": 3049, "G22": 13344, "G49": 6000, "G50": 5880, "G55": 10282, "G70": 9559,
}


def load_gset(path: str):
    """Parse a Gset file. Handles both the standard '<n> <m>' header format and headerless
    edge lists ('u v w', 1-indexed). Returns (n, edges[list of (u,v,w)]) with 0-indexed nodes."""
    with open(path) as f:
        lines = [ln.split() for ln in f if ln.strip()]
    edges = []
    start = 0
    n = None
    if len(lines[0]) == 2:  # header: n m
        n = int(lines[0][0]); start = 1
    for parts in lines[start:]:
        if len(parts) < 3:
            continue
        u, v, w = int(parts[0]) - 1, int(parts[1]) - 1, float(parts[2])
        edges.append((u, v, w))
    if n is None:
        n = 1 + max(max(u, v) for u, v, _ in edges)
    return n, edges


def maxcut_qubo(path: str) -> QUBO:
    name = os.path.splitext(os.path.basename(path))[0]
    n, edges = load_gset(path)
    Q = np.zeros((n, n), dtype=np.float64)
    for u, v, w in edges:
        Q[u, v] += w
        Q[v, u] += w
        Q[u, u] -= w
        Q[v, v] -= w
    return QUBO(Q, offset=0.0, name=name, meta={"n": n, "m": len(edges), "edges": edges})


def cut_value(qubo: QUBO, x: np.ndarray) -> float:
    """Cut value = -energy for a maxcut QUBO."""
    return -qubo.energy(x)
