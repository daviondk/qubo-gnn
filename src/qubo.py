"""Core QUBO container and utilities.

A QUBO is: minimize  x^T Q x + offset,  x in {0,1}^n,  with Q a symmetric dense matrix.
We keep Q symmetric (NOT upper-triangular) so that energy = x^T Q x is unambiguous and the
GNN loss (probs^T Q probs) matches the discrete energy exactly. This is deliberately different
from the original notebook, which mixed an upper-triangular dict with a symmetrized tensor and
was a source of confusion.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class QUBO:
    Q: np.ndarray            # (n, n) symmetric
    offset: float = 0.0
    name: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.Q = np.asarray(self.Q, dtype=np.float64)
        assert self.Q.ndim == 2 and self.Q.shape[0] == self.Q.shape[1], "Q must be square"
        # enforce symmetry
        self.Q = 0.5 * (self.Q + self.Q.T)

    @property
    def n(self) -> int:
        return self.Q.shape[0]

    def energy(self, x: np.ndarray) -> float:
        """x^T Q x + offset for a binary vector x."""
        x = np.asarray(x, dtype=np.float64).ravel()
        return float(x @ self.Q @ x + self.offset)

    def to_dimod(self):
        """Return a dimod BinaryQuadraticModel (for neal / tabu samplers)."""
        import dimod
        Q = self.Q
        n = self.n
        lin = {i: float(Q[i, i]) for i in range(n)}
        quad = {}
        for i in range(n):
            for j in range(i + 1, n):
                c = Q[i, j] + Q[j, i]  # full off-diagonal coefficient
                if c != 0.0:
                    quad[(i, j)] = float(c)
        return dimod.BinaryQuadraticModel(lin, quad, self.offset, dimod.BINARY)

    def edge_index_weight(self):
        """Return (edge_index [2,E], edge_weight [E]) for off-diagonal nonzeros (PyG/DGL).

        Symmetric edges (both directions) are emitted so message passing sees the graph
        as undirected.
        """
        Q = self.Q.copy()
        np.fill_diagonal(Q, 0.0)
        rows, cols = np.nonzero(Q)
        ew = Q[rows, cols].astype(np.float32)
        ei = np.vstack([rows, cols]).astype(np.int64)
        return ei, ew


def random_binary(n: int, rng: np.random.Generator) -> np.ndarray:
    return (rng.random(n) < 0.5).astype(np.int8)


def local_search_1flip(qubo: QUBO, x: np.ndarray, max_passes: int = 50) -> tuple[np.ndarray, float]:
    """Greedy 1-flip local search (steepest descent over single-bit flips).

    Uses incremental delta-energy: flipping bit k changes energy by
        delta_k = (1 - 2 x_k) * (2 * (Q x)_k_offdiag + Q_kk)
    Recompute Qx lazily per pass for simplicity & numerical safety on dense Q.
    """
    Q = qubo.Q
    x = np.asarray(x, dtype=np.float64).copy()
    n = qubo.n
    diag = np.diag(Q)
    for _ in range(max_passes):
        Qx = Q @ x
        # Flip bit k: x_k -> 1-x_k, i.e. change s_k = 1-2x_k (in {+1,-1}).
        # dE_k = energy(x') - energy(x) = 2*s_k*(Qx)_k + Q_kk   (since s_k^2 = 1)
        s = 1.0 - 2.0 * x
        dE = 2.0 * s * Qx + diag
        k = int(np.argmin(dE))
        if dE[k] < -1e-12:
            x[k] = 1.0 - x[k]
        else:
            break
    return (x > 0.5).astype(np.int8), qubo.energy(x)
