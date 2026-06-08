"""QUBO builders for portfolio optimization.

Two formulations:

1) selection_qubo  -- cardinality-constrained selection over N binary vars z_i (pick K assets),
   using an EQUAL-WEIGHT surrogate (w_i = z_i/K). Clean, small (N vars). Used both directly and
   as the support-selector for the HYBRID (GNN/QUBO selects K assets -> convex QP sets weights).
   This is the genuinely NP-hard, combinatorially interesting case (MIQP).

2) weight_qubo     -- weights discretized with B bits/asset, budget + risk + return penalties,
   NO cardinality. This is the discretized *convex* problem; it CANNOT beat the convex optimum
   and is used ONLY as a dense-graph stress test of the QUBO solver (GNN vs SA vs tabu on energy).

All QUBOs are returned with a symmetric Q (see qubo.QUBO) so energy == relaxation loss.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qubo import QUBO


# ----------------------------- selection QUBO (cardinality) -----------------------------

def selection_qubo(mu: np.ndarray, Sigma: np.ndarray, K: int, *,
                   risk_aversion: float = 1.0, return_weight: float = 1.0,
                   card_penalty: float | None = None, penalty_factor: float = 10.0) -> QUBO:
    """min  risk_aversion*(1/K^2) z^T Sigma z  -  return_weight*(1/K) mu^T z
       s.t. sum z = K   (enforced via penalty card_penalty*(sum z - K)^2).

    card_penalty defaults to an auto value above the objective's scale so the
    cardinality constraint is never profitably violated.
    """
    n = len(mu)
    Sigma = np.asarray(Sigma, float)
    mu = np.asarray(mu, float)
    risk = risk_aversion / (K * K)
    ret = return_weight / K

    Q = risk * Sigma.copy()                      # off-diag + diag from risk
    np.fill_diagonal(Q, np.diag(Q) - ret * mu)   # return term on diagonal

    if card_penalty is None:
        # scale: make a single wrong (de)selection cost more than the largest objective swing
        obj_scale = risk * np.abs(Sigma).sum() / max(n, 1) + ret * np.abs(mu).max()
        card_penalty = penalty_factor * (obj_scale + 1e-9)

    A = card_penalty
    # penalty A*(sum z - K)^2 = A*[ sum z_i + 2 sum_{i<j} z_i z_j - 2K sum z_i + K^2 ]
    Q[np.diag_indices(n)] += A * (1.0 - 2.0 * K)
    Q += A * (np.ones((n, n)) - np.eye(n))       # adds A to every off-diagonal entry (symmetric)
    offset = A * K * K
    return QUBO(Q, offset=offset, name=f"sel_card_K{K}",
                meta={"kind": "selection", "K": K, "n_assets": n,
                      "risk_aversion": risk_aversion, "return_weight": return_weight,
                      "card_penalty": A, "mu": mu, "Sigma": Sigma})


def decode_selection(x: np.ndarray):
    return np.flatnonzero(np.asarray(x).ravel() > 0.5)


def tracking_qubo(Sigma, b, K, *, penalty_factor: float = 10.0) -> QUBO:
    """Index-tracking selection QUBO: pick K assets to minimize tracking-error variance
    (w - b)^T Sigma (w - b) under an equal-weight surrogate w_i = z_i/K.
    TE(z) = (1/K^2) z'Sigma z - (2/K) (Sigma b)'z + b'Sigma b  + cardinality penalty.
    """
    Sigma = np.asarray(Sigma, float); b = np.asarray(b, float); n = len(b)
    Q = (1.0 / (K * K)) * Sigma.copy()
    Sb = Sigma @ b
    Q[np.diag_indices(n)] += -(2.0 / K) * Sb
    obj_scale = (1.0 / K**2) * np.abs(Sigma).sum() / max(n, 1) + (2.0 / K) * np.abs(Sb).max()
    A = penalty_factor * (obj_scale + 1e-12)
    Q[np.diag_indices(n)] += A * (1.0 - 2.0 * K)
    Q += A * (np.ones((n, n)) - np.eye(n))
    offset = A * K * K + float(b @ Sigma @ b)
    return QUBO(Q, offset=offset, name=f"track_K{K}",
                meta={"kind": "tracking", "K": K, "Sigma": Sigma, "b": b})


# ----------------------------- generic squared penalty helper -----------------------------

def add_squared_penalty(Q, offset, coeffs: dict, target: float, A: float):
    """Add A*(sum_j coeffs[j]*x_j - target)^2 to a symmetric QUBO Q (in place) and return new offset.
    Uses x_j^2 = x_j (binary). coeffs: {var_index: coefficient}."""
    items = list(coeffs.items())
    for j, cj in items:
        Q[j, j] += A * (cj * cj - 2.0 * target * cj)
    for a in range(len(items)):
        ja, ca = items[a]
        for b in range(a + 1, len(items)):
            jb, cb = items[b]
            Q[ja, jb] += A * ca * cb
            Q[jb, ja] += A * ca * cb
    return offset + A * target * target


# ----------------- sector-capped cardinality selection QUBO (non-modular) -----------------

def selection_qubo_sector_caps(mu, Sigma, K, sector_of, cap, *, risk_aversion=1.0, return_weight=1.0,
                               penalty=None):
    """Cardinality selection with per-sector UPPER caps: choose exactly K assets, at most `cap`
    per sector. Genuinely non-modular (greedy is myopic about sector budgets).

    Variables: z_0..z_{N-1} (selection) + slack bits per sector to turn 'sum_{i in g} z_i <= cap'
    into the equality 'sum z_i + slack_g = cap'. Returns (QUBO, n_assets, var_layout).
    """
    n = len(mu); mu = np.asarray(mu, float); Sigma = np.asarray(Sigma, float)
    sector_of = np.asarray(sector_of, int)
    sectors = sorted(set(sector_of.tolist()))
    n_slack_bits = int(np.ceil(np.log2(cap + 1)))
    slack_base = {}
    nv = n
    for g in sectors:
        slack_base[g] = nv
        nv += n_slack_bits
    Q = np.zeros((nv, nv)); offset = 0.0

    ra = risk_aversion / (K * K); rw = return_weight / K
    # objective on z: ra*z'Sigma z - rw*mu'z
    Q[:n, :n] += ra * Sigma
    for i in range(n):
        Q[i, i] -= rw * mu[i]

    if penalty is None:
        obj_scale = ra * np.abs(Sigma).sum() / max(n, 1) + rw * np.abs(mu).max()
        penalty = 10.0 * (obj_scale + 1e-9)
    A = penalty

    # global cardinality: (sum z - K)^2
    offset = add_squared_penalty(Q, offset, {i: 1.0 for i in range(n)}, K, A)
    # per-sector cap with slack: (sum_{i in g} z_i + sum_k 2^k slackbit - cap)^2
    mant = (2.0 ** np.arange(n_slack_bits))
    for g in sectors:
        coeffs = {i: 1.0 for i in range(n) if sector_of[i] == g}
        for k in range(n_slack_bits):
            coeffs[slack_base[g] + k] = float(mant[k])
        offset = add_squared_penalty(Q, offset, coeffs, float(cap), A)

    layout = {"n_assets": n, "nv": nv, "n_slack_bits": n_slack_bits, "sectors": sectors,
              "cap": cap, "K": K, "sector_of": sector_of}
    return QUBO(Q, offset=offset, name=f"sector_caps_K{K}_cap{cap}",
                meta={"kind": "sector_caps", "mu": mu, "Sigma": Sigma, **layout}), layout


# ----------------------------- weight QUBO (dense stress test) ---------------------------

@dataclass
class WeightSpec:
    n_assets: int
    n_bits: int
    coeff: np.ndarray  # per-bit weight contribution, shape (n_bits,)


def weight_qubo(mu: np.ndarray, Sigma: np.ndarray, *, n_bits: int = 4,
                risk_aversion: float = 1.0, return_weight: float = 1.0,
                budget_penalty: float | None = None, w_max: float = 1.0):
    """Discretize w_i = sum_b coeff_b * x_{i,b},  coeff_b = w_max * 2^b / (2^B - 1).
    Objective: risk_aversion*w^T Sigma w - return_weight*mu^T w + budget_penalty*(sum w - 1)^2.
    Returns (QUBO, WeightSpec).
    """
    n = len(mu)
    B = n_bits
    coeff = w_max * (2.0 ** np.arange(B)) / (2 ** B - 1)
    nv = n * B
    # beta[i*B+b] maps variable -> its weight contribution
    beta = np.tile(coeff, n)                       # (nv,)
    # asset index per variable
    aidx = np.repeat(np.arange(n), B)

    # risk: w^T Sigma w = sum_{u,v} beta_u beta_v Sigma[a_u,a_v] x_u x_v
    S = Sigma[np.ix_(aidx, aidx)]                  # (nv,nv)
    Qrisk = risk_aversion * (np.outer(beta, beta) * S)

    if budget_penalty is None:
        budget_penalty = 10.0 * (risk_aversion * np.abs(Sigma).max() + return_weight * np.abs(mu).max() + 1e-9)
    A = budget_penalty
    # budget: A*(sum_u beta_u x_u - 1)^2 = A[ (beta x)^2 - 2 beta x + 1 ]
    Qbud = A * np.outer(beta, beta)
    Q = Qrisk + Qbud
    # linear terms go on the diagonal (x^2 = x): return -return_weight*mu, budget -2A*beta
    lin = -return_weight * (mu[aidx] * beta) - 2.0 * A * beta
    Q[np.diag_indices(nv)] += lin
    offset = A * 1.0
    spec = WeightSpec(n_assets=n, n_bits=B, coeff=coeff)
    return (QUBO(Q, offset=offset, name=f"weight_B{B}",
                 meta={"kind": "weight", "spec": spec, "mu": mu, "Sigma": Sigma}), spec)


def decode_weights(x: np.ndarray, spec: WeightSpec, normalize: bool = True) -> np.ndarray:
    x = np.asarray(x).ravel().reshape(spec.n_assets, spec.n_bits)
    w = x @ spec.coeff
    if normalize and w.sum() > 0:
        w = w / w.sum()
    return w
