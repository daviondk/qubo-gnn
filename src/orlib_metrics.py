"""Exact Cura (2009) MED / VRE% / MRE% metric for OR-Library cardinality portfolio benchmark.

Setup (Chang 2000 / Cura 2009): K=10, eps=0.01, delta=1, unconstrained frontier = 2000 points,
heuristic frontier = 51 lambda points. MED in (variance, return) space via nearest standard point.
"""
from __future__ import annotations

import numpy as np


def unconstrained_frontier(mu, Sigma, n_points=2000):
    """Standard Markowitz frontier (budget + no-short only). Returns (variance[], return[])."""
    import cvxpy as cp
    n = len(mu)
    lams = np.linspace(0.0, 1.0, n_points)
    V, R = [], []
    for L in lams:
        xv = cp.Variable(n, nonneg=True)
        cp.Problem(cp.Minimize(L * cp.quad_form(xv, cp.psd_wrap(Sigma)) - (1 - L) * (mu @ xv)),
                   [cp.sum(xv) == 1]).solve(solver=cp.CLARABEL)
        if xv.value is not None:
            w = np.maximum(xv.value, 0); w = w / w.sum()
            V.append(float(w @ Sigma @ w)); R.append(float(mu @ w))
    return np.array(V), np.array(R)


def cura_metrics(heur_var, heur_ret, std_var, std_ret):
    """MED, VRE%, MRE% for heuristic frontier points vs the standard frontier."""
    heur_var = np.asarray(heur_var); heur_ret = np.asarray(heur_ret)
    std_pts = np.column_stack([std_var, std_ret])
    meds, vres, mres = [], [], []
    for v, r in zip(heur_var, heur_ret):
        d = np.sqrt((std_var - v) ** 2 + (std_ret - r) ** 2)
        j = int(np.argmin(d))
        meds.append(d[j])
        if abs(v) > 1e-15:
            vres.append(100.0 * abs(std_var[j] - v) / abs(v))
        if abs(r) > 1e-15:
            mres.append(100.0 * abs(std_ret[j] - r) / abs(r))
    return {"MED": float(np.mean(meds)), "VRE": float(np.mean(vres)) if vres else float("nan"),
            "MRE": float(np.mean(mres)) if mres else float("nan"), "n_points": len(meds)}


def lambda_grid(n=51):
    return np.linspace(0.0, 1.0, n)
