"""Portfolio + QUBO evaluation metrics. No normalization rescues hidden here -- feasibility
is reported, not patched."""
from __future__ import annotations

import numpy as np


def portfolio_metrics(w, mu, Sigma, rf=0.0):
    w = np.asarray(w, float)
    ret = float(mu @ w)
    vol = float(np.sqrt(max(w @ Sigma @ w, 0.0)))
    sharpe = (ret - rf) / vol if vol > 1e-12 else float("nan")
    return {"return": ret, "vol": vol, "sharpe": sharpe}


def cardinality_feasible(x, K):
    x = np.asarray(x).ravel()
    k = int((x > 0.5).sum())
    return k == K, k


def optimality_gap(energy, best_energy):
    """Relative gap to the best (most negative) energy. 0 = matched best; positive = worse."""
    denom = abs(best_energy) if abs(best_energy) > 1e-12 else 1.0
    return (energy - best_energy) / denom


def frontier_deviation(ret, vol, front_rets, front_vols):
    """Min Euclidean distance from (vol, ret) to the unconstrained frontier polyline (%)."""
    if len(front_rets) == 0:
        return float("nan")
    d = np.sqrt((front_vols - vol) ** 2 + (front_rets - ret) ** 2)
    return float(d.min())
