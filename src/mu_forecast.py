"""ML return forecaster (cross-sectional) for the backtest — strictly walk-forward, no lookahead.

At each rebalance index t it builds standard cross-sectional features from past returns R[:t], trains
a model on past (features -> forward mean-daily-return) samples (all with horizon ending < t), and
predicts mu for date t. Returns mu in DAILY-return units (comparable to the historical-mean baseline).

Also a Ledoit-Wolf covariance estimator (shrinkage), standard for out-of-sample robustness.
"""
from __future__ import annotations

import numpy as np


def _features_at(R, s):
    """Cross-sectional features for all assets using only R[:s]. Returns (n, F) or None if too early."""
    if s < 252:
        return None
    def csum(k):
        return R[s - k:s].sum(axis=0)
    m21, m63, m126, m252 = csum(21), csum(63), csum(126), csum(252)
    m252_21 = R[s - 252:s - 21].sum(axis=0)        # classic 12-1 momentum
    vol63 = R[s - 63:s].std(axis=0)
    rev5 = -R[s - 5:s].sum(axis=0)                   # short-term reversal
    F = np.column_stack([m21, m63, m126, m252, m252_21, vol63, rev5])
    # cross-sectional z-score (per feature, across assets) for scale invariance
    mu = F.mean(axis=0); sd = F.std(axis=0) + 1e-12
    return (F - mu) / sd


def make_ml_mu_forecaster(model="ridge", train_window=1260, horizon=63, step=21, alpha=10.0):
    """Return f(R, t, lookback) -> predicted mu vector (daily-return units) for rebalance index t."""
    from sklearn.linear_model import Ridge
    if model == "hgb":
        from sklearn.ensemble import HistGradientBoostingRegressor

    def forecaster(R, t, lookback):
        n = R.shape[1]
        Xs, ys = [], []
        s_start = max(252, t - train_window)
        for s in range(s_start, t - horizon, step):
            F = _features_at(R, s)
            if F is None:
                continue
            target = R[s:s + horizon].mean(axis=0)   # mean daily return over next horizon
            Xs.append(F); ys.append(target)
        Ft = _features_at(R, t)
        if not Xs or Ft is None:
            return R[t - lookback:t].mean(axis=0)     # fallback: historical mean
        X = np.vstack(Xs); y = np.concatenate(ys)
        if model == "hgb":
            m = HistGradientBoostingRegressor(max_depth=3, max_iter=150, learning_rate=0.05,
                                              min_samples_leaf=30)
        else:
            m = Ridge(alpha=alpha)
        m.fit(X, y)
        return m.predict(Ft)

    return forecaster


def momentum_mu(R, t, lookback):
    """12-1 momentum as a selection signal (mean daily return from t-252 to t-21), per asset."""
    if t < 252:
        return R[t - lookback:t].mean(axis=0)
    return R[t - 252:t - 21].mean(axis=0)


def ledoit_wolf_cov(window):
    from sklearn.covariance import LedoitWolf
    try:
        return LedoitWolf().fit(window).covariance_
    except Exception:
        S = np.cov(window, rowvar=False)
        return 0.5 * (S + S.T)
