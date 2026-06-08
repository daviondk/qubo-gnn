"""Walk-forward portfolio-optimization backtest harness on live equity data (DOW30 / S&P100),
so our strategies' Sharpe/Sortino/MaxDD/turnover sit next to the 2025-26 literature
(Hwang & Zohren SIT 2510.03129, decision-focused MV, DRL/diffusion papers).

Strategies: equal-weight, min-variance, max-Sharpe, Markowitz(lambda), and GNN-QUBO cardinality
(pick K assets via the unsupervised GNN selection QUBO, then convex re-weight = the hybrid).

Run in .venv (yfinance, cvxpy, torch+pyg). Data cached under data/.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

# DOW 30 (liquid, long history); some recent additions trimmed by dropna -> effective universe reported
DOW30 = ["AAPL","AMGN","AXP","BA","CAT","CSCO","CVX","DIS","GS","HD","HON","IBM","INTC","JNJ","JPM",
         "KO","MCD","MMM","MRK","MSFT","NKE","PG","TRV","UNH","VZ","WMT","HPQ","XOM","WBA","CL"]

SP100 = ["AAPL","MSFT","AMZN","GOOGL","META","NVDA","BRK-B","JPM","JNJ","V","PG","UNH","HD","MA","XOM",
         "CVX","ABBV","PFE","KO","PEP","MRK","WMT","BAC","TMO","CSCO","MCD","ABT","CRM","ACN","DHR",
         "DIS","ADBE","NKE","TXN","NEE","WFC","PM","BMY","RTX","LIN","UNP","HON","QCOM","LOW","INTC",
         "IBM","GS","CAT","AMGN","SBUX","DE","AXP","MDT","BLK","GILD","ADP","C","MMM","CVS","MO","CB",
         "T","VZ","AMT","LMT","SO","BKNG","MDLZ","ADI","SYK","PLD","TJX","CI","DUK","BDX","MMC","USB"]


def load_prices(tickers, start="2000-01-01", end="2024-12-31", cache_dir="data"):
    os.makedirs(cache_dir, exist_ok=True)
    key = os.path.join(cache_dir, f"bt_{len(tickers)}_{start}_{end}.pkl")
    if os.path.exists(key):
        return pd.read_pickle(key)
    import yfinance as yf
    px = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    px = px.dropna(axis=1, how="any").dropna()
    px.to_pickle(key)
    return px


# ---------------- metrics ----------------

def perf_metrics(daily_ret: np.ndarray, rf=0.0, ppy=252):
    r = np.asarray(daily_ret, float)
    ann_ret = r.mean() * ppy
    ann_vol = r.std(ddof=1) * np.sqrt(ppy)
    sharpe = (ann_ret - rf) / ann_vol if ann_vol > 1e-12 else float("nan")
    downside = r[r < 0]
    dvol = downside.std(ddof=1) * np.sqrt(ppy) if len(downside) > 1 else float("nan")
    sortino = (ann_ret - rf) / dvol if dvol and dvol > 1e-12 else float("nan")
    cum = np.cumprod(1 + r)
    peak = np.maximum.accumulate(cum)
    maxdd = float(((cum - peak) / peak).min())
    return {"ann_return": float(ann_ret), "ann_vol": float(ann_vol), "sharpe": float(sharpe),
            "sortino": float(sortino), "maxdd": maxdd, "final_wealth": float(cum[-1])}


# ---------------- strategies (return weight vector over the universe) ----------------

def s_equal(mu, Sigma, **kw):
    n = len(mu); return np.ones(n) / n


def _convex(mu, Sigma, lam=None, target=None, long_only=True):
    import cvxpy as cp
    n = len(mu); w = cp.Variable(n, nonneg=long_only)
    cons = [cp.sum(w) == 1]
    if target is not None:
        cons.append(mu @ w >= target)
        obj = cp.Minimize(cp.quad_form(w, cp.psd_wrap(Sigma)))
    else:
        obj = cp.Minimize(lam * cp.quad_form(w, cp.psd_wrap(Sigma)) - (1 - lam) * (mu @ w))
    try:
        cp.Problem(obj, cons).solve(solver=cp.CLARABEL)
        if w.value is None:
            return np.ones(n) / n
        v = np.maximum(w.value, 0); return v / v.sum()
    except Exception:
        return np.ones(n) / n


def s_minvar(mu, Sigma, **kw):
    return _convex(mu, Sigma, lam=1.0)


def s_maxsharpe(mu, Sigma, **kw):
    # grid over lambda, pick max realized in-sample Sharpe (simple, robust)
    best_w, best_s = None, -np.inf
    for lam in np.linspace(0.05, 0.95, 19):
        w = _convex(mu, Sigma, lam=lam)
        v = float(w @ Sigma @ w)
        s = (mu @ w) / np.sqrt(v) if v > 1e-12 else -np.inf
        if s > best_s:
            best_s, best_w = s, w
    return best_w


def s_markowitz(mu, Sigma, lam=0.5, **kw):
    return _convex(mu, Sigma, lam=lam)


def make_gnn_cardinality(K, lam=0.5, device="cuda"):
    from qubo_portfolio import selection_qubo, decode_selection
    from baselines import convex_reweight
    from gnn_solver import solve_qubo_gnn, GNNHypers
    h = GNNHypers(model="qrf", epochs=600, hidden=96, dim_embedding=20, n_layers=3, lr=2e-3,
                  anneal_rate=0.0, eval_every=50, patience=250, ls_passes=60, n_round_samples=10,
                  refine_sa=True, refine_reads=15)

    def strat(mu, Sigma, **kw):
        n = len(mu)
        q = selection_qubo(mu, Sigma, min(K, n), risk_aversion=lam, return_weight=(1 - lam))
        r = solve_qubo_gnn(q, h, device=device, seed=0)
        S = list(decode_selection(r["x"]))
        if len(S) != min(K, n):
            S = list(np.argsort(-r["x"])[:min(K, n)])
        return convex_reweight(mu, Sigma, S, risk_aversion=lam, return_weight=(1 - lam))
    return strat


def s_mom_minvar(K):
    """Select top-K by the signal in `mu` (momentum), then MIN-VARIANCE weight them (robust combo)."""
    def strat(mu, Sigma, **kw):
        n = len(mu); S = np.argsort(-mu)[:min(K, n)]
        s = Sigma[np.ix_(S, S)]
        import cvxpy as cp
        x = cp.Variable(len(S), nonneg=True)
        try:
            cp.Problem(cp.Minimize(cp.quad_form(x, cp.psd_wrap(s))), [cp.sum(x) == 1]).solve(solver=cp.CLARABEL)
            w = np.zeros(n); w[S] = np.maximum(x.value, 0); return w / w.sum()
        except Exception:
            w = np.zeros(n); w[S] = 1.0 / len(S); return w
    return strat


def make_gnn_mom_minvar(K, device="cuda"):
    """GNN selects K assets on a momentum+risk QUBO, then MIN-VARIANCE re-weights the chosen support."""
    from qubo_portfolio import selection_qubo, decode_selection
    from baselines import convex_reweight
    from gnn_solver import solve_qubo_gnn, GNNHypers
    h = GNNHypers(model="qrf", epochs=600, hidden=96, dim_embedding=20, n_layers=3, lr=2e-3,
                  anneal_rate=0.0, eval_every=50, patience=250, ls_passes=60, n_round_samples=10,
                  refine_sa=True, refine_reads=15)

    def strat(mu, Sigma, **kw):
        n = len(mu); Kk = min(K, n)
        q = selection_qubo(mu, Sigma, Kk, risk_aversion=1.0, return_weight=1.0)
        r = solve_qubo_gnn(q, h, device=device, seed=0)
        S = list(decode_selection(r["x"]))
        if len(S) != Kk:
            S = list(np.argsort(-r["x"])[:Kk])
        return convex_reweight(mu, Sigma, S, risk_aversion=1.0, return_weight=0.0)  # min-var weights
    return strat


def make_sa_cardinality(K, lam=0.5):
    from qubo_portfolio import selection_qubo, decode_selection
    from baselines import convex_reweight, sa_qubo

    def strat(mu, Sigma, **kw):
        n = len(mu)
        q = selection_qubo(mu, Sigma, min(K, n), risk_aversion=lam, return_weight=(1 - lam))
        S = list(decode_selection(sa_qubo(q, num_reads=80, seed=0)["x"]))
        if len(S) != min(K, n):
            return np.ones(n) / n
        return convex_reweight(mu, Sigma, S, risk_aversion=lam, return_weight=(1 - lam))
    return strat


# ---------------- walk-forward engine ----------------

def backtest(prices: pd.DataFrame, strategies: dict, lookback=252, rebalance=21,
             start_idx=None, ann_mu=True, mu_forecaster=None, cov_fn=None):
    rets = prices.pct_change().dropna()
    R = rets.values
    dates = rets.index
    T, n = R.shape
    if start_idx is None:
        start_idx = lookback
    out = {name: {"daily": [], "weights_prev": None, "turnover": []} for name in strategies}
    t = start_idx
    while t < T:
        window = R[t - lookback:t]
        mu = mu_forecaster(R, t, lookback) if mu_forecaster is not None else window.mean(axis=0)
        Sigma = cov_fn(window) if cov_fn is not None else np.cov(window, rowvar=False)
        Sigma = 0.5 * (Sigma + Sigma.T) + 1e-8 * np.eye(n)
        horizon = min(rebalance, T - t)
        for name, fn in strategies.items():
            w = np.asarray(fn(mu, Sigma), float)
            if w.sum() <= 0 or not np.isfinite(w).all():
                w = np.ones(n) / n
            w = w / w.sum()
            # turnover vs previous drifted weights
            prev = out[name]["weights_prev"]
            if prev is not None:
                out[name]["turnover"].append(float(np.abs(w - prev).sum()))
            # accumulate daily portfolio returns over the holding horizon, drift weights
            wt = w.copy()
            for h_ in range(horizon):
                day = R[t + h_]
                out[name]["daily"].append(float(wt @ day))
                wt = wt * (1 + day); wt = wt / wt.sum()
            out[name]["weights_prev"] = wt
        t += horizon
    results = {}
    for name in strategies:
        m = perf_metrics(np.array(out[name]["daily"]))
        m["turnover"] = float(np.mean(out[name]["turnover"])) if out[name]["turnover"] else float("nan")
        m["n_rebal"] = len(out[name]["turnover"]) + 1
        results[name] = m
    return results, dates[start_idx], dates[-1]


def main():
    import json, time
    uni = sys.argv[1] if len(sys.argv) > 1 else "dow"
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    rebalance = int(sys.argv[3]) if len(sys.argv) > 3 else 63
    start = sys.argv[4] if len(sys.argv) > 4 else "2005-01-01"
    mu_mode = sys.argv[5] if len(sys.argv) > 5 else "hist"   # hist | ridge | hgb
    tickers = DOW30 if uni == "dow" else SP100
    px = load_prices(tickers, start, "2024-12-31")
    mu_forecaster, cov_fn = None, None
    if mu_mode in ("ridge", "hgb"):
        from mu_forecast import make_ml_mu_forecaster, ledoit_wolf_cov
        mu_forecaster = make_ml_mu_forecaster(model=mu_mode, horizon=rebalance)
        cov_fn = ledoit_wolf_cov
    elif mu_mode == "mom":
        from mu_forecast import momentum_mu, ledoit_wolf_cov
        mu_forecaster = momentum_mu
        cov_fn = ledoit_wolf_cov
    print(f"=== backtest {uni} | universe {px.shape[1]} assets | K={K} | rebalance={rebalance}d | "
          f"mu={mu_mode} cov={'LW' if cov_fn else 'sample'} | "
          f"{px.index[0].date()}->{px.index[-1].date()} ===", flush=True)
    lam = float(os.environ.get("BT_LAM", "0.5"))
    if mu_mode == "mom":
        # momentum = SELECTION signal; weighting = min-variance (robust factor combo)
        strats = {
            "EqualWeight": s_equal, "MinVar": s_minvar,
            f"Mom-MinVar(K{K})": s_mom_minvar(K),
            f"GNN-Mom-MinVar(K{K})": make_gnn_mom_minvar(K),
        }
    else:
        strats = {
            "EqualWeight": s_equal, "MinVar": s_minvar, "MaxSharpe": s_maxsharpe,
            f"Markowitz{lam}": (lambda mu, S, **k: s_markowitz(mu, S, lam=lam)),
            f"SA-card(K{K})": make_sa_cardinality(K, lam=lam),
            f"GNN-card(K{K})": make_gnn_cardinality(K, lam=lam),
        }
    t0 = time.time()
    res, d0, d1 = backtest(px, strats, lookback=252, rebalance=rebalance,
                           mu_forecaster=mu_forecaster, cov_fn=cov_fn)
    print(f"OOS {d0.date()} -> {d1.date()}  ({time.time()-t0:.0f}s)")
    hdr = f"{'strategy':<16}{'Sharpe':>8}{'Sortino':>9}{'AnnRet':>9}{'Vol':>8}{'MaxDD':>8}{'Turn':>8}{'FinW':>8}"
    print(hdr); print("-" * len(hdr))
    for k, m in res.items():
        print(f"{k:<16}{m['sharpe']:>8.3f}{m['sortino']:>9.3f}{m['ann_return']:>9.3f}{m['ann_vol']:>8.3f}"
              f"{m['maxdd']:>8.3f}{m['turnover']:>8.3f}{m['final_wealth']:>8.2f}")
    os.makedirs("results/backtest", exist_ok=True)
    json.dump({k: v for k, v in res.items()},
              open(f"results/backtest/{uni}_K{K}_reb{rebalance}_{mu_mode}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
