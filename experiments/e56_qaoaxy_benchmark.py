"""E56 (loop, USER DIRECTIVE): head-to-head vs QAOA-XY (Mancilla 2026, arXiv:2602.14827) on THEIR task +
THEIR metrics. Exact replication: K=5 of N=10 tickers, cost min 0.3*w'Sigma_ann*w - 0.7*mu_ann*w with
w=x/K (equal-weight on selected), 180-day lookback, MONTHLY rebalance over 2025, 5 bps * turnover.
We slot OUR solvers (greedy / tabu / SA / GNN+LS) in place of their SA/QAOA and report their metrics
(total return, Sharpe, ann vol, MaxDD, monthly turnover). Their numbers: QAOA Sharpe 1.81 / SA 1.31 / HRP 0.98.
Run in .venv.
"""
import os, sys, json, numpy as np
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE"); os.environ.setdefault("OMP_NUM_THREADS", "6")
HERE = os.path.dirname(__file__); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import pandas as pd
from backtest import load_prices
from baselines import tabu_qubo, sa_qubo, greedy_selection
from qubo_portfolio import selection_qubo, decode_selection
from gnn_solver import solve_qubo_gnn, GNNHypers
TICKERS = ["AAPL","MSFT","GOOGL","AMZN","JPM","V","TSLA","UNH","LLY","XOM"]
K, Q = 5, 0.3


def metrics(daily):
    d = np.asarray(daily); ann = 252
    tot = float(np.prod(1 + d) - 1); vol = float(d.std() * np.sqrt(ann))
    sharpe = float(d.mean() / (d.std() + 1e-12) * np.sqrt(ann))
    eq = np.cumprod(1 + d); mdd = float((eq / np.maximum.accumulate(eq) - 1).min())
    return tot, sharpe, vol, mdd


def select(method, q, mu, S, h):
    if method == "greedy":
        return decode_selection(greedy_selection(mu, S, K, risk_aversion=Q, return_weight=1 - Q)["x"])
    if method == "tabu":
        return decode_selection(tabu_qubo(q, num_reads=200, seed=0)["x"])
    if method == "sa":
        return decode_selection(sa_qubo(q, num_reads=200, seed=0)["x"])
    if method == "gnn":
        return decode_selection(solve_qubo_gnn(q, h, device="cuda", seed=0)["x"])


def main():
    px = load_prices(TICKERS, "2023-12-01", "2025-12-31")
    px = px[[c for c in TICKERS if c in px.columns]]
    rets = px.pct_change().dropna()
    dates = rets.index
    # monthly rebalance points within 2025 (first trading day of each month)
    reb = []
    y2025 = dates[(dates >= "2025-01-01")]
    cur_m = None
    for dt in y2025:
        if dt.month != cur_m: reb.append(dt); cur_m = dt.month
    lookback = 180
    h = GNNHypers(model="qrf", epochs=1500, hidden=128, dim_embedding=16, n_layers=3, lr=1e-3, anneal_rate=0.0,
                  eval_every=50, patience=400, ls_passes=100, n_round_samples=16, refine_sa=True, refine_reads=30)
    methods = ["greedy", "tabu", "sa", "gnn"]
    print(f"E56 QAOA-XY replication: N={px.shape[1]} K={K} q={Q}, {len(reb)} monthly rebalances 2025", flush=True)
    acc = {m: {"daily": [], "prevw": None, "turn": []} for m in methods}
    Rv = rets.values; idx_of = {d: i for i, d in enumerate(dates)}
    for ri, rd in enumerate(reb):
        i = idx_of[rd]
        if i < lookback: continue
        w_hist = Rv[i - lookback:i]
        mu = w_hist.mean(0) * 252; S = np.cov(w_hist, rowvar=False) * 252; S = 0.5 * (S + S.T) + 1e-10 * np.eye(px.shape[1])
        q = selection_qubo(mu, S, K, risk_aversion=Q, return_weight=1 - Q)
        end = idx_of[reb[ri + 1]] if ri + 1 < len(reb) else len(dates)
        seg = Rv[i:end]
        for m in methods:
            sel = select(m, q, mu, S, h)
            w = np.zeros(px.shape[1]);
            if len(sel) == K: w[sel] = 1 / K
            else: w[np.argsort(-mu)[:K]] = 1 / K
            prevw = acc[m]["prevw"]; turn = float(np.abs(w - prevw).sum()) if prevw is not None else 1.0
            seg_d = seg @ w
            if len(seg_d) > 0: seg_d = seg_d.copy(); seg_d[0] -= 5e-4 * turn  # 5 bps * turnover
            acc[m]["daily"].extend(seg_d.tolist()); acc[m]["turn"].append(turn); acc[m]["prevw"] = w
    print(f"\n{'method':<10}{'TotRet':>9}{'Sharpe':>8}{'AnnVol':>8}{'MaxDD':>8}{'MoTurn':>8}", flush=True)
    print(f"{'(QAOA-XY)':<10}{'30.09%':>9}{1.81:>8}{'18.55%':>8}{'-8.27%':>8}{'76.8%':>8}  <- their paper", flush=True)
    print(f"{'(SA-paper)':<10}{'24.17%':>9}{1.31:>8}{'19.53%':>8}{'-9.26%':>8}{'21.0%':>8}  <- their paper", flush=True)
    print(f"{'(HRP)':<10}{'10.88%':>9}{0.98:>8}{'10.65%':>8}{'-8.40%':>8}{'21.6%':>8}  <- their paper", flush=True)
    out = {}
    for m in methods:
        tot, sh, vol, mdd = metrics(acc[m]["daily"]); mt = float(np.mean(acc[m]["turn"]))
        out[m] = {"total_return": tot, "sharpe": sh, "ann_vol": vol, "maxdd": mdd, "monthly_turnover": mt}
        print(f"{('ours:'+m):<10}{tot*100:>8.2f}%{sh:>8.2f}{vol*100:>7.2f}%{mdd*100:>7.2f}%{mt*100:>7.1f}%", flush=True)
    json.dump(out, open(os.path.join(HERE, "results", "e56_qaoaxy.json"), "w"), indent=2); print("saved", flush=True)


if __name__ == "__main__":
    main()
