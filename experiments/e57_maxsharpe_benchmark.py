"""E57 (USER DIRECTIVE): max-Sharpe financial task (DSL/Kim 2025 objective) on a real universe (S&P100),
causal monthly walk-forward, DSL protocol (40 bps), reporting DSL's metrics (CR, Sharpe, Sortino, MaxDD,
turnover). Tests whether our QUBO cardinality solver handles the max-Sharpe task, and reproduces the
optimizer-vs-investor effect on THEIR objective:
  - EqualWeight (their baseline)
  - PlugIn-MaxSharpe-ALL (in-sample tangency on all assets, causal next month) = the "naive optimum"
  - Ours: cardinality-K QUBO selection (tabu / GNN) + max-Sharpe (tangency) reweight on the support
DSL reported: S&P-Top30 Sharpe 1.10; S&P500-rolling 0.47; our DSL-code run on S&P100 = 0.672.
Run in .venv.
"""
import os, sys, json, numpy as np
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE"); os.environ.setdefault("OMP_NUM_THREADS", "6")
HERE = os.path.dirname(__file__); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import cvxpy as cp
from backtest import load_prices, SP100, perf_metrics
from baselines import tabu_qubo
from qubo_portfolio import selection_qubo, decode_selection
from gnn_solver import solve_qubo_gnn, GNNHypers
K = 20


def tangency(sel, mu, S):
    k = len(sel); muS = mu[sel]; SS = S[np.ix_(sel, sel)]
    if (muS <= 0).all(): return np.ones(k) / k
    y = cp.Variable(k)
    try:
        cp.Problem(cp.Minimize(cp.quad_form(y, cp.psd_wrap(SS))), [muS @ y == 1, y >= 0]).solve(solver=cp.CLARABEL)
        if y.value is None or y.value.sum() <= 0: return np.ones(k) / k
        w = np.clip(y.value, 0, None); return w / w.sum()
    except Exception:
        return np.ones(k) / k


def metrics(daily, turns):
    d = np.asarray(daily); ann = 252
    cr = float(np.prod(1 + d) - 1); sh = float(d.mean() / (d.std() + 1e-12) * np.sqrt(ann))
    dn = d[d < 0]; so = float(d.mean() / (dn.std() + 1e-12) * np.sqrt(ann)) if len(dn) else float("nan")
    eq = np.cumprod(1 + d); mdd = float((eq / np.maximum.accumulate(eq) - 1).min())
    return {"CR": cr, "Sharpe": sh, "Sortino": so, "MaxDD": mdd, "turn": float(np.mean(turns))}


def main():
    R = load_prices(SP100, "2005-01-01", "2025-12-31").pct_change().dropna()
    dates = R.index; Rv = R.values; N = Rv.shape[1]
    # monthly rebalance from 2015 (causal), 252d trailing for mu/Sigma
    reb = []; cm = None
    for dt in dates[dates >= "2015-01-01"]:
        if dt.month != cm: reb.append(dt); cm = dt.month
    idx = {d: i for i, d in enumerate(dates)}; lb = 252
    h = GNNHypers(model="qrf", epochs=1200, hidden=128, dim_embedding=16, n_layers=3, lr=1e-3, anneal_rate=0.0,
                  eval_every=50, patience=300, ls_passes=100, n_round_samples=16, refine_sa=True, refine_reads=30)
    methods = ["EqualWeight", "PlugIn-MaxSharpe-ALL", "Ours-tabu-K20", "Ours-GNN-K20"]
    acc = {m: {"daily": [], "turn": [], "prev": None} for m in methods}
    print(f"E57 max-Sharpe on S&P100, causal monthly, 40bps, {len(reb)} rebalances from 2015", flush=True)
    LAM = 0.5
    for ri, rd in enumerate(reb):
        i = idx[rd]
        if i < lb: continue
        wn = Rv[i - lb:i]; mu = wn.mean(0) * 252; S = np.cov(wn, rowvar=False) * 252; S = 0.5 * (S + S.T) + 1e-8 * np.eye(N)
        end = idx[reb[ri + 1]] if ri + 1 < len(reb) else len(dates); seg = Rv[i:end]
        q = selection_qubo(mu, S, K, risk_aversion=LAM, return_weight=1 - LAM)
        sel_tabu = decode_selection(tabu_qubo(q, num_reads=120, seed=0)["x"])
        sel_gnn = decode_selection(solve_qubo_gnn(q, h, device="cuda", seed=0)["x"])
        for m in methods:
            w = np.zeros(N)
            if m == "EqualWeight":
                w[:] = 1 / N
            elif m == "PlugIn-MaxSharpe-ALL":
                w = tangency(np.arange(N), mu, S)
            elif m == "Ours-tabu-K20":
                sel = sel_tabu if len(sel_tabu) == K else np.argsort(-mu)[:K]; w[sel] = tangency(np.array(sel), mu, S)
            else:
                sel = sel_gnn if len(sel_gnn) == K else np.argsort(-mu)[:K]; w[sel] = tangency(np.array(sel), mu, S)
            prev = acc[m]["prev"]; turn = float(np.abs(w - prev).sum()) if prev is not None else 1.0
            sd = (seg @ w).copy();
            if len(sd): sd[0] -= 40e-4 * turn  # 40 bps * turnover (DSL)
            acc[m]["daily"].extend(sd.tolist()); acc[m]["turn"].append(turn); acc[m]["prev"] = w
    print(f"\n{'method':<22}{'CR':>9}{'Sharpe':>8}{'Sortino':>9}{'MaxDD':>8}{'Turn':>7}", flush=True)
    print(f"{'(DSL S&P-Top30)':<22}{'8.50':>9}{1.10:>8}{1.78:>9}{'n/a':>8}{'n/a':>7}  <- their paper", flush=True)
    print(f"{'(DSL S&P500-roll)':<22}{'2.24':>9}{0.47:>8}{0.75:>9}{'n/a':>8}{'n/a':>7}  <- their paper", flush=True)
    out = {}
    for m in methods:
        mm = metrics(acc[m]["daily"], acc[m]["turn"]); out[m] = mm
        print(f"{m:<22}{mm['CR']*100:>8.1f}%{mm['Sharpe']:>8.2f}{mm['Sortino']:>9.2f}{mm['MaxDD']*100:>7.1f}%{mm['turn']:>7.2f}", flush=True)
    json.dump(out, open(os.path.join(HERE, "results", "e57_maxsharpe.json"), "w"), indent=2); print("saved", flush=True)


if __name__ == "__main__":
    main()
