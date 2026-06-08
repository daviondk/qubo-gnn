"""E51 (loop): is the min-variance regime (lambda=0.99, where the GNN earns its keep, E46-E50) actually
investable? Walk-forward S&P100 backtest comparing the min-variance cardinality selection (greedy+tabu vs
GNN+LS) against balanced (lambda=0.5) and equal-weight. Does min-variance deliver lower realized volatility
/ good Sharpe (low-vol anomaly), and do the solvers tie in realized investment terms? Run in .venv.
"""
import os, sys, json, numpy as np
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE"); os.environ.setdefault("OMP_NUM_THREADS", "6")
HERE = os.path.dirname(__file__); sys.path.insert(0, os.path.join(HERE, "..", "src"))
from backtest import load_prices, SP100, perf_metrics
from baselines import tabu_qubo, greedy_selection, convex_reweight
from qubo_portfolio import selection_qubo, decode_selection
from gnn_solver import solve_qubo_gnn, GNNHypers
from tabu import TabuSampler


def warm_tabu(q, x0, k=8):
    init = [{i: int(x0[i]) for i in range(q.n)} for _ in range(k)]
    b = TabuSampler().sample(q.to_dimod(), num_reads=k, seed=0, initial_states=init).first
    return np.array([b.sample[i] for i in range(q.n)], np.int8)


def main():
    R = load_prices(SP100, "2005-01-01", "2024-12-31").pct_change().dropna().values; N = R.shape[1]
    lb, step, cost = 252, 63, 10 / 1e4
    reb = list(range(lb, len(R) - step, step)); te = reb[int(0.5 * len(reb)):]
    def est(t):
        w = R[t - lb:t]; mu = w.mean(0); S = np.cov(w, rowvar=False); return mu, 0.5 * (S + S.T) + 1e-8 * np.eye(N)
    K = 15
    h = GNNHypers(model="qrf", epochs=1500, hidden=128, dim_embedding=20, n_layers=3, lr=1e-3, anneal_rate=0.0,
                  eval_every=50, patience=400, ls_passes=150, n_round_samples=16, refine_sa=True, refine_reads=30)
    print(f"E51 min-variance investability ({len(te)} OOS rebalances)", flush=True)
    str3 = {
        "MinVar(0.99) greedy+tabu": ("gt", 0.99),
        "MinVar(0.99) GNN+LS": ("gnn", 0.99),
        "Balanced(0.5) greedy+tabu": ("gt", 0.5),
        "EqualWeight": ("ew", None),
    }
    acc = {k: {"net": [], "turn": [], "prev": None} for k in str3}
    for t in te:
        mu, S = est(t); rn = R[t:t + step]
        sel_cache = {}
        for nm, (kind, lam) in str3.items():
            if kind == "ew":
                w = np.ones(N) / N
            else:
                q = selection_qubo(mu, S, K, risk_aversion=lam, return_weight=1 - lam)
                key = (kind, lam)
                if key not in sel_cache:
                    if kind == "gt":
                        xg = np.asarray(greedy_selection(mu, S, K, risk_aversion=lam, return_weight=1 - lam)["x"])
                        x = warm_tabu(q, xg, 8); sel = np.flatnonzero(x > 0.5)
                    else:
                        sel = decode_selection(solve_qubo_gnn(q, h, device="cuda", seed=0)["x"])
                    sel_cache[key] = sel
                sel = sel_cache[key]
                w = convex_reweight(mu, S, sel, risk_aversion=lam, return_weight=1 - lam, eps=0.0, delta=0.25) if len(sel) == K else np.ones(N) / N
            prev = acc[nm]["prev"]; turn = float(np.abs(w - prev).sum()) if prev is not None else 1.0
            d = rn @ w; d[0] -= cost * turn; acc[nm]["net"].extend(d.tolist()); acc[nm]["turn"].append(turn); acc[nm]["prev"] = w
    print(f"\n{'strategy':<26}{'Sharpe':>8}{'AnnVol':>8}{'MaxDD':>8}{'Turn':>7}", flush=True)
    out = {}
    for nm in str3:
        dd = np.asarray(acc[nm]["net"]); m = perf_metrics(dd); vol = float(dd.std() * np.sqrt(252))
        out[nm] = {"sharpe": m["sharpe"], "annvol": vol, "maxdd": m["maxdd"], "turn": float(np.mean(acc[nm]["turn"]))}
        print(f"{nm:<26}{m['sharpe']:>8.3f}{vol:>8.3f}{m['maxdd']:>8.3f}{np.mean(acc[nm]['turn']):>7.2f}", flush=True)
    json.dump(out, open(os.path.join(HERE, "results", "e51_minvar_backtest.json"), "w"), indent=2); print("saved", flush=True)


if __name__ == "__main__":
    main()
