"""E47 (loop): per-instance solver comparison in the HARD (risk-dominated, high-lambda) regime that E46
identified. At lambda in {0.9, 0.99} on S&P100 windows, compare greedy / tabu(200) / SA(200) / GNN+LS
(solve_qubo_gnn) on selection-QUBO energy (gap vs best-found across all). Does the full GNN+local-search
beat greedy/tabu here -- the one regime where a per-instance GNN win might appear? Run in .venv.
"""
import os, sys, json, numpy as np
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE"); os.environ.setdefault("OMP_NUM_THREADS", "6")
HERE = os.path.dirname(__file__); sys.path.insert(0, os.path.join(HERE, "..", "src"))
from backtest import load_prices, SP100
from baselines import tabu_qubo, sa_qubo, greedy_selection
from qubo_portfolio import selection_qubo
from gnn_solver import solve_qubo_gnn, GNNHypers


def main():
    R = load_prices(SP100, "2005-01-01", "2024-12-31").pct_change().dropna().values; N = R.shape[1]
    lb = 252; ts = list(range(lb, len(R), 90))[-12:]  # 12 windows
    def est(t):
        w = R[t - lb:t]; mu = w.mean(0); S = np.cov(w, rowvar=False); return mu, 0.5 * (S + S.T) + 1e-8 * np.eye(N)
    K = 15
    h = GNNHypers(model="qrf", epochs=2500, hidden=128, dim_embedding=20, n_layers=3, lr=1e-3, anneal_rate=0.0,
                  eval_every=50, patience=500, ls_passes=200, n_round_samples=20, refine_sa=True, refine_reads=40)
    print("E47 hard-regime per-instance solvers (high lambda)", flush=True)
    out = {}
    for lam in [0.9, 0.99]:
        acc = {k: [] for k in ["greedy", "tabu200", "sa200", "gnn+ls"]}
        for t in ts:
            mu, S = est(t); q = selection_qubo(mu, S, K, risk_aversion=lam, return_weight=1 - lam)
            res = {}
            res["greedy"] = greedy_selection(mu, S, K, risk_aversion=lam, return_weight=1 - lam)["energy"]
            res["tabu200"] = tabu_qubo(q, num_reads=200, seed=0)["energy"]
            res["sa200"] = sa_qubo(q, num_reads=200, seed=0)["energy"]
            res["gnn+ls"] = solve_qubo_gnn(q, h, device="cuda", seed=0)["energy"]
            best = min(res.values())
            for k, e in res.items(): acc[k].append((e - best) / abs(best) * 100 if abs(best) > 1e-12 else 0.0)
        row = {k: float(np.mean(v)) for k, v in acc.items()}
        out[lam] = row
        win = min(row, key=row.get)
        print(f"  lambda={lam}: " + " | ".join(f"{k} {row[k]:.3f}%" for k in row) + f"  -> best: {win}", flush=True)
    json.dump(out, open(os.path.join(HERE, "results", "e47_hard_regime.json"), "w"), indent=2); print("saved", flush=True)


if __name__ == "__main__":
    main()
