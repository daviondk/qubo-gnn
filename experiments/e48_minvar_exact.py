"""E48 (loop): definitive hard-regime (min-variance-like, lambda=0.99) characterization vs EXACT.
On S&P100 windows, compare SCIP global-QUBO (exact, true optimum) vs greedy / tabu(200) / GNN+LS on the
selection-QUBO. Is the min-variance cardinality QUBO hard for exact? Which heuristic is closest? Confirms
the E46/E47 hard-regime picture against ground truth. Run in .venv.
"""
import os, sys, json, numpy as np
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE"); os.environ.setdefault("OMP_NUM_THREADS", "6")
HERE = os.path.dirname(__file__); sys.path.insert(0, os.path.join(HERE, "..", "src"))
from backtest import load_prices, SP100
from baselines import tabu_qubo, sa_qubo, greedy_selection, scip_qubo
from qubo_portfolio import selection_qubo
from gnn_solver import solve_qubo_gnn, GNNHypers


def main():
    R = load_prices(SP100, "2005-01-01", "2024-12-31").pct_change().dropna().values; N = R.shape[1]
    lb = 252; ts = list(range(lb, len(R), 120))[-8:]
    def est(t):
        w = R[t - lb:t]; mu = w.mean(0); S = np.cov(w, rowvar=False); return mu, 0.5 * (S + S.T) + 1e-8 * np.eye(N)
    K, lam = 15, 0.99
    h = GNNHypers(model="qrf", epochs=2500, hidden=128, dim_embedding=20, n_layers=3, lr=1e-3, anneal_rate=0.0,
                  eval_every=50, patience=500, ls_passes=200, n_round_samples=20, refine_sa=True, refine_reads=40)
    print(f"E48 min-variance regime (lambda={lam}) vs EXACT (SCIP), {len(ts)} windows", flush=True)
    acc = {k: [] for k in ["scip_exact", "greedy", "tabu200", "gnn+ls"]}; scip_to = 0
    for t in ts:
        mu, S = est(t); q = selection_qubo(mu, S, K, risk_aversion=lam, return_weight=1 - lam)
        rs = scip_qubo(q, time_limit=60)
        e_scip = rs["energy"]; scip_to += int(rs.get("status", "") != "optimal")
        e_greedy = greedy_selection(mu, S, K, risk_aversion=lam, return_weight=1 - lam)["energy"]
        e_tabu = tabu_qubo(q, num_reads=200, seed=0)["energy"]
        e_gnn = solve_qubo_gnn(q, h, device="cuda", seed=0)["energy"]
        best = min(e_scip, e_greedy, e_tabu, e_gnn)  # best-found = ground-truth proxy
        for k, e in [("scip_exact", e_scip), ("greedy", e_greedy), ("tabu200", e_tabu), ("gnn+ls", e_gnn)]:
            acc[k].append((e - best) / abs(best) * 100 if abs(best) > 1e-12 else 0.0)
    row = {k: float(np.mean(v)) for k, v in acc.items()}
    print(f"  gaps vs best-found: " + " | ".join(f"{k} {row[k]:.3f}%" for k in row), flush=True)
    print(f"  (SCIP non-optimal/timeout on {scip_to}/{len(ts)} windows)", flush=True)
    json.dump({"row": row, "scip_timeouts": scip_to, "n": len(ts)}, open(os.path.join(HERE, "results", "e48_minvar_exact.json"), "w"), indent=2)
    print("saved", flush=True)


if __name__ == "__main__":
    main()
