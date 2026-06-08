"""E49 (loop): a regime-robust per-instance solver. E47/E48 showed cold tabu can fail in the risk-dominated
regime while greedy is robust. Test greedy-warm-started tabu (init tabu from greedy's solution) vs cold
tabu / greedy / GNN+LS across lambda in {0.5,0.9,0.99} on S&P100. Hypothesis: greedy+tabu is robust
everywhere (good init + refinement). Run in .venv.
"""
import os, sys, json, numpy as np
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE"); os.environ.setdefault("OMP_NUM_THREADS", "6")
HERE = os.path.dirname(__file__); sys.path.insert(0, os.path.join(HERE, "..", "src"))
from backtest import load_prices, SP100
from baselines import tabu_qubo, greedy_selection
from qubo_portfolio import selection_qubo
from gnn_solver import solve_qubo_gnn, GNNHypers
from tabu import TabuSampler


def warm_tabu(q, x0, k=8):
    init = [{i: int(x0[i]) for i in range(q.n)} for _ in range(k)]
    res = TabuSampler().sample(q.to_dimod(), num_reads=k, seed=0, initial_states=init); b = res.first
    return q.energy(np.array([b.sample[i] for i in range(q.n)], np.int8))


def main():
    R = load_prices(SP100, "2005-01-01", "2024-12-31").pct_change().dropna().values; N = R.shape[1]
    lb = 252; ts = list(range(lb, len(R), 120))[-8:]
    def est(t):
        w = R[t - lb:t]; mu = w.mean(0); S = np.cov(w, rowvar=False); return mu, 0.5 * (S + S.T) + 1e-8 * np.eye(N)
    K = 15
    h = GNNHypers(model="qrf", epochs=2000, hidden=128, dim_embedding=20, n_layers=3, lr=1e-3, anneal_rate=0.0,
                  eval_every=50, patience=400, ls_passes=150, n_round_samples=16, refine_sa=True, refine_reads=30)
    print("E49 regime-robust solver across lambda", flush=True)
    out = {}
    for lam in [0.5, 0.9, 0.99]:
        acc = {k: [] for k in ["cold_tabu200", "greedy", "greedy+tabu8", "gnn+ls"]}
        for t in ts:
            mu, S = est(t); q = selection_qubo(mu, S, K, risk_aversion=lam, return_weight=1 - lam)
            gres = greedy_selection(mu, S, K, risk_aversion=lam, return_weight=1 - lam)
            e_greedy = gres["energy"]; xg = np.asarray(gres["x"])
            e_cold = tabu_qubo(q, num_reads=200, seed=0)["energy"]
            e_gt = warm_tabu(q, xg, k=8)
            e_gnn = solve_qubo_gnn(q, h, device="cuda", seed=0)["energy"]
            best = min(e_cold, e_greedy, e_gt, e_gnn)
            for k, e in [("cold_tabu200", e_cold), ("greedy", e_greedy), ("greedy+tabu8", e_gt), ("gnn+ls", e_gnn)]:
                acc[k].append((e - best) / abs(best) * 100 if abs(best) > 1e-12 else 0.0)
        row = {k: float(np.mean(v)) for k, v in acc.items()}; out[lam] = row
        print(f"  lambda={lam}: " + " | ".join(f"{k} {row[k]:.3f}%" for k in row) + f"  -> best: {min(row,key=row.get)}", flush=True)
    # worst-case across lambda per method (robustness)
    methods = list(next(iter(out.values())).keys())
    worst = {m: max(out[l][m] for l in out) for m in methods}
    print("  WORST-CASE gap across lambda (robustness): " + " | ".join(f"{m} {worst[m]:.3f}%" for m in methods), flush=True)
    print(f"  => most robust (lowest worst-case): {min(worst,key=worst.get)}", flush=True)
    json.dump({"per_lambda": out, "worst_case": worst}, open(os.path.join(HERE, "results", "e49_robust_solver.json"), "w"), indent=2)
    print("saved", flush=True)


if __name__ == "__main__":
    main()
