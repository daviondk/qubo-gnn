"""Scaling demo: synthetic factor-model market where exact MIQP is infeasible (free Gurobi license).
GNN vs greedy/SA/random on the cardinality selection QUBO. Saves results/scaling/synth_N{N}_K{K}.json.
"""
from __future__ import annotations

import sys, os, json
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from qubo_portfolio import selection_qubo, decode_selection
from baselines import sa_qubo, greedy_selection, random_selection, convex_reweight
from gnn_solver import solve_qubo_gnn, GNNHypers


def synth_market(N, n_factors=8, seed=7):
    rng = np.random.default_rng(seed)
    B = rng.standard_normal((N, n_factors)) * 0.02
    d = np.abs(rng.standard_normal(N)) * 0.01 + 0.005
    Sigma = B @ B.T + np.diag(d ** 2)
    mu = rng.standard_normal(N) * 0.004 + 0.003
    return mu, 0.5 * (Sigma + Sigma.T)


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    ra, rw = 1.0, 0.5
    mu, Sigma = synth_market(N)
    q = selection_qubo(mu, Sigma, K, risk_aversion=ra, return_weight=rw)

    def obj(x):
        s = decode_selection(x)
        if len(s) != K:
            return float("nan")
        w = convex_reweight(mu, Sigma, s, risk_aversion=ra, return_weight=rw)
        return float(ra * (w @ Sigma @ w) - rw * (mu @ w))

    res = {}
    r = greedy_selection(mu, Sigma, K, risk_aversion=ra, return_weight=rw); res["Greedy"] = {"obj": obj(r["x"]), "time": r["time"]}
    r = sa_qubo(q, num_reads=200, seed=0); res["SA"] = {"obj": obj(r["x"]), "time": r["time"]}
    r = random_selection(mu, Sigma, K, risk_aversion=ra, return_weight=rw, n_tries=5000, seed=0); res["Random"] = {"obj": obj(r["x"]), "time": r["time"]}
    h = GNNHypers(model="qrf", epochs=1500, hidden=128, dim_embedding=32, n_layers=3, lr=1e-3,
                  anneal_rate=0.0, eval_every=50, patience=500, ls_passes=120, n_round_samples=16,
                  refine_sa=True, refine_reads=30)
    r = solve_qubo_gnn(q, h, device="cuda", seed=0); res["GNN"] = {"obj": obj(r["x"]), "time": r["time"]}

    best = min(v["obj"] for v in res.values())
    for v in res.values():
        v["gap_pct"] = (v["obj"] - best) / abs(best) * 100
    print(f"Synthetic factor market N={N} K={K} (exact MIQP infeasible on free license)")
    print(f"{'method':<10}{'obj':>12}{'gap%':>9}{'t(s)':>8}")
    for m, v in res.items():
        print(f"{m:<10}{v['obj']:>12.6f}{v['gap_pct']:>9.3f}{v['time']:>8.2f}")
    os.makedirs("results/scaling", exist_ok=True)
    with open(f"results/scaling/synth_N{N}_K{K}.json", "w") as f:
        json.dump({"N": N, "K": K, "results": res}, f, indent=2)


if __name__ == "__main__":
    main()
