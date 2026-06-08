"""OR-Library cardinality portfolio: compute the EXACT Cura (2009) MED/VRE/MRE for our methods and
compare against published numbers (GA/TS/SA/PSO from Cura 2009; IPSO-SA; Firefly mFA).

Setup: K=10, eps=0.01, delta=1, unconstrained frontier = 2000 pts, heuristic frontier = 51 lambdas.
For each lambda the cardinality problem is min lambda*w'Sigma w - (1-lambda)*mu'w.
Methods: GNN (explore+exploit), exact MIQP (where Gurobi license allows), SA, Greedy.
"""
from __future__ import annotations

import sys, os, json, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from portfolio_data import download_orlib, load_orlib, ORLIB_FILES
from qubo_portfolio import selection_qubo, decode_selection
from baselines import miqp_cardinality, convex_reweight, sa_qubo, greedy_selection
from gnn_solver import solve_qubo_gnn, GNNHypers
from orlib_metrics import unconstrained_frontier, cura_metrics, lambda_grid

K = 10
EPS, DELTA = 0.01, 1.0
PUBLISHED = {  # Cura 2009 MED, plus best-known (IPSO-SA / Firefly)
    "port1": {"GA": 0.0040, "TS": 0.0040, "SA*": 0.0040, "PSO": 0.0049, "IPSO-SA": 0.0001, "Firefly": 0.0003},
    "port2": {"GA": 0.0076, "TS": 0.0082, "SA*": 0.0078, "PSO": 0.0090, "IPSO-SA": 0.0001, "Firefly": 0.0009},
    "port3": {"GA": 0.0020, "TS": 0.0021, "SA*": 0.0021, "PSO": 0.0022, "IPSO-SA": 0.0000, "Firefly": 0.0004},
    "port4": {"GA": 0.0041, "TS": 0.0041, "SA*": 0.0041, "PSO": 0.0052, "IPSO-SA": 0.0001, "Firefly": 0.0003},
    "port5": {"GA": 0.0093, "TS": 0.0010, "SA*": 0.0010, "PSO": 0.0019, "IPSO-SA": 0.0000, "Firefly": 0.0000},
}


def card_point(method, mu, Sigma, lam, device="cuda"):
    """Return (variance, return) of method's cardinality portfolio for tradeoff lam."""
    ra, rw = float(lam), float(1 - lam)
    if method == "MIQP":
        mi = miqp_cardinality(mu, Sigma, K, risk_aversion=ra, return_weight=rw, eps=EPS, delta=DELTA)
        w = mi["weights"]
    else:
        q = selection_qubo(mu, Sigma, K, risk_aversion=ra, return_weight=rw)
        if method == "SA":
            x = sa_qubo(q, num_reads=100, seed=0)["x"]
        elif method == "Greedy":
            x = greedy_selection(mu, Sigma, K, risk_aversion=ra, return_weight=rw)["x"]
        elif method == "GNN":
            h = GNNHypers(model="qrf", epochs=1000, hidden=128, dim_embedding=24, n_layers=3, lr=1e-3,
                          anneal_rate=0.0, eval_every=50, patience=400, ls_passes=80,
                          n_round_samples=12, refine_sa=True, refine_reads=20)
            x = solve_qubo_gnn(q, h, device=device, seed=0)["x"]
        sup = decode_selection(x)
        w = convex_reweight(mu, Sigma, sup, risk_aversion=ra, return_weight=rw, eps=EPS, delta=DELTA)
    return float(w @ Sigma @ w), float(mu @ w)


def run_instance(name, methods, device="cuda"):
    mu, Sigma, _ = load_orlib(download_orlib("data/orlib")[name])
    n = len(mu)
    sv, sr = unconstrained_frontier(mu, Sigma, n_points=2000)
    lams = lambda_grid(51)
    out = {}
    for m in methods:
        t0 = time.time(); V, R = [], []
        for lam in lams:
            try:
                v, r = card_point(m, mu, Sigma, lam, device=device)
                V.append(v); R.append(r)
            except Exception as e:
                pass
        met = cura_metrics(V, R, sv, sr)
        met["time"] = time.time() - t0; met["frontier_var"] = V; met["frontier_ret"] = R
        out[m] = met
        print(f"  {name} {m:<7} MED={met['MED']:.4f} VRE={met['VRE']:.3f}% MRE={met['MRE']:.3f}% "
              f"pts={met['n_points']}/51 t={met['time']:.1f}s")
    return out, (sv, sr)


def main():
    names = sys.argv[1:] if len(sys.argv) > 1 else ["port1", "port2", "port3", "port4", "port5"]
    methods = ["GNN", "MIQP", "SA", "Greedy"]
    allres = {}
    for name in names:
        print(f"=== {name} ({ORLIB_FILES[name][1]}) ===")
        res, _ = run_instance(name, methods)
        allres[name] = {m: {k: v for k, v in d.items() if k not in ("frontier_var", "frontier_ret")}
                        for m, d in res.items()}
    os.makedirs("results/orlib_med", exist_ok=True)
    with open("results/orlib_med/med_results.json", "w") as f:
        json.dump(allres, f, indent=2)

    # comparison table vs published
    print("\n\n=== MED COMPARISON (ours vs published Cura2009 / IPSO-SA / Firefly) ===")
    cols = ["GNN", "MIQP", "SA", "Greedy", "GA", "TS", "SA*", "PSO", "IPSO-SA", "Firefly"]
    print(f"{'instance':<10}" + "".join(f"{c:>9}" for c in cols))
    for name in names:
        row = f"{name:<10}"
        for c in cols:
            if c in allres.get(name, {}):
                row += f"{allres[name][c]['MED']:>9.4f}"
            elif c in PUBLISHED.get(name, {}):
                row += f"{PUBLISHED[name][c]:>9.4f}"
            else:
                row += f"{'-':>9}"
        print(row)


if __name__ == "__main__":
    main()
