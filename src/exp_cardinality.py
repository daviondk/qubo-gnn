"""Cardinality-constrained portfolio experiment.

For a given (instance, K, risk_aversion, return_weight):
  * Build the selection QUBO (equal-weight surrogate over N binary vars).
  * Solve it with: GNN (explore+exploit), SA, tabu, greedy, random.
  * Decode each selection, then HYBRID convex-reweight the chosen K assets (cvxpy).
  * Compare the resulting true objective  obj(w) = ra*w'Sigma w - rw*mu'w  against the EXACT
    Gurobi MIQP optimum (same objective, continuous weights, cardinality = K).
  * Report optimality gap, feasibility, return/vol/Sharpe, wall-clock.

This is the honest test: same objective, exact ground truth, no normalization rescue.
"""
from __future__ import annotations

import sys, os, json, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # avoid OpenMP double-load abort (torch+neal)
os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from portfolio_data import download_orlib, load_orlib, ORLIB_FILES
from qubo_portfolio import selection_qubo, decode_selection
from baselines import (miqp_cardinality, convex_reweight, sa_qubo, tabu_qubo,
                       greedy_selection, random_selection)
from gnn_solver import solve_qubo_gnn, GNNHypers
from metrics import portfolio_metrics


def obj_value(w, mu, Sigma, ra, rw):
    return float(ra * (w @ Sigma @ w) - rw * (mu @ w))


def run_point(mu, Sigma, K, ra, rw, device="cuda", seed=0, gnn_epochs=1500):
    n = len(mu)
    q = selection_qubo(mu, Sigma, K, risk_aversion=ra, return_weight=rw)
    out = {}

    # exact MIQP ground truth -- may be unavailable for large N (free Gurobi license size limit)
    obj_mi = None
    try:
        mi = miqp_cardinality(mu, Sigma, K, risk_aversion=ra, return_weight=rw)
        w_mi = mi["weights"]; obj_mi = obj_value(w_mi, mu, Sigma, ra, rw)
        pm = portfolio_metrics(w_mi, mu, Sigma)
        out["MIQP(exact)"] = {**pm, "obj": obj_mi, "k": len(mi["support"]),
                              "feasible": len(mi["support"]) == K, "time": mi["time"]}
    except Exception as e:
        out["MIQP(exact)"] = {"obj": float("nan"), "note": f"unavailable: {str(e)[:60]}",
                              "k": 0, "feasible": False, "time": 0.0}

    def record(name, x, t):
        sup = decode_selection(x)
        feasible = len(sup) == K
        w = convex_reweight(mu, Sigma, sup, risk_aversion=ra, return_weight=rw) if len(sup) else np.zeros(n)
        o = obj_value(w, mu, Sigma, ra, rw) if len(sup) else float("inf")
        pm = portfolio_metrics(w, mu, Sigma)
        out[name] = {**pm, "obj": o, "k": len(sup), "feasible": feasible,
                     "qubo_energy": q.energy(x), "time": t}

    r = sa_qubo(q, num_reads=100, seed=seed); record("SA", r["x"], r["time"])
    r = tabu_qubo(q, num_reads=50, seed=seed); record("Tabu", r["x"], r["time"])
    r = greedy_selection(mu, Sigma, K, risk_aversion=ra, return_weight=rw); record("Greedy", r["x"], r["time"])
    r = random_selection(mu, Sigma, K, risk_aversion=ra, return_weight=rw, n_tries=2000, seed=seed); record("Random", r["x"], r["time"])
    h = GNNHypers(model="qrf", epochs=gnn_epochs, hidden=128, dim_embedding=32, n_layers=3,
                  lr=1e-3, anneal_rate=0.0, eval_every=50, patience=500, ls_passes=100,
                  n_round_samples=16, refine_sa=True, refine_reads=20)
    r = solve_qubo_gnn(q, h, device=device, seed=seed); record("GNN", r["x"], r["time"])
    out["GNN"]["qubo_energy_pre_refine"] = r.get("energy_pre_refine")

    # reference objective: exact MIQP if available, else best objective found by any method
    ref = obj_mi if obj_mi is not None else min(
        d["obj"] for k, d in out.items() if k != "MIQP(exact)" and np.isfinite(d.get("obj", np.nan)))
    for k, d in out.items():
        o = d.get("obj", float("nan"))
        d["gap"] = ((o - ref) / abs(ref)) if (np.isfinite(o) and abs(ref) > 1e-12) else float("nan")
    out["_ref"] = {"obj_ref": ref, "exact": obj_mi is not None}
    return out


def main():
    paths = download_orlib("data/orlib")
    K = 10
    ra, rw = 1.0, 0.5
    name = sys.argv[1] if len(sys.argv) > 1 else "port1"
    mu, Sigma, _ = load_orlib(paths[name])
    print(f"=== {name} ({ORLIB_FILES[name][1]}) N={len(mu)} K={K} ra={ra} rw={rw} ===")
    res = run_point(mu, Sigma, K, ra, rw)
    print(f"reference: {'exact MIQP' if res['_ref']['exact'] else 'best-found (MIQP unavailable)'}")
    hdr = f"{'method':<14}{'obj':>12}{'gap%':>9}{'return':>10}{'vol':>9}{'sharpe':>9}{'k':>4}{'feas':>6}{'t(s)':>8}"
    print(hdr); print("-" * len(hdr))
    for m, d in res.items():
        if m == "_ref":
            continue
        obj = d.get("obj", float("nan")); gap = d.get("gap", float("nan"))
        print(f"{m:<14}{obj:>12.6f}{gap*100:>9.3f}{d.get('return',float('nan')):>10.5f}"
              f"{d.get('vol',float('nan')):>9.5f}{d.get('sharpe',float('nan')):>9.4f}"
              f"{d['k']:>4}{str(d['feasible']):>6}{d['time']:>8.2f}")
    os.makedirs("results/cardinality", exist_ok=True)
    with open(f"results/cardinality/{name}_K{K}.json", "w") as f:
        json.dump({k: {kk: (float(vv) if isinstance(vv, (np.floating, np.integer)) else vv)
                       for kk, vv in v.items()} for k, v in res.items()}, f, indent=2, default=str)


if __name__ == "__main__":
    main()
