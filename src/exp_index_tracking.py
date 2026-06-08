"""Axis B (1/3): cardinality INDEX TRACKING — pick K assets + weights to replicate a benchmark
(equal-weight of all N assets), minimizing tracking-error variance (w-b)'Sigma(w-b).
Compare GNN / SA / tabu / greedy vs SCIP-exact (the tracking-error optimum). Metric = annualized
tracking-error % and optimality gap. 2026-relevant (Mancilla/THRML index tracking; Dhingra review).
Run in .venv.
"""
from __future__ import annotations

import sys, os, json
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE"); os.environ.setdefault("OMP_NUM_THREADS", "8")
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import cvxpy as cp

from portfolio_data import download_orlib, load_orlib
from datasets import get_returns
from qubo_portfolio import tracking_qubo, decode_selection
from baselines import sa_qubo, tabu_qubo
from gnn_solver import solve_qubo_gnn, GNNHypers


def te(w, b, Sigma):
    d = w - b
    return float(np.sqrt(max(d @ Sigma @ d, 0)) * np.sqrt(252)) * 100  # annualized TE %


def reweight_track(S, b, Sigma, eps=0.0):
    n = len(b); S = np.asarray(S, int)
    if len(S) == 0:
        return np.ones(n) / n
    x = cp.Variable(len(S), nonneg=True)
    bs = b[S]; Ss = Sigma[np.ix_(S, S)]
    cp.Problem(cp.Minimize(cp.quad_form(x - bs / max(bs.sum(), 1e-9) * 0 + 0, cp.psd_wrap(Ss))
                           if False else cp.quad_form(x, cp.psd_wrap(Ss)) - 2 * (Ss @ bs) @ x),
               [cp.sum(x) == 1, x >= eps]).solve(solver=cp.CLARABEL)
    w = np.zeros(n)
    if x.value is not None:
        w[S] = np.maximum(x.value, 0); w[S] /= max(w[S].sum(), 1e-9)
    return w


def scip_track(b, Sigma, K, time_limit=60):
    from pyscipopt import Model, quicksum
    n = len(b); m = Model(); m.hideOutput(); m.setParam("limits/time", time_limit)
    w = {i: m.addVar(lb=0.0, ub=1.0) for i in range(n)}; z = {i: m.addVar(vtype="B") for i in range(n)}
    m.addCons(quicksum(w[i] for i in range(n)) == 1); m.addCons(quicksum(z[i] for i in range(n)) == K)
    for i in range(n):
        m.addCons(w[i] <= z[i])
    # (w-b)'Sigma(w-b) = w'Sigma w - 2 b'Sigma w + b'Sigma b ; minimize via epigraph on the quadratic
    d = {i: w[i] - float(b[i]) for i in range(n)}
    quad = quicksum(Sigma[i, i] * d[i] * d[i] for i in range(n)) + \
        quicksum(2 * Sigma[i, j] * d[i] * d[j] for i in range(n) for j in range(i + 1, n) if Sigma[i, j] != 0)
    t = m.addVar(lb=0.0); m.addCons(quad <= t); m.setObjective(t, "minimize")
    m.optimize()
    wv = np.array([m.getVal(w[i]) for i in range(n)])
    return wv


def run(name, mu, Sigma, K):
    n = len(mu); b = np.ones(n) / n      # benchmark = equal-weight of all assets
    q = tracking_qubo(Sigma, b, K)
    res = {}
    w = scip_track(b, Sigma, K); res["SCIP(exact)"] = te(w, b, Sigma)
    def rec(nm, x):
        S = decode_selection(x)
        res[nm] = te(reweight_track(S, b, Sigma), b, Sigma) if len(S) == K else float("nan")
    rec("SA", sa_qubo(q, num_reads=100, seed=0)["x"])
    rec("Tabu", tabu_qubo(q, num_reads=50, seed=0)["x"])
    h = GNNHypers(model="qrf", epochs=1200, hidden=128, dim_embedding=24, n_layers=3, lr=1e-3,
                  anneal_rate=0.0, eval_every=50, patience=400, ls_passes=100, n_round_samples=16,
                  refine_sa=True, refine_reads=20)
    rec("GNN", solve_qubo_gnn(q, h, device="cuda", seed=0)["x"])
    best = min(v for v in res.values() if np.isfinite(v))
    print(f"\n=== index tracking {name}: N={n} K={K} | benchmark=equal-weight ===")
    print(f"{'method':<12}{'ann.TE%':>10}{'gap%':>9}")
    for m_, v in res.items():
        print(f"{m_:<12}{v:>10.4f}{(v-best)/abs(best)*100:>9.2f}")
    return res


def main():
    paths = download_orlib("data/orlib")
    out = {}
    for name, K in [("port2", 10), ("port4", 10)]:
        mu, Sig, _ = load_orlib(paths[name]); out[name] = run(name, mu, Sig, K)
    for ds, K in [("french49", 10), ("nasdaq100", 10)]:
        R = get_returns(ds); mu = R.mean().values; Sig = R.cov().values; Sig = 0.5 * (Sig + Sig.T)
        out[ds] = run(ds, mu, Sig, K)
    os.makedirs("results/index_tracking", exist_ok=True)
    json.dump(out, open("results/index_tracking/results.json", "w"), indent=2)


if __name__ == "__main__":
    main()
