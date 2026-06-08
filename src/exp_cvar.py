"""Axis 2: CVaR (scenario-based, Rockafellar-Uryasev) cardinality portfolio -- the genuinely 'hard'
class where the exact scenario MILP strains. min CVaR_alpha(loss) s.t. sum w=1, cardinality=K, w>=0.

Methods:
 - exact mean-CVaR MILP (SCIP): binary z + w + eta + per-scenario slacks (large, time-limited).
 - hybrid: select K via a downside-risk (semivariance) selection QUBO solved by GNN / tabu, then
   set weights by the CVaR LP on the chosen support (cvxpy).
 - equal-weight baseline.
Metric: achieved CVaR (lower=better), gap vs exact, time. Run in .venv.
"""
from __future__ import annotations
import sys, os, json, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE"); os.environ.setdefault("OMP_NUM_THREADS", "8")
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np, cvxpy as cp

from datasets import get_returns
from qubo_portfolio import selection_qubo, decode_selection
from baselines import tabu_qubo
from gnn_solver import solve_qubo_gnn, GNNHypers

ALPHA = 0.05  # 95% CVaR


def cvar_of(w, scen):
    losses = -(scen @ w)
    var = np.quantile(losses, 1 - ALPHA)
    tail = losses[losses >= var]
    return float(tail.mean()) if len(tail) else float(var)


def cvar_lp(support, scen, alpha=ALPHA):
    n = scen.shape[1]; S = np.asarray(support, int)
    if len(S) == 0:
        return np.ones(n) / n
    r = scen[:, S]; m = r.shape[0]
    x = cp.Variable(len(S), nonneg=True); eta = cp.Variable(); u = cp.Variable(m, nonneg=True)
    cons = [cp.sum(x) == 1, u >= -(r @ x) - eta]
    cp.Problem(cp.Minimize(eta + (1.0 / (alpha * m)) * cp.sum(u)), cons).solve(solver=cp.CLARABEL)
    w = np.zeros(n)
    if x.value is not None:
        w[S] = np.maximum(x.value, 0); w[S] /= max(w[S].sum(), 1e-9)
    return w


def scip_cvar(scen, K, alpha=ALPHA, time_limit=120):
    from pyscipopt import Model, quicksum
    m, n = scen.shape
    M = Model(); M.hideOutput(); M.setParam("limits/time", time_limit)
    w = {i: M.addVar(lb=0.0, ub=1.0) for i in range(n)}; z = {i: M.addVar(vtype="B") for i in range(n)}
    eta = M.addVar(lb=-10, ub=10); u = {s: M.addVar(lb=0.0) for s in range(m)}
    M.addCons(quicksum(w[i] for i in range(n)) == 1); M.addCons(quicksum(z[i] for i in range(n)) == K)
    for i in range(n):
        M.addCons(w[i] <= z[i])
    for s in range(m):
        M.addCons(u[s] >= -quicksum(float(scen[s, i]) * w[i] for i in range(n)) - eta)
    M.setObjective(eta + (1.0 / (alpha * m)) * quicksum(u[s] for s in range(m)), "minimize")
    t0 = time.time(); M.optimize(); dt = time.time() - t0
    wv = np.array([M.getVal(w[i]) for i in range(n)])
    return wv, dt, M.getGap(), M.getStatus()


def run(name, R, K, n_scen=400):
    scen = R[-n_scen:]               # recent scenarios
    mu = scen.mean(0); n = scen.shape[1]
    d = np.minimum(scen - mu, 0.0); semicov = (d.T @ d) / len(scen); semicov = 0.5 * (semicov + semicov.T)
    q = selection_qubo(mu, semicov, K, risk_aversion=1.0, return_weight=0.0)  # downside-risk selection
    res = {}
    w, dt, gap, st = scip_cvar(scen, K, time_limit=120)
    res["SCIP-MILP(exact)"] = (cvar_of(w, scen), dt, f"gap={gap:.3f} {st}")
    res["EqualWeight"] = (cvar_of(np.ones(n) / n, scen), 0.0, "")
    St = decode_selection(tabu_qubo(q, num_reads=100, seed=0)["x"])
    res["Tabu+CVaR-LP"] = (cvar_of(cvar_lp(St, scen), scen), 0.0, "") if len(St) == K else (np.nan, 0, "infeas")
    h = GNNHypers(model="qrf", epochs=1200, hidden=128, dim_embedding=24, n_layers=3, lr=1e-3,
                  anneal_rate=0.0, eval_every=50, patience=400, ls_passes=100, n_round_samples=16,
                  refine_sa=True, refine_reads=20)
    Sg = decode_selection(solve_qubo_gnn(q, h, device="cuda", seed=0)["x"])
    res["GNN+CVaR-LP"] = (cvar_of(cvar_lp(Sg, scen), scen), 0.0, "") if len(Sg) == K else (np.nan, 0, "infeas")
    best = min(v[0] for v in res.values() if np.isfinite(v[0]))
    print(f"\n=== CVaR(95%) {name}: N={n} K={K} scen={len(scen)} ===")
    print(f"{'method':<18}{'CVaR':>10}{'gap%':>8}{'t(s)':>8}  note")
    for k, (c, t, note) in res.items():
        print(f"{k:<18}{c:>10.5f}{(c-best)/abs(best)*100:>8.2f}{t:>8.1f}  {note}")
    return {k: {"cvar": v[0], "time": v[1], "note": v[2]} for k, v in res.items()}


def main():
    out = {}
    for ds, K in [("french49", 10), ("nasdaq100", 10)]:
        out[ds] = run(ds, get_returns(ds).values, K)
    os.makedirs("results/cvar", exist_ok=True)
    json.dump(out, open("results/cvar/results.json", "w"), indent=2, default=str)


if __name__ == "__main__":
    main()
