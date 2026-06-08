"""HARD portfolio benchmark on a REAL dataset (S&P 500) where the optimum is UNKNOWN:
cardinality-constrained CVaR(95%) with a LARGE Monte-Carlo / bootstrap scenario set.
This is how the risk class of portfolio problems is actually solved (scenario-based CVaR,
Rockafellar-Uryasev). With many scenarios + cardinality, the exact MILP does NOT close the gap
(optimum unknown) -> we compare on BEST-FOUND and report the MILP's proved gap & time.

Methods: exact MILP (SCIP, time-limited; report its proved gap), GNN-select + CVaR-LP,
tabu-select + CVaR-LP, equal-weight. Run in .venv.
"""
from __future__ import annotations
import os, sys, json, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE"); os.environ.setdefault("OMP_NUM_THREADS", "8")
HERE = os.path.dirname(__file__); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, cvxpy as cp
from datasets import get_returns
from qubo_portfolio import selection_qubo, decode_selection
from baselines import tabu_qubo
from gnn_solver import solve_qubo_gnn, GNNHypers
ALPHA = 0.05


def gen_scenarios(R, S, seed=0):
    """Block-free bootstrap of historical daily returns -> S scenarios (preserves cross-asset deps + fat tails)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(R), size=S)
    return R[idx]


def cvar(w, scen):
    losses = -(scen @ w); var = np.quantile(losses, 1 - ALPHA)
    tail = losses[losses >= var]; return float(tail.mean()) if len(tail) else float(var)


def cvar_lp(S_idx, scen):
    n = scen.shape[1]; S = np.asarray(S_idx, int)
    if len(S) == 0:
        return np.ones(n) / n
    r = scen[:, S]; m = r.shape[0]
    x = cp.Variable(len(S), nonneg=True); eta = cp.Variable(); u = cp.Variable(m, nonneg=True)
    cp.Problem(cp.Minimize(eta + (1.0 / (ALPHA * m)) * cp.sum(u)),
               [cp.sum(x) == 1, u >= -(r @ x) - eta]).solve(solver=cp.CLARABEL)
    w = np.zeros(n)
    if x.value is not None:
        w[S] = np.maximum(x.value, 0); w[S] /= max(w[S].sum(), 1e-9)
    return w


def scip_cvar_card(scen, K, time_limit=240):
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
    M.setObjective(eta + (1.0 / (ALPHA * m)) * quicksum(u[s] for s in range(m)), "minimize")
    t0 = time.time(); M.optimize(); dt = time.time() - t0
    wv = np.array([M.getVal(w[i]) for i in range(n)])
    return wv, dt, M.getGap(), M.getStatus()


def main():
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    Ssc = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
    R = get_returns("sp500").values
    N = R.shape[1]
    scen = gen_scenarios(R, Ssc, seed=0)
    mu = scen.mean(0)
    d = np.minimum(scen - mu, 0.0); semicov = (d.T @ d) / len(scen); semicov = 0.5 * (semicov + semicov.T)
    print(f"=== HARD CVaR(95%): S&P500 N={N}, K={K}, {Ssc} bootstrap scenarios (optimum unknown) ===", flush=True)
    res = {}
    w, dt, gap, st = scip_cvar_card(scen, K, time_limit=240)
    res["SCIP-MILP"] = {"cvar": cvar(w, scen), "t": dt, "proved_gap": gap, "status": st}
    res["EqualWeight"] = {"cvar": cvar(np.ones(N) / N, scen), "t": 0.0}
    q = selection_qubo(mu, semicov, K, risk_aversion=1.0, return_weight=0.0)
    t0 = time.time(); St = decode_selection(tabu_qubo(q, num_reads=100, seed=0)["x"])
    res["Tabu+CVaR-LP"] = {"cvar": cvar(cvar_lp(St, scen), scen) if len(St) == K else float("nan"), "t": time.time() - t0}
    h = GNNHypers(model="qrf", epochs=1500, hidden=128, dim_embedding=24, n_layers=3, lr=1e-3,
                  anneal_rate=0.0, eval_every=50, patience=400, ls_passes=120, n_round_samples=16,
                  refine_sa=True, refine_reads=30)
    t0 = time.time(); Sg = decode_selection(solve_qubo_gnn(q, h, device="cuda", seed=0)["x"])
    res["GNN+CVaR-LP"] = {"cvar": cvar(cvar_lp(Sg, scen), scen) if len(Sg) == K else float("nan"), "t": time.time() - t0}
    best = min(v["cvar"] for v in res.values() if np.isfinite(v["cvar"]))
    print(f"{'method':<16}{'CVaR':>10}{'gap%toBest':>12}{'t(s)':>8}  note")
    for m_, v in res.items():
        g = (v["cvar"] - best) / abs(best) * 100 if np.isfinite(v["cvar"]) else float("nan")
        note = (f"MILP proved_gap={v['proved_gap']:.2f} {v['status']}" if m_ == "SCIP-MILP" else "")
        print(f"{m_:<16}{v['cvar']:>10.5f}{g:>11.2f}%{v['t']:>8.1f}  {note}", flush=True)
        v["gap_to_best%"] = g
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(res, open(os.path.join(HERE, "results", f"hard_cvar_K{K}_S{Ssc}.json"), "w"), indent=2, default=str)


if __name__ == "__main__":
    main()
