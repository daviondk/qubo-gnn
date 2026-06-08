"""Fixed-charge mean-variance portfolio (non-convex): pay a fixed cost c per held position; the number
of positions is endogenous. Objective for a support S:
    F(S) = min_{w on S, sum w=1, w>=0} [ lambda w'Sigma w - (1-lambda) mu'w ] + c*|S|
Compare: SCIP exact MIQP, forward-greedy (re-solving the QP), and GNN via a cardinality sweep
(GNN picks the best K'-subset for each K', then we add c*K' and take the best K').

Run in .venv (torch+pyg+scip+cvxpy).
"""
from __future__ import annotations

import sys, os, json, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "8")
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import cvxpy as cp

from qubo_portfolio import selection_qubo, decode_selection
from gnn_solver import solve_qubo_gnn, GNNHypers


def qp_value(S, mu, Sigma, lam):
    """min lambda w'Sigma w - (1-lambda) mu'w  on support S (sum w=1, w>=0). Returns (obj, w)."""
    S = np.asarray(S, int)
    if len(S) == 0:
        return np.inf, None
    s = Sigma[np.ix_(S, S)]; m = mu[S]
    x = cp.Variable(len(S), nonneg=True)
    cp.Problem(cp.Minimize(lam * cp.quad_form(x, cp.psd_wrap(s)) - (1 - lam) * (m @ x)),
               [cp.sum(x) == 1]).solve(solver=cp.CLARABEL)
    if x.value is None:
        return np.inf, None
    w = np.zeros(len(mu)); w[S] = np.maximum(x.value, 0); w[S] /= w[S].sum()
    return float(lam * (w @ Sigma @ w) - (1 - lam) * (mu @ w)), w


def F(S, mu, Sigma, lam, c):
    v, _ = qp_value(S, mu, Sigma, lam)
    return v + c * len(S)


def greedy_forward(mu, Sigma, lam, c, kmax):
    n = len(mu); S = []; cur = np.inf; t0 = time.time()
    improved = True
    while improved and len(S) < kmax:
        improved = False; best_i, best_F = None, cur
        for i in range(n):
            if i in S:
                continue
            f = F(S + [i], mu, Sigma, lam, c)
            if f < best_F - 1e-12:
                best_F, best_i = f, i
        if best_i is not None:
            S.append(best_i); cur = best_F; improved = True
    return sorted(S), cur, time.time() - t0


def scip_fixed_charge(mu, Sigma, lam, c, kmax, time_limit=180):
    from pyscipopt import Model, quicksum
    n = len(mu); m = Model(); m.hideOutput(); m.setParam("limits/time", time_limit)
    w = {i: m.addVar(lb=0.0, ub=1.0) for i in range(n)}
    z = {i: m.addVar(vtype="B") for i in range(n)}
    m.addCons(quicksum(w[i] for i in range(n)) == 1)
    for i in range(n):
        m.addCons(w[i] <= z[i])
    m.addCons(quicksum(z[i] for i in range(n)) <= kmax)
    risk = quicksum(Sigma[i, i] * w[i] * w[i] for i in range(n)) + \
        quicksum(2 * Sigma[i, j] * w[i] * w[j] for i in range(n) for j in range(i + 1, n) if Sigma[i, j] != 0)
    t = m.addVar(lb=0.0); m.addCons(risk <= t)
    ret = quicksum(mu[i] * w[i] for i in range(n))
    m.setObjective(lam * t - (1 - lam) * ret + c * quicksum(z[i] for i in range(n)), "minimize")
    t0 = time.time(); m.optimize(); dt = time.time() - t0
    S = sorted([i for i in range(n) if m.getVal(z[i]) > 0.5])
    return S, F(S, mu, Sigma, lam, c), dt, m.getGap(), m.getStatus()


def gnn_ksweep(mu, Sigma, lam, c, kgrid):
    n = len(mu); best_S, best_F = None, np.inf; t0 = time.time()
    h = GNNHypers(model="qrf", epochs=800, hidden=96, dim_embedding=24, n_layers=3, lr=2e-3,
                  anneal_rate=0.0, eval_every=40, patience=300, ls_passes=80, n_round_samples=12,
                  refine_sa=True, refine_reads=20)
    for K in kgrid:
        q = selection_qubo(mu, Sigma, K, risk_aversion=lam, return_weight=(1 - lam))
        r = solve_qubo_gnn(q, h, device="cuda", seed=0)
        S = list(decode_selection(r["x"]))
        if len(S) != K:  # enforce size K via top-K not needed; selection penalty usually exact
            S = list(np.argsort(-r["x"])[:K])
        f = F(S, mu, Sigma, lam, c)
        if f < best_F:
            best_F, best_S = f, sorted(S)
    return best_S, best_F, time.time() - t0


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    c = float(sys.argv[2]) if len(sys.argv) > 2 else 2e-4   # fixed cost per position
    lam = 0.5
    kmax = min(N, 40)
    rng = np.random.default_rng(3)
    nf = 6
    B = rng.standard_normal((N, nf)) * 0.02; d = np.abs(rng.standard_normal(N)) * 0.01 + 0.005
    Sigma = B @ B.T + np.diag(d ** 2); Sigma = 0.5 * (Sigma + Sigma.T)
    mu = rng.standard_normal(N) * 0.004 + 0.003
    print(f"=== fixed-charge: N={N} c={c} lam={lam} kmax={kmax} ===")

    res = {}
    S, f, dt, gap, st = scip_fixed_charge(mu, Sigma, lam, c, kmax)
    res["SCIP(exact)"] = (f, len(S), dt, f"gap={gap:.3f} {st}")
    S, f, dt = greedy_forward(mu, Sigma, lam, c, kmax)
    res["Greedy(QP)"] = (f, len(S), dt, "")
    kgrid = list(range(2, kmax + 1, 2))
    S, f, dt = gnn_ksweep(mu, Sigma, lam, c, kgrid)
    res["GNN(Ksweep)"] = (f, len(S), dt, f"grid step2")

    best = min(v[0] for v in res.values())
    print(f"{'method':<14}{'F(obj)':>13}{'gap%':>9}{'|S|':>5}{'t(s)':>9}  note")
    for k, (f, ns, t, note) in res.items():
        print(f"{k:<14}{f:>13.6f}{(f-best)/abs(best)*100:>9.3f}{ns:>5}{t:>9.2f}  {note}")
    os.makedirs("results/fixed_charge", exist_ok=True)
    json.dump({k: {"F": v[0], "size": v[1], "time": v[2], "note": v[3]} for k, v in res.items()},
              open(f"results/fixed_charge/N{N}_c{c}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
