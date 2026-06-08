"""Baselines: exact MIQP (Gurobi), convex QP (cvxpy), QUBO heuristics (SA, tabu), greedy, random.

All cardinality solvers use the SAME objective convention as selection_qubo:
    minimize  risk_aversion*(1/K^2) z^T Sigma z  -  return_weight*(1/K) mu^T z,   sum z = K
so energies are directly comparable. The financial evaluation (after picking a support) uses
convex_reweight to set optimal weights on the chosen assets (the hybrid).
"""
from __future__ import annotations

import time

import numpy as np

from qubo import QUBO, local_search_1flip


# ------------------------------- exact MIQP (Gurobi) -------------------------------

def miqp_cardinality(mu, Sigma, K, *, risk_aversion=1.0, return_weight=1.0,
                     eps=0.0, delta=1.0, time_limit=300.0):
    """Exact MIQP:  min ra*w'Sigma w - rw*mu'w  s.t. sum w=1, sum z=K, eps*z<=w<=delta*z, z binary.
    Returns dict(weights, support, objective, time, gap, status)."""
    import gurobipy as gp
    from gurobipy import GRB
    n = len(mu)
    m = gp.Model("miqp_card")
    m.setParam("OutputFlag", 0)
    m.setParam("TimeLimit", time_limit)
    w = m.addVars(n, lb=0.0, ub=delta, name="w")
    z = m.addVars(n, vtype=GRB.BINARY, name="z")
    m.addConstr(gp.quicksum(w[i] for i in range(n)) == 1.0)
    m.addConstr(gp.quicksum(z[i] for i in range(n)) == K)
    for i in range(n):
        m.addConstr(w[i] <= delta * z[i])
        m.addConstr(w[i] >= eps * z[i])
    risk = gp.quicksum(Sigma[i, j] * w[i] * w[j] for i in range(n) for j in range(n))
    ret = gp.quicksum(mu[i] * w[i] for i in range(n))
    m.setObjective(risk_aversion * risk - return_weight * ret, GRB.MINIMIZE)
    t0 = time.time()
    m.optimize()
    dt = time.time() - t0
    wv = np.array([w[i].X for i in range(n)])
    support = np.flatnonzero(np.array([z[i].X for i in range(n)]) > 0.5)
    return {"weights": wv, "support": support, "objective": m.ObjVal, "time": dt,
            "gap": m.MIPGap, "status": int(m.Status)}


# ------------------------------- exact MIQP (SCIP, free/unlimited) -------------------------------

def scip_cardinality(mu, Sigma, K, *, risk_aversion=1.0, return_weight=1.0,
                     eps=0.0, delta=1.0, time_limit=120.0):
    """Exact (or time-limited) cardinality MIQP via SCIP/PySCIPOpt -- free, no size limit.
    min ra*w'Sigma w - rw*mu'w  s.t. sum w=1, sum z=K, eps*z<=w<=delta*z, z binary."""
    from pyscipopt import Model, quicksum
    n = len(mu)
    m = Model("scip_card"); m.hideOutput()
    m.setParam("limits/time", time_limit)
    w = {i: m.addVar(lb=0.0, ub=delta, vtype="C", name=f"w{i}") for i in range(n)}
    z = {i: m.addVar(vtype="B", name=f"z{i}") for i in range(n)}
    m.addCons(quicksum(w[i] for i in range(n)) == 1.0)
    m.addCons(quicksum(z[i] for i in range(n)) == K)
    for i in range(n):
        m.addCons(w[i] <= delta * z[i]); m.addCons(w[i] >= eps * z[i])
    # epigraph: SCIP needs a LINEAR objective, so push the convex quadratic into a constraint
    # risk uses upper triangle with factor 2 for off-diagonal (Sigma symmetric)
    risk = quicksum(Sigma[i, i] * w[i] * w[i] for i in range(n)) \
        + quicksum(2.0 * Sigma[i, j] * w[i] * w[j] for i in range(n) for j in range(i + 1, n) if Sigma[i, j] != 0)
    t = m.addVar(lb=0.0, ub=None, vtype="C", name="t")
    m.addCons(risk <= t)
    ret = quicksum(mu[i] * w[i] for i in range(n))
    m.setObjective(risk_aversion * t - return_weight * ret, "minimize")
    t0 = time.time(); m.optimize(); dt = time.time() - t0
    wv = np.array([m.getVal(w[i]) for i in range(n)])
    support = np.flatnonzero(np.array([m.getVal(z[i]) for i in range(n)]) > 0.5)
    gap = m.getGap()
    return {"weights": wv, "support": support, "objective": m.getObjVal(), "time": dt,
            "gap": gap, "status": m.getStatus()}


# ------------------------------- exact QUBO (SCIP global MINLP) -------------------------------

def scip_qubo(qubo, time_limit=120.0):
    """Exact/time-limited global solve of min x'Q x (x binary) via SCIP spatial B&B.
    Returns dict(x, energy, time, gap, status). Handles indefinite Q (nonconvex)."""
    from pyscipopt import Model, quicksum
    import numpy as _np
    Q = qubo.Q; n = qubo.n
    m = Model("scip_qubo"); m.hideOutput(); m.setParam("limits/time", time_limit)
    x = {i: m.addVar(vtype="B") for i in range(n)}
    quad = quicksum(float(Q[i, i]) * x[i] * x[i] for i in range(n)) + \
        quicksum(2.0 * float(Q[i, j]) * x[i] * x[j]
                 for i in range(n) for j in range(i + 1, n) if Q[i, j] != 0)
    # SCIP objective must be linear -> epigraph (nonconvex quad constraint, handled by spatial B&B)
    t = m.addVar(lb=-1e9, ub=1e9, vtype="C", name="t")
    m.addCons(quad <= t)
    m.setObjective(t, "minimize")
    t0 = time.time(); m.optimize(); dt = time.time() - t0
    xv = _np.array([round(m.getVal(x[i])) for i in range(n)], dtype=_np.int8)
    return {"x": xv, "energy": qubo.energy(xv) if hasattr(qubo, "energy") else float(m.getObjVal()),
            "time": dt, "gap": m.getGap(), "status": m.getStatus()}


# ------------------------------- convex QP (cvxpy) -------------------------------

def convex_reweight(mu, Sigma, support, *, risk_aversion=1.0, return_weight=1.0, eps=0.0, delta=1.0):
    """Optimal continuous weights ON a fixed support (the hybrid's second stage), respecting the
    OR-Library bounds eps <= w_i <= delta for selected assets."""
    import cvxpy as cp
    support = np.asarray(support, int)
    k = len(support)
    if k == 0:
        return np.zeros(len(mu))
    s = Sigma[np.ix_(support, support)]
    msub = mu[support]
    x = cp.Variable(k)
    obj = risk_aversion * cp.quad_form(x, cp.psd_wrap(s)) - return_weight * (msub @ x)
    cons = [cp.sum(x) == 1, x >= eps, x <= delta]
    cp.Problem(cp.Minimize(obj), cons).solve(solver=cp.CLARABEL)
    w = np.zeros(len(mu))
    if x.value is not None:
        w[support] = np.maximum(x.value, 0)
        if w.sum() > 0:
            w = w / w.sum()
    return w


def convex_unconstrained_frontier(mu, Sigma, n_points=50):
    """Continuous mean-variance frontier (no cardinality) -- the reference for % deviation."""
    import cvxpy as cp
    n = len(mu)
    rets, vols = [], []
    targets = np.linspace(mu.min(), mu.max(), n_points)
    for tr in targets:
        x = cp.Variable(n, nonneg=True)
        try:
            cp.Problem(cp.Minimize(cp.quad_form(x, cp.psd_wrap(Sigma))),
                       [cp.sum(x) == 1, mu @ x >= tr]).solve(solver=cp.CLARABEL)
            if x.value is not None:
                w = np.maximum(x.value, 0); w = w / w.sum()
                rets.append(float(mu @ w)); vols.append(float(np.sqrt(w @ Sigma @ w)))
        except Exception:
            pass
    return np.array(rets), np.array(vols)


# ------------------------------- QUBO heuristics -------------------------------

def sa_qubo(qubo: QUBO, num_reads=100, seed=0, **kw):
    import neal
    t0 = time.time()
    res = neal.SimulatedAnnealingSampler().sample(qubo.to_dimod(), num_reads=num_reads, seed=seed)
    best = res.first
    x = np.array([best.sample[i] for i in range(qubo.n)], dtype=np.int8)
    return {"x": x, "energy": qubo.energy(x), "time": time.time() - t0}


def tabu_qubo(qubo: QUBO, num_reads=50, seed=0, **kw):
    from tabu import TabuSampler
    t0 = time.time()
    res = TabuSampler().sample(qubo.to_dimod(), num_reads=num_reads, seed=seed)
    best = res.first
    x = np.array([best.sample[i] for i in range(qubo.n)], dtype=np.int8)
    return {"x": x, "energy": qubo.energy(x), "time": time.time() - t0}


# ------------------------------- selection heuristics -------------------------------

def greedy_selection(mu, Sigma, K, *, risk_aversion=1.0, return_weight=1.0):
    """Forward-greedy: add the asset that most reduces the selection objective each step."""
    from qubo_portfolio import selection_qubo
    n = len(mu)
    q = selection_qubo(mu, Sigma, K, risk_aversion=risk_aversion, return_weight=return_weight)
    t0 = time.time()
    chosen = []
    remaining = set(range(n))
    for _ in range(K):
        best_i, best_e = None, np.inf
        for i in remaining:
            x = np.zeros(n);
            for c in chosen: x[c] = 1
            x[i] = 1
            e = q.energy(x)
            if e < best_e:
                best_e, best_i = e, i
        chosen.append(best_i); remaining.discard(best_i)
    x = np.zeros(n, dtype=np.int8); x[chosen] = 1
    return {"x": x, "energy": q.energy(x), "time": time.time() - t0}


def random_selection(mu, Sigma, K, *, risk_aversion=1.0, return_weight=1.0, n_tries=1000, seed=0):
    from qubo_portfolio import selection_qubo
    n = len(mu)
    q = selection_qubo(mu, Sigma, K, risk_aversion=risk_aversion, return_weight=return_weight)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    best_x, best_e = None, np.inf
    for _ in range(n_tries):
        idx = rng.choice(n, size=K, replace=False)
        x = np.zeros(n, dtype=np.int8); x[idx] = 1
        e = q.energy(x)
        if e < best_e:
            best_e, best_x = e, x
    return {"x": best_x, "energy": best_e, "time": time.time() - t0}
