"""Sector-capped cardinality portfolio (diversification): choose exactly K assets, at most `cap` per
sector. Non-modular -> greedy is myopic about sector budgets, exact gets harder. Compare:
GNN (PyG, explore+exploit) vs SA vs tabu vs cap-aware greedy vs SCIP exact.

Objective (equal-weight selection surrogate): f(z) = (ra/K^2) z'Sigma z - (rw/K) mu'z.
Run in .venv (torch+pyg+scip+neal+tabu).
"""
from __future__ import annotations

import sys, os, json, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "8")
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from qubo_portfolio import selection_qubo_sector_caps
from baselines import sa_qubo, tabu_qubo
from gnn_solver import solve_qubo_gnn, GNNHypers


def synth_sectors(N, S, seed=0):
    """Factor model: strong within-sector correlation; sector-level return tilts."""
    rng = np.random.default_rng(seed)
    sector_of = np.repeat(np.arange(S), N // S)[:N]
    if len(sector_of) < N:
        sector_of = np.concatenate([sector_of, np.full(N - len(sector_of), S - 1)])
    sector_factor = rng.standard_normal((S, 1))
    B = np.zeros((N, S))
    for i in range(N):
        B[i, sector_of[i]] = 0.03 + 0.01 * rng.standard_normal()  # within-sector load
    common = rng.standard_normal((N, 2)) * 0.01
    L = np.hstack([B, common])
    d = np.abs(rng.standard_normal(N)) * 0.008 + 0.004
    Sigma = L @ L.T + np.diag(d ** 2)
    sector_mu = rng.standard_normal(S) * 0.004
    mu = sector_mu[sector_of] + rng.standard_normal(N) * 0.002 + 0.003
    return mu, 0.5 * (Sigma + Sigma.T), sector_of


def obj(z, mu, Sigma, K, ra, rw):
    z = np.asarray(z, float)
    return float((ra / K**2) * (z @ Sigma @ z) - (rw / K) * (mu @ z))


def feasible(z, sector_of, K, cap):
    z = np.asarray(z, int)
    if z.sum() != K:
        return False
    for g in set(sector_of.tolist()):
        if z[sector_of == g].sum() > cap:
            return False
    return True


def greedy_caps(mu, Sigma, K, sector_of, cap, ra, rw):
    n = len(mu); chosen = []; cnt = {g: 0 for g in set(sector_of.tolist())}
    t0 = time.time()
    for _ in range(K):
        best_i, best_e = None, np.inf
        for i in range(n):
            if i in chosen or cnt[sector_of[i]] >= cap:
                continue
            z = np.zeros(n); z[chosen] = 1; z[i] = 1
            e = obj(z, mu, Sigma, K, ra, rw)
            if e < best_e:
                best_e, best_i = e, i
        if best_i is None:
            break
        chosen.append(best_i); cnt[sector_of[best_i]] += 1
    z = np.zeros(n, dtype=int); z[chosen] = 1
    return z, time.time() - t0


def scip_caps(mu, Sigma, K, sector_of, cap, ra, rw, time_limit=120):
    from pyscipopt import Model, quicksum
    n = len(mu); m = Model(); m.hideOutput(); m.setParam("limits/time", time_limit)
    z = {i: m.addVar(vtype="B") for i in range(n)}
    m.addCons(quicksum(z[i] for i in range(n)) == K)
    for g in set(sector_of.tolist()):
        m.addCons(quicksum(z[i] for i in range(n) if sector_of[i] == g) <= cap)
    risk = quicksum(Sigma[i, i] * z[i] * z[i] for i in range(n)) + \
        quicksum(2 * Sigma[i, j] * z[i] * z[j] for i in range(n) for j in range(i + 1, n) if Sigma[i, j] != 0)
    t = m.addVar(lb=0.0); m.addCons(risk <= t)
    ret = quicksum(mu[i] * z[i] for i in range(n))
    m.setObjective((ra / K**2) * t - (rw / K) * ret, "minimize")
    t0 = time.time(); m.optimize(); dt = time.time() - t0
    zv = np.array([round(m.getVal(z[i])) for i in range(n)], dtype=int)
    return zv, dt, m.getGap(), m.getStatus()


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 90
    S = int(sys.argv[2]) if len(sys.argv) > 2 else 9
    K = int(sys.argv[3]) if len(sys.argv) > 3 else 18
    cap = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    ra, rw = 1.0, 0.5
    mu, Sigma, sector_of = synth_sectors(N, S, seed=1)
    print(f"=== sector-capped: N={N} S={S} K={K} cap={cap} (sum caps={S*cap} >= K) ===")
    q, layout = selection_qubo_sector_caps(mu, Sigma, K, sector_of, cap, risk_aversion=ra, return_weight=rw)
    n = N
    res = {}

    zs, dt, gap, st = scip_caps(mu, Sigma, K, sector_of, cap, ra, rw, time_limit=180)
    res["SCIP(exact)"] = (obj(zs, mu, Sigma, K, ra, rw), feasible(zs, sector_of, K, cap), dt, f"gap={gap:.3f} {st}")

    zg, dt = greedy_caps(mu, Sigma, K, sector_of, cap, ra, rw)
    res["Greedy(caps)"] = (obj(zg, mu, Sigma, K, ra, rw), feasible(zg, sector_of, K, cap), dt, "")

    r = sa_qubo(q, num_reads=200, seed=0); zq = np.asarray(r["x"])[:n]
    res["SA"] = (obj(zq, mu, Sigma, K, ra, rw), feasible(zq, sector_of, K, cap), r["time"], "")
    r = tabu_qubo(q, num_reads=100, seed=0); zq = np.asarray(r["x"])[:n]
    res["Tabu"] = (obj(zq, mu, Sigma, K, ra, rw), feasible(zq, sector_of, K, cap), r["time"], "")

    h = GNNHypers(model="qrf", epochs=3000, hidden=128, dim_embedding=32, n_layers=3, lr=1e-3,
                  anneal_rate=0.0, eval_every=50, patience=600, ls_passes=150, n_round_samples=24,
                  refine_sa=True, refine_reads=40)
    r = solve_qubo_gnn(q, h, device="cuda", seed=0); zq = np.asarray(r["x"])[:n]
    res["GNN"] = (obj(zq, mu, Sigma, K, ra, rw), feasible(zq, sector_of, K, cap), r["time"], "")

    best = min(v[0] for v in res.values() if v[1])  # best among FEASIBLE
    print(f"{'method':<14}{'obj':>12}{'gap%':>9}{'feasible':>10}{'t(s)':>8}  note")
    for m_, (o, feas, t, note) in res.items():
        g = (o - best) / abs(best) * 100 if feas else float('nan')
        print(f"{m_:<14}{o:>12.6f}{g:>9.3f}{str(feas):>10}{t:>8.2f}  {note}")
    os.makedirs("results/sectors", exist_ok=True)
    json.dump({k: {"obj": v[0], "feasible": bool(v[1]), "time": v[2], "note": v[3]} for k, v in res.items()},
              open(f"results/sectors/N{N}_S{S}_K{K}_cap{cap}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
