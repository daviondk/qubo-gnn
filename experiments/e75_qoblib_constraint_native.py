"""E75 (IMPROVE our GNN on QOBLIB): CONSTRAINT-NATIVE decode. Instead of penalty+naive-repair, decode the
GNN's continuous probabilities per period into a feasible-by-construction selection: pick top-<=b_tot assets
(by GNN prob) per period, set slacks to satisfy budget+cardinality (always feasible since <=b_tot selected).
Uses GNN's learned preferences, respects constraints by design. Run in .venv."""
import sys, os, json, numpy as np
sys.path.insert(0, "experiments"); sys.path.insert(0, "src")
from e70b_qoblib_feasible import load_qs, load_tbl, feasible
from e72_qoblib_evalobj import eval_obj, load_data
import e72_qoblib_evalobj as E
from qubo import QUBO
from gnn_solver import solve_qubo_gnn, GNNHypers
from baselines import tabu_qubo
T, B_TOT, B_CSH = 10, 4, 10
CS1, CS2 = list(range(4)), list(range(5))


def cn_decode(p, mp, n, kcap=B_TOT, thr=None):
    x = np.zeros(n, dtype=np.int8)
    s1idx = {(v[1], v[2]): i for i, v in mp.items() if v[0] == "s1"}
    s2idx = {(v[1], v[2]): i for i, v in mp.items() if v[0] == "s2"}
    for t in range(T):
        cands = sorted([(i, p[i], v[3]) for i, v in mp.items() if v[0] == "x" and v[4] == t], key=lambda c: -c[1])
        chosen = cands[:kcap] if thr is None else [c for c in cands[:kcap] if c[1] > thr]
        for (i, _, _) in chosen: x[i] = 1
        cnt = len(chosen); r3 = B_TOT - cnt
        for k in CS2: x[s2idx[(k, t)]] = (r3 >> k) & 1
        signed = sum(sl for (_, _, sl) in chosen)  # #long - #short
        r2 = B_CSH - signed
        r2 = max(0, min(15, r2))
        for k in CS1: x[s1idx[(k, t)]] = (r2 >> k) & 1
    return x


def main():
    mp = load_tbl("x"); S, p_data, cov = load_data()
    q, n = load_qs("experiments/results/qoblib_qs/a010_q0.qs"); BK = -110525; qv = 0.0
    E.PBOOST = 1e7; qb = E.boost(q.Q, mp); scale = np.abs(qb.Q[qb.Q != 0]).mean(); qn = QUBO(qb.Q / scale)
    print(f"QOBLIB orig q=0 best-known={BK}", flush=True)
    best = None
    for seed in range(5):
        h = GNNHypers(model="qrf", epochs=4000, hidden=128, dim_embedding=20, n_layers=3, lr=1e-3, anneal_rate=2e-4,
                      eval_every=400, patience=4000, ls_passes=200, n_round_samples=30, refine_sa=False)
        r = solve_qubo_gnn(qn, h, device="cuda", seed=seed); pc = r["p_continuous"]
        for kcap in [B_TOT]:
            for thr in [None, 0.5]:
                x = cn_decode(pc, mp, n, kcap, thr); f, v = feasible(x, mp); o = eval_obj(x, mp, S, p_data, cov, qv)
                if f and (best is None or o < best[0]): best = (o, seed, thr)
    bo, bs, bt = best
    print(f"  BEST constraint-native: obj={bo:.0f} feasible=True gap={(bo-BK)/abs(BK)*100:+.1f}% (seed {bs}, thr {bt})", flush=True)
    print(f"  vs best-known {BK} | (prev: penalty+repair feasible -64395 = +41.7%)", flush=True)
    json.dump({"best_obj": bo, "gap%": (bo - BK) / abs(BK) * 100, "bk": BK}, open("experiments/results/e75_cn.json", "w"), indent=2)


if __name__ == "__main__":
    main()
