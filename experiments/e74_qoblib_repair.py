"""E74 (USER DIRECTIVE - IMPROVE our solver on QOBLIB): constraint-native slack-REPAIR. Generic penalty-QUBO
gives near-best objective but infeasible (slack bits uncoordinated). We take the solver's x-selection and
RECOMPUTE slacks (s1,s2) per period to satisfy c2/c3; if cardinality over budget, drop lowest x. Slacks only
affect cash-interest (minor) -> feasible + near-best objective. Compare to README best-known. Run in .venv."""
import sys, os, json, numpy as np
sys.path.insert(0, "experiments"); sys.path.insert(0, "src")
from e70b_qoblib_feasible import load_qs, load_tbl, feasible, load_sol
from e72_qoblib_evalobj import eval_obj, load_data, boost
import e72_qoblib_evalobj as E
from baselines import tabu_qubo, sa_qubo
from gnn_solver import solve_qubo_gnn, GNNHypers
T, B_TOT, B_CSH = 10, 4, 10
CS1, CS2 = list(range(4)), list(range(5))  # s1: 0..15, s2: 0..31
BK = {"0": -110525, "0.00001": None, "0.001": None}  # orig best-knowns (q0 known)


def repair(x, mp, scores=None):
    """Given x (full vector), per period: drop excess x to meet cardinality, recompute slack bits to satisfy c2/c3."""
    x = x.copy()
    xidx = {(v[1], v[2], v[3], v[4]): i for i, v in mp.items() if v[0] == "x"}
    s1idx = {(v[1], v[2]): i for i, v in mp.items() if v[0] == "s1"}
    s2idx = {(v[1], v[2]): i for i, v in mp.items() if v[0] == "s2"}
    assets = sorted({k[0] for k in xidx}); ms = sorted({k[1] for k in xidx}); sls = sorted({k[2] for k in xidx})
    for t in range(T):
        sel = [(a, m, sl) for a in assets for m in ms for sl in sls if x[xidx[(a, m, sl, t)]] == 1]
        # cardinality c3: need count <= B_TOT (slack fills the rest)
        while len(sel) > B_TOT:
            # drop the one whose removal least hurts (by score if given, else arbitrary)
            drop = min(sel, key=lambda ams: (scores[xidx[(ams[0], ams[1], ams[2], t)]] if scores is not None else 0))
            x[xidx[(drop[0], drop[1], drop[2], t)]] = 0; sel.remove(drop)
        cnt = len(sel)
        r3 = B_TOT - cnt  # must be 0..31 -> always since 0<=cnt<=4
        for k in CS2: x[s2idx[(k, t)]] = (r3 >> k) & 1
        # cash c2: sum sl*x + sum 2^k s1 = B_CSH ; residual r2 = B_CSH - sum(sl*x), need 0..15
        signed = sum(sl for (a, m, sl) in sel)
        r2 = B_CSH - signed
        if 0 <= r2 <= 15:
            for k in CS1: x[s1idx[(k, t)]] = (r2 >> k) & 1
        else:
            # adjust: drop a short (sl=-1) if r2>15, or a long if r2<0, to bring into range
            while r2 > 15 and any(sl == -1 for (a, m, sl) in sel):
                d = next((a, m, sl) for (a, m, sl) in sel if sl == -1); x[xidx[(d[0], d[1], d[2], t)]] = 0; sel.remove(d)
                r3 = B_TOT - len(sel)
                for k in CS2: x[s2idx[(k, t)]] = (r3 >> k) & 1
                r2 = B_CSH - sum(sl for (a, m, sl) in sel)
            while r2 < 0 and any(sl == 1 for (a, m, sl) in sel):
                d = next((a, m, sl) for (a, m, sl) in sel if sl == 1); x[xidx[(d[0], d[1], d[2], t)]] = 0; sel.remove(d)
                r3 = B_TOT - len(sel)
                for k in CS2: x[s2idx[(k, t)]] = (r3 >> k) & 1
                r2 = B_CSH - sum(sl for (a, m, sl) in sel)
            r2 = max(0, min(15, r2))
            for k in CS1: x[s1idx[(k, t)]] = (r2 >> k) & 1
    return x


def main():
    mp = load_tbl("x"); S, p, cov = load_data(); h = GNNHypers(model="qrf", epochs=3000, hidden=128, dim_embedding=20, n_layers=3, lr=1e-3, anneal_rate=0.0, eval_every=50, patience=600, ls_passes=300, n_round_samples=32, refine_sa=True, refine_reads=60)
    q, n = load_qs("experiments/results/qoblib_qs/a010_q0.qs"); bk = BK["0"]; qv = 0.0
    print(f"QOBLIB orig q=0, best-known={bk}", flush=True)
    out = {}
    for P in [1e7, 2e7, 3e7]:
        E.PBOOST = P - 1e7 if P > 1e7 else 0.0; qb = boost(q.Q, mp) if P > 1e7 else q
        for nm, getx in [("tabu", lambda: np.asarray(tabu_qubo(qb, num_reads=500, seed=0)["x"])),
                         ("GNN", lambda: np.asarray(solve_qubo_gnn(qb, h, device="cuda", seed=0)["x"]))]:
            x = getx().astype(np.int8)[:n]
            f0, v0 = feasible(x, mp); o0 = eval_obj(x, mp, S, p, cov, qv)
            xr = repair(x, mp); fr, vr = feasible(xr, mp); orr = eval_obj(xr, mp, S, p, cov, qv)
            gap = (orr - bk) / abs(bk) * 100 if fr else None
            print(f"  P={P:.0e} {nm}: raw feasible={f0} obj={o0:.0f} | REPAIRED feasible={fr} viol={vr} obj={orr:.0f} gap={f'{gap:+.1f}%' if fr else '-'}", flush=True)
            out[f"{P:.0e}_{nm}"] = {"raw_feasible": bool(f0), "raw_obj": float(o0), "repaired_feasible": bool(fr), "repaired_obj": float(orr), "gap%": gap}
    json.dump(out, open("experiments/results/e74_qoblib_repair.json", "w"), indent=2); print("saved", flush=True)


if __name__ == "__main__":
    main()
