"""E72 (USER DIRECTIVE): correct QOBLIB benchmark using THEIR exact objective (bqp_eval.zpl: obj=q*risk-profit
on normalized up). Their .sol.mst is suboptimal (evals to -110534 vs README best-known -231106), so we
compute THEIR objective directly for OUR feasible solutions (constraint-aware boosted solve) and compare to
the README best-known. Run in .venv."""
import sys, os, json, gzip, numpy as np
sys.path.insert(0, "experiments"); sys.path.insert(0, "src")
from e70b_qoblib_feasible import load_qs, load_tbl, feasible, load_sol
from qubo import QUBO
from baselines import tabu_qubo
from gnn_solver import solve_qubo_gnn, GNNHypers
T, B_TOT, B_CSH, PBOOST = 10, 4, 10, 5e8
DELTA, RHO_C, RHO_S, UNIT, UPSCALE = 0.001, 0.0001, 0.000025, 1e5, 1
SC, SL, CS1, CS2 = [1, 2, 3], [1, -1], range(4), range(5)
README_BK = {"0": -231106, "0.00001": -204799, "0.001": -30169}
h = GNNHypers(model="qrf", epochs=2500, hidden=128, dim_embedding=20, n_layers=3, lr=1e-3, anneal_rate=0.0,
              eval_every=50, patience=500, ls_passes=200, n_round_samples=24, refine_sa=True, refine_reads=40)


def load_data(name="po_a010_t10_orig"):
    pdir = os.path.join("competitors/QOBLIB/06-portfolio/instances", name)
    S = []; p = {}
    for ln in gzip.open(pdir + "/stock_prices.txt.gz", "rt"):
        t, a, v = ln.split(); t = int(t); p[(a, t)] = float(v)
        if a not in S: S.append(a)
    cov = {}
    for ln in gzip.open(pdir + "/covariance_matrices.txt.gz", "rt"):
        t, i, j, v = ln.split(); cov[(i, j, int(t))] = float(v)
    return S, p, cov


def eval_obj(x, mp, S, p, cov, q):
    """THEIR objective obj = q*risk - profit (bqp_eval_u3_c10.zpl), normalized up."""
    up = {(a, t): p[(a, t)] * (UNIT / p[(a, 0)]) for a in S for t in range(T)}
    X = {}; s1 = {}
    for i, v in mp.items():
        if v[0] == "x": X[(v[1], v[2], v[3], v[4])] = int(x[i])
        elif v[0] == "s1": s1[(v[1], v[2])] = int(x[i])
    SX = [(ai, m, sl) for ai in range(len(S)) for m in SC for sl in SL]
    risk = 0.0
    if q != 0:
     for t in range(T):
        for (ai, m, s1l) in SX:
            xi = X.get((ai, m, s1l, t), 0)
            if not xi: continue
            for (aj, nn, s2l) in SX:
                xj = X.get((aj, nn, s2l, t), 0)
                if xj: risk += s1l * s2l * cov.get((S[ai], S[aj], t), 0.0) * up[(S[ai], t)] * up[(S[aj], t)] * xi * xj
    profit = 0.0; trans = 0.0
    for t in range(T):
        profit += RHO_C * UNIT * sum((2 ** k) * s1.get((k, t), 0) for k in CS1)
        profit -= RHO_S * sum(up[(S[ai], t)] * X.get((ai, m, -1, t), 0) for (ai, m, sl) in SX if sl == -1)
        if 0 < t < T - 1:
            for (ai, m, sl) in SX:
                profit += sl * (up[(S[ai], t + 1)] - up[(S[ai], t)]) * X.get((ai, m, sl, t), 0)
    for (ai, m, sl) in SX:
        profit += sl * (up[(S[ai], 1)] - up[(S[ai], 0)]) * X.get((ai, m, sl, 0), 0)  # first day
    for t in range(T):
        if 0 < t < T - 1:
            for (ai, m, sl) in SX:
                trans += DELTA * up[(S[ai], t)] * (X.get((ai, m, sl, t - 1), 0) + X.get((ai, m, sl, t), 0) - 2 * X.get((ai, m, sl, t - 1), 0) * X.get((ai, m, sl, t), 0))
    for (ai, m, sl) in SX:
        trans += DELTA * up[(S[ai], 0)] * X.get((ai, m, sl, 0), 0)
        trans += DELTA * up[(S[ai], T - 1)] * X.get((ai, m, sl, T - 1), 0)
    profit -= trans
    return UPSCALE * (q * risk - profit)


def boost(Q, mp):
    Q2 = Q.copy()
    for t in range(T):
        for rhs, sel in [(B_TOT, [(i, 1) for i, v in mp.items() if v[0] == "x" and v[4] == t] + [(i, 2 ** v[1]) for i, v in mp.items() if v[0] == "s2" and v[2] == t]),
                         (B_CSH, [(i, v[3]) for i, v in mp.items() if v[0] == "x" and v[4] == t] + [(i, 2 ** v[1]) for i, v in mp.items() if v[0] == "s1" and v[2] == t])]:
            for a, ca in sel:
                Q2[a, a] += PBOOST * (ca * ca - 2 * rhs * ca)
                for b, cb in sel:
                    if a < b: Q2[a, b] += PBOOST * ca * cb; Q2[b, a] += PBOOST * ca * cb
    return QUBO(Q2)


def main():
    mp = load_tbl("x"); S, p, cov = load_data(); out = {}
    for qtag, qv in [("0", 0.0), ("0.00001", 1e-5), ("0.001", 1e-3)]:
        qs = f"experiments/results/qoblib_qs/a010_q{qtag}.qs"
        if not os.path.exists(qs): continue
        q, n = load_qs(qs); qb = boost(q.Q, mp); bk = README_BK[qtag]
        their = load_sol(qtag)[:n]; their_obj = eval_obj(their, mp, S, p, cov, qv)
        print(f"[q={qtag}] README best-known={bk} | their .mst evals to {their_obj:.0f} (their solver objective)", flush=True)
        res = {"readme_bk": bk, "their_mst_obj": float(their_obj)}
        for name, getx in [("GNN", lambda: np.asarray(solve_qubo_gnn(qb, h, device="cuda", seed=0)["x"])),
                           ("tabu", lambda: np.asarray(tabu_qubo(qb, num_reads=300, seed=0)["x"]))]:
            x = getx().astype(np.int8)[:n]; f_, v_ = feasible(x, mp); o_ = eval_obj(x, mp, S, p, cov, qv)
            gap = (o_ - bk) / abs(bk) * 100 if f_ else None
            res[name] = {"feasible": bool(f_), "obj": float(o_), "gap_vs_readme%": gap}
            print(f"   {name}: feasible={f_} their-objective={o_:.0f} -> {f'gap {gap:+.2f}% vs README best-known' if f_ else 'INFEASIBLE'}", flush=True)
        out[qtag] = res
    json.dump(out, open("experiments/results/e72_qoblib_evalobj.json", "w"), indent=2)
    print("saved", flush=True)


if __name__ == "__main__":
    main()
