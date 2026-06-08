"""E78 (TUNE GNN on QOBLIB - close the gap): feasible-preserving local search on QIGNN solution.
QIGNN finds near-optimal raw (-106460, ~3.7%) but infeasible; repair loses objective. Here: start from a
feasible solution (QIGNN->repair) and HILL-CLIMB via per-period swaps (deselect held / select unheld),
recomputing slacks to stay feasible, accept if THEIR objective improves. Goal: approach best-known -110525.
Run in .venv."""
import sys, os, json, time, numpy as np, torch
sys.path.insert(0, "experiments"); sys.path.insert(0, "src")
from e70b_qoblib_feasible import load_qs, load_tbl, feasible
from e72_qoblib_evalobj import eval_obj, load_data
import e72_qoblib_evalobj as E
from e74_qoblib_repair import repair
from e76_qignn import solve_qignn
from qubo import QUBO
T, B_TOT, B_CSH = 10, 4, 10
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def set_slacks(x, mp, t, s1idx, s2idx, xkeys_t):
    cnt = sum(x[i] for i in xkeys_t); r3 = max(0, min(31, B_TOT - cnt))
    for k in range(5): x[s2idx[(k, t)]] = (r3 >> k) & 1
    signed = sum(mp[i][3] * x[i] for i in xkeys_t)  # #long-#short among selected
    r2 = max(0, min(15, B_CSH - signed))
    for k in range(4): x[s1idx[(k, t)]] = (r2 >> k) & 1


def feasible_ls(x, mp, S, p, cov, qv, rounds=6):
    x = x.copy()
    s1idx = {(v[1], v[2]): i for i, v in mp.items() if v[0] == "s1"}
    s2idx = {(v[1], v[2]): i for i, v in mp.items() if v[0] == "s2"}
    xvars_t = {t: [i for i, v in mp.items() if v[0] == "x" and v[4] == t] for t in range(T)}
    cur = eval_obj(x, mp, S, p, cov, qv)
    for _ in range(rounds):
        improved = False
        for t in range(T):
            xkeys = xvars_t[t]
            held = [i for i in xkeys if x[i] == 1]; unheld = [i for i in xkeys if x[i] == 0]
            # try: swap each held<->unheld; also try add (if <B_TOT) and remove
            moves = []
            for h in held:
                for u in unheld: moves.append(("swap", h, u))
                moves.append(("rem", h, None))
            if len(held) < B_TOT:
                for u in unheld: moves.append(("add", None, u))
            best_mv = None; best_o = cur
            for mv, h, u in moves:
                if mv == "swap": x[h] = 0; x[u] = 1
                elif mv == "rem": x[h] = 0
                else: x[u] = 1
                set_slacks(x, mp, t, s1idx, s2idx, xkeys)
                o = eval_obj(x, mp, S, p, cov, qv)
                if o < best_o - 1e-9: best_o = o; best_mv = (mv, h, u)
                # revert
                if mv == "swap": x[h] = 1; x[u] = 0
                elif mv == "rem": x[h] = 1
                else: x[u] = 0
                set_slacks(x, mp, t, s1idx, s2idx, xkeys)
            if best_mv:
                mv, h, u = best_mv
                if mv == "swap": x[h] = 0; x[u] = 1
                elif mv == "rem": x[h] = 0
                else: x[u] = 1
                set_slacks(x, mp, t, s1idx, s2idx, xkeys); cur = best_o; improved = True
        if not improved: break
    return x, cur


def main():
    mp = load_tbl("x"); S, p, cov = load_data()
    q, n = load_qs("experiments/results/qoblib_qs/a010_q0.qs"); BK = -110525; qv = 0.0
    E.PBOOST = 1e7; qb = E.boost(q.Q, mp)  # Pboost=2e7 total
    Qd = np.asarray(qb.Q); diag = np.diag(Qd).reshape(-1, 1)
    sf = np.column_stack([(diag - diag.mean()) / (diag.std() + 1e-9), np.ones((n, 1))]).astype(np.float32)
    A = Qd - np.diag(np.diag(Qd)); rr, cc = np.nonzero(A); ei = torch.tensor(np.vstack([rr, cc]), dtype=torch.long, device=DEV)
    best = None
    for seed in range(4):
        r = solve_qignn(qb, sf, ei, epochs=4000, anneal=2e-4, eval_every=300, ls_passes=300, n_round=40, seed=seed)
        x0 = repair(np.asarray(r["x"]).astype(np.int8)[:n], mp)
        o0 = eval_obj(x0, mp, S, p, cov, qv)
        t0 = time.time(); xls, ols = feasible_ls(x0, mp, S, p, cov, qv, rounds=8); dt = time.time() - t0
        feas = feasible(xls, mp)[0]
        print(f"  seed{seed}: QIGNN+repair {o0:.0f} -> +feasible-LS {ols:.0f} (feas {feas}, {dt:.0f}s) gap={(ols-BK)/abs(BK)*100:+.1f}%", flush=True)
        if feas and (best is None or ols < best): best = ols
    print(f"=> BEST feasible (QIGNN+repair+LS): {best:.0f} gap={(best-BK)/abs(BK)*100:+.1f}% (BK {BK}; prior best -64395=+41.7%)", flush=True)
    json.dump({"best": best, "gap%": (best - BK) / abs(BK) * 100, "bk": BK}, open("experiments/results/e78_feasible_ls.json", "w"), indent=2)


if __name__ == "__main__":
    main()
