"""E79 (DOES THE GNN ADD VALUE AT SCALE?): on a050 (best-known -501631, b_tot=20), compare
LS-from-random vs QIGNN+LS. If LS-alone reaches best-known, GNN unneeded (like a010); if GNN-init helps,
GNN adds value. Generalized for n_assets/b_tot. Run in .venv."""
import sys, os, json, time, gzip, numpy as np, torch
sys.path.insert(0, "experiments"); sys.path.insert(0, "src")
from e70b_qoblib_feasible import load_qs
from e76_qignn import solve_qignn
DEV = "cuda" if torch.cuda.is_available() else "cpu"
T, NA, B_TOT, B_CSH = 10, 50, 20, 10
DELTA, RHO_C, RHO_S, UNIT = 0.001, 0.0001, 0.000025, 1e5
SC, SL, CS1, CS2 = [1, 2, 3], [1, -1], list(range(4)), list(range(5))
INST = "po_a050_t10_orig"; BK = -501631


def build_map(na):
    mp = {}; i = 0
    for a in range(na):
        for m in SC:
            for sl in SL:
                for t in range(T): mp[i] = ("x", a, m, sl, t); i += 1
    for k in CS1:
        for t in range(T): mp[i] = ("s1", k, t); i += 1
    for k in CS2:
        for t in range(T): mp[i] = ("s2", k, t); i += 1
    return mp


def load_data(name):
    pdir = os.path.join("competitors/QOBLIB/06-portfolio/instances", name); S = []; p = {}
    for ln in gzip.open(pdir + "/stock_prices.txt.gz", "rt"):
        t, a, v = ln.split(); t = int(t); p[(a, t)] = float(v)
        if a not in S: S.append(a)
    return S, p


def eval_q0(x, mp, S, up):
    X = {}; s1 = {}
    for i, v in mp.items():
        if x[i]:
            if v[0] == "x": X[(v[1], v[2], v[3], v[4])] = 1
            elif v[0] == "s1": s1[(v[1], v[2])] = 1
    SX = [(ai, m, sl) for ai in range(NA) for m in SC for sl in SL]
    profit = 0.0; trans = 0.0
    for t in range(T):
        profit += RHO_C * UNIT * sum((2 ** k) for k in CS1 if s1.get((k, t), 0))
        for (ai, m, sl) in SX:
            if X.get((ai, m, sl, t), 0):
                if sl == -1: profit -= RHO_S * up[ai, t]
                if 0 < t < T - 1: profit += sl * (up[ai, t + 1] - up[ai, t])
    for (ai, m, sl) in SX:
        if X.get((ai, m, sl, 0), 0): profit += sl * (up[ai, 1] - up[ai, 0])
    for t in range(T):
        if 0 < t < T - 1:
            for (ai, m, sl) in SX:
                a = X.get((ai, m, sl, t - 1), 0); b = X.get((ai, m, sl, t), 0)
                if a or b: trans += DELTA * up[ai, t] * (a + b - 2 * a * b)
    for (ai, m, sl) in SX:
        if X.get((ai, m, sl, 0), 0): trans += DELTA * up[ai, 0]
        if X.get((ai, m, sl, T - 1), 0): trans += DELTA * up[ai, T - 1]
    return -(profit - trans)


def feasible(x, mp):
    for t in range(T):
        c3 = sum(x[i] for i, v in mp.items() if v[0] == "x" and v[4] == t) + sum((2 ** v[1]) * x[i] for i, v in mp.items() if v[0] == "s2" and v[2] == t)
        c2 = sum(v[3] * x[i] for i, v in mp.items() if v[0] == "x" and v[4] == t) + sum((2 ** v[1]) * x[i] for i, v in mp.items() if v[0] == "s1" and v[2] == t)
        if c3 != B_TOT or c2 != B_CSH: return False
    return True


def set_slacks(x, mp, t, s1idx, s2idx, xkeys):
    cnt = sum(x[i] for i in xkeys); r3 = max(0, min(31, B_TOT - cnt))
    for k in CS2: x[s2idx[(k, t)]] = (r3 >> k) & 1
    signed = sum(mp[i][3] * x[i] for i in xkeys); r2 = max(0, min(15, B_CSH - signed))
    for k in CS1: x[s1idx[(k, t)]] = (r2 >> k) & 1


def feasible_ls(x, mp, S, up, rounds=5, cap=30):
    x = x.copy(); rng = np.random.default_rng(0)
    s1idx = {(v[1], v[2]): i for i, v in mp.items() if v[0] == "s1"}; s2idx = {(v[1], v[2]): i for i, v in mp.items() if v[0] == "s2"}
    xv = {t: [i for i, v in mp.items() if v[0] == "x" and v[4] == t] for t in range(T)}
    cur = eval_q0(x, mp, S, up)
    for _ in range(rounds):
        imp = False
        for t in range(T):
            xk = xv[t]; held = [i for i in xk if x[i] == 1]; unheld = [i for i in xk if x[i] == 0]
            cand_u = list(rng.choice(unheld, min(cap, len(unheld)), replace=False)) if len(unheld) > cap else unheld
            bo = cur; bm = None
            for h in held:
                for u in cand_u:
                    x[h] = 0; x[u] = 1; set_slacks(x, mp, t, s1idx, s2idx, xk); o = eval_q0(x, mp, S, up)
                    if o < bo - 1e-9: bo = o; bm = (h, u)
                    x[h] = 1; x[u] = 0
                set_slacks(x, mp, t, s1idx, s2idx, xk)
            if bm: x[bm[0]] = 0; x[bm[1]] = 1; set_slacks(x, mp, t, s1idx, s2idx, xk); cur = bo; imp = True
        if not imp: break
    return x, cur


def main():
    S, p = load_data(INST); up = np.array([[p[(S[a], t)] * (UNIT / p[(S[a], 0)]) for t in range(T)] for a in range(NA)])
    mp = build_map(NA); q, n = load_qs(f"experiments/results/qoblib_qs/a050_q0.qs")
    s1idx = {(v[1], v[2]): i for i, v in mp.items() if v[0] == "s1"}; s2idx = {(v[1], v[2]): i for i, v in mp.items() if v[0] == "s2"}
    xv = {t: [i for i, v in mp.items() if v[0] == "x" and v[4] == t] for t in range(T)}
    rng = np.random.default_rng(0)
    print(f"a050 q=0 best-known={BK}, n={n}", flush=True)
    # LS-from-random
    rnd_best = None
    for tr in range(3):
        x = np.zeros(n, np.int8)
        for t in range(T):
            for i in rng.choice(xv[t], B_TOT, replace=False): x[i] = 1
            set_slacks(x, mp, t, s1idx, s2idx, xv[t])
        t0 = time.time(); xls, o = feasible_ls(x, mp, S, up, rounds=5); dt = time.time() - t0
        rnd_best = o if rnd_best is None else min(rnd_best, o)
        print(f"  LS-from-random tr{tr}: {o:.0f} gap={(o-BK)/abs(BK)*100:+.1f}% ({dt:.0f}s)", flush=True)
    # GNN + LS
    Qd = np.asarray(q.Q); diag = np.diag(Qd).reshape(-1, 1)
    sf = np.column_stack([(diag - diag.mean()) / (diag.std() + 1e-9), np.ones((n, 1))]).astype(np.float32)
    A = Qd - np.diag(np.diag(Qd)); rr, cc = np.nonzero(A); ei = torch.tensor(np.vstack([rr, cc]), dtype=torch.long, device=DEV)
    r = solve_qignn(q, sf, ei, epochs=3000, anneal=2e-4, eval_every=300, ls_passes=200, n_round=30, seed=0)
    xg = np.asarray(r["x"]).astype(np.int8)[:n]
    # repair to feasible per period (keep top-B_TOT by GNN raw x, set slacks)
    for t in range(T):
        held = [i for i in xv[t] if xg[i] == 1]
        if len(held) > B_TOT:
            for i in held[B_TOT:]: xg[i] = 0
        set_slacks(xg, mp, t, s1idx, s2idx, xv[t])
    xgl, og = feasible_ls(xg, mp, S, up, rounds=5)
    print(f"  GNN+LS: {og:.0f} gap={(og-BK)/abs(BK)*100:+.1f}% feasible={feasible(xgl,mp)}", flush=True)
    print(f"=> LS-random best {rnd_best:.0f} ({(rnd_best-BK)/abs(BK)*100:+.1f}%) vs GNN+LS {og:.0f} ({(og-BK)/abs(BK)*100:+.1f}%)", flush=True)
    json.dump({"bk": BK, "ls_random": rnd_best, "gnn_ls": og}, open("experiments/results/e79_a050.json", "w"), indent=2)


if __name__ == "__main__":
    main()
