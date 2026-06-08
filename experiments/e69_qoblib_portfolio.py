"""E69 (USER DIRECTIVE): test OUR solver on QOBLIB 06-portfolio (the ready hard-portfolio benchmark).
Reconstruct their exact UQO QUBO from the .zpl (validated: var-count + min/max coeff match their published
metrics, penalty7=1e7), run our GNN-QUBO + SB + tabu, compare to their BEST-KNOWN (gap + feasibility + time).
QOBLIB metric: gap to best-known. Run in .venv.
"""
import sys, os, gzip, json, time, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from qubo import QUBO
from baselines import tabu_qubo, sa_qubo
from gnn_solver import solve_qubo_gnn, GNNHypers
import simulated_bifurcation as sbif, torch
BENCH = os.path.join(os.path.dirname(__file__), "..", "competitors", "QOBLIB", "06-portfolio")
PEN = 1e7  # penalty7 (validated against published min/max coeffs)
CASH, UNIT, DELTA, RHO_C, RHO_S, UB, UPSCALE = 1e6, 1e5, 0.001, 0.0001, 0.000025, 3, 1
CS1 = list(range(4)); CS2 = list(range(5)); SC = [1, 2, 3]; SL = [1, -1]


def load_inst(name):
    pdir = os.path.join(BENCH, "instances", name)
    p = {}; assets = []
    with gzip.open(os.path.join(pdir, "stock_prices.txt.gz"), "rt") as f:
        for ln in f:
            t, a, v = ln.split(); t = int(t); p[(a, t)] = float(v)
            if a not in assets: assets.append(a)
    cov = {}
    with gzip.open(os.path.join(pdir, "covariance_matrices.txt.gz"), "rt") as f:
        for ln in f:
            t, i, j, v = ln.split(); cov[(i, j, int(t))] = float(v)
    T = sorted({tt for (_, tt) in p}); return assets, T, p, cov


def build_qubo(name, b_tot, q):
    S, T, p, cov = load_inst(name); tb, te = T[0], T[-1]; b_csh = CASH / UNIT
    ucnt = {s: UNIT / p[(s, tb)] for s in S}
    up = {(s, t): p[(s, t)] * ucnt[s] for s in S for t in T}
    # variable index
    idx = {}; n = 0
    for s in S:
        for m in SC:
            for sl in SL:
                for t in T: idx[("x", s, m, sl, t)] = n; n += 1
    for k in CS1:
        for t in T: idx[("s1", k, t)] = n; n += 1
    for k in CS2:
        for t in T: idx[("s2", k, t)] = n; n += 1
    Q = np.zeros((n, n)); lin = np.zeros(n); const = 0.0
    def add2(a, b, v):
        Q[a, b] += v / 2; Q[b, a] += v / 2
    SX = [(s, m, sl) for s in S for m in SC for sl in SL]
    # risk (all t)
    for t in T:
        for (i, m, s1) in SX:
            for (j, nn, s2) in SX:
                v = q * s1 * s2 * cov.get((i, j, t), 0.0) * up[(i, t)] * up[(j, t)]
                if v != 0: add2(idx[("x", i, m, s1, t)], idx[("x", j, nn, s2, t)], v)
        for k in CS1: lin[idx[("s1", k, t)]] += -RHO_C * UNIT * (2 ** k)        # cash interest
        for (i, m, sl) in SX:
            if sl == -1: lin[idx[("x", i, m, -1, t)]] += RHO_S * up[(i, t)]       # short interest
    # profit + tx fee, middle t
    for t in T:
        if t == tb or t == te: continue
        for (i, m, sl) in SX:
            lin[idx[("x", i, m, sl, t)]] += -sl * (up[(i, t + 1)] - up[(i, t)])    # profit
            lin[idx[("x", i, m, sl, t)]] += DELTA * up[(i, t)]                      # tx fee linear (x[t])
            lin[idx[("x", i, m, sl, t - 1)]] += DELTA * up[(i, t)]                  # tx fee linear (x[t-1])
            add2(idx[("x", i, m, sl, t - 1)], idx[("x", i, m, sl, t)], -2 * DELTA * up[(i, t)])  # -2 x[t-1]x[t]
    # first day (t=tb)
    for (i, m, sl) in SX:
        lin[idx[("x", i, m, sl, tb)]] += -sl * (up[(i, tb + 1)] - up[(i, tb)])
        lin[idx[("x", i, m, sl, tb)]] += DELTA * up[(i, tb)]
    # last day fee (t=te)
    for (i, m, sl) in SX:
        lin[idx[("x", i, m, sl, te)]] += DELTA * up[(i, te)]
    # scale objective by upscale
    Q *= UPSCALE; lin *= UPSCALE
    # penalties (penalty7), per t
    for t in T:
        # c2: sum sl*x + sum 2^k s1 == b_csh
        terms = [(idx[("x", i, m, sl, t)], sl) for (i, m, sl) in SX] + [(idx[("s1", k, t)], 2 ** k) for k in CS1]
        rhs = b_csh
        for (a, ca) in terms:
            lin[a] += PEN * (ca * ca - 2 * rhs * ca)
            for (b, cb) in terms:
                if a < b: add2(a, b, PEN * 2 * ca * cb)
        const += PEN * rhs * rhs
        # c3: sum x + sum 2^k s2 == b_tot
        terms = [(idx[("x", i, m, sl, t)], 1) for (i, m, sl) in SX] + [(idx[("s2", k, t)], 2 ** k) for k in CS2]
        rhs = b_tot
        for (a, ca) in terms:
            lin[a] += PEN * (ca * ca - 2 * rhs * ca)
            for (b, cb) in terms:
                if a < b: add2(a, b, PEN * 2 * ca * cb)
        const += PEN * rhs * rhs
    np.fill_diagonal(Q, np.diag(Q) + lin)  # linear -> diagonal (x^2=x)
    return QUBO(0.5 * (Q + Q.T), offset=const), n


def energy(q, x): return q.energy(np.asarray(x))


def main():
    name, b_tot = "po_a010_t10_orig", 4
    BEST = {0.0: -231106, 1e-6: -227475, 1e-5: -204799, 5e-5: -169143, 1e-4: -142817, 5e-4: -59937, 1e-3: -30169, 1e-2: -1000}
    h = GNNHypers(model="qrf", epochs=2500, hidden=128, dim_embedding=20, n_layers=3, lr=1e-3, anneal_rate=0.0,
                  eval_every=50, patience=500, ls_passes=200, n_round_samples=24, refine_sa=True, refine_reads=40)
    out = {}
    for q in [0.0, 1e-5, 1e-3]:
        qubo, n = build_qubo(name, b_tot, q)
        if q == 0.0:
            offdiag = qubo.Q - np.diag(np.diag(qubo.Q)); dmax = max(qubo.Q.max(), (2 * offdiag).max()); dmin = min(np.diag(qubo.Q).min(), (2 * offdiag).min())
            print(f"  [validate] n={n} (pub 690), coeff range ~[{dmin:.3g},{dmax:.3g}] (pub [-9.6e8, 1.28e9])", flush=True)
        bk = BEST[q]
        eg = energy(qubo, solve_qubo_gnn(qubo, h, device="cuda", seed=0)["x"])
        x, _ = sbif.minimize(torch.tensor(qubo.Q.astype(np.float64)), domain="binary", agents=128, max_steps=30000, best_only=True); es = energy(qubo, np.asarray(x.cpu()).reshape(-1))
        et = energy(qubo, tabu_qubo(qubo, num_reads=100, seed=0)["x"])
        gap = lambda e: (e - bk) / abs(bk) * 100
        out[q] = {"best_known": bk, "GNN": eg, "SB": es, "tabu": et}
        print(f"  q={q}: BK={bk} | GNN {eg:.0f}({gap(eg):+.1f}%) | SB {es:.0f}({gap(es):+.1f}%) | tabu {et:.0f}({gap(et):+.1f}%)", flush=True)
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "results", "e69_qoblib.json"), "w"), indent=2)
    print("saved", flush=True)


if __name__ == "__main__":
    main()
