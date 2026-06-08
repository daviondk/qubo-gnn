"""E70b (USER DIRECTIVE): VALID QOBLIB benchmark with FEASIBILITY checking. The penalized .qs has
lower-energy INFEASIBLE points; QOBLIB best-known is the best FEASIBLE objective. We parse the .tbl
(qs-index -> asset#m#sl#t / s1 / s2), run our solvers on the exact .qs, and report FEASIBILITY (c2/c3
per period) + objective. Run in .venv."""
import sys, os, json, time, numpy as np, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from qubo import QUBO
from baselines import tabu_qubo, sa_qubo
from gnn_solver import solve_qubo_gnn, GNNHypers
import simulated_bifurcation as sbif, torch
QSDIR = os.path.join(os.path.dirname(__file__), "results", "qoblib_qs")
SOLDIR = os.path.join(os.path.dirname(__file__), "..", "competitors", "QOBLIB", "06-portfolio",
                      "solutions", "uqo", "_x_a010", "a010_t10_orig_b004")
B_TOT, B_CSH = 4, 10


def load_qs(path):
    lines = [ln for ln in open(path) if not ln.startswith("#") and ln.strip()]
    n = int(lines[0].split()[0]); Q = np.zeros((n, n))
    for ln in lines[1:]:
        a, b, v = ln.split(); a = int(a) - 1; b = int(b) - 1; v = float(v)
        if a == b: Q[a, a] += v
        else: Q[a, b] += v / 2; Q[b, a] += v / 2
    return QUBO(0.5 * (Q + Q.T)), n


def load_tbl(path, n_assets=10, T=10, SC=(1, 2, 3), SL=(1, -1), CS1=range(4), CS2=range(5)):
    # ZIMPL truncates long var names in .tbl, so build index->var map from the KNOWN declaration ORDER
    # (validated: x[asset,m,sl,t] t-fastest; then s1[k,t]; then s2[k,t]). path arg kept for signature compat.
    mp = {}; i = 0
    for a in range(n_assets):
        for m in SC:
            for sl in SL:
                for t in range(T): mp[i] = ("x", a, m, sl, t); i += 1
    for k in CS1:
        for t in range(T): mp[i] = ("s1", k, t); i += 1
    for k in CS2:
        for t in range(T): mp[i] = ("s2", k, t); i += 1
    return mp


def feasible(x, mp):
    Ts = sorted({v[-1] for v in mp.values() if v[0] in ("x", "s1", "s2")})
    ok = True; viol = 0
    for t in Ts:
        c3 = sum(x[i] for i, v in mp.items() if v[0] == "x" and v[4] == t) \
            + sum((2 ** v[1]) * x[i] for i, v in mp.items() if v[0] == "s2" and v[2] == t)
        c2 = sum(v[3] * x[i] for i, v in mp.items() if v[0] == "x" and v[4] == t) \
            + sum((2 ** v[1]) * x[i] for i, v in mp.items() if v[0] == "s1" and v[2] == t)
        if c3 != B_TOT: viol += 1; ok = False
        if c2 != B_CSH: viol += 1; ok = False
    return ok, viol


def load_sol(qtag):
    f = os.path.join(SOLDIR, f"uqo_a010_t10_q{qtag}_b004.sol.mst"); vals = {}
    for ln in open(f):
        p = ln.split()
        if len(p) >= 2 and p[0].startswith("x#"): vals[int(p[0][2:]) - 1] = int(round(float(p[1])))
    return np.array([vals[k] for k in range(len(vals))], dtype=np.int8)


def main():
    h = GNNHypers(model="qrf", epochs=2500, hidden=128, dim_embedding=20, n_layers=3, lr=1e-3, anneal_rate=0.0,
                  eval_every=50, patience=500, ls_passes=200, n_round_samples=24, refine_sa=True, refine_reads=40)
    out = {}
    for qtag in ["0", "0.00001", "0.001"]:
        qs = os.path.join(QSDIR, f"a010_q{qtag}.qs")
        if not os.path.exists(qs): continue
        q, n = load_qs(qs); mp = load_tbl(os.path.join(QSDIR, f"a010_q{qtag}.tbl"))
        xs = load_sol(qtag); fk, vk = feasible(xs[:n], mp); ek = q.energy(xs[:n])
        print(f"[q={qtag}] THEIR best-known: feasible={fk} viol={vk} energy={ek:.0f}", flush=True)
        res = {"their": [bool(fk), int(vk), float(ek)]}
        for name, getx in [("GNN", lambda: np.asarray(solve_qubo_gnn(q, h, device="cuda", seed=0)["x"])),
                           ("SB", lambda: np.asarray(sbif.minimize(torch.tensor(q.Q.astype(np.float64)), domain="binary", agents=128, max_steps=30000, best_only=True)[0].cpu()).reshape(-1)),
                           ("tabu", lambda: np.asarray(tabu_qubo(q, num_reads=100, seed=0)["x"]))]:
            x = getx().astype(np.int8)[:n]; f_, v_ = feasible(x, mp); e_ = q.energy(x)
            res[name] = [bool(f_), int(v_), float(e_)]
            print(f"   {name}: feasible={f_} viol={v_} energy={e_:.0f}", flush=True)
        out[qtag] = res
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "results", "e70b_qoblib_feasible.json"), "w"), indent=2)
    print("saved", flush=True)


if __name__ == "__main__":
    main()
