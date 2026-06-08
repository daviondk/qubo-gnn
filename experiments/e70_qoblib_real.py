"""E70 (USER DIRECTIVE): REAL QOBLIB 06-portfolio benchmark using their EXACT ZIMPL-generated QUBO (.qs,
via WSL). Validate against their best-known, then run our GNN-QUBO + SB + tabu (QOBLIB metric: gap to
best-known). Run in .venv."""
import sys, os, json, time, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from qubo import QUBO
from baselines import tabu_qubo, sa_qubo
from gnn_solver import solve_qubo_gnn, GNNHypers
import simulated_bifurcation as sbif, torch
QSDIR = os.path.join(os.path.dirname(__file__), "results", "qoblib_qs")
SOLDIR = os.path.join(os.path.dirname(__file__), "..", "competitors", "QOBLIB", "06-portfolio",
                      "solutions", "uqo", "_x_a010", "a010_t10_orig_b004")
BEST = {"0": -231106, "0.00001": -204799, "0.001": -30169}


def load_qs(path):
    with open(path) as f:
        lines = [ln for ln in f if not ln.startswith("#") and ln.strip()]
    n, nnz = map(int, lines[0].split())
    Q = np.zeros((n, n))
    for ln in lines[1:]:
        a, b, v = ln.split(); a = int(a) - 1; b = int(b) - 1; v = float(v)
        if a == b: Q[a, a] += v
        else: Q[a, b] += v / 2; Q[b, a] += v / 2
    return QUBO(0.5 * (Q + Q.T)), n


def load_sol(qtag):
    f = os.path.join(SOLDIR, f"uqo_a010_t10_q{qtag}_b004.sol.mst")
    vals = {}
    for ln in open(f):
        p = ln.split()
        if len(p) >= 2 and p[0].startswith("x#"): vals[int(p[0][2:]) - 1] = int(round(float(p[1])))
    return np.array([vals[k] for k in range(len(vals))], dtype=np.int8)


def main():
    h = GNNHypers(model="qrf", epochs=2500, hidden=128, dim_embedding=20, n_layers=3, lr=1e-3, anneal_rate=0.0,
                  eval_every=50, patience=500, ls_passes=200, n_round_samples=24, refine_sa=True, refine_reads=40)
    out = {}
    for qtag in ["0", "0.00001", "0.001"]:
        qspath = os.path.join(QSDIR, f"a010_q{qtag}.qs")
        if not os.path.exists(qspath):
            print(f"  q={qtag}: .qs missing (generate via WSL)", flush=True); continue
        q, n = load_qs(qspath)
        xs = load_sol(qtag); bk = q.energy(xs[:n])  # reference = their best-known solution's .qs energy (same units)
        print(f"  [q={qtag}] n={n}; reference (THEIR best-known sol .qs energy) = {bk:.0f}", flush=True)
        # run our solvers
        t0 = time.time(); eg = q.energy(solve_qubo_gnn(q, h, device="cuda", seed=0)["x"]); tg = time.time() - t0
        x, _ = sbif.minimize(torch.tensor(q.Q.astype(np.float64)), domain="binary", agents=128, max_steps=30000, best_only=True); es = q.energy(np.asarray(x.cpu()).reshape(-1))
        et = q.energy(tabu_qubo(q, num_reads=100, seed=0)["x"])
        gap = lambda e: (e - bk) / abs(bk) * 100
        out[qtag] = {"bk": bk, "GNN": [eg, gap(eg), tg], "SB": [es, gap(es)], "tabu": [et, gap(et)]}
        print(f"    -> GNN {eg:.0f}({gap(eg):+.1f}%) {tg:.0f}s | SB {es:.0f}({gap(es):+.1f}%) | tabu {et:.0f}({gap(et):+.1f}%)", flush=True)
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "results", "e70_qoblib_real.json"), "w"), indent=2)
    print("saved", flush=True)


if __name__ == "__main__":
    main()
