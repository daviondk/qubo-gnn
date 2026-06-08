"""E83 (PURE GNN on a genuinely-hard, finance-adjacent, reproducible benchmark): QOBLIB Market Split
(Cornuejols-Dawande hard 0-1 programs; multi-dim subset-sum = cash-flow matching/basket replication).
QUBO min||Ax-b||^2 is UNCONSTRAINED (like MaxCut), optimum=0 (feasibility). Test PURE GNN (annealing, NO
tabu/SA crutch) -- does it reach 0? Compare vs strong tabu. Run in .venv."""
import sys, os, glob, numpy as np
sys.path.insert(0, "src")
from qubo import QUBO
from gnn_solver import solve_qubo_gnn, GNNHypers
from baselines import tabu_qubo

MSDIR = "competitors/QOBLIB/01-marketsplit/instances"


def load_ms(path):
    toks = open(path).read().split()
    m = int(toks[0]); n = int(toks[1]); idx = 2
    A = np.zeros((m, n)); b = np.zeros(m)
    for i in range(m):
        for j in range(n): A[i, j] = float(toks[idx]); idx += 1
        b[i] = float(toks[idx]); idx += 1
    return A, b, m, n


def ms_qubo(A, b):
    # min ||Ax-b||^2 = x'(A'A)x - 2 b'A x + b'b ; x binary so x_j^2=x_j
    n = A.shape[1]; Q = A.T @ A
    lin = -2.0 * (A.T @ b)
    Qd = Q.copy(); np.fill_diagonal(Qd, np.diag(Qd) + lin)
    return QUBO(0.5 * (Qd + Qd.T)), float(b @ b)


def main():
    insts = []
    for m in [3, 4, 5, 6, 7, 8]:
        g = sorted(glob.glob(f"{MSDIR}/ms_{m:02d}_050_*.dat"))
        if g: insts.append(g[0])
    h = GNNHypers(model="qrf", epochs=6000, hidden=128, dim_embedding=20, n_layers=3, lr=1e-3,
                  anneal_rate=1e-4, eval_every=200, patience=6000, ls_passes=300, n_round_samples=50, refine_sa=False)
    for path in insts:
        A, b, m, n = load_ms(path); q, const = ms_qubo(A, b)
        # PURE GNN: best over restarts; energy(x)+const = ||Ax-b||^2
        gnn_best = min(q.energy(np.asarray(solve_qubo_gnn(q, h, device="cuda", seed=s)["x"])) + const for s in range(5))
        tb = q.energy(np.asarray(tabu_qubo(q, num_reads=3000, seed=0)["x"])) + const
        name = os.path.basename(path)
        print(f"[{name} m={m} n={n}] PURE-GNN ||Ax-b||^2={gnn_best:.0f} ({'SOLVED' if gnn_best<0.5 else 'not solved'}) | tabu={tb:.0f} ({'SOLVED' if tb<0.5 else 'no'}) (opt=0)", flush=True)


if __name__ == "__main__":
    main()
