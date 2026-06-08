"""E86 (PURE GNN on NEW hard finance QUBO NOT in QIGNN): Number Partitioning. Finance = fair division of
assets/cash-flows into two equal-value groups. NP-hard (Karp-21), native QUBO min(sum a_i s_i)^2, s in {+1,-1}.
Approximate = minimize discrepancy (standard). Simple GNN (original style) vs tabu/SA. Reproducible (seeded).
Run in .venv."""
import sys, numpy as np
sys.path.insert(0, "src")
from qubo import QUBO
from gnn_solver import solve_qubo_gnn, GNNHypers
from baselines import tabu_qubo, sa_qubo


def npp_qubo(a):
    n = len(a)
    # min (a's)^2, s=2x-1 -> (a'(2x-1))^2 = (2 a'x - a'1)^2 ; Q = 4 a a', lin = -4 (a'1) a
    A = a.reshape(-1, 1)
    Q = 4.0 * (A @ A.T)
    lin = -4.0 * a.sum() * a
    Qd = Q.copy(); np.fill_diagonal(Qd, np.diag(Qd) + lin)
    return QUBO(0.5 * (Qd + Qd.T)), float(a.sum() ** 2)


def disc(a, x):  # discrepancy |sum a_i s_i|
    return abs(float(a @ (2 * x - 1)))


def main():
    rng = np.random.default_rng(42)
    for n in [100, 300, 500]:
        a = rng.integers(1, 10 ** 6, size=n).astype(np.float64)  # hard-phase: large range
        a = a / a.mean()  # well-conditioned
        q, const = npp_qubo(a)
        dt = disc(a, np.asarray(tabu_qubo(q, num_reads=3000, seed=0)["x"]))
        ds = disc(a, np.asarray(sa_qubo(q, num_reads=3000, seed=0)["x"]))
        ref = min(dt, ds)
        h = GNNHypers(model="qrf", epochs=5000, hidden=128, dim_embedding=20, n_layers=3, lr=1e-3,
                      anneal_rate=1e-4, eval_every=200, patience=5000, ls_passes=200, n_round_samples=30, refine_sa=False)
        dg = min(disc(a, np.asarray(solve_qubo_gnn(q, h, device="cuda", seed=s)["x"])) for s in range(5))
        scale = a.sum()
        print(f"[NPP n={n}] SIMPLE-GNN disc={dg:.4f} | tabu={dt:.4f} SA={ds:.4f} (rel to total {scale:.0f}) -> GNN {'BEST' if dg<=ref+1e-9 else f'{(dg-ref):.4f} above best'} | tabu-SA differ: {abs(dt-ds)>1e-9}", flush=True)


if __name__ == "__main__":
    main()
