"""E87 (PURE GNN on established hard QUBO NOT in QIGNN): Sherrington-Kirkpatrick spin glass. Canonical
genuinely-hard frustrated Ising, heavily used in ML-for-CO benchmarking (Schuetz/PI-GNN, DiffUCO, etc.) ->
directly comparable to other learned solvers. min s'Js, J_ij ~ N(0,1), s in {+1,-1}. Native quadratic,
simple GNN (original style). Metric: energy/N vs strong baseline (genuinely hard => baselines disagree).
Reproducible (seeded). Run in .venv."""
import sys, numpy as np
sys.path.insert(0, "src")
from qubo import QUBO
from gnn_solver import solve_qubo_gnn, GNNHypers
from baselines import tabu_qubo, sa_qubo


def sk_qubo(n, seed):
    rng = np.random.default_rng(seed)
    J = rng.standard_normal((n, n)); J = np.triu(J, 1); J = J + J.T
    J = J / np.sqrt(n)  # SK normalization
    # min s'Js, s=2x-1 -> energy in s-space; build QUBO in x
    # s'Js = (2x-1)'J(2x-1) = 4x'Jx -4(J1)'x + 1'J1
    Q = 4.0 * J; lin = -4.0 * (J @ np.ones(n))
    Qd = Q.copy(); np.fill_diagonal(Qd, np.diag(Qd) + lin)
    return QUBO(0.5 * (Qd + Qd.T)), J


def e_per_spin(J, x):
    s = 2 * x - 1; return float(s @ J @ s) / len(s)


def main():
    for n in [100, 200, 400]:
        q, J = sk_qubo(n, seed=1)
        et = e_per_spin(J, np.asarray(tabu_qubo(q, num_reads=4000, seed=0)["x"]))
        es = e_per_spin(J, np.asarray(sa_qubo(q, num_reads=4000, seed=0)["x"]))
        ref = min(et, es)
        h = GNNHypers(model="qrf", epochs=6000, hidden=128, dim_embedding=20, n_layers=3, lr=1e-3,
                      anneal_rate=1e-4, eval_every=200, patience=6000, ls_passes=200, n_round_samples=30, refine_sa=False)
        eg = min(e_per_spin(J, np.asarray(solve_qubo_gnn(q, h, device="cuda", seed=s)["x"])) for s in range(5))
        gap = (eg - ref) / abs(ref) * 100
        print(f"[SK n={n}] SIMPLE-GNN E/N={eg:.4f} | tabu={et:.4f} SA={es:.4f} (Parisi~-0.763) -> GNN gap {gap:+.2f}% | tabu-SA differ {abs(et-es)>1e-4} ({'HARD' if abs(et-es)>1e-4 else 'easy'})", flush=True)


if __name__ == "__main__":
    main()
