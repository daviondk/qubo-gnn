"""E85 (PURE GNN on genuinely-hard finance QUBO NOT in QIGNN): long/short portfolio Ising ground state.
s_i in {+1 long, -1 short}; minimize risk - return = s'Sigma s - lambda*mu's. Mixed-sign s_i s_j ->
FRUSTRATED spin glass (genuinely hard, unlike PSD long-only), native QUBO, approximate metric (energy),
finance (long/short), reproducible (OR-Library). Simple GNN (original style) vs tabu/SA. Run in .venv."""
import sys, numpy as np
sys.path.insert(0, "src"); sys.path.insert(0, "experiments")
from e82_pure_gnn_portfolio import load_port
from qubo import QUBO
from gnn_solver import solve_qubo_gnn, GNNHypers
from baselines import tabu_qubo, sa_qubo


def longshort_qubo(mu, Sigma, lam):
    # s in {+1,-1}, s = 2x-1. minimize s'Sigma s - lam mu's
    n = len(mu)
    # s'Sigma s = (2x-1)'Sigma(2x-1) = 4 x'Sigma x -4 (Sigma1)'x + 1'Sigma1
    # -lam mu's = -lam mu'(2x-1) = -2lam mu'x + lam mu'1
    Q = 4.0 * Sigma.copy()
    lin = -4.0 * (Sigma @ np.ones(n)) - 2.0 * lam * mu
    Qd = Q.copy(); np.fill_diagonal(Qd, np.diag(Qd) + lin)
    const = float(np.ones(n) @ Sigma @ np.ones(n) + lam * mu.sum())
    return QUBO(0.5 * (Qd + Qd.T)), const


def energy_s(mu, Sigma, x, lam):
    s = 2 * x - 1
    return float(s @ Sigma @ s - lam * mu @ s)


def main():
    for pf in ["port3", "port5"]:
        n, mu, Sigma = load_port(f"competitors/orlib_portfolio/{pf}.txt")
        lam = 0.5 * np.abs(Sigma).mean() / (np.abs(mu).mean() + 1e-12)  # balance risk/return scale
        q, const = longshort_qubo(mu, Sigma, lam)
        # references
        et = energy_s(mu, Sigma, np.asarray(tabu_qubo(q, num_reads=4000, seed=0)["x"]), lam)
        es = energy_s(mu, Sigma, np.asarray(sa_qubo(q, num_reads=4000, seed=0)["x"]), lam)
        ref = min(et, es)
        # SIMPLE GNN (original style: light, annealing, NO refine_sa, modest restarts)
        h = GNNHypers(model="qrf", epochs=5000, hidden=128, dim_embedding=20, n_layers=3, lr=1e-3,
                      anneal_rate=1e-4, eval_every=200, patience=5000, ls_passes=200, n_round_samples=30, refine_sa=False)
        eg = min(energy_s(mu, Sigma, np.asarray(solve_qubo_gnn(q, h, device="cuda", seed=s)["x"]), lam) for s in range(5))
        # hardness check: spread between tabu and SA (if they disagree, it's genuinely hard)
        gap = (eg - ref) / abs(ref) * 100
        print(f"[{pf} n={n}] SIMPLE-GNN E={eg:.2f} | tabu E={et:.2f} SA E={es:.2f} -> GNN gap vs best {gap:+.2f}% | tabu-SA spread {abs(et-es)/abs(ref)*100:.1f}% ({'HARD' if abs(et-es)>1e-6 else 'easy?'})", flush=True)


if __name__ == "__main__":
    main()
