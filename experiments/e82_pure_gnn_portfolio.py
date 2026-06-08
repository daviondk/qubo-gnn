"""E82 (find where PURE GNN solves a portfolio problem, like MaxCut). Clean cardinality-constrained
min-variance QUBO from OR-Library (one constraint, well-conditioned coeffs). PURE GNN (original method:
annealing, multi-threshold round + 1-flip; NO tabu/SA crutch) vs strong reference. Run in .venv."""
import sys, os, numpy as np
sys.path.insert(0, "src")
from qubo import QUBO
from gnn_solver import solve_qubo_gnn, GNNHypers
from baselines import tabu_qubo, sa_qubo


def load_port(path):
    toks = open(path).read().split()
    idx = 0; n = int(toks[idx]); idx += 1
    mean = np.zeros(n); std = np.zeros(n)
    for i in range(n):
        mean[i] = float(toks[idx]); std[i] = float(toks[idx + 1]); idx += 2
    C = np.eye(n)
    while idx + 2 < len(toks):
        i = int(toks[idx]) - 1; j = int(toks[idx + 1]) - 1; c = float(toks[idx + 2]); idx += 3
        C[i, j] = c; C[j, i] = c
    Sigma = C * np.outer(std, std)
    return n, mean, Sigma


def cardinality_minvar_qubo(Sigma, K, P):
    n = Sigma.shape[0]
    Q = Sigma.copy()
    # + P*(sum x - K)^2 = P*(x'11'x - 2K 1'x + K^2); diag gets P*(1-2K), offdiag P
    Q += P * np.ones((n, n))
    np.fill_diagonal(Q, np.diag(Q) + P * (1 - 2 * K) - P)  # adjust: diag had +P from ones; want P(1-2K)
    return QUBO(0.5 * (Q + Q.T)), P


def card(x, K): return int(x.sum())


def main():
    for pf, K in [("port1", 10), ("port2", 10)]:
        n, mean, Sigma = load_port(f"competitors/orlib_portfolio/{pf}.txt")
        P = 5 * np.abs(Sigma).mean() * n  # penalty ~ objective scale (well-conditioned)
        q, _ = cardinality_minvar_qubo(Sigma, K, P)
        # reference: strong tabu + SA (many reads)
        xt = np.asarray(tabu_qubo(q, num_reads=2000, seed=0)["x"]); xs = np.asarray(sa_qubo(q, num_reads=2000, seed=0)["x"])
        ref_x = xt if q.energy(xt) < q.energy(xs) else xs
        ref_var = float(ref_x @ Sigma @ ref_x); ref_k = card(ref_x, K)
        # PURE GNN (original method: annealing, NO refine_sa)
        h = GNNHypers(model="qrf", epochs=5000, hidden=128, dim_embedding=20, n_layers=3, lr=1e-3,
                      anneal_rate=1e-4, eval_every=200, patience=5000, ls_passes=200, n_round_samples=40, refine_sa=False)
        best = None
        for s in range(5):
            x = np.asarray(solve_qubo_gnn(q, h, device="cuda", seed=s)["x"])
            if card(x, K) == K and (best is None or x @ Sigma @ x < best[0]): best = (float(x @ Sigma @ x), card(x, K))
        gnn_var, gnn_k = best if best else (None, None)
        print(f"[{pf} n={n} K={K}] PURE-GNN var={gnn_var} (card {gnn_k}=={K}?) | ref(tabu/SA) var={ref_var:.6f} (card {ref_k})", flush=True)
        if gnn_var is not None:
            print(f"   -> PURE-GNN feasible & {'MATCHES' if abs(gnn_var-ref_var)/ref_var<0.02 else f'gap {(gnn_var-ref_var)/ref_var*100:+.1f}%'} reference", flush=True)
        else:
            print(f"   -> PURE-GNN found NO feasible (card={K}) solution in 5 restarts", flush=True)


if __name__ == "__main__":
    main()
