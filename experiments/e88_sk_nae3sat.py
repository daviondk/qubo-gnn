"""E88 (STANDARD QUBO-solver benchmarks NOT in QIGNN, literature-aligned): SK spin glass (dim 128, 10 seeds,
as QIS3 2506.04596) + NAE-3SAT (critical ratio 2.11). Simple GNN (original style) vs tabu/SA best-found.
Metrics: mean energy + %-optimality (SK), %clauses-satisfied (NAE-3SAT). Run in .venv."""
import sys, numpy as np
sys.path.insert(0, "src")
from qubo import QUBO
from gnn_solver import solve_qubo_gnn, GNNHypers
from baselines import tabu_qubo, sa_qubo
H = GNNHypers(model="qrf", epochs=4000, hidden=128, dim_embedding=20, n_layers=3, lr=1e-3, anneal_rate=1e-4,
              eval_every=200, patience=4000, ls_passes=200, n_round_samples=30, refine_sa=False)


def sk_qubo(n, seed):
    rng = np.random.default_rng(seed)
    J = rng.standard_normal((n, n)); J = np.triu(J, 1); J = J + J.T
    Q = 4.0 * J; lin = -4.0 * (J @ np.ones(n))
    Qd = Q.copy(); np.fill_diagonal(Qd, np.diag(Qd) + lin)
    return QUBO(0.5 * (Qd + Qd.T)), J


def sk_energy(J, x): s = 2 * x - 1; return float(s @ J @ s)


def nae3sat_qubo(nv, seed, ratio=2.11):
    rng = np.random.default_rng(seed); m = int(ratio * nv); clauses = []
    Q = np.zeros((nv, nv))
    for _ in range(m):
        idx = rng.choice(nv, 3, replace=False); sgn = rng.choice([-1, 1], 3)
        clauses.append((idx, sgn))
        # E_clause = sum over pairs of (sgn_a s_a)(sgn_b s_b); s=2x-1. accumulate spin coupling J then convert
        for p in range(3):
            for r in range(p + 1, 3):
                a, b = idx[p], idx[r]; w = sgn[p] * sgn[r]
                Q[a, b] += w; Q[b, a] += w
    # convert spin J (Q here in s-space, pairwise) to x-space: s=2x-1
    Js = 0.5 * Q  # since we added both (a,b) and (b,a), Js symmetric with J_ab = w per clause-pair
    Qx = 4.0 * Js; lin = -4.0 * (Js @ np.ones(nv))
    Qd = Qx.copy(); np.fill_diagonal(Qd, np.diag(Qd) + lin)
    return QUBO(0.5 * (Qd + Qd.T)), clauses, m


def nae_sat_frac(clauses, x):
    s = 2 * x - 1; ok = 0
    for idx, sgn in clauses:
        vals = [sgn[i] * s[idx[i]] for i in range(3)]
        if not (vals[0] == vals[1] == vals[2]): ok += 1
    return ok / len(clauses)


def main():
    print("=== SK spin glass (dim 128, 10 seeds) -- standard QUBO-solver benchmark ===", flush=True)
    gnn_gaps = []
    for seed in range(10):
        q, J = sk_qubo(128, seed)
        et = sk_energy(J, np.asarray(tabu_qubo(q, num_reads=3000, seed=0)["x"]))
        es = sk_energy(J, np.asarray(sa_qubo(q, num_reads=3000, seed=0)["x"]))
        ref = min(et, es)
        eg = min(sk_energy(J, np.asarray(solve_qubo_gnn(q, H, device="cuda", seed=s)["x"])) for s in range(3))
        gap = (eg - ref) / abs(ref) * 100; gnn_gaps.append(gap)
        print(f"  seed{seed}: GNN={eg:.1f} best(tabu/SA)={ref:.1f} gap={gap:+.2f}%", flush=True)
    print(f"  => SK mean GNN gap vs best-found: {np.mean(gnn_gaps):+.2f}% (n=128, 10 seeds)", flush=True)
    print("=== NAE-3SAT (ratio 2.11) -- standard QUBO-solver benchmark ===", flush=True)
    for nv in [100, 200]:
        q, clauses, m = nae3sat_qubo(nv, 0)
        ft = nae_sat_frac(clauses, np.asarray(tabu_qubo(q, num_reads=3000, seed=0)["x"]))
        fg = max(nae_sat_frac(clauses, np.asarray(solve_qubo_gnn(q, H, device="cuda", seed=s)["x"])) for s in range(3))
        print(f"  nv={nv} m={m}: GNN {fg*100:.2f}% clauses NAE-sat | tabu {ft*100:.2f}%", flush=True)


if __name__ == "__main__":
    main()
