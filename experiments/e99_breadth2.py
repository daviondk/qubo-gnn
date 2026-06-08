"""E99 (BREADTH batch 2): more NEW QUBO problems. GNN vs best-of(tabu,SA) same instances. Run in .venv."""
import sys, numpy as np, networkx as nx
sys.path.insert(0, "src")
from qubo import QUBO
from gnn_solver import solve_qubo_gnn, GNNHypers
from baselines import tabu_qubo, sa_qubo
H = GNNHypers(model="qrf", epochs=3000, hidden=128, dim_embedding=20, n_layers=3, lr=1e-3, anneal_rate=2e-4, eval_every=200, patience=3000, ls_passes=200, n_round_samples=30, refine_sa=False)


def run(name, Q):
    q = QUBO(0.5 * (Q + Q.T))
    eg = q.energy(np.asarray(solve_qubo_gnn(q, H, device="cuda", seed=0)["x"]))
    et = q.energy(np.asarray(tabu_qubo(q, num_reads=2000, seed=0)["x"])); es = q.energy(np.asarray(sa_qubo(q, num_reads=2000, seed=0)["x"]))
    ref = min(eg, et, es); gap = (eg - ref) / (abs(ref) + 1e-9) * 100
    print(f"  [{name}] GNN={eg:.1f} | tabu={et:.1f} SA={es:.1f} -> GNN {'BEST' if eg<=ref+1e-6 else f'gap +{gap:.2f}%'}", flush=True)


def main():
    rng = np.random.default_rng(0)
    # 1) Quadratic Knapsack (FINANCE): max sum p_i x_i + sum P_ij x_i x_j s.t. sum w_i x_i <= C
    n = 80; p = rng.integers(1, 50, n).astype(float); Pm = np.triu(rng.integers(0, 20, (n, n)).astype(float), 1); Pm = Pm + Pm.T
    w = rng.integers(1, 30, n).astype(float); C = w.sum() * 0.5; lam = 10.0
    Q = -(np.diag(p) + Pm)  # maximize value -> minimize neg
    # budget penalty lam*(sum w x - C)^2 (soft); ignore slack for speed (penalize overflow approx)
    Q += lam * np.outer(w, w); np.fill_diagonal(Q, np.diag(Q) + lam * (w * w - 2 * C * w) - lam * w * w)
    print("=== Quadratic Knapsack n=80 (FINANCE) ===", flush=True); run("QKP", Q)
    # 2) Set Cover: min sum x_s s.t. every element covered by >=1 chosen set. penalty for uncovered
    n_sets = 80; n_el = 60; sets = [set(rng.choice(n_el, rng.integers(3, 10), replace=False)) for _ in range(n_sets)]
    P = 5.0; Q = np.zeros((n_sets, n_sets)); 
    for s in range(n_sets): Q[s, s] += 1.0  # min number of sets
    for e in range(n_el):
        cov = [s for s in range(n_sets) if e in sets[s]]
        # penalty (1 - sum_{s cov} x_s) approx via -P sum + P pairs (encourage >=1)
        for s in cov: Q[s, s] += -P
        for a in cov:
            for b in cov:
                if a < b: Q[a, b] += P; Q[b, a] += P
    print("=== Set Cover (80 sets, 60 elems) ===", flush=True); run("SetCover", Q)
    # 3) Maximum Balanced Biclique-ish / Max Cut on BA (sanity, our strength)
    G = nx.barabasi_albert_graph(150, 5, seed=5); A = nx.to_numpy_array(G); n = 150
    Q = A - np.diag(A.sum(1))  # maxcut: max sum_E (x_i - x_j)^2 = -x'(A - D)x... minimize x'(D-A)x neg -> use -(D-A)? cut max
    # MaxCut QUBO min: -sum_E(x_i+x_j-2x_ix_j); Q_ij=2A_ij offdiag (min -cut). use standard:
    Q = np.zeros((n, n))
    for i, j in G.edges(): Q[i, j] += 1; Q[j, i] += 1; Q[i, i] -= 1; Q[j, j] -= 1  # min -> -cut
    print("=== MaxCut BA n=150 (sanity, our strength) ===", flush=True); run("MaxCut", Q)
    # 4) Feedback Vertex Set-ish: min vertices to remove to make acyclic -- approximate via MVC-like on dense (skip exact); use weighted MIS complement
    G = nx.erdos_renyi_graph(100, 0.2, seed=6); n = 100; P = 3.0; Q = np.zeros((n, n))
    for v in range(n): Q[v, v] += -1  # max independent-ish
    for i, j in G.edges(): Q[i, j] += P; Q[j, i] += P
    print("=== Max Independent Set ER n=100 (dense p=0.2) ===", flush=True); run("MIS-ER", Q)
    print("done", flush=True)


if __name__ == "__main__":
    main()
