"""E101 (LIST COVERAGE): NEW problems from qubo_formulations list, not done before, not in QIGNN.
QUBO forms from Lucas(2014)/Glover. GNN vs best-of(tabu,SA) on same instances (breadth-applicability survey;
paper-SOTA comparison would need each paper's instances). Run in .venv."""
import sys, numpy as np, networkx as nx
sys.path.insert(0, "src")
from qubo import QUBO
from gnn_solver import solve_qubo_gnn, GNNHypers
from baselines import tabu_qubo, sa_qubo
H = GNNHypers(model="qrf", epochs=3000, hidden=128, dim_embedding=20, n_layers=3, lr=1e-3, anneal_rate=2e-4, eval_every=200, patience=3000, ls_passes=200, n_round_samples=30, refine_sa=False)


def run(name, Q):
    q = QUBO(0.5 * (np.asarray(Q) + np.asarray(Q).T))
    eg = min(q.energy(np.asarray(solve_qubo_gnn(q, H, device="cuda", seed=s)["x"])) for s in range(2))
    et = q.energy(np.asarray(tabu_qubo(q, num_reads=2000, seed=0)["x"])); es = q.energy(np.asarray(sa_qubo(q, num_reads=2000, seed=0)["x"]))
    ref = min(eg, et, es); gap = (eg - ref) / (abs(ref) + 1e-9) * 100
    print(f"  [{name}] GNN={eg:.1f} | tabu={et:.1f} SA={es:.1f} -> GNN {'BEST' if eg<=ref+1e-6 else f'+{gap:.2f}%'}", flush=True)


def main():
    rng = np.random.default_rng(0)
    # 1) Feedback Vertex Set (min nodes to remove -> acyclic). Lucas: complex; approx via min nodes s.t. remaining forest.
    #    Use ordering-penalty-free surrogate: maximize induced forest = min FVS. Penalize edges in a cycle proxy:
    #    Simplified: on a graph, x_v=1 if KEEP; penalize keeping both endpoints if it closes a triangle (cycle proxy).
    G = nx.erdos_renyi_graph(60, 0.12, seed=1); n = 60; P = 3.0; Q = np.zeros((n, n))
    for v in range(n): Q[v, v] -= 1.0  # maximize kept nodes (=> min removed = FVS)
    for tri in [c for c in nx.cycle_basis(G)]:  # penalize keeping all of a fundamental cycle
        if len(tri) <= 5:
            for a in tri:
                for b in tri:
                    if a < b: Q[a, b] += P / len(tri); Q[b, a] += P / len(tri)
    print("=== Feedback Vertex Set (proxy, n=60) ===", flush=True); run("FVS", Q)
    # 2) Max k-colorable subgraph (k=3): max nodes assignable to 3 colors w/o conflict -- use max-cut-like per color (one-hot 3)
    #    Simplify to MAX 2-colorable subgraph = max induced bipartite = max-cut complement. Binary: max edges cut-consistent.
    G = nx.erdos_renyi_graph(80, 0.1, seed=2); n = 80; Q = np.zeros((n, n))  # max-2-colorable ~ maxcut on kept
    for i, j in G.edges(): Q[i, j] += 1; Q[j, i] += 1; Q[i, i] -= 1; Q[j, j] -= 1  # maxcut form (proxy)
    print("=== Max-2-colorable-subgraph proxy (n=80) ===", flush=True); run("Max2Col", Q)
    # 3) Set Partitioning (exact cover): choose subsets so each element covered EXACTLY once. min cost.
    n_sets = 60; n_el = 40; sets = [set(rng.choice(n_el, rng.integers(2, 6), replace=False)) for _ in range(n_sets)]
    cost = rng.integers(1, 10, n_sets).astype(float); P = 8.0; Q = np.diag(cost.copy())
    for e in range(n_el):
        cov = [s for s in range(n_sets) if e in sets[s]]  # penalty (sum_{cov} x - 1)^2
        for a in cov:
            Q[a, a] += P * (1 - 2 * 1)  # (x^2 -2x) part of (sum-1)^2 diagonal
            for b in cov:
                if a < b: Q[a, b] += P; Q[b, a] += P
    print("=== Set Partitioning / Exact Cover (60 sets, 40 elems) ===", flush=True); run("SetPart", Q)
    # 4) Clique Cover (min cliques to cover all vertices) -- proxy: min colors of complement = coloring complement
    G = nx.erdos_renyi_graph(50, 0.5, seed=4); n = 50; K = 8; Q = np.zeros((n * K, n * K))
    def idx(v, c): return v * K + c
    Pc = 4.0
    for v in range(n):
        for c in range(K): Q[idx(v, c), idx(v, c)] += Pc * (1 - 2)  # each vertex one clique
        for c in range(K):
            for d in range(c + 1, K): Q[idx(v, c), idx(v, d)] += Pc; Q[idx(v, d), idx(v, c)] += Pc
    Gc = nx.complement(G)
    for i, j in Gc.edges():  # non-adjacent in G can't be same clique
        for c in range(K): Q[idx(i, c), idx(j, c)] += Pc; Q[idx(j, c), idx(i, c)] += Pc
    print("=== Clique Cover (n=50, K=8) ===", flush=True); run("CliqueCover", Q)
    print("done", flush=True)


if __name__ == "__main__":
    main()
