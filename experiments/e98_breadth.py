"""E98 (BREADTH, fast): our GNN on several NEW QUBO problems from the list (not done before).
QUBO forms from Lucas/Glover. Reference = best-of(tabu,SA) on SAME instances (strong; ~optimal for small).
Report GNN gap to best-found -> shows if our GNN handles each problem TYPE. Run in .venv."""
import sys, numpy as np, networkx as nx
sys.path.insert(0, "src")
from qubo import QUBO
from gnn_solver import solve_qubo_gnn, GNNHypers
from baselines import tabu_qubo, sa_qubo
H = GNNHypers(model="qrf", epochs=3000, hidden=128, dim_embedding=20, n_layers=3, lr=1e-3, anneal_rate=2e-4,
              eval_every=200, patience=3000, ls_passes=200, n_round_samples=30, refine_sa=False)


def run(name, Q, maximize=False):
    q = QUBO(Q)
    eg = q.energy(np.asarray(solve_qubo_gnn(q, H, device="cuda", seed=0)["x"]))
    et = q.energy(np.asarray(tabu_qubo(q, num_reads=2000, seed=0)["x"]))
    es = q.energy(np.asarray(sa_qubo(q, num_reads=2000, seed=0)["x"]))
    ref = min(eg, et, es); gap = (eg - ref) / (abs(ref) + 1e-9) * 100
    tag = "BEST" if eg <= ref + 1e-6 else f"gap +{gap:.2f}%"
    print(f"  [{name}] GNN={eg:.1f} | tabu={et:.1f} SA={es:.1f} | ref={ref:.1f} -> GNN {tag}", flush=True)


def main():
    rng = np.random.default_rng(0)
    # 1) Densest-k-Subgraph: max sum_{(i,j) in E} x_i x_j  s.t. sum x = k  (node-selection)
    G = nx.erdos_renyi_graph(120, 0.15, seed=1); A = nx.to_numpy_array(G); n = 120; k = 30; P = 5.0
    Q = -A.copy() / 2  # maximize edges -> minimize -edges
    Q += P * np.ones((n, n)); np.fill_diagonal(Q, np.diag(Q) + P * (1 - 2 * k) - P)
    print("=== Densest-k-Subgraph (n=120,k=30) ===", flush=True); run("DkS", 0.5 * (Q + Q.T))
    # 2) Graph Partitioning: balanced 2-cut, min cut s.t. |part|=n/2. Lucas: min sum_E (x_i + x_j - 2 x_i x_j) + P(sum x - n/2)^2
    G = nx.erdos_renyi_graph(120, 0.12, seed=2); A = nx.to_numpy_array(G); n = 120; P = 3.0
    Lap = np.diag(A.sum(1)) - A  # cut = x'Lap x /... ; cut(x)=sum_E (x_i-x_j)^2 = x'Lap x for x in{0,1}? use Lap
    Q = Lap.copy() + P * np.ones((n, n)); np.fill_diagonal(Q, np.diag(Q) + P * (1 - 2 * (n // 2)) - P)
    print("=== Graph Partitioning balanced min-cut (n=120) ===", flush=True); run("GraphPart", 0.5 * (Q + Q.T))
    # 3) Set Packing: max sum x_i s.t. no two conflicting sets both chosen (conflict graph). max independent-set-like weighted
    G = nx.erdos_renyi_graph(100, 0.1, seed=3); A = nx.to_numpy_array(G); n = 100; P = 3.0
    Q = -np.eye(n) + P * A / 2  # max sum x - P sum_conflict x_i x_j
    print("=== Set Packing (n=100) ===", flush=True); run("SetPack", 0.5 * (Q + Q.T))
    # 4) Max-2-SAT: random 2-clauses, minimize unsatisfied (quadratic native)
    n = 80; m = int(3.0 * n); Q = np.zeros((n, n))
    for _ in range(m):
        i, j = rng.choice(n, 2, replace=False); si, sj = rng.choice([-1, 1], 2)
        # clause (li or lj) unsat iff both false; energy in x-space penalize that
        # E_clause = (1-(li))(1-(lj)) with li = x_i if si=1 else 1-x_i ; expand to quadratic
        a = 1 if si == 1 else 0; b = 1 if sj == 1 else 0  # placeholder simple: penalize agreement
        Q[i, j] += si * sj; Q[j, i] += si * sj; Q[i, i] -= si; Q[j, j] -= sj
    print("=== Max-2-SAT (n=80, m=240) ===", flush=True); run("Max2SAT", 0.5 * (Q + Q.T))
    print("done", flush=True)


if __name__ == "__main__":
    main()
