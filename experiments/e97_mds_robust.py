"""E97: RIGOROUS MDS vs DiffUCO 106.61 -- 50 BA-large graphs from DiffUCO's EXACT distribution
(n=randint(401)+800, barabasi_albert_graph(n,4)), best-of-3 samples/graph. Robust mean +- SE. Run in .venv."""
import sys, time, numpy as np, networkx as nx
sys.path.insert(0, "experiments"); sys.path.insert(0, "src")
from e89_mds_vs_diffuco import solve_mds
print("=== RIGOROUS MDS BA-large (50 graphs, DiffUCO exact dist, best-of-3) vs DiffUCO 106.61 ===", flush=True)
sizes = []
for g in range(50):
    rng = np.random.default_rng(9000 + g); n = int(rng.integers(0, 401)) + 800  # == DiffUCO randint(401)+800
    G = nx.barabasi_albert_graph(n, 4, seed=9000 + g)
    s = min(solve_mds(G, epochs=2000, beta=2.0, seed=sd) for sd in range(3)); sizes.append(s)
    if g % 5 == 4: print(f"  [{g+1}/50] running mean={np.mean(sizes):.2f} +- {np.std(sizes)/np.sqrt(len(sizes)):.2f}", flush=True)
m, se = np.mean(sizes), np.std(sizes) / np.sqrt(len(sizes))
print(f"=> MDS BA-large = {m:.2f} +- {se:.2f} (SE, n=50, best-of-3) | DiffUCO 106.61, EGN 116.76, Gurobi 103.80", flush=True)
print(f"   verdict: {'BEAT' if m < 106.61 - se else 'TIE' if abs(m-106.61) < 2 else 'behind'} DiffUCO", flush=True)
print("done", flush=True)
