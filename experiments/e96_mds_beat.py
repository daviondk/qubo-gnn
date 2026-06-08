"""E96: try to BEAT DiffUCO on MDS BA-large via best-of-3 restarts (tie 106.87 -> beat 106.61?).
Lean: 15 BA-large graphs, repair only at eval, per-graph progress. Run in .venv."""
import sys, time, numpy as np, networkx as nx
sys.path.insert(0, "experiments"); sys.path.insert(0, "src")
from e89_mds_vs_diffuco import solve_mds
print("=== MDS BA-large best-of-3 vs DiffUCO 106.61 / Gurobi 103.80 ===", flush=True)
sizes = []
for g in range(15):
    rng = np.random.default_rng(5000 + g); n = int(rng.integers(800, 1201))
    G = nx.barabasi_albert_graph(n, 4, seed=5000 + g)
    t0 = time.time(); s = min(solve_mds(G, epochs=2500, beta=2.0, seed=sd) for sd in range(3)); dt = time.time() - t0
    sizes.append(s); print(f"  g{g} (n={n}): {s} ({dt:.0f}s) running-mean={np.mean(sizes):.2f}", flush=True)
print(f"=> MDS BA-large best-of-3 mean = {np.mean(sizes):.2f} +- {np.std(sizes):.2f} | DiffUCO 106.61, single-run was 106.87", flush=True)
print("done", flush=True)
