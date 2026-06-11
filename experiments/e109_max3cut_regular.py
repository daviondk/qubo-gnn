"""E109: Max-3-Cut on random regular graphs vs ROS Table 2 (k=3, N=100). ROS test = 60 graphs N=100
(20 each of 3/5/7-regular per their setup). SAME metric (avg cut value, higher=better). Reproducible.
Reuses e108 solver. Run in .venv."""
import sys, numpy as np, networkx as nx
sys.path.insert(0, "experiments")
from e108_max3cut_ros import solve, cut_value, node_move_ls

def reg_graph(d, n, seed):
    G = nx.random_regular_graph(d, n, seed=seed)
    ei = np.array(G.edges(), dtype=np.int64); w = np.ones(len(ei))
    return n, ei, w

def main():
    # ROS Table 2, k=3, N=100: MD 235.50, Genetic 235.50, BQP 239.70, ANYCSP 247.90(best), ROS 240.30
    print("=== Max-3-Cut random regular N=100 vs ROS Table 2 (k=3, cut value higher=better) ===", flush=True)
    allres = []
    for d in [3, 5, 7]:
        res = []
        for s in range(20):
            n, ei, w = reg_graph(d, 100, 1000 + s)
            res.append(solve(n, ei, w, restarts=6, epochs=1500))
        m = float(np.mean(res)); allres += res
        print(f"  {d}-regular: OURS avg cut = {m:.2f} ({len(res)} graphs)", flush=True)
    M = float(np.mean(allres))
    v = "BEAT ANYCSP(best)" if M > 247.90 else ("beat ROS" if M > 240.30 else ("~ROS" if M > 239 else "behind"))
    print(f"OURS overall avg = {M:.2f} | ROS 240.30, ANYCSP 247.90, BQP 239.70, MD/Gen 235.50 -> {v}", flush=True)
    print("done", flush=True)

if __name__ == "__main__":
    main()
