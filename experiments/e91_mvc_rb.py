"""E91 (SOTA comparison NOT-in-QIGNN core): Minimum Vertex Cover on RB-200 graphs (DiffUCO benchmark vs
Sanokowski 2023). Metric = approximation ratio AR = size/optimum (DiffUCO ~1.003). RB-200 config from DiffUCO.
MVC opt = V - n_parts (RB MIS = n). Pure GNN: min sum(p)+P*sum_edges(1-p_i)(1-p_j), round + greedy repair.
Run in .venv."""
import sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, networkx as nx
sys.path.insert(0, "src"); sys.path.insert(0, "experiments")
from e90_maxclique_rb import rb_edges, GNN
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def repair_cover(G, cover):
    cover = set(cover)
    for u, v in G.edges():
        if u not in cover and v not in cover:
            cover.add(u if G.degree(u) >= G.degree(v) else v)
    return cover


def solve_mvc(G, epochs=2500, P=2.0, seed=0):
    torch.manual_seed(seed); n = G.number_of_nodes()
    deg = np.array([d for _, d in sorted(G.degree())]).reshape(-1, 1)
    sf = torch.tensor(np.column_stack([(deg - deg.mean()) / (deg.std() + 1e-9), np.ones((n, 1))]), dtype=torch.float32, device=DEV)
    E = np.array(G.edges()).T; ei = torch.tensor(E, dtype=torch.long, device=DEV); ei2 = torch.cat([ei, ei.flip(0)], 1)
    net = GNN().to(DEV); opt = torch.optim.Adam(net.parameters(), lr=1e-3); hd = torch.zeros((n, net.hidden), device=DEV); best = n
    eidx = torch.tensor(E, dtype=torch.long, device=DEV)
    for ep in range(epochs):
        net.train(); p, hnew = net(sf, ei2, hd)
        # min sum p + P sum_edges (1-p_i)(1-p_j)
        loss = p.sum() + P * ((1 - p[eidx[0]]) * (1 - p[eidx[1]])).sum()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step(); hd = hnew.detach()
        if ep % 200 == 0:
            pv = p.detach().cpu().numpy(); cover = [v for v in range(n) if pv[v] > 0.5]
            c = repair_cover(G, cover); best = min(best, len(c))
    return best


def main():
    print("=== MVC on RB-200 vs DiffUCO (AR~1.003) / Sanokowski ===", flush=True)
    ars = []
    for g in range(20):
        rng = np.random.default_rng(3000 + g); n = int(rng.integers(20, 25)); k = int(rng.integers(9, 10)); p = float(rng.uniform(0.25, 1))
        v, edges = rb_edges(n, k, p, 3000 + g)
        G = nx.Graph(); G.add_nodes_from(range(v)); G.add_edges_from(edges)
        opt_mvc = v - n  # RB: MIS = n_parts
        size = solve_mvc(G, seed=0); ar = size / opt_mvc; ars.append(ar)
    print(f"  OUR-GNN MVC RB-200 mean AR = {np.mean(ars):.4f} (20 graphs) | DiffUCO ~1.003, optimum AR=1.000", flush=True)


if __name__ == "__main__":
    main()
