"""E90 (DIRECT SOTA comparison, NOT in QIGNN): Maximum Clique on RB-model graphs (DiffUCO/X2GNN benchmark).
RB generator reimplemented exactly from DiffUCO (Xu instances). MaxClique QUBO/loss: max sum(p) - P*[sum over
NON-edges p_i p_j]. Pure GNN + greedy clique extraction. Compare to DiffUCO 16.30, X2GNN(~1.2% of opt),
Gurobi 19.05 (RB-small). Run in .venv."""
import sys, itertools, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, networkx as nx
sys.path.insert(0, "src")
from torch_geometric.nn import SAGEConv
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def rb_edges(n, k, p, seed):
    rng = np.random.default_rng(seed); random.seed(seed)
    a = np.log(k) / np.log(n); r = -a / np.log(1 - p)
    v = k * n; s = int(p * (n ** (2 * a))); iters = int(r * n * np.log(n) - 1)
    parts = np.arange(v).reshape(n, k)
    nand = []
    for i in parts: nand += list(itertools.combinations(i.tolist(), 2))
    edges = set()
    for _ in range(iters):
        i, j = rng.choice(n, 2, replace=False)
        allp = set(itertools.product(parts[i].tolist(), parts[j].tolist())) - edges
        if allp: edges |= set(random.sample(tuple(allp), k=min(s, len(allp))))
    nand += list(edges)
    return v, [(min(e), max(e)) for e in nand]


class GNN(nn.Module):
    def __init__(self, hidden=128, n_layers=3, dropout=0.1):
        super().__init__(); self.dropout = dropout; self.hidden = hidden
        self.blocks = nn.ModuleList(); self.norms = nn.ModuleList(); self.proj = nn.ModuleList()
        cur = 2 + hidden
        for _ in range(n_layers):
            self.blocks.append(nn.ModuleList([SAGEConv(cur, hidden, "mean"), SAGEConv(cur, hidden, "max")]))
            self.norms.append(nn.LayerNorm(hidden)); self.proj.append(nn.Linear(cur, hidden) if cur != hidden else nn.Identity()); cur = hidden
        self.out = nn.Linear(cur, 1); self.act = nn.LeakyReLU()

    def forward(self, xs, ei, hd):
        x = torch.cat([xs, hd], 1)
        for (cm, cx), norm, proj in zip(self.blocks, self.norms, self.proj):
            h = norm(cm(x, ei) + cx(x, ei)); x = self.act(h + proj(x)); x = F.dropout(x, self.dropout, self.training)
        return torch.sigmoid(self.out(x)).squeeze(-1), x


def extract_clique(G, p):
    order = sorted(G.nodes(), key=lambda v: -p[v]); clique = []
    for v in order:
        if all(G.has_edge(v, u) for u in clique): clique.append(v)
    return clique


def solve_maxclique(G, epochs=2500, P=3.0, seed=0):
    torch.manual_seed(seed); n = G.number_of_nodes()
    deg = np.array([d for _, d in G.degree()]).reshape(-1, 1)
    sf = torch.tensor(np.column_stack([(deg - deg.mean()) / (deg.std() + 1e-9), np.ones((n, 1))]), dtype=torch.float32, device=DEV)
    ei = torch.tensor(np.array(G.edges()).T, dtype=torch.long, device=DEV); ei = torch.cat([ei, ei.flip(0)], 1)
    eidx = torch.tensor(np.array(G.edges()).T, dtype=torch.long, device=DEV)
    net = GNN().to(DEV); opt = torch.optim.Adam(net.parameters(), lr=1e-3); hd = torch.zeros((n, net.hidden), device=DEV); best = 0
    for ep in range(epochs):
        net.train(); p, hnew = net(sf, ei, hd)
        sump = p.sum(); edge_pp = (p[eidx[0]] * p[eidx[1]]).sum()
        nonedge_pp = (sump * sump - (p * p).sum()) / 2 - edge_pp
        loss = -sump + P * nonedge_pp
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step(); hd = hnew.detach()
        if ep % 200 == 0:
            cl = extract_clique(G, p.detach().cpu().numpy()); best = max(best, len(cl))
    return best


def main():
    print("=== Maximum Clique on RB-small graphs vs DiffUCO/X2GNN/Gurobi ===", flush=True)
    sizes = []
    for g in range(20):
        rng = np.random.default_rng(2000 + g); n = int(rng.integers(20, 25)); k = int(rng.integers(5, 12)); p = float(rng.uniform(0.3, 1))
        v, edges = rb_edges(n, k, p, 2000 + g)
        G = nx.Graph(); G.add_nodes_from(range(v)); G.add_edges_from(edges)
        sizes.append(solve_maxclique(G, seed=0))
    print(f"  OUR-GNN MaxClique RB-small mean = {np.mean(sizes):.2f} (20 graphs) | ref: Gurobi 19.05, X2GNN ~18.8, DiffUCO 16.30, LTFT 16.24, EGN 12.02", flush=True)


if __name__ == "__main__":
    main()
