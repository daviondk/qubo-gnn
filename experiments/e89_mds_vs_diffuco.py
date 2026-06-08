"""E89 (DIRECT comparison to SOTA paper DiffUCO ICML2024, NOT in QIGNN): Minimum Dominating Set on
Barabasi-Albert graphs. DiffUCO setup: BA-small 200-300 nodes, BA-large 800-1200 nodes. Published BA-large:
Gurobi 103.80, DiffUCO 106.61, EGN-Anneal 111.50, EGN 116.76 (size, lower=better). Our GNN: EGN/DiffUCO-style
differentiable MDS loss (sum p + beta*ReLU(1-(A+I)p)^2), round + greedy repair to valid dominating set.
Run in .venv."""
import sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, networkx as nx
sys.path.insert(0, "src")
from torch_geometric.nn import SAGEConv
DEV = "cuda" if torch.cuda.is_available() else "cpu"


class GNN(nn.Module):
    def __init__(self, hidden=128, n_layers=3, dropout=0.1):
        super().__init__(); self.dropout = dropout
        self.blocks = nn.ModuleList(); self.norms = nn.ModuleList(); self.proj = nn.ModuleList()
        cur = 2 + hidden
        for _ in range(n_layers):
            self.blocks.append(nn.ModuleList([SAGEConv(cur, hidden, aggr="mean"), SAGEConv(cur, hidden, aggr="max")]))
            self.norms.append(nn.LayerNorm(hidden)); self.proj.append(nn.Linear(cur, hidden) if cur != hidden else nn.Identity()); cur = hidden
        self.out = nn.Linear(cur, 1); self.act = nn.LeakyReLU(); self.hidden = hidden

    def forward(self, xs, ei, hdyn):
        x = torch.cat([xs, hdyn], 1)
        for (cm, cx), norm, proj in zip(self.blocks, self.norms, self.proj):
            h = norm(cm(x, ei) + cx(x, ei)); x = self.act(h + proj(x)); x = F.dropout(x, self.dropout, self.training)
        return torch.sigmoid(self.out(x)).squeeze(-1), x


def greedy_repair(G, sel):
    sel = set(sel); dominated = set()
    for v in sel: dominated.add(v); dominated.update(G.neighbors(v))
    undom = set(G.nodes()) - dominated
    while undom:
        # pick node covering most undominated (among undom and their neighbors)
        cand = set(undom); [cand.update(G.neighbors(u)) for u in list(undom)]
        best = max(cand, key=lambda c: len((set([c]) | set(G.neighbors(c))) & undom))
        sel.add(best); newd = set([best]) | set(G.neighbors(best)); undom -= newd
    return sel


def solve_mds(G, epochs=2500, hidden=128, beta=2.0, seed=0):
    torch.manual_seed(seed); n = G.number_of_nodes()
    A = nx.to_numpy_array(G); AI = torch.tensor(A + np.eye(n), dtype=torch.float32, device=DEV)
    deg = A.sum(1, keepdims=True); sf = torch.tensor(np.column_stack([(deg - deg.mean()) / (deg.std() + 1e-9), np.ones((n, 1))]), dtype=torch.float32, device=DEV)
    ei = torch.tensor(np.vstack(np.nonzero(A)), dtype=torch.long, device=DEV)
    net = GNN(hidden).to(DEV); opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    hd = torch.zeros((n, hidden), device=DEV); best = None
    for ep in range(epochs):
        net.train(); p, hnew = net(sf, ei, hd)
        cover = AI @ p
        loss = p.sum() + beta * (F.relu(1.0 - cover) ** 2).sum()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
        hd = hnew.detach()
        if ep % 200 == 0:
            pv = p.detach().cpu().numpy()
            sel = [v for v in range(n) if pv[v] > 0.5]
            ds = greedy_repair(G, sel)
            if best is None or len(ds) < best: best = len(ds)
    return best


def main():
    print("=== MDS on BA graphs vs DiffUCO (ICML2024) ===", flush=True)
    for tag, lo, hi, K, ref in [("BA-small", 200, 300, 30, "Gurobi 27.89 / DiffUCO 28.20"),
                                 ("BA-large", 800, 1200, 15, "Gurobi 103.80 / DiffUCO 106.61 / EGN-A 111.50 / EGN 116.76")]:
        sizes = []
        for g in range(K):
            rng = np.random.default_rng(1000 + g); n = int(rng.integers(lo, hi + 1))
            G = nx.barabasi_albert_graph(n, 4, seed=1000 + g)
            sizes.append(solve_mds(G, seed=0))
        print(f"  {tag}: OUR-GNN mean DS size = {np.mean(sizes):.2f} (over {K} graphs) | ref: {ref}", flush=True)


if __name__ == "__main__":
    main()
