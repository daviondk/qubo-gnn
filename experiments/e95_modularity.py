"""E95 (NEW not-in-QIGNN candidate, GNN-native): Modularity Maximization / Community Detection.
QUBO/Potts modularity loss Q=(1/2m)[sum_edges P_i.P_j - (1/2m)|sum_i d_i P_i|^2], GNN outputs C-way softmax.
Compare modularity Q + NMI to Louvain (gold-standard heuristic) on SBM graphs (reproducible, planted
communities). Refs: DMoN, DGCLUSTER, "Analyzing Modularity Max in GNN". Run in .venv."""
import sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, networkx as nx
sys.path.insert(0, "src")
from torch_geometric.nn import SAGEConv
DEV = "cuda" if torch.cuda.is_available() else "cpu"


class ClusterGNN(nn.Module):
    def __init__(self, C, hidden=128, n_layers=3, dropout=0.1):
        super().__init__(); self.dropout = dropout; self.hidden = hidden
        self.blocks = nn.ModuleList(); self.norms = nn.ModuleList(); self.proj = nn.ModuleList()
        cur = 18 + hidden  # degree+ones+16 random feats (symmetry breaking)
        for _ in range(n_layers):
            self.blocks.append(nn.ModuleList([SAGEConv(cur, hidden, "mean"), SAGEConv(cur, hidden, "max")]))
            self.norms.append(nn.LayerNorm(hidden)); self.proj.append(nn.Linear(cur, hidden) if cur != hidden else nn.Identity()); cur = hidden
        self.out = nn.Linear(cur, C); self.act = nn.LeakyReLU()

    def forward(self, xs, ei, hd):
        x = torch.cat([xs, hd], 1)
        for (cm, cx), norm, proj in zip(self.blocks, self.norms, self.proj):
            h = norm(cm(x, ei) + cx(x, ei)); x = self.act(h + proj(x)); x = F.dropout(x, self.dropout, self.training)
        return F.softmax(self.out(x), dim=1), x


def modularity_Q(G, labels):
    return nx.algorithms.community.modularity(G, _parts(labels))


def _parts(labels):
    from collections import defaultdict
    d = defaultdict(set)
    for v, c in enumerate(labels): d[c].add(v)
    return list(d.values())


def solve_modularity(G, C=15, epochs=2500, seed=0):
    torch.manual_seed(seed); n = G.number_of_nodes(); m = G.number_of_edges()
    A = nx.to_numpy_array(G); d = A.sum(1)
    rng = np.random.default_rng(seed); deg = d.reshape(-1, 1)
    rfeat = rng.standard_normal((n, 16))  # random features break SBM degree-symmetry
    sf = torch.tensor(np.column_stack([(deg - deg.mean()) / (deg.std() + 1e-9), np.ones((n, 1)), rfeat]), dtype=torch.float32, device=DEV)
    E = np.array(G.edges()).T; ei = torch.tensor(np.hstack([E, E[::-1]]), dtype=torch.long, device=DEV)
    eidx = torch.tensor(E, dtype=torch.long, device=DEV)
    dt = torch.tensor(d, dtype=torch.float32, device=DEV)
    net = ClusterGNN(C).to(DEV); opt = torch.optim.Adam(net.parameters(), lr=1e-3); hd = torch.zeros((n, net.hidden), device=DEV)
    best = -1; best_lab = None
    for ep in range(epochs):
        net.train(); P, hnew = net(sf, ei, hd)
        edge_term = (P[eidx[0]] * P[eidx[1]]).sum()  # sum over edges (one dir) of P_i . P_j
        deg_vec = (dt.unsqueeze(1) * P).sum(0)  # sum_i d_i P_i  (C,)
        Q = (2 * edge_term - (deg_vec * deg_vec).sum() / (2 * m)) / (2 * m)  # 2*edge_term: both dirs
        collapse = (C ** 0.5 / n) * torch.norm(P.sum(0)) - 1.0  # DMoN collapse regularizer (balance)
        loss = -Q + 1.0 * collapse
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step(); hd = hnew.detach()
        if ep % 200 == 0:
            lab = P.detach().argmax(1).cpu().numpy(); q = modularity_Q(G, lab)
            if q > best: best = q; best_lab = lab
    return best, best_lab


def main():
    from sklearn.metrics import normalized_mutual_info_score as nmi
    print("=== Modularity Maximization (community detection) on SBM: OUR-GNN vs Louvain ===", flush=True)
    for C_true, n_per, pin, pout in [(5, 100, 0.10, 0.01), (10, 80, 0.12, 0.008)]:
        gnn_q, gnn_nmi, lou_q, lou_nmi = [], [], [], []
        for g in range(8):
            sizes = [n_per] * C_true
            G = nx.stochastic_block_model(sizes, [[pin if i == j else pout for j in range(C_true)] for i in range(C_true)], seed=6000 + g)
            G = nx.Graph(G); planted = sum([[c] * n_per for c in range(C_true)], [])
            # ours
            q, lab = solve_modularity(G, C=C_true + 5, seed=0); gnn_q.append(q); gnn_nmi.append(nmi(planted, lab))
            # louvain
            lc = nx.algorithms.community.louvain_communities(G, seed=0); ll = np.zeros(G.number_of_nodes(), int)
            for ci, com in enumerate(lc):
                for v in com: ll[v] = ci
            lou_q.append(nx.algorithms.community.modularity(G, lc)); lou_nmi.append(nmi(planted, ll))
        print(f"  SBM C={C_true} n={n_per*C_true}: OUR-GNN Q={np.mean(gnn_q):.4f} NMI={np.mean(gnn_nmi):.3f} | Louvain Q={np.mean(lou_q):.4f} NMI={np.mean(lou_nmi):.3f}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
