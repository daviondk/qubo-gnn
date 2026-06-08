"""GNN architectures for unsupervised QUBO solving (PI-GNN / QRF-GNN lineage), in PyG.

Both models take a TRAINABLE node embedding as input (the key PI-GNN ingredient: the per-node
embedding is optimized jointly with the network, giving the model freedom to assign each node).
Optionally a recurrent channel (previous soft assignment) is concatenated (QRF-GNN).

Message passing is structural; the QUBO edge weights enter through the loss (probs^T Q probs).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, SAGEConv


class PIGNN(nn.Module):
    """Clean PI-GNN: GraphConv x n_layers -> sigmoid. Robust baseline that converges well."""

    def __init__(self, in_dim: int, hidden: int = 64, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        cur = in_dim
        for _ in range(n_layers - 1):
            self.convs.append(GraphConv(cur, hidden))
            cur = hidden
        self.out = GraphConv(cur, 1)

    def forward(self, x, edge_index, edge_weight=None, h0=None):
        if h0 is not None:
            x = torch.cat([x, h0], dim=1)
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_weight))
            x = F.dropout(x, p=self.dropout, training=self.training)
        logits = self.out(x, edge_index, edge_weight)
        return torch.sigmoid(logits), logits


class QRFGNN(nn.Module):
    """QRF-GNN style: parallel SAGE (mean+max) residual blocks; recurrent channel handled by solver
    (pass h0 = previous probs). LayerNorm instead of BatchNorm for full-graph stability."""

    def __init__(self, in_dim: int, hidden: int = 64, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.dropout = dropout
        self.blocks = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.proj = nn.ModuleList()
        cur = in_dim
        for _ in range(n_layers):
            self.blocks.append(nn.ModuleList([
                SAGEConv(cur, hidden, aggr="mean"),
                SAGEConv(cur, hidden, aggr="max"),
            ]))
            self.norms.append(nn.LayerNorm(hidden))
            self.proj.append(nn.Linear(cur, hidden) if cur != hidden else nn.Identity())
            cur = hidden
        self.out = SAGEConv(cur, 1, aggr="mean")
        self.act = nn.LeakyReLU()

    def forward(self, x, edge_index, edge_weight=None, h0=None):
        if h0 is not None:
            x = torch.cat([x, h0], dim=1)
        for (cmean, cmax), norm, proj in zip(self.blocks, self.norms, self.proj):
            h = norm(cmean(x, edge_index) + cmax(x, edge_index))
            x = self.act(h + proj(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
        logits = self.out(x, edge_index)
        return torch.sigmoid(logits), logits


def pagerank_vector(edge_index, n: int, device) -> torch.Tensor:
    import networkx as nx
    g = nx.Graph(); g.add_nodes_from(range(n))
    ei = edge_index.cpu().numpy()
    g.add_edges_from(zip(ei[0].tolist(), ei[1].tolist()))
    pr = nx.pagerank(g) if g.number_of_edges() else {i: 1.0 / n for i in range(n)}
    v = torch.zeros((n, 1), device=device)
    for k, val in pr.items():
        v[k, 0] = val
    return v
