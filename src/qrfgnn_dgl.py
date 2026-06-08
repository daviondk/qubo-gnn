"""Faithful, verbatim reimplementation of the original notebook's QRF-GNN (DGL) method.

Copied as-is from `original_from_paper_gnn_example_Copy1-2.ipynb` /
`qubo_portofolio_opt.ipynb` so we reproduce the paper exactly (no library substitution).
Only a thin Gset runner + argument parsing is added at the bottom.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import dgl
import torch
import random
import glob
import numpy as np
import networkx as nx
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict, defaultdict
from dgl.nn.pytorch import GraphConv, SAGEConv
from itertools import chain, islice, combinations
from time import time

TORCH_DEVICE = torch.device('cpu')  # DGL CUDA wheels are Linux-only; cut quality is device-independent
TORCH_DTYPE = torch.float32


# ----------------------------- MaxCut QUBO (verbatim) -----------------------------
def gen_q_dict_maxcut(nx_G, penalty=2):
    Q_dic = defaultdict(int)
    Adj = nx.adjacency_matrix(nx_G).toarray()
    for (u, v) in nx_G.edges:
        Q_dic[(u, v)] = penalty * nx_G[u][v]["weight"]
    for u in nx_G.nodes:
        Q_dic[(u, u)] = -Adj[u].sum()
    return Q_dic


def get_cut_edges(G):
    weights = 0
    n_cut = 0
    for u, v, d in G.edges(data='weight'):
        if G.nodes[u]['subset'] != G.nodes[v]['subset']:
            n_cut += 1
            weights += G[u][v]["weight"]
    return n_cut, weights


# ----------------------------- model (verbatim) -----------------------------
class SAGEResBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, feat_drop=0.):
        super(SAGEResBlock, self).__init__()
        self.sage1 = SAGEConv(in_channels, out_channels, aggregator_type='mean', feat_drop=feat_drop, bias=False)
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.sage2 = SAGEConv(in_channels, out_channels, aggregator_type='pool', feat_drop=feat_drop, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.LeakyReLU()

    def forward(self, graph, x, edge_weight=None):
        residual = x
        out1 = self.sage1(graph, x, edge_weight)
        out1 = self.bn1(out1)
        out2 = self.sage2(graph, x, edge_weight)
        out2 = self.bn2(out2)
        out = self.relu(out1 + out2)
        return out


class ResSAGE(torch.nn.Module):
    def __init__(self, in_feats, hidden_sizes, number_classes, dropout, device):
        super(ResSAGE, self).__init__()
        self.dropout_frac = dropout
        self.layers = nn.ModuleList()
        current_dim = in_feats
        self.relu = torch.nn.LeakyReLU()
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]
        for hdim in hidden_sizes:
            self.layers.append(SAGEResBlock(current_dim, hdim).to(device))
            self.layers.append(torch.nn.LeakyReLU())
            current_dim = hdim
        self.layers.append(SAGEConv(current_dim, number_classes, aggregator_type='mean').to(device))

    def forward(self, graph, h, h0, edge_weight=None):
        h = torch.cat([h, h0], 1)
        for i, (layer, norm) in enumerate(zip(self.layers[:-1][::2], self.layers[:-1][1::2])):
            h = layer(graph, h, edge_weight)
            h = norm(h)
        h = F.dropout(h, p=self.dropout_frac)
        h0 = self.layers[-1](graph, h, edge_weight)
        h = torch.sigmoid(h0)
        return h, h0


# ----------------------------- utilities (verbatim) -----------------------------
def qubo_dict_to_torch(nx_G, Q, torch_dtype=None, torch_device=None):
    n_nodes = len(nx_G.nodes)
    Q_mat = torch.zeros(n_nodes, n_nodes)
    for (x_coord, y_coord), val in Q.items():
        Q_mat[x_coord][y_coord] = val
    if torch_dtype is not None:
        Q_mat = Q_mat.type(torch_dtype)
    if torch_device is not None:
        Q_mat = Q_mat.to(torch_device)
    return Q_mat


def gen_combinations(combs, chunk_size):
    yield from iter(lambda: list(islice(combs, chunk_size)), [])


def pagerank(nx_graph, feature_dim=10):
    features = torch.zeros((nx_graph.number_of_nodes(), feature_dim))
    pr = nx.pagerank(nx.Graph(nx_graph))
    for k, v in pr.items():
        features[k, :] = v
    return features


def loss_func(probs, Q_mat, epoch=0):
    probs_ = torch.unsqueeze(probs, 1)
    lbd = epoch / 1e4
    penalty = (probs - 1) * probs
    cost = (probs_.T @ Q_mat @ probs_).squeeze() + lbd * penalty.abs().sum()
    return cost


def get_gnn(n_nodes, gnn_hypers, opt_params, torch_device, torch_dtype):
    dim_embedding = gnn_hypers['dim_embedding']
    hidden_dim = gnn_hypers['hidden_dim']
    dropout = gnn_hypers['dropout']
    number_classes = gnn_hypers['number_classes']
    net = ResSAGE(dim_embedding + 1 * number_classes + 4 * dim_embedding, hidden_dim, number_classes, dropout, torch_device)
    net = net.type(torch_dtype).to(torch_device)
    embed = nn.Embedding(n_nodes, dim_embedding)
    embed = embed.type(torch_dtype).to(torch_device)
    params = chain(net.parameters(), embed.parameters())
    optimizer = torch.optim.Adam(params, **opt_params)
    return net, embed, optimizer


def run_gnn_training(q_torch, dgl_graph, net, embed, optimizer, number_epochs, tol, patience, prob_threshold):
    edge_weight = (q_torch - torch.diag(q_torch.diag(0))) / 2
    edge_weight = edge_weight + edge_weight.T
    edge_weight = edge_weight[dgl_graph.edges()[0], dgl_graph.edges()[1]]

    inputs = torch.rand((dgl_graph.number_of_nodes(), 10), dtype=q_torch.dtype).to(q_torch.device)
    walk = pagerank(dgl_graph.cpu().to_networkx(), 2 * inputs.shape[1])
    inputs = torch.cat([inputs, torch.ones_like(inputs), torch.ones_like(inputs), walk.to(q_torch.device)], 1)

    h0 = torch.zeros(dgl_graph.number_of_nodes(), 1).to(q_torch.device)
    prev_loss = 1.
    count = 0
    best_epoch = 0
    best_sums = 0
    best_bitstring = torch.zeros((dgl_graph.number_of_nodes(),)).type(q_torch.dtype).to(q_torch.device)
    best_loss = loss_func(best_bitstring.float(), q_torch)
    best_probs = torch.zeros((dgl_graph.number_of_nodes(),)).type(q_torch.dtype).to(q_torch.device)
    t_gnn_start = time()

    for epoch in range(number_epochs):
        probs, h0 = net(dgl_graph, inputs, h0.detach(), edge_weight)
        probs = probs.squeeze()
        loss = loss_func(probs, q_torch, epoch)
        loss_ = loss.detach().item()
        bitstring = (probs.detach() >= prob_threshold) * 1
        if loss < best_loss:
            sums = -loss_func(bitstring.to(torch.float32), q_torch)
            if best_sums < sums:
                best_loss = max(loss, -sums)
                best_bitstring = bitstring
                best_probs = probs
                best_epoch = epoch
                best_sums = sums
        if epoch % 1000 == 0:
            print(f'Epoch: {epoch}, Loss: {loss_}, Best Loss: {best_loss}, Best sum: {best_sums}')
        if (abs(loss_ - prev_loss) <= tol) | ((loss_ - prev_loss) > 0):
            count += 1
        else:
            count = 0
        if count >= patience:
            print(f'Stopping early on epoch {epoch} (patience: {patience})')
            break
        prev_loss = loss_
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), max_norm=2.0, norm_type=2)
        optimizer.step()

    t_gnn = time() - t_gnn_start
    print(f'GNN training (n={dgl_graph.number_of_nodes()}) took {round(t_gnn, 3)}')
    final_bitstring = (probs.detach() >= prob_threshold) * 1
    return net, best_epoch, best_probs, best_bitstring, best_loss


# ----------------------------- Gset runner (mirrors notebook cell 21) -----------------------------
GSET_BEST = {"G14": 3064, "G15": 3050, "G22": 13359, "G49": 6000, "G50": 5880, "G55": 10294, "G70": 9541}
QRF_PAPER = {"G14": 3058, "G15": 3049, "G22": 13344, "G49": 6000, "G50": 5880, "G55": 10282, "G70": 9559}


def run_gset(pattern="Gset/G14*", n_seeds=5, number_epochs=20000, lr=0.014):
    import sys
    for file in glob.glob(pattern):
        name = os.path.splitext(os.path.basename(file))[0]
        nx_graph = nx.read_edgelist(file, nodetype=int, data=(('weight', float),))
        nx_graph = nx.to_undirected(nx_graph)
        mapping = dict(zip(nx_graph.nodes(), np.arange(nx_graph.number_of_nodes())))
        nx_graph = nx.relabel_nodes(nx_graph, mapping)
        graph_dgl = dgl.from_networkx(nx_graph=nx_graph).to(TORCH_DEVICE)
        n = nx_graph.number_of_nodes()
        q_torch = qubo_dict_to_torch(nx_graph, gen_q_dict_maxcut(nx_graph), torch_dtype=TORCH_DTYPE, torch_device=TORCH_DEVICE)
        best_cut = -1
        for s in range(1, n_seeds + 1):
            random.seed(s); np.random.seed(s); torch.manual_seed(s)
            gnn_hypers = {'dim_embedding': 10, 'hidden_dim': 51, 'dropout': 0.5,
                          'number_classes': 1, 'prob_threshold': 0.5,
                          'number_epochs': number_epochs, 'tolerance': 1e-4, 'patience': 100}
            net, embed, optimizer = get_gnn(n, gnn_hypers, {'lr': lr}, TORCH_DEVICE, TORCH_DTYPE)
            _, epoch, _, bitstring, loss = run_gnn_training(
                q_torch, graph_dgl, net, embed, optimizer, number_epochs, 1e-4, 100, 0.5)
            membership = dict(zip(nx_graph.nodes(), bitstring.detach().cpu().numpy()))
            nx.set_node_attributes(nx_graph, membership, 'subset')
            n_cuts, _ = get_cut_edges(nx_graph)
            best_cut = max(best_cut, n_cuts)
            print(f'  {name} seed {s}: cut={n_cuts} best={best_cut}')
        bk = GSET_BEST.get(name, 0); pp = QRF_PAPER.get(name, 0)
        print(f'>>> {name} n={n}: GNN best cut={best_cut} | best-known={bk} ({best_cut/bk:.4f}) | paper-QRF={pp}')


if __name__ == "__main__":
    import sys
    pat = sys.argv[1] if len(sys.argv) > 1 else "Gset/G14*"
    seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    eps = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
    print(f"device={TORCH_DEVICE}")
    run_gset(pat, n_seeds=seeds, number_epochs=eps)
