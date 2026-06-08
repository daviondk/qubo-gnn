"""Architecture lab (separate from src/, for the improvement track).

Goal: ISOLATE the GNN's own contribution and test architecture fixes from Krylova (2024) + our findings.
Self-contained GNN with toggles: optimizer {adam,rprop}, layer {sage,gcn}, graph {full,knn}, recurrent,
binarization-anneal. Reports, per instance, on the SELECTION-QUBO energy (lower=better):
  - GNN-alone (round p>0.5, NO local search)         -> the GNN's raw quality
  - GNN top-K (enforce |S|=K from p, NO local search) -> the GNN's ranking quality
  - GNN + 1-flip local search                         -> with polish
vs references: random multistart+LS, tabu, exact(best). Gaps to best-found.

Usage: python experiments/arch_lab.py            (default config sweep on port4,port5,synth300)
Every result is appended to experiments/results/arch_lab.jsonl with the config, for the LOG.
"""
from __future__ import annotations
import os, sys, json, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE"); os.environ.setdefault("OMP_NUM_THREADS", "8")
HERE = os.path.dirname(__file__); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GraphConv
from portfolio_data import download_orlib, load_orlib
from qubo_portfolio import selection_qubo, decode_selection
from qubo import local_search_1flip, random_binary
from baselines import tabu_qubo
DEV = "cuda" if torch.cuda.is_available() else "cpu"
LAM, K_DEFAULT = 0.5, 10


def knn_edges(Q, k=10):
    n = Q.shape[0]; A = np.abs(Q - np.diag(np.diag(Q))); np.fill_diagonal(A, -1)
    rows, cols = [], []
    for i in range(n):
        for j in np.argsort(-A[i])[:k]:
            rows += [i, int(j)]; cols += [int(j), i]
    return np.array([rows, cols], dtype=np.int64)


def full_edges(Q):
    off = Q - np.diag(np.diag(Q)); r, c = np.nonzero(off)
    return np.vstack([r, c]).astype(np.int64)


class Net(nn.Module):
    def __init__(self, in_dim, hidden, layers, kind, recurrent):
        super().__init__(); self.recurrent = recurrent
        Conv = SAGEConv if kind == "sage" else GraphConv
        self.convs = nn.ModuleList(); cur = in_dim + (1 if recurrent else 0)
        for _ in range(layers):
            self.convs.append(Conv(cur, hidden)); cur = hidden
        self.out = Conv(cur, 1)

    def forward(self, x, ei, h0):
        if self.recurrent and h0 is not None:
            x = torch.cat([x, h0], 1)
        for c in self.convs:
            x = F.relu(c(x, ei))
        return torch.sigmoid(self.out(x, ei))


def run_gnn(qubo, K, cfg, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    n = qubo.n; Q = torch.tensor(qubo.Q, dtype=torch.float32, device=DEV)
    ei = knn_edges(qubo.Q, cfg.get("knn", 10)) if cfg["graph"] == "knn" else full_edges(qubo.Q)
    ei = torch.tensor(ei, dtype=torch.long, device=DEV)
    emb = nn.Embedding(n, cfg["embed"]).to(DEV)
    net = Net(cfg["embed"], cfg["hidden"], cfg["layers"], cfg["layer"], cfg["recurrent"]).to(DEV)
    params = list(net.parameters()) + list(emb.parameters())
    opt = (torch.optim.Rprop(params, lr=cfg.get("lr", 0.01)) if cfg["opt"] == "rprop"
           else torch.optim.Adam(params, lr=cfg.get("lr", 1e-3)))
    idx = torch.arange(n, device=DEV); h0 = torch.zeros((n, 1), device=DEV)
    t0 = time.time()
    for ep in range(cfg["epochs"]):
        net.train(); p = net(emb(idx), ei, h0).squeeze(-1)
        loss = p @ (Q @ p) + cfg.get("anneal", 0.0) * ep * (p * (1 - p)).sum()
        if not torch.isfinite(loss):
            h0 = torch.zeros((n, 1), device=DEV); continue
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0); opt.step()
        if cfg["recurrent"]:
            h0 = p.detach().unsqueeze(-1)
    pn = p.detach().cpu().numpy()
    x_bare = (pn > 0.5).astype(np.int8)
    x_topk = np.zeros(n, dtype=np.int8); x_topk[np.argsort(-pn)[:K]] = 1
    x_ls, e_ls = local_search_1flip(qubo, x_bare.copy(), 100)
    return {"bare_E": qubo.energy(x_bare), "bare_k": int(x_bare.sum()),
            "topk_E": qubo.energy(x_topk), "ls_E": e_ls, "t": time.time() - t0}


def main():
    paths = download_orlib(os.path.join(HERE, "..", "data", "orlib"))
    insts = {}
    for nm, K in [("port4", 10), ("port5", 10)]:
        mu, S, _ = load_orlib(paths[nm]); insts[nm] = (mu, S, K)
    rng = np.random.default_rng(5); N = 300; B = rng.standard_normal((N, 6)) * 0.02
    Sig = B @ B.T + np.diag((np.abs(rng.standard_normal(N)) * 0.01 + 0.005) ** 2)
    insts["synth300"] = (rng.standard_normal(N) * 0.004 + 0.003, 0.5 * (Sig + Sig.T), 20)

    base = dict(opt="adam", layer="sage", graph="full", recurrent=True, embed=24, hidden=128,
                layers=3, epochs=1500, anneal=0.0, lr=1e-3, knn=10)
    configs = {
        "A_adam_full":   {**base},
        "B_rprop_full":  {**base, "opt": "rprop", "lr": 0.01},
        "C_adam_knn":    {**base, "graph": "knn"},
        "D_rprop_knn":   {**base, "opt": "rprop", "lr": 0.01, "graph": "knn"},
        "E_gcn_full":    {**base, "layer": "gcn"},
    }
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    fout = open(os.path.join(HERE, "results", "arch_lab.jsonl"), "a")
    for inst, (mu, Sig, K) in insts.items():
        q = selection_qubo(mu, Sig, K, risk_aversion=LAM, return_weight=1 - LAM)
        # references
        tb = tabu_qubo(q, num_reads=200, seed=0); e_tabu = tb["energy"]
        e_rand = min(local_search_1flip(q, random_binary(q.n, np.random.default_rng(s)), 100)[1] for s in range(20))
        best = min(e_tabu, e_rand)
        print(f"\n=== {inst} N={len(mu)} K={K} | tabu={e_tabu:.5f} randLS={e_rand:.5f} ===", flush=True)
        print(f"{'config':<14}{'GNN-alone':>11}{'GNN-topK':>10}{'GNN+LS':>10}{'feas|S|':>8}{'t(s)':>7}")
        for name, cfg in configs.items():
            r = run_gnn(q, K, cfg, seed=0)
            def gap(e): return (e - best) / abs(best) * 100
            print(f"{name:<14}{gap(r['bare_E']):>10.1f}%{gap(r['topk_E']):>9.2f}%{gap(r['ls_E']):>9.2f}%"
                  f"{r['bare_k']:>8}{r['t']:>7.1f}", flush=True)
            fout.write(json.dumps({"inst": inst, "N": len(mu), "K": K, "config": name, "cfg": cfg,
                                   "gap_bare%": gap(r["bare_E"]), "gap_topk%": gap(r["topk_E"]),
                                   "gap_ls%": gap(r["ls_E"]), "bare_k": r["bare_k"], "t": r["t"],
                                   "e_tabu": e_tabu}) + "\n"); fout.flush()
    fout.close()


if __name__ == "__main__":
    main()
