"""Exp 2: kill the bare-GNN collapse with a CARDINALITY-AWARE design.
Two levers on top of Exp1's best (SAGE + kNN graph):
  - out_bias: initialize a learnable scalar output bias to logit(K/N) so the model STARTS near |S|=K
    (not at the all-zero saddle).
  - knorm: evaluate the QUBO loss on K-normalized probabilities  p~ = clip(K * p / sum(p), 0, 1),
    so ~K units of selection mass are always allocated (prevents collapse to p=0).
Reports GNN-alone / GNN-topK / GNN+LS gap to best-found, isolating the GNN's OWN quality.
"""
from __future__ import annotations
import os, sys, json, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE"); os.environ.setdefault("OMP_NUM_THREADS", "8")
HERE = os.path.dirname(__file__); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from arch_lab import knn_edges, full_edges
from portfolio_data import download_orlib, load_orlib
from qubo_portfolio import selection_qubo
from qubo import local_search_1flip, random_binary
from baselines import tabu_qubo
DEV = "cuda" if torch.cuda.is_available() else "cpu"
LAM = 0.5


class Net2(nn.Module):
    def __init__(self, in_dim, hidden, layers, out_bias_init):
        super().__init__()
        self.convs = nn.ModuleList(); cur = in_dim
        for _ in range(layers):
            self.convs.append(SAGEConv(cur, hidden)); cur = hidden
        self.out = SAGEConv(cur, 1)
        self.obias = nn.Parameter(torch.tensor(float(out_bias_init)))

    def forward(self, x, ei):
        for c in self.convs:
            x = F.relu(c(x, ei))
        return torch.sigmoid(self.out(x, ei).squeeze(-1) + self.obias)


def run(qubo, K, cfg, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    n = qubo.n; Q = torch.tensor(qubo.Q, dtype=torch.float32, device=DEV)
    ei = knn_edges(qubo.Q, cfg.get("knn", 10)) if cfg["graph"] == "knn" else full_edges(qubo.Q)
    ei = torch.tensor(ei, dtype=torch.long, device=DEV)
    bias0 = float(np.log((K / n) / (1 - K / n))) if cfg["out_bias"] else 0.0
    emb = nn.Embedding(n, cfg["embed"]).to(DEV)
    net = Net2(cfg["embed"], cfg["hidden"], cfg["layers"], bias0).to(DEV)
    params = list(net.parameters()) + list(emb.parameters())
    opt = torch.optim.Adam(params, lr=cfg.get("lr", 1e-3))
    idx = torch.arange(n, device=DEV)
    for ep in range(cfg["epochs"]):
        net.train(); p = net(emb(idx), ei)
        pe = torch.clamp(K * p / (p.sum() + 1e-9), 0, 1) if cfg["knorm"] else p
        loss = pe @ (Q @ pe)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 5.0); opt.step()
    pn = p.detach().cpu().numpy()
    pe = (np.clip(K * pn / (pn.sum() + 1e-9), 0, 1) if cfg["knorm"] else pn)
    x_bare = (pe > 0.5).astype(np.int8)
    x_topk = np.zeros(n, dtype=np.int8); x_topk[np.argsort(-pn)[:K]] = 1
    _, e_ls = local_search_1flip(qubo, x_bare.copy(), 100)
    return {"bare_E": qubo.energy(x_bare), "bare_k": int(x_bare.sum()),
            "topk_E": qubo.energy(x_topk), "ls_E": e_ls}


def main():
    paths = download_orlib(os.path.join(HERE, "..", "data", "orlib"))
    insts = {}
    for nm, K in [("port4", 10), ("port5", 10)]:
        mu, S, _ = load_orlib(paths[nm]); insts[nm] = (mu, S, K)
    rng = np.random.default_rng(5); N = 300; B = rng.standard_normal((N, 6)) * 0.02
    Sig = B @ B.T + np.diag((np.abs(rng.standard_normal(N)) * 0.01 + 0.005) ** 2)
    insts["synth300"] = (rng.standard_normal(N) * 0.004 + 0.003, 0.5 * (Sig + Sig.T), 20)
    base = dict(graph="knn", embed=24, hidden=128, layers=3, epochs=1500, lr=1e-3, knn=10,
                out_bias=False, knorm=False)
    configs = {
        "C_knn(base)":      {**base},
        "F_knn+bias":       {**base, "out_bias": True},
        "G_knn+knorm":      {**base, "knorm": True},
        "H_knn+bias+knorm": {**base, "out_bias": True, "knorm": True},
        "I_full+bias+knorm":{**base, "graph": "full", "out_bias": True, "knorm": True},
    }
    fout = open(os.path.join(HERE, "results", "arch_lab2.jsonl"), "a")
    for inst, (mu, Sig, K) in insts.items():
        q = selection_qubo(mu, Sig, K, risk_aversion=LAM, return_weight=1 - LAM)
        e_tabu = tabu_qubo(q, num_reads=200, seed=0)["energy"]
        e_rand = min(local_search_1flip(q, random_binary(q.n, np.random.default_rng(s)), 100)[1] for s in range(20))
        best = min(e_tabu, e_rand)
        print(f"\n=== {inst} N={len(mu)} K={K} | best(tabu)={best:.5f} ===", flush=True)
        print(f"{'config':<18}{'GNN-alone':>11}{'GNN-topK':>10}{'GNN+LS':>10}{'bare|S|':>8}")
        for name, cfg in configs.items():
            r = run(q, K, cfg, seed=0)
            g = lambda e: (e - best) / abs(best) * 100
            print(f"{name:<18}{g(r['bare_E']):>10.1f}%{g(r['topk_E']):>9.2f}%{g(r['ls_E']):>9.2f}%{r['bare_k']:>8}", flush=True)
            fout.write(json.dumps({"inst": inst, "config": name, "cfg": cfg, "gap_bare%": g(r["bare_E"]),
                                   "gap_topk%": g(r["topk_E"]), "gap_ls%": g(r["ls_E"]), "bare_k": r["bare_k"]}) + "\n")
            fout.flush()
    fout.close()


if __name__ == "__main__":
    main()
