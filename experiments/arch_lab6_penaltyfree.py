"""E6 (Phase A): penalty-free encoding (Lozano 2605.17628 idea). The cardinality penalty (sum z - K)^2
adds a dense rank-one all-ones coupling that makes the QUBO graph complete and (we showed) induces the
p=0 collapse. Here we train the unsupervised GNN on the OBJECTIVE-ONLY QUBO (risk - return, no penalty),
then enforce cardinality by top-K projection at decode. Hypothesis: removing the dense penalty term
improves the GNN's ranking signal (topK gap), even if it can't beat greedy per-instance on easy MV.

Compares, per instance: penalized base (top-K) vs penalty-free (top-K) vs +LS. Run in .venv.
"""
from __future__ import annotations
import os, sys, json
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


class Net(nn.Module):
    def __init__(self, din, h, L):
        super().__init__(); self.cs = nn.ModuleList(); c = din
        for _ in range(L):
            self.cs.append(SAGEConv(c, h)); c = h
        self.o = SAGEConv(c, 1)

    def forward(self, x, ei):
        for c in self.cs:
            x = F.relu(c(x, ei))
        return torch.sigmoid(self.o(x, ei).squeeze(-1))


def train_gnn(Qtrain, ei, n, K, epochs=2000, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    Qn = Qtrain / (np.abs(Qtrain[Qtrain != 0]).mean() + 1e-12)
    Qn = torch.tensor(Qn, dtype=torch.float32, device=DEV)
    emb = nn.Embedding(n, 24).to(DEV); net = Net(24, 128, 3).to(DEV)
    params = list(net.parameters()) + list(emb.parameters())
    opt = torch.optim.Adam(params, lr=1e-3); idx = torch.arange(n, device=DEV)
    for ep in range(epochs):
        net.train(); p = net(emb(idx), ei); loss = p @ (Qn @ p)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 5.0); opt.step()
    return p.detach().cpu().numpy()


def main():
    paths = download_orlib(os.path.join(HERE, "..", "data", "orlib"))
    insts = {}
    for nm, K in [("port4", 10), ("port5", 10)]:
        mu, S, _ = load_orlib(paths[nm]); insts[nm] = (mu, S, K)
    rng = np.random.default_rng(5); N = 300; B = rng.standard_normal((N, 6)) * 0.02
    Sig = B @ B.T + np.diag((np.abs(rng.standard_normal(N)) * 0.01 + 0.005) ** 2)
    insts["synth300"] = (rng.standard_normal(N) * 0.004 + 0.003, 0.5 * (Sig + Sig.T), 20)
    fout = open(os.path.join(HERE, "results", "arch_lab6.jsonl"), "a")
    for inst, (mu, Sig, K) in insts.items():
        q = selection_qubo(mu, Sig, K, risk_aversion=LAM, return_weight=1 - LAM)            # penalized (true)
        qf = selection_qubo(mu, Sig, K, risk_aversion=LAM, return_weight=1 - LAM, penalty_factor=0.0)  # objective-only
        e_tabu = tabu_qubo(q, num_reads=200, seed=0)["energy"]
        e_rand = min(local_search_1flip(q, random_binary(q.n, np.random.default_rng(s)), 100)[1] for s in range(20))
        best = min(e_tabu, e_rand); n = q.n
        print(f"\n=== {inst} N={n} K={K} | best={best:.5f} ===", flush=True)
        print(f"{'encoding':<16}{'GNN-topK':>10}{'GNN+LS':>10}{'bare|S|':>8}")
        g = lambda e: (e - best) / abs(best) * 100
        for name, Qtr in [("penalized", q.Q), ("penalty-free", qf.Q)]:
            # build graph from the covariance structure (kNN on the OBJECTIVE Q so it's the same sparse graph)
            ei = torch.tensor(knn_edges(qf.Q, 12), dtype=torch.long, device=DEV)
            p = train_gnn(Qtr, ei, n, K)
            x_topk = np.zeros(n, np.int8); x_topk[np.argsort(-p)[:K]] = 1
            x_bare = (p > 0.5).astype(np.int8)
            _, e_ls = local_search_1flip(q, x_topk.copy(), 100)   # LS from the feasible top-K, on TRUE objective
            print(f"{name:<16}{g(q.energy(x_topk)):>9.2f}%{g(e_ls):>9.2f}%{int(x_bare.sum()):>8}", flush=True)
            fout.write(json.dumps({"inst": inst, "encoding": name, "gap_topk%": g(q.energy(x_topk)),
                                   "gap_ls%": g(e_ls), "bare_k": int(x_bare.sum())}) + "\n"); fout.flush()
    fout.close()


if __name__ == "__main__":
    main()
