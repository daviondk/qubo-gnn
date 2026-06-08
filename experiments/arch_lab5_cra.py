"""E5 (Phase A): fix the relaxation->binary COLLAPSE with Continuous Relaxation Annealing (CRA,
Ichikawa 2309.16965) on the dense portfolio selection QUBO.

CRA adds a penalty Phi(p)=sum_i (1-(2 p_i-1)^2) and anneals its coefficient gamma from NEGATIVE
(maximize Phi -> push p toward 0.5: smooth, escape the p=0 saddle/all-zero collapse) to POSITIVE
(minimize Phi -> push p to {0,1}: rounding-free binarization). loss = p^T Qn p + gamma_t * Phi(p).

Measures GNN-ALONE gap (bare round, NO local search) -> tests whether CRA makes the GNN itself good.
Compares base (no CRA) vs CRA at several gamma coefficients, on port4/port5/synth300. Run in .venv.
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


def run(qubo, K, gamma_coef, graph="knn", embed=24, hidden=128, layers=3, epochs=2000, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    n = qubo.n
    Qn = qubo.Q / (np.abs(qubo.Q[qubo.Q != 0]).mean() + 1e-12)   # normalize energy scale ~O(1)
    Qn = torch.tensor(Qn, dtype=torch.float32, device=DEV)
    ei = knn_edges(qubo.Q, 12) if graph == "knn" else full_edges(qubo.Q)
    ei = torch.tensor(ei, dtype=torch.long, device=DEV)
    emb = nn.Embedding(n, embed).to(DEV); net = Net(embed, hidden, layers).to(DEV)
    params = list(net.parameters()) + list(emb.parameters())
    opt = torch.optim.Adam(params, lr=1e-3); idx = torch.arange(n, device=DEV)
    for ep in range(epochs):
        net.train(); p = net(emb(idx), ei)
        energy = p @ (Qn @ p)
        phi = (1.0 - (2.0 * p - 1.0) ** 2).sum()          # in [0,n]; max at p=0.5, min at p in {0,1}
        gamma = gamma_coef * (2.0 * ep / epochs - 1.0)     # anneal from -gamma_coef to +gamma_coef
        loss = energy + gamma * phi
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0); opt.step()
    pn = p.detach().cpu().numpy()
    x_bare = (pn > 0.5).astype(np.int8)
    x_topk = np.zeros(n, np.int8); x_topk[np.argsort(-pn)[:K]] = 1
    _, e_ls = local_search_1flip(qubo, x_bare.copy(), 100)
    return {"bare_E": qubo.energy(x_bare), "bare_k": int(x_bare.sum()),
            "topk_E": qubo.energy(x_topk), "ls_E": e_ls, "pmean": float(pn.mean())}


def main():
    paths = download_orlib(os.path.join(HERE, "..", "data", "orlib"))
    insts = {}
    for nm, K in [("port4", 10), ("port5", 10)]:
        mu, S, _ = load_orlib(paths[nm]); insts[nm] = (mu, S, K)
    rng = np.random.default_rng(5); N = 300; B = rng.standard_normal((N, 6)) * 0.02
    Sig = B @ B.T + np.diag((np.abs(rng.standard_normal(N)) * 0.01 + 0.005) ** 2)
    insts["synth300"] = (rng.standard_normal(N) * 0.004 + 0.003, 0.5 * (Sig + Sig.T), 20)
    gammas = [("base(no CRA)", 0.0), ("CRA g=0.5", 0.5), ("CRA g=1", 1.0), ("CRA g=2", 2.0), ("CRA g=4", 4.0)]
    fout = open(os.path.join(HERE, "results", "arch_lab5_cra.jsonl"), "a")
    for inst, (mu, Sig, K) in insts.items():
        q = selection_qubo(mu, Sig, K, risk_aversion=LAM, return_weight=1 - LAM)
        e_tabu = tabu_qubo(q, num_reads=200, seed=0)["energy"]
        e_rand = min(local_search_1flip(q, random_binary(q.n, np.random.default_rng(s)), 100)[1] for s in range(20))
        best = min(e_tabu, e_rand)
        print(f"\n=== {inst} N={len(mu)} K={K} | best={best:.5f} ===", flush=True)
        print(f"{'config':<14}{'GNN-alone':>11}{'GNN-topK':>10}{'GNN+LS':>10}{'bare|S|':>8}{'pmean':>7}")
        for name, gc in gammas:
            r = run(q, K, gc, seed=0)
            g = lambda e: (e - best) / abs(best) * 100
            print(f"{name:<14}{g(r['bare_E']):>10.1f}%{g(r['topk_E']):>9.2f}%{g(r['ls_E']):>9.2f}%{r['bare_k']:>8}{r['pmean']:>7.3f}", flush=True)
            fout.write(json.dumps({"inst": inst, "config": name, "gamma": gc, "gap_bare%": g(r["bare_E"]),
                                   "gap_topk%": g(r["topk_E"]), "gap_ls%": g(r["ls_E"]), "bare_k": r["bare_k"]}) + "\n")
            fout.flush()
    fout.close()


if __name__ == "__main__":
    main()
