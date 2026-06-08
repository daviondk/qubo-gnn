"""E24 (loop): newest architecture idea — ITERATIVE SOLUTION REFINEMENT at inference (QRF-GNN successor,
ICLR 2026). Train unsupervised GNN on the QUBO; at inference, run T rounds where each round feeds the
current solution (rounded probs) back as an extra node feature and re-infers, refining the selection.
Measure GNN-alone top-K gap across refinement rounds on port4/port5 + a frustrated synthetic instance.
Tests whether iterative refinement improves the bare-GNN ranking. Run in .venv.
"""
from __future__ import annotations
import os, sys, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE"); os.environ.setdefault("OMP_NUM_THREADS", "6")
HERE = os.path.dirname(__file__); sys.path.insert(0, os.path.join(HERE, "..", "src"))
from torch_geometric.nn import SAGEConv
from qubo import QUBO
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def knn_ei(Q, k=12):
    n = Q.shape[0]; A = np.abs(Q - np.diag(np.diag(Q))); np.fill_diagonal(A, -1); r, c = [], []
    for i in range(n):
        for j in np.argsort(-A[i])[:k]:
            r += [i, int(j)]; c += [int(j), i]
    return torch.tensor(np.array([r, c], np.int64), device=DEV)


class Net(nn.Module):
    def __init__(s, din, h=128, L=3):
        super().__init__(); s.cs = nn.ModuleList(); c = din
        for _ in range(L): s.cs.append(SAGEConv(c, h)); c = h
        s.o = SAGEConv(c, 1)
    def forward(s, x, ei):
        for c in s.cs: x = F.relu(c(x, ei))
        return torch.sigmoid(s.o(x, ei).squeeze(-1))


def refine(Q, K, T=6, epochs=1500, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    n = Q.shape[0]; Qn = torch.tensor(Q / (np.abs(Q[Q != 0]).mean() + 1e-12), dtype=torch.float32, device=DEV)
    ei = knn_ei(Q); emb = nn.Embedding(n, 16).to(DEV)
    cur = torch.zeros(n, 1, device=DEV)  # current-solution feedback feature
    net = Net(16 + 1, 128, 3).to(DEV); params = list(net.parameters()) + list(emb.parameters())
    opt = torch.optim.Adam(params, lr=1e-3); idx = torch.arange(n, device=DEV)
    gaps = []
    qubo = QUBO(Q)
    for ep in range(epochs):
        net.train(); x = torch.cat([emb(idx), cur], 1); p = net(x, ei)
        loss = p @ (Qn @ p)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 5.0); opt.step()
        if ep % (epochs // T) == 0:
            pn = p.detach().cpu().numpy(); xk = np.zeros(n, np.int8); xk[np.argsort(-pn)[:K]] = 1
            cur = torch.tensor(xk.reshape(-1, 1).astype(np.float32), device=DEV)  # feed back current solution
            gaps.append(qubo.energy(xk))
    return gaps


def main():
    insts = {}
    for nm in ["port4", "port5"]:
        d = np.load(os.path.join(HERE, "results", f"_inst_{nm}.npz")); insts[nm] = (d["Q"], int(d["K"]), float(d["obj_ref"]))
    # frustrated synthetic
    rng = np.random.default_rng(0); N = 150; B = rng.standard_normal((N, 6)) * 0.02
    Sig = B @ B.T + np.diag((np.abs(rng.standard_normal(N)) * 0.01 + 0.005) ** 2); Sig /= np.abs(Sig).mean()
    Rm = rng.standard_normal((N, N)); Rm = 0.5 * (Rm + Rm.T); np.fill_diagonal(Rm, 0); Rm /= np.abs(Rm[Rm != 0]).mean()
    Qo = 0.7 * Rm + 0.3 * Sig; K = 20; A = 4 * np.abs(Qo[Qo != 0]).mean()
    Qf = Qo + A * (np.ones((N, N)) - np.eye(N)); np.fill_diagonal(Qf, np.diag(Qf) + A * (1 - 2 * K))
    from baselines import tabu_qubo
    insts["frustr150"] = (Qf, K, tabu_qubo(QUBO(Qf), num_reads=150, seed=0)["energy"])
    out = {}
    for nm, (Q, K, ref) in insts.items():
        gaps_e = refine(Q, K, T=6)
        g = [(e - ref) / abs(ref) * 100 if abs(ref) > 1e-9 else 0.0 for e in gaps_e]
        out[nm] = g
        print(f"{nm}: top-K gap by refine round = {[round(x,1) for x in g]}", flush=True)
    json.dump(out, open(os.path.join(HERE, "results", "e24_iterative_refine.json"), "w"), indent=2)
    print("saved", flush=True)


if __name__ == "__main__":
    main()
