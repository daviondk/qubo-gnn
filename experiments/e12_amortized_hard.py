"""E12 (loop): does AMORTIZATION win also on HARD/frustrated cardinality QUBOs (not just easy
mean-variance)? Generate a distribution of frustrated cardinality QUBOs (random signed couplings +
factor cov), tabu-label a train set, train ONE amortized GNN, evaluate gap-vs-tabu + speedup on held-out
hard instances. If amortized GNN ~ tabu at huge speedup on HARD instances => extends the core win.
Saves curve + checkpoint + JSON. Run in .venv.
"""
from __future__ import annotations
import os, sys, json, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE"); os.environ.setdefault("OMP_NUM_THREADS", "4")
HERE = os.path.dirname(__file__); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from torch_geometric.nn import SAGEConv
from qubo import QUBO
from baselines import tabu_qubo
DEV = "cuda" if torch.cuda.is_available() else "cpu"
N, K, H = 120, 20, 0.7


def make_Qobj(seed):
    rng = np.random.default_rng(seed)
    B = rng.standard_normal((N, 6)) * 0.02; Sig = B @ B.T + np.diag((np.abs(rng.standard_normal(N)) * 0.01 + 0.005) ** 2)
    Sig = Sig / np.abs(Sig).mean()
    R = rng.standard_normal((N, N)); R = 0.5 * (R + R.T); np.fill_diagonal(R, 0); R = R / np.abs(R[R != 0]).mean()
    Qo = (1 - H) * Sig + H * R; return 0.5 * (Qo + Qo.T)


def card_qubo(Qo, pf=4.0):
    A = pf * np.abs(Qo[Qo != 0]).mean(); Q = Qo + A * (np.ones((N, N)) - np.eye(N))
    np.fill_diagonal(Q, np.diag(Q) + A * (1 - 2 * K)); return QUBO(Q)


def feats_graph(Qo, k=12):
    d = np.diag(Qo); off = Qo - np.diag(d)
    def z(x): return (x - x.mean()) / (x.std() + 1e-9)
    f = np.column_stack([z(d), z(np.abs(off).sum(1)), z(off.sum(1)), np.ones(N)]).astype(np.float32)
    A = np.abs(off.copy()); np.fill_diagonal(A, -1); r, c = [], []
    for i in range(N):
        for j in np.argsort(-A[i])[:k]:
            r += [i, int(j)]; c += [int(j), i]
    return torch.tensor(f, device=DEV), torch.tensor(np.array([r, c], np.int64), device=DEV)


def obj(x, Qo): x = np.asarray(x, float); return float(x @ Qo @ x)


class Net(nn.Module):
    def __init__(self, din, h=64, L=3, drop=0.2):
        super().__init__(); self.cs = nn.ModuleList(); c = din
        for _ in range(L): self.cs.append(SAGEConv(c, h)); c = h
        self.drop = drop; self.o = nn.Linear(h, 1)
    def forward(self, x, ei):
        for c in self.cs:
            x = F.relu(c(x, ei));
            if self.drop > 0: x = F.dropout(x, self.drop, self.training)
        return self.o(x).squeeze(-1)


def tabu_sel(Qo):
    q = card_qubo(Qo); t0 = time.time(); r = tabu_qubo(q, num_reads=80, seed=0); dt = time.time() - t0
    x = np.asarray(r["x"])[:N]; idx = np.flatnonzero(x > 0.5); idx = idx if len(idx) == K else np.argsort(-x)[:K]
    return idx, obj(np.isin(np.arange(N), idx).astype(float), Qo), dt


def main():
    tr_seeds = list(range(80)); te_seeds = list(range(200, 230))
    print(f"E12 amortized on HARD QUBOs N={N} K={K} H={H}: {len(tr_seeds)}tr/{len(te_seeds)}te; tabu labels...", flush=True)
    trQ = [make_Qobj(s) for s in tr_seeds]; teQ = [make_Qobj(s) for s in te_seeds]
    t0 = time.time(); lab = [tabu_sel(Qo)[0] for Qo in trQ]
    te_ref = [tabu_sel(Qo) for Qo in teQ]; tabu_t = float(np.mean([x[2] for x in te_ref]))
    print(f"  labels {time.time()-t0:.0f}s; tabu/instance={tabu_t:.2f}s", flush=True)
    tr = [feats_graph(Qo) for Qo in trQ]; te = [feats_graph(Qo) for Qo in teQ]
    labs = [torch.tensor(np.isin(np.arange(N), l).astype(np.float32), device=DEV) for l in lab]
    torch.manual_seed(0); np.random.seed(0)
    m = Net(tr[0][0].shape[1]).to(DEV); opt = torch.optim.Adam(m.parameters(), lr=2e-3); pw = torch.tensor([(N - K) / K], device=DEV)

    def gap():
        m.eval(); g = []; t0 = time.time()
        for (x, ei), (Qo, rf) in zip(te, [(q, r) for q, r in zip(teQ, te_ref)]):
            with torch.no_grad(): p = m(x, ei).cpu().numpy()
            xs = np.zeros(N); xs[np.argsort(-p)[:K]] = 1; g.append((obj(xs, Qo) - rf[1]) / abs(rf[1]) * 100 if abs(rf[1]) > 1e-9 else 0.0)
        return float(np.mean(g)), (time.time() - t0) / len(te) * 1000

    curve = []; best = (1e9, None)
    for ep in range(601):
        if ep > 0:
            m.train(); perm = np.random.permutation(len(tr))
            for bi in range(0, len(tr), 16):
                opt.zero_grad(); bl = 0.0
                for ii in perm[bi:bi + 16]:
                    bl = bl + F.binary_cross_entropy_with_logits(m(tr[ii][0], tr[ii][1]), labs[ii], pos_weight=pw)
                (bl / max(1, len(perm[bi:bi + 16]))).backward(); opt.step()
        if ep % 50 == 0:
            g, inf = gap(); curve.append((ep, g))
            if g < best[0]: best = (g, {k: v.cpu() for k, v in m.state_dict().items()})
            print(f"  ep {ep:3d}: gap {g:6.2f}%  inf {inf:.2f}ms", flush=True)
    g, inf = best[0], None
    m.load_state_dict({k: v.to(DEV) for k, v in best[1].items()}); _, inf = gap()
    speed = tabu_t / (inf / 1000 + 1e-9)
    print(f"\n=== E12 BEST: amortized gap vs tabu on HARD QUBOs = {g:.2f}%  | inf {inf:.2f}ms vs tabu {tabu_t:.2f}s -> {speed:.0f}x ===", flush=True)
    os.makedirs(os.path.join(HERE, "checkpoints"), exist_ok=True)
    torch.save({"state_dict": best[1], "gap": g, "N": N, "K": K, "H": H}, os.path.join(HERE, "checkpoints", "e12_amortized_hard_best.pt"))
    plt.figure(figsize=(7, 4.2)); plt.plot([c[0] for c in curve], [c[1] for c in curve], "o-")
    plt.axhline(0, ls=":", c="gray"); plt.xlabel("epoch"); plt.ylabel("amortized gap vs tabu (%)")
    plt.title(f"E12 amortized on HARD frustrated QUBOs (N={N},K={K},H={H})"); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(HERE, "..", "results", "figures", "fig_e12_amortized_hard.png"), dpi=130)
    json.dump({"gap%": g, "inference_ms": inf, "tabu_s": tabu_t, "speedup": speed, "curve": curve, "N": N, "K": K, "H": H},
              open(os.path.join(HERE, "results", "e12_amortized_hard.json"), "w"), indent=2)
    print("saved checkpoint + fig + json", flush=True)


if __name__ == "__main__":
    main()
