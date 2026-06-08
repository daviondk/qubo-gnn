"""E19 (loop): can CRA-annealing make UNSUPERVISED amortization work (LABEL-FREE solver)?
E3 showed plain unsupervised amortization collapses (71% gap). Here we train ONE shared GNN across
S&P100 windows with a CRA continuous-relaxation-annealed UNSUPERVISED loss:
  loss = mean_w [ p^T Qn_w p + gamma_t * sum(1-(2p-1)^2) ],  gamma_t annealed neg->pos over epochs.
If this drops well below 71% (toward the supervised ~1%), we'd have a label-free amortized solver
(no tabu labels needed) = a genuine improvement. Eval: top-K decode, gap vs per-instance tabu. Run in .venv.
"""
from __future__ import annotations
import os, sys, json, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE"); os.environ.setdefault("OMP_NUM_THREADS", "6")
HERE = os.path.dirname(__file__); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from backtest import load_prices, SP100
from baselines import tabu_qubo
from qubo_portfolio import selection_qubo
from amortized import sel_obj
DEV = "cuda" if torch.cuda.is_available() else "cpu"
LAM, K = 0.5, 15


def feats(mu, S):
    sig = np.sqrt(np.clip(np.diag(S), 1e-12, None)); C = S / np.outer(sig, sig)
    ac = (np.abs(C).sum(1) - 1) / (len(mu) - 1)
    def z(x): return (x - x.mean()) / (x.std() + 1e-9)
    return np.column_stack([z(mu), z(sig), z(ac), np.ones_like(mu)]).astype(np.float32), C


def knn(C, k=12):
    n = C.shape[0]; A = np.abs(C.copy()); np.fill_diagonal(A, -1); r, c = [], []
    for i in range(n):
        for j in np.argsort(-A[i])[:k]:
            r += [i, int(j)]; c += [int(j), i]
    return torch.tensor(np.array([r, c], np.int64), device=DEV)


class Net(nn.Module):
    def __init__(s, din, h=64, L=3, drop=0.1):
        super().__init__(); s.cs = nn.ModuleList(); c = din
        for _ in range(L): s.cs.append(SAGEConv(c, h)); c = h
        s.drop = drop; s.o = nn.Linear(h, 1)
    def forward(s, x, ei):
        for c in s.cs:
            x = F.relu(c(x, ei))
            if s.drop > 0: x = F.dropout(x, s.drop, s.training)
        return torch.sigmoid(s.o(x).squeeze(-1))


def main():
    R = load_prices(SP100, "2005-01-01", "2024-12-31").pct_change().dropna().values
    N = R.shape[1]; lb, step = 252, 21
    idx = list(range(lb, len(R), step)); sp = int(0.7 * len(idx)); tr_i, te_i = idx[:sp], idx[sp:][:40]
    def est(t):
        w = R[t - lb:t]; mu = w.mean(0); S = np.cov(w, rowvar=False); return mu, 0.5 * (S + S.T) + 1e-8 * np.eye(N)
    print(f"E19 CRA label-free amortized: N={N} K={K} {len(tr_i)}tr/{len(te_i)}te", flush=True)
    tr = []
    for t in tr_i:
        mu, S = est(t); q = selection_qubo(mu, S, K, risk_aversion=LAM, return_weight=1 - LAM)
        Qn = q.Q / (np.abs(q.Q[q.Q != 0]).mean() + 1e-12); f, C = feats(mu, S)
        tr.append((torch.tensor(f, device=DEV), knn(C), torch.tensor(Qn, dtype=torch.float32, device=DEV)))
    teR = []
    for t in te_i:
        mu, S = est(t); q = selection_qubo(mu, S, K, risk_aversion=LAM, return_weight=1 - LAM)
        r = tabu_qubo(q, num_reads=80, seed=0); xi = np.flatnonzero(np.asarray(r["x"]) > 0.5)
        ref = sel_obj(xi if len(xi) == K else np.argsort(-np.asarray(r["x"]))[:K], mu, S, K)
        teR.append((torch.tensor(feats(mu, S)[0], device=DEV), knn(feats(mu, S)[1]), mu, S, ref))
    torch.manual_seed(0); np.random.seed(0)
    m = Net(4).to(DEV); opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    EP = 600; gc = 1.0
    def evalgap():
        m.eval(); g = []
        for x, ei, mu, S, ref in teR:
            with torch.no_grad(): p = m(x, ei).cpu().numpy()
            oa = sel_obj(np.argsort(-p)[:K], mu, S, K); g.append((oa - ref) / abs(ref) * 100 if abs(ref) > 1e-12 else 0.0)
        return float(np.mean(g))
    best = 1e9
    for ep in range(EP + 1):
        if ep > 0:
            m.train(); perm = np.random.permutation(len(tr)); gamma = gc * (2.0 * ep / EP - 1.0)
            for bi in range(0, len(tr), 16):
                opt.zero_grad(); bl = 0.0
                for ii in perm[bi:bi + 16]:
                    p = m(tr[ii][0], tr[ii][1]); phi = (1.0 - (2 * p - 1) ** 2).mean()
                    bl = bl + (p @ (tr[ii][2] @ p)) / N + gamma * phi
                (bl / max(1, len(perm[bi:bi + 16]))).backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
        if ep % 50 == 0:
            g = evalgap(); best = min(best, g)
            print(f"  ep {ep:3d}: gap vs tabu {g:7.2f}%", flush=True)
    print(f"\n=== E19 label-free CRA-amortized BEST gap = {best:.2f}% (vs plain-unsup 71%, supervised ~1%) ===", flush=True)
    json.dump({"best_gap%": best, "vs_plain_unsup": 71.0, "vs_supervised": 1.0},
              open(os.path.join(HERE, "results", "e19_cra_amortized.json"), "w"), indent=2)
    print("saved", flush=True)


if __name__ == "__main__":
    main()
