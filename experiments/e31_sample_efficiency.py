"""E31 (loop): SAMPLE EFFICIENCY of the amortized GNN — how many labeled training windows are needed to
reach tabu-quality? Train on the last {5,10,20,40,80,159} S&P100 train-windows, eval gap vs per-instance
tabu on a fixed held-out test set. Practical: the one-time training cost (tabu labels) shrinks if few
windows suffice. Saves a sample-efficiency curve. Run in .venv.
"""
import os, sys, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE"); os.environ.setdefault("OMP_NUM_THREADS", "6")
HERE = os.path.dirname(__file__); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from torch_geometric.nn import SAGEConv
from backtest import load_prices, SP100
from baselines import tabu_qubo
from qubo_portfolio import selection_qubo
from amortized import sel_obj
DEV = "cuda" if torch.cuda.is_available() else "cpu"; LAM, K = 0.5, 15


def feats(mu, S):
    sig = np.sqrt(np.clip(np.diag(S), 1e-12, None)); C = S / np.outer(sig, sig); ac = (np.abs(C).sum(1) - 1) / (len(mu) - 1)
    z = lambda x: (x - x.mean()) / (x.std() + 1e-9)
    return np.column_stack([z(mu), z(sig), z(ac), np.ones_like(mu)]).astype(np.float32), C


def knn(C, k=12):
    n = C.shape[0]; A = np.abs(C.copy()); np.fill_diagonal(A, -1); r, c = [], []
    for i in range(n):
        for j in np.argsort(-A[i])[:k]: r += [i, int(j)]; c += [int(j), i]
    return torch.tensor(np.array([r, c], np.int64), device=DEV)


class Net(nn.Module):
    def __init__(s, d, h=64, L=3, dr=0.24):
        super().__init__(); s.cs = nn.ModuleList(); c = d
        for _ in range(L): s.cs.append(SAGEConv(c, h)); c = h
        s.dr = dr; s.o = nn.Linear(h, 1)
    def forward(s, x, ei):
        for c in s.cs:
            x = F.relu(c(x, ei))
            if s.dr > 0: x = F.dropout(x, s.dr, s.training)
        return s.o(x).squeeze(-1)


def main():
    R = load_prices(SP100, "2005-01-01", "2024-12-31").pct_change().dropna().values; N = R.shape[1]
    lb, step = 252, 21; idx = list(range(lb, len(R), step)); sp = int(0.7 * len(idx))
    tr_i, te_i = idx[:sp], idx[sp:][:40]
    def est(t):
        w = R[t - lb:t]; mu = w.mean(0); S = np.cov(w, rowvar=False); return mu, 0.5 * (S + S.T) + 1e-8 * np.eye(N)
    def prep(ts):
        out = []
        for t in ts:
            mu, S = est(t); q = selection_qubo(mu, S, K, risk_aversion=LAM, return_weight=1 - LAM)
            r = tabu_qubo(q, num_reads=80, seed=0); xi = np.flatnonzero(np.asarray(r["x"]) > 0.5)
            sel = xi if len(xi) == K else np.argsort(-np.asarray(r["x"]))[:K]; f, C = feats(mu, S)
            out.append({"f": torch.tensor(f, device=DEV), "ei": knn(C), "mu": mu, "S": S, "sel": sel, "ref": sel_obj(sel, mu, S, K)})
        return out
    print(f"E31 sample efficiency: {len(tr_i)} train avail, {len(te_i)} test; building refs...", flush=True)
    TR = prep(tr_i); TE = prep(te_i)
    def train_eval(ntr):
        sub = TR[-ntr:]
        torch.manual_seed(0); np.random.seed(0)
        m = Net(4).to(DEV); opt = torch.optim.Adam(m.parameters(), lr=1.3e-3); pw = torch.tensor([(N - K) / K], device=DEV)
        labs = [torch.tensor(np.isin(np.arange(N), it["sel"]).astype(np.float32), device=DEV) for it in sub]
        for ep in range(500):
            m.train(); perm = np.random.permutation(len(sub))
            for bi in range(0, len(sub), 16):
                opt.zero_grad(); bl = 0.0
                for ii in perm[bi:bi + 16]:
                    bl = bl + F.binary_cross_entropy_with_logits(m(sub[ii]["f"], sub[ii]["ei"]), labs[ii], pos_weight=pw)
                (bl / max(1, len(perm[bi:bi + 16]))).backward(); opt.step()
        m.eval(); g = []
        for it in TE:
            with torch.no_grad(): p = m(it["f"], it["ei"]).cpu().numpy()
            g.append((sel_obj(np.argsort(-p)[:K], it["mu"], it["S"], K) - it["ref"]) / abs(it["ref"]) * 100 if abs(it["ref"]) > 1e-12 else 0.0)
        return float(np.mean(g))
    sizes = [5, 10, 20, 40, 80, len(TR)]; curve = []
    for n in sizes:
        g = train_eval(n); curve.append((n, g)); print(f"  n_train={n:>4}: gap {g:.3f}%", flush=True)
    json.dump(curve, open(os.path.join(HERE, "results", "e31_sample_efficiency.json"), "w"), indent=2)
    plt.figure(figsize=(7, 4.2)); plt.plot([c[0] for c in curve], [c[1] for c in curve], "o-")
    plt.axhline(0, ls=":", c="gray"); plt.xscale("log"); plt.xlabel("# labeled training windows (log)")
    plt.ylabel("amortized gap vs tabu (%)"); plt.title("Sample efficiency of the amortized GNN (S&P100)")
    plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(os.path.join(HERE, "..", "results", "figures", "fig_e31_sample_eff.png"), dpi=130)
    print("saved fig_e31_sample_eff.png", flush=True)


if __name__ == "__main__":
    main()
