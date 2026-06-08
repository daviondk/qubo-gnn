"""E16 (loop): extend amortization to a THIRD objective — cardinality INDEX TRACKING (the THRML task).
Per S&P100 rolling window: index b = equal-weight of all N; per-instance reference = tabu-select on the
tracking QUBO (w-b)'Sigma(w-b) + min-TE reweight on the support -> achieved tracking error. Train an
amortized GNN to imitate the tabu tracking-selection; eval amortized top-K -> reweight -> TE gap vs
per-instance + speedup. If small => amortization extends across a 3rd objective. Run in .venv.
"""
from __future__ import annotations
import os, sys, json, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE"); os.environ.setdefault("OMP_NUM_THREADS", "4")
HERE = os.path.dirname(__file__); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, cvxpy as cp
from torch_geometric.nn import SAGEConv
from backtest import load_prices, SP100
from baselines import tabu_qubo
from qubo_portfolio import tracking_qubo, decode_selection
DEV = "cuda" if torch.cuda.is_available() else "cpu"
K = 15


def feats(mu, S, b=None):
    sig = np.sqrt(np.clip(np.diag(S), 1e-12, None)); C = S / np.outer(sig, sig)
    ac = (np.abs(C).sum(1) - 1) / (len(mu) - 1)
    def z(x): return (x - x.mean()) / (x.std() + 1e-9)
    bcorr = (C @ b) if b is not None else np.zeros_like(mu)
    beta = (S @ b) / (b @ S @ b + 1e-12) if b is not None else np.zeros_like(mu)
    return np.column_stack([z(mu), z(sig), z(ac), z(bcorr), z(beta), np.ones_like(mu)]).astype(np.float32), C


def knn(C, k=12):
    n = C.shape[0]; A = np.abs(C.copy()); np.fill_diagonal(A, -1); r, c = [], []
    for i in range(n):
        for j in np.argsort(-A[i])[:k]:
            r += [i, int(j)]; c += [int(j), i]
    return torch.tensor(np.array([r, c], np.int64), device=DEV)


class Net(nn.Module):
    def __init__(s, din, h=64, L=3, drop=0.24):
        super().__init__(); s.cs = nn.ModuleList(); c = din
        for _ in range(L): s.cs.append(SAGEConv(c, h)); c = h
        s.drop = drop; s.o = nn.Linear(h, 1)
    def forward(s, x, ei):
        for c in s.cs:
            x = F.relu(c(x, ei))
            if s.drop > 0: x = F.dropout(x, s.drop, s.training)
        return s.o(x).squeeze(-1)


def te_reweight(S, Sig, b):
    """min-tracking-error weights on support S vs index b."""
    n = Sig.shape[0]; S = np.asarray(S, int)
    if len(S) == 0: return b.copy()
    x = cp.Variable(len(S), nonneg=True); d = np.zeros(n)
    # minimize (x_S - b_S)'Sig_SS(x_S-b_S) + b_rest mass... track full index: w has support S, compare to b
    w = cp.Variable(n, nonneg=True)
    cons = [cp.sum(w) == 1, w[[i for i in range(n) if i not in set(S.tolist())]] == 0]
    cp.Problem(cp.Minimize(cp.quad_form(w - b, cp.psd_wrap(Sig)))).solve(solver=cp.CLARABEL)  # constraints below
    return None


def te(w, Sig, b):
    d = w - b; return float(np.sqrt(max(d @ Sig @ d, 0)))


def reweight(S, Sig, b):
    n = Sig.shape[0]; S = np.asarray(S, int)
    if len(S) != K: return b.copy()
    w = cp.Variable(K, nonneg=True)
    SS = Sig[np.ix_(S, S)]; bb = b[S]; brest = b.copy()
    # track full b: w_full has mass only on S; minimize (w_full-b)'Sig(w_full-b)
    full = np.zeros(n)
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(SS)) - 2 * (Sig[np.ix_(S, range(n))] @ b) @ w), [cp.sum(w) == 1])
    prob.solve(solver=cp.CLARABEL)
    out = np.zeros(n)
    if w.value is not None: out[S] = np.maximum(w.value, 0); out[S] /= max(out[S].sum(), 1e-9)
    else: out[S] = 1.0 / K
    return out


def main():
    R = load_prices(SP100, "2005-01-01", "2024-12-31").pct_change().dropna().values
    N = R.shape[1]; lb, step = 252, 21
    idx = list(range(lb, len(R), step)); sp = int(0.7 * len(idx))
    b = np.ones(N) / N   # equal-weight index
    def est(t):
        w = R[t - lb:t]; mu = w.mean(0); S = np.cov(w, rowvar=False); return mu, 0.5 * (S + S.T) + 1e-8 * np.eye(N)
    tr_i, te_i = idx[:sp], idx[sp:][:30]
    print(f"E17 tracking index-aware-feats: N={N} K={K} {len(tr_i)}tr/{len(te_i)}te; refs...", flush=True)
    t0 = time.time()
    def ref(t):
        mu, S = est(t); q = tracking_qubo(S, b, K); sel = decode_selection(tabu_qubo(q, num_reads=80, seed=0)["x"])
        return sel, (te(reweight(sel, S, b), S, b) if len(sel) == K else np.nan), mu, S
    trR = [ref(t) for t in tr_i]; teR = [ref(t) for t in te_i]
    tpi = (time.time() - t0) / (len(tr_i) + len(te_i))
    tr = [(torch.tensor(feats(r[2], r[3], b)[0], device=DEV), knn(feats(r[2], r[3], b)[1]),
           torch.tensor(np.isin(np.arange(N), r[0]).astype(np.float32), device=DEV)) for r in trR]
    print(f"  refs {time.time()-t0:.0f}s (~{tpi:.2f}s/inst)", flush=True)
    torch.manual_seed(0); np.random.seed(0)
    m = Net(6).to(DEV); opt = torch.optim.Adam(m.parameters(), lr=1.3e-3); pw = torch.tensor([(N - K) / K], device=DEV)
    best = (1e9, None)
    for ep in range(401):
        if ep > 0:
            m.train(); perm = np.random.permutation(len(tr))
            for bi in range(0, len(tr), 16):
                opt.zero_grad(); bl = 0.0
                for ii in perm[bi:bi + 16]:
                    bl = bl + F.binary_cross_entropy_with_logits(m(tr[ii][0], tr[ii][1]), tr[ii][2], pos_weight=pw)
                (bl / max(1, len(perm[bi:bi + 16]))).backward(); opt.step()
        if ep % 50 == 0:
            m.eval(); g = []
            for r in teR:
                if not np.isfinite(r[1]): continue
                f, C = feats(r[2], r[3], b)
                with torch.no_grad(): p = m(torch.tensor(f, device=DEV), knn(C)).cpu().numpy()
                te_a = te(reweight(np.argsort(-p)[:K], r[3], b), r[3], b)
                g.append((te_a - r[1]) / abs(r[1]) * 100)
            gg = float(np.mean(g))
            if gg < best[0]: best = (gg, ep)
            print(f"  ep {ep:3d}: TE gap vs per-instance {gg:6.2f}% (median {np.median(g):.2f}%)", flush=True)
    print(f"\n=== E17 BEST amortized index-tracking TE gap = {best[0]:.2f}% (ep{best[1]}); per-inst ~{tpi:.2f}s, amortized ms ===", flush=True)
    json.dump({"best_te_gap%": best[0], "per_instance_s": tpi}, open(os.path.join(HERE, "results", "e17_tracking_idxfeats.json"), "w"), indent=2)
    print("saved", flush=True)


if __name__ == "__main__":
    main()
