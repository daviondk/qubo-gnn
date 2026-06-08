"""E23 (loop, paper rigor): multi-seed error bars on the headline amortization backtest (E15).
Per-instance tabu+reweight backtest computed ONCE (deterministic); amortized GNN trained over 5 seeds,
each backtested. Report amortized OOS Sharpe mean+/-std vs the fixed per-instance Sharpe. Run in .venv.
"""
from __future__ import annotations
import os, sys, json, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE"); os.environ.setdefault("OMP_NUM_THREADS", "6")
HERE = os.path.dirname(__file__); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from backtest import load_prices, SP100, perf_metrics
from baselines import tabu_qubo, convex_reweight
from qubo_portfolio import selection_qubo, decode_selection
DEV = "cuda" if torch.cuda.is_available() else "cpu"
LAM, K, UB = 0.5, 15, 0.25


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
    def __init__(s, din, h=64, L=3, drop=0.24):
        super().__init__(); s.cs = nn.ModuleList(); c = din
        for _ in range(L): s.cs.append(SAGEConv(c, h)); c = h
        s.drop = drop; s.o = nn.Linear(h, 1)
    def forward(s, x, ei):
        for c in s.cs:
            x = F.relu(c(x, ei))
            if s.drop > 0: x = F.dropout(x, s.drop, s.training)
        return s.o(x).squeeze(-1)


def sharpe_of(net, turns, cost):
    d = np.asarray(net); m = perf_metrics(d); return m["sharpe"], m["sortino"], m["maxdd"]


def backtest(weights_fn, te_i, est, R, step, cost, N):
    net, turns, prev = [], [], None
    for t in te_i:
        mu, S = est(t); rn = R[t:t + step]; w = weights_fn(mu, S)
        turn = float(np.abs(w - prev).sum()) if prev is not None else 1.0
        daily = rn @ w; daily[0] -= cost * turn; net.extend(daily.tolist()); turns.append(turn); prev = w
    return sharpe_of(net, turns, cost)


def main():
    R = load_prices(SP100, "2005-01-01", "2024-12-31").pct_change().dropna().values
    N = R.shape[1]; lb, step, cost = 252, 63, 10 / 1e4
    reb = list(range(lb, len(R) - step, step)); split = int(0.55 * len(reb)); tr_i, te_i = reb[:split], reb[split:]
    def est(t):
        w = R[t - lb:t]; mu = w.mean(0); S = np.cov(w, rowvar=False); return mu, 0.5 * (S + S.T) + 1e-8 * np.eye(N)
    print(f"E23 multi-seed amortized backtest: {len(tr_i)}tr/{len(te_i)}te", flush=True)
    # per-instance tabu reference (once) + amortized labels
    tr = []
    for t in tr_i:
        mu, S = est(t); q = selection_qubo(mu, S, K, risk_aversion=LAM, return_weight=1 - LAM)
        lab = decode_selection(tabu_qubo(q, num_reads=80, seed=0)["x"]); f, C = feats(mu, S)
        tr.append((torch.tensor(f, device=DEV), knn(C), torch.tensor(np.isin(np.arange(N), lab).astype(np.float32), device=DEV)))
    def tabu_w(mu, S):
        q = selection_qubo(mu, S, K, risk_aversion=LAM, return_weight=1 - LAM); Ss = decode_selection(tabu_qubo(q, num_reads=80, seed=0)["x"])
        return convex_reweight(mu, S, Ss, risk_aversion=LAM, return_weight=1 - LAM, eps=0.0, delta=UB) if len(Ss) == K else np.ones(N) / N
    ts, to, td = backtest(tabu_w, te_i, est, R, step, cost, N)
    print(f"  per-instance Tabu+rw: Sharpe {ts:.3f} Sortino {to:.3f} MaxDD {td:.3f}", flush=True)
    amort_sh = []
    for seed in range(5):
        torch.manual_seed(seed); np.random.seed(seed)
        m = Net(4).to(DEV); opt = torch.optim.Adam(m.parameters(), lr=1.3e-3); pw = torch.tensor([(N - K) / K], device=DEV)
        for ep in range(400):
            m.train(); perm = np.random.permutation(len(tr))
            for bi in range(0, len(tr), 16):
                opt.zero_grad(); bl = 0.0
                for ii in perm[bi:bi + 16]:
                    bl = bl + F.binary_cross_entropy_with_logits(m(tr[ii][0], tr[ii][1]), tr[ii][2], pos_weight=pw)
                (bl / max(1, len(perm[bi:bi + 16]))).backward(); opt.step()
        m.eval()
        def am_w(mu, S, m=m):
            f, C = feats(mu, S)
            with torch.no_grad(): p = m(torch.tensor(f, device=DEV), knn(C)).cpu().numpy()
            Sa = np.argsort(-p)[:K]
            return convex_reweight(mu, S, Sa, risk_aversion=LAM, return_weight=1 - LAM, eps=0.0, delta=UB)
        sh, so, dd = backtest(am_w, te_i, est, R, step, cost, N); amort_sh.append(sh)
        print(f"  seed {seed}: amortized Sharpe {sh:.3f}", flush=True)
    print(f"\n=== E23: amortized Sharpe {np.mean(amort_sh):.3f} +/- {np.std(amort_sh):.3f} (5 seeds) vs per-instance tabu {ts:.3f} ===", flush=True)
    json.dump({"per_instance_sharpe": ts, "amortized_sharpe_mean": float(np.mean(amort_sh)),
               "amortized_sharpe_std": float(np.std(amort_sh)), "seeds": amort_sh},
              open(os.path.join(HERE, "results", "e23_amortized_seeds_backtest.json"), "w"), indent=2)
    print("saved", flush=True)


if __name__ == "__main__":
    main()
