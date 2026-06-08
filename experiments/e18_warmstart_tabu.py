"""E18 (loop): amortized GNN as a WARM-START for tabu (novel practical hybrid). Train amortized GNN on
in-sample S&P100 windows; on test windows compare on the selection-QUBO energy:
  (a) cold tabu (num_reads=80)            — full budget
  (b) amortized-alone (top-K)             — ms, no search
  (c) amortized warm-start + SHORT tabu (initial_state = GNN selection, num_reads=4)
Report mean energy gap vs cold-tabu best + wall-clock. If (c) ~ (a) at much lower time => the amortized
model accelerates tabu. Run in .venv.
"""
from __future__ import annotations
import os, sys, json, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE"); os.environ.setdefault("OMP_NUM_THREADS", "6")
HERE = os.path.dirname(__file__); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from backtest import load_prices, SP100
from baselines import tabu_qubo
from qubo_portfolio import selection_qubo, decode_selection
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
    def __init__(s, din, h=64, L=3, drop=0.24):
        super().__init__(); s.cs = nn.ModuleList(); c = din
        for _ in range(L): s.cs.append(SAGEConv(c, h)); c = h
        s.drop = drop; s.o = nn.Linear(h, 1)
    def forward(s, x, ei):
        for c in s.cs:
            x = F.relu(c(x, ei))
            if s.drop > 0: x = F.dropout(x, s.drop, s.training)
        return s.o(x).squeeze(-1)


def warm_tabu(q, x0, num_reads=4, seed=0):
    from tabu import TabuSampler
    t0 = time.time()
    init = [{i: int(x0[i]) for i in range(q.n)} for _ in range(num_reads)]
    res = TabuSampler().sample(q.to_dimod(), num_reads=num_reads, seed=seed, initial_states=init)
    b = res.first; x = np.array([b.sample[i] for i in range(q.n)], np.int8)
    return x, q.energy(x), time.time() - t0


def main():
    R = load_prices(SP100, "2005-01-01", "2024-12-31").pct_change().dropna().values
    N = R.shape[1]; lb, step = 252, 21
    idx = list(range(lb, len(R), step)); sp = int(0.7 * len(idx)); tr_i, te_i = idx[:sp], idx[sp:][:40]
    def est(t):
        w = R[t - lb:t]; mu = w.mean(0); S = np.cov(w, rowvar=False); return mu, 0.5 * (S + S.T) + 1e-8 * np.eye(N)
    print(f"E18 warm-start: N={N} K={K} {len(tr_i)}tr/{len(te_i)}te; training amortized...", flush=True)
    tr = []
    for t in tr_i:
        mu, S = est(t); q = selection_qubo(mu, S, K, risk_aversion=LAM, return_weight=1 - LAM)
        lab = decode_selection(tabu_qubo(q, num_reads=80, seed=0)["x"]); f, C = feats(mu, S)
        tr.append((torch.tensor(f, device=DEV), knn(C), torch.tensor(np.isin(np.arange(N), lab).astype(np.float32), device=DEV)))
    torch.manual_seed(0); np.random.seed(0)
    m = Net(4).to(DEV); opt = torch.optim.Adam(m.parameters(), lr=1.3e-3); pw = torch.tensor([(N - K) / K], device=DEV)
    for ep in range(400):
        m.train(); perm = np.random.permutation(len(tr))
        for bi in range(0, len(tr), 16):
            opt.zero_grad(); bl = 0.0
            for ii in perm[bi:bi + 16]:
                bl = bl + F.binary_cross_entropy_with_logits(m(tr[ii][0], tr[ii][1]), tr[ii][2], pos_weight=pw)
            (bl / max(1, len(perm[bi:bi + 16]))).backward(); opt.step()
    m.eval(); print("  trained.", flush=True)
    acc = {x: {"gap": [], "t": 0.0} for x in ["cold_tabu80", "amortized_alone", "warm_tabu4"]}
    for t in te_i:
        mu, S = est(t); q = selection_qubo(mu, S, K, risk_aversion=LAM, return_weight=1 - LAM)
        rc = tabu_qubo(q, num_reads=80, seed=0); ec = rc["energy"]; acc["cold_tabu80"]["t"] += rc["time"]
        t1 = time.time(); f, C = feats(mu, S)
        with torch.no_grad(): p = m(torch.tensor(f, device=DEV), knn(C)).cpu().numpy()
        xa = np.zeros(N, np.int8); xa[np.argsort(-p)[:K]] = 1; ea = q.energy(xa); ta = time.time() - t1
        acc["amortized_alone"]["t"] += ta
        xw, ew, tw = warm_tabu(q, xa, num_reads=4); acc["warm_tabu4"]["t"] += ta + tw
        best = min(ec, ea, ew)
        for nm, e in [("cold_tabu80", ec), ("amortized_alone", ea), ("warm_tabu4", ew)]:
            acc[nm]["gap"].append((e - best) / abs(best) * 100 if abs(best) > 1e-12 else 0.0)
    print(f"\n{'method':<18}{'gap%vsBest':>12}{'total_t(s)':>12}{'per_inst(s)':>12}", flush=True)
    out = {}
    for nm in acc:
        g = float(np.mean(acc[nm]["gap"])); tt = acc[nm]["t"]
        out[nm] = {"gap%": g, "total_s": tt, "per_inst_s": tt / len(te_i)}
        print(f"{nm:<18}{g:>12.3f}{tt:>12.2f}{tt/len(te_i):>12.3f}", flush=True)
    json.dump(out, open(os.path.join(HERE, "results", "e18_warmstart.json"), "w"), indent=2)
    print("saved", flush=True)


if __name__ == "__main__":
    main()
