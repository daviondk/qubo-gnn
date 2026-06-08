"""E15 (loop): show the amortization win in the ACTUAL INVESTMENT metric. Walk-forward S&P100 backtest
(K=15, 10bps): compare per-instance Tabu+reweight, per-instance GNN-QUBO+reweight, and AMORTIZED-GNN+
reweight (trained on the in-sample rebalances, applied OOS by single forward pass). Report OOS Sharpe/
Sortino/MaxDD/turnover + per-rebalance solve time. If amortized ~ per-instance Sharpe at ms inference =>
the amortization win shows up in the investment metric, not just QUBO gap. Run in .venv.
"""
from __future__ import annotations
import os, sys, json, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE"); os.environ.setdefault("OMP_NUM_THREADS", "4")
HERE = os.path.dirname(__file__); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from backtest import load_prices, SP100, perf_metrics
from baselines import tabu_qubo, convex_reweight
from qubo_portfolio import selection_qubo, decode_selection
from gnn_solver import solve_qubo_gnn, GNNHypers
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


def metrics(net, turns):
    d = np.asarray(net); m = perf_metrics(d); var = np.quantile(d, 0.05)
    m["cvar5"] = float(d[d <= var].mean()) if (d <= var).any() else float(var); m["turn"] = float(np.mean(turns)); return m


def main():
    R = load_prices(SP100, "2005-01-01", "2024-12-31").pct_change().dropna().values
    N = R.shape[1]; lb, step, cost = 252, 63, 10 / 1e4
    reb = list(range(lb, len(R) - step, step)); split = int(0.55 * len(reb)); tr_i, te_i = reb[:split], reb[split:]
    def est(t):
        w = R[t - lb:t]; mu = w.mean(0); S = np.cov(w, rowvar=False); S = 0.5 * (S + S.T) + 1e-8 * np.eye(N); return mu, S
    print(f"E15 amortized backtest: N={N} K={K} {len(tr_i)}tr/{len(te_i)}te rebalances", flush=True)
    # train amortized GNN on in-sample rebalances (tabu labels)
    t0 = time.time(); tr = []
    for t in tr_i:
        mu, S = est(t); q = selection_qubo(mu, S, K, risk_aversion=LAM, return_weight=1 - LAM)
        lab = decode_selection(tabu_qubo(q, num_reads=80, seed=0)["x"]); f, C = feats(mu, S)
        tr.append((torch.tensor(f, device=DEV), knn(C), torch.tensor(np.isin(np.arange(N), lab).astype(np.float32), device=DEV)))
    torch.manual_seed(0); np.random.seed(0)
    m = Net(tr[0][0].shape[1]).to(DEV); opt = torch.optim.Adam(m.parameters(), lr=1.3e-3); pw = torch.tensor([(N - K) / K], device=DEV)
    for ep in range(400):
        m.train(); perm = np.random.permutation(len(tr))
        for bi in range(0, len(tr), 16):
            opt.zero_grad(); bl = 0.0
            for ii in perm[bi:bi + 16]:
                bl = bl + F.binary_cross_entropy_with_logits(m(tr[ii][0], tr[ii][1]), tr[ii][2], pos_weight=pw)
            (bl / max(1, len(perm[bi:bi + 16]))).backward(); opt.step()
    m.eval(); print(f"  amortized trained {time.time()-t0:.0f}s", flush=True)
    gh = GNNHypers(model="qrf", epochs=1200, hidden=128, dim_embedding=24, n_layers=3, lr=1e-3, anneal_rate=0.0,
                   eval_every=50, patience=400, ls_passes=80, n_round_samples=16, refine_sa=True, refine_reads=20)
    acc = {x: {"net": [], "turn": [], "prev": None, "t": 0.0} for x in ["Tabu+rw", "GNN-QUBO+rw", "Amortized+rw"]}
    for t in te_i:
        mu, S = est(t); rn = R[t:t + step]; q = selection_qubo(mu, S, K, risk_aversion=LAM, return_weight=1 - LAM)
        sols = {}
        t1 = time.time(); St = decode_selection(tabu_qubo(q, num_reads=80, seed=0)["x"]); acc["Tabu+rw"]["t"] += time.time() - t1
        sols["Tabu+rw"] = St
        t1 = time.time(); Sg = decode_selection(solve_qubo_gnn(q, gh, device=DEV, seed=0)["x"]); acc["GNN-QUBO+rw"]["t"] += time.time() - t1
        sols["GNN-QUBO+rw"] = Sg
        f, C = feats(mu, S); t1 = time.time()
        with torch.no_grad(): p = m(torch.tensor(f, device=DEV), knn(C)).cpu().numpy()
        Sa = np.argsort(-p)[:K]; acc["Amortized+rw"]["t"] += time.time() - t1; sols["Amortized+rw"] = Sa
        for nm, Ssel in sols.items():
            w = convex_reweight(mu, S, Ssel, risk_aversion=LAM, return_weight=1 - LAM, eps=0.0, delta=UB) if len(Ssel) == K else np.ones(N) / N
            prev = acc[nm]["prev"]; turn = float(np.abs(w - prev).sum()) if prev is not None else 1.0
            daily = rn @ w; daily[0] -= cost * turn
            acc[nm]["net"].extend(daily.tolist()); acc[nm]["turn"].append(turn); acc[nm]["prev"] = w
    print(f"\n{'method':<14}{'Sharpe':>8}{'Sortino':>8}{'MaxDD':>8}{'Turn':>7}{'CVaR5':>8}{'solve/reb':>11}", flush=True)
    out = {}
    for nm in acc:
        mm = metrics(acc[nm]["net"], acc[nm]["turn"]); spr = acc[nm]["t"] / len(te_i)
        out[nm] = {**mm, "solve_per_reb_s": spr}
        print(f"{nm:<14}{mm['sharpe']:>8.3f}{mm['sortino']:>8.3f}{mm['maxdd']:>8.3f}{mm['turn']:>7.2f}{mm['cvar5']:>8.4f}{spr:>11.3f}", flush=True)
    json.dump(out, open(os.path.join(HERE, "results", "e15_amortized_backtest.json"), "w"), indent=2)
    print("saved e15_amortized_backtest.json", flush=True)


if __name__ == "__main__":
    main()
