"""E28 (loop): amortized deployable system on a SECOND real market (DOW 30) — does the headline
(amortized = per-instance OOS Sharpe at ms inference) generalize beyond S&P100? Walk-forward backtest,
K=8, 10bps: per-instance tabu+reweight vs amortized+reweight. Run in .venv.
"""
import os, sys, json, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE"); os.environ.setdefault("OMP_NUM_THREADS", "6")
HERE = os.path.dirname(__file__); sys.path.insert(0, os.path.join(HERE, "..", "src"))
from torch_geometric.nn import SAGEConv
from backtest import load_prices, perf_metrics
from baselines import tabu_qubo, convex_reweight
from qubo_portfolio import selection_qubo, decode_selection
DEV = "cuda" if torch.cuda.is_available() else "cpu"; LAM, K, UB = 0.5, 8, 0.30
DOW = ["AAPL","MSFT","JPM","V","WMT","JNJ","PG","HD","CVX","KO","MRK","CSCO","MCD","CRM","DIS",
       "AXP","IBM","GS","CAT","VZ","AMGN","HON","BA","MMM","NKE","TRV","AMZN","INTC"]


def feats(mu, S):
    sig = np.sqrt(np.clip(np.diag(S), 1e-12, None)); C = S / np.outer(sig, sig); ac = (np.abs(C).sum(1) - 1) / (len(mu) - 1)
    z = lambda x: (x - x.mean()) / (x.std() + 1e-9)
    return np.column_stack([z(mu), z(sig), z(ac), np.ones_like(mu)]).astype(np.float32), C


def knn(C, k=10):
    n = C.shape[0]; A = np.abs(C.copy()); np.fill_diagonal(A, -1); r, c = [], []
    for i in range(n):
        for j in np.argsort(-A[i])[:min(k, n - 1)]: r += [i, int(j)]; c += [int(j), i]
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


def metrics(net, turns):
    d = np.asarray(net); m = perf_metrics(d); var = np.quantile(d, 0.05)
    m["cvar5"] = float(d[d <= var].mean()) if (d <= var).any() else float(var); m["turn"] = float(np.mean(turns)); return m


def main():
    px = load_prices(DOW, "2005-01-01", "2024-12-31"); R = px.pct_change().dropna().values; N = R.shape[1]
    lb, step, cost = 252, 63, 10 / 1e4
    reb = list(range(lb, len(R) - step, step)); split = int(0.55 * len(reb)); tr_i, te_i = reb[:split], reb[split:]
    def est(t):
        w = R[t - lb:t]; mu = w.mean(0); S = np.cov(w, rowvar=False); return mu, 0.5 * (S + S.T) + 1e-8 * np.eye(N)
    print(f"E28 DOW amortized backtest: N={N} K={K} {len(tr_i)}tr/{len(te_i)}te", flush=True)
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
    m.eval()
    acc = {x: {"net": [], "turn": [], "prev": None, "t": 0.0} for x in ["Tabu+rw", "Amortized+rw"]}
    for t in te_i:
        mu, S = est(t); rn = R[t:t + step]; q = selection_qubo(mu, S, K, risk_aversion=LAM, return_weight=1 - LAM)
        t1 = time.time(); St = decode_selection(tabu_qubo(q, num_reads=80, seed=0)["x"]); acc["Tabu+rw"]["t"] += time.time() - t1
        f, C = feats(mu, S); t1 = time.time()
        with torch.no_grad(): p = m(torch.tensor(f, device=DEV), knn(C)).cpu().numpy()
        Sa = np.argsort(-p)[:K]; acc["Amortized+rw"]["t"] += time.time() - t1
        for nm, Ssel in [("Tabu+rw", St), ("Amortized+rw", Sa)]:
            w = convex_reweight(mu, S, Ssel, risk_aversion=LAM, return_weight=1 - LAM, eps=0.0, delta=UB) if len(Ssel) == K else np.ones(N) / N
            prev = acc[nm]["prev"]; turn = float(np.abs(w - prev).sum()) if prev is not None else 1.0
            daily = rn @ w; daily[0] -= cost * turn; acc[nm]["net"].extend(daily.tolist()); acc[nm]["turn"].append(turn); acc[nm]["prev"] = w
    print(f"\n{'method':<14}{'Sharpe':>8}{'Sortino':>8}{'MaxDD':>8}{'Turn':>7}{'solve/reb':>11}", flush=True)
    out = {}
    for nm in acc:
        mm = metrics(acc[nm]["net"], acc[nm]["turn"]); spr = acc[nm]["t"] / len(te_i)
        out[nm] = {**mm, "solve_per_reb_s": spr}
        print(f"{nm:<14}{mm['sharpe']:>8.3f}{mm['sortino']:>8.3f}{mm['maxdd']:>8.3f}{mm['turn']:>7.2f}{spr:>11.4f}", flush=True)
    json.dump(out, open(os.path.join(HERE, "results", "e28_amortized_dow.json"), "w"), indent=2); print("saved", flush=True)


if __name__ == "__main__":
    main()
