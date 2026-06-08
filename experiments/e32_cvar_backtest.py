"""E32 (loop): live CVaR-objective backtest (completes the CVaR contribution; parallels E15 for MV).
Walk-forward S&P100: per rebalance, scenarios = lookback returns; weights via CVaR-LP on the selected
support, selection by per-instance hybrid (tabu on downside-risk QUBO) vs amortized-CVaR GNN. Hold step
days; report realized OOS Sharpe/Sortino/MaxDD/CVaR(5%)/turnover + per-rebalance solve time. Tests whether
the amortized CVaR selector matches the per-instance hybrid in REALIZED investment metrics at ms inference.
Run in .venv.
"""
import os, sys, json, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE"); os.environ.setdefault("OMP_NUM_THREADS", "6")
HERE = os.path.dirname(__file__); sys.path.insert(0, os.path.join(HERE, "..", "src"))
from torch_geometric.nn import SAGEConv
from backtest import load_prices, SP100, perf_metrics
from baselines import tabu_qubo
from qubo_portfolio import selection_qubo, decode_selection
from exp_cvar import cvar_lp
DEV = "cuda" if torch.cuda.is_available() else "cpu"; K = 15


def basic_feats(mu, S):
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


def downside_select(scen):
    mu = scen.mean(0); d = np.minimum(scen - mu, 0.0); semi = (d.T @ d) / len(scen); semi = 0.5 * (semi + semi.T)
    q = selection_qubo(mu, semi, K, risk_aversion=1.0, return_weight=0.0)
    return decode_selection(tabu_qubo(q, num_reads=80, seed=0)["x"]), semi


def metrics(net, turns):
    d = np.asarray(net); m = perf_metrics(d); var = np.quantile(d, 0.05)
    m["cvar5"] = float(d[d <= var].mean()) if (d <= var).any() else float(var); m["turn"] = float(np.mean(turns)); return m


def main():
    R = load_prices(SP100, "2005-01-01", "2024-12-31").pct_change().dropna().values; N = R.shape[1]
    lb, step = 252, 63; reb = list(range(lb, len(R) - step, step)); split = int(0.55 * len(reb)); tr_i, te_i = reb[:split], reb[split:]
    print(f"E32 CVaR backtest: {len(tr_i)}tr/{len(te_i)}te", flush=True)
    # train amortized-CVaR (imitate downside-select) on in-sample
    tr = []
    for t in tr_i:
        scen = R[t - lb:t]; mu = scen.mean(0); sel, semi = downside_select(scen); f, C = basic_feats(mu, semi)
        tr.append((torch.tensor(f, device=DEV), knn(C), torch.tensor(np.isin(np.arange(N), sel).astype(np.float32), device=DEV)))
    torch.manual_seed(0); np.random.seed(0)
    m = Net(4).to(DEV); opt = torch.optim.Adam(m.parameters(), lr=1.3e-3); pw = torch.tensor([(N - K) / K], device=DEV)
    for ep in range(400):
        m.train(); perm = np.random.permutation(len(tr))
        for bi in range(0, len(tr), 16):
            opt.zero_grad(); bl = 0.0
            for ii in perm[bi:bi + 16]:
                bl = bl + F.binary_cross_entropy_with_logits(m(tr[ii][0], tr[ii][1]), tr[ii][2], pos_weight=pw)
            (bl / max(1, len(perm[bi:bi + 16]))).backward(); opt.step()
    m.eval(); print("  amortized-CVaR trained", flush=True)
    acc = {x: {"net": [], "turn": [], "prev": None, "t": 0.0} for x in ["PerInst-CVaR", "Amortized-CVaR"]}
    for t in te_i:
        scen = R[t - lb:t]; mu = scen.mean(0); rn = R[t:t + step]
        t1 = time.time(); Sp, semi = downside_select(scen); acc["PerInst-CVaR"]["t"] += time.time() - t1
        f, C = basic_feats(mu, semi); t1 = time.time()
        with torch.no_grad(): p = m(torch.tensor(f, device=DEV), knn(C)).cpu().numpy()
        Sa = np.argsort(-p)[:K]; acc["Amortized-CVaR"]["t"] += time.time() - t1
        for nm, Ssel in [("PerInst-CVaR", Sp), ("Amortized-CVaR", Sa)]:
            w = cvar_lp(Ssel, scen) if len(Ssel) == K else np.ones(N) / N
            prev = acc[nm]["prev"]; turn = float(np.abs(w - prev).sum()) if prev is not None else 1.0
            daily = rn @ w; daily[0] -= 10 / 1e4 * turn; acc[nm]["net"].extend(daily.tolist()); acc[nm]["turn"].append(turn); acc[nm]["prev"] = w
    print(f"\n{'method':<16}{'Sharpe':>8}{'Sortino':>8}{'MaxDD':>8}{'Turn':>7}{'CVaR5':>9}{'solve/reb':>11}", flush=True)
    out = {}
    for nm in acc:
        mm = metrics(acc[nm]["net"], acc[nm]["turn"]); spr = acc[nm]["t"] / len(te_i)
        out[nm] = {**mm, "solve_per_reb_s": spr}
        print(f"{nm:<16}{mm['sharpe']:>8.3f}{mm['sortino']:>8.3f}{mm['maxdd']:>8.3f}{mm['turn']:>7.2f}{mm['cvar5']:>9.4f}{spr:>11.4f}", flush=True)
    json.dump(out, open(os.path.join(HERE, "results", "e32_cvar_backtest.json"), "w"), indent=2); print("saved", flush=True)


if __name__ == "__main__":
    main()
