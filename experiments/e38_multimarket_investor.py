"""E38 (loop): can MULTI-MARKET training fix the investor-mode OOD transfer failure (E37)? Train the
decision-focused amortized GNN on POOLED windows from S&P100 + NASDAQ100 + French49, then test on a fully
held-out market (DOW) backtest. Compare DF Sharpe vs equal-weight and vs the single-market baseline
(E37: DOW DF 0.797 ~ EW 0.811). If multi-market training -> DF beats EW on held-out DOW, the boundary is
(partly) fixed. Run in .venv.
"""
import os, sys, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE"); os.environ.setdefault("OMP_NUM_THREADS", "6")
HERE = os.path.dirname(__file__); sys.path.insert(0, os.path.join(HERE, "..", "src"))
from torch_geometric.nn import SAGEConv
from backtest import load_prices, SP100, perf_metrics
from datasets import get_returns
DEV = "cuda" if torch.cuda.is_available() else "cpu"; K, UB, cost = 15, 0.25, 10 / 1e4
DOW = ["AAPL","MSFT","JPM","V","WMT","JNJ","PG","HD","CVX","KO","MRK","CSCO","MCD","CRM","DIS",
       "AXP","IBM","GS","CAT","VZ","AMGN","HON","BA","MMM","NKE","TRV","AMZN","INTC"]


def feats(mu, S):
    sig = np.sqrt(np.clip(np.diag(S), 1e-12, None)); C = S / np.outer(sig, sig); ac = (np.abs(C).sum(1) - 1) / (len(mu) - 1)
    z = lambda x: (x - x.mean()) / (x.std() + 1e-9)
    return np.column_stack([z(mu), z(sig), z(ac), np.ones_like(mu)]).astype(np.float32), C


def knn(C, k=12):
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


def sample_topk(lg, k, greedy=False):
    l = lg.clone(); ch = []; lp = torch.tensor(0.0, device=l.device)
    for _ in range(k):
        p = F.log_softmax(l, 0); i = int(torch.argmax(l)) if greedy else int(torch.distributions.Categorical(logits=l).sample())
        lp = lp + p[i]; ch.append(i); l = l.clone(); l[i] = -1e9
    return ch, lp


def wfrom(lg, idx, n):
    w = torch.zeros(n, device=lg.device); w[idx] = torch.softmax(lg[idx], 0)
    for _ in range(8):
        w = torch.clamp(w, 0, UB); s = w.sum()
        if s > 0: w = w / s
    return w


def windows(R, lb=252, step=63):
    idx = list(range(lb, len(R) - step, step)); out = []
    for t in idx:
        w = R[t - lb:t]; mu = w.mean(0); S = np.cov(w, rowvar=False); out.append((mu, 0.5 * (S + S.T) + 1e-8 * np.eye(R.shape[1]), R[t:t + step]))
    return out


def to_train(W):
    return [(torch.tensor(feats(mu, S)[0], device=DEV), knn(feats(mu, S)[1]), torch.tensor(rn.sum(0), dtype=torch.float32, device=DEV), mu.shape[0]) for mu, S, rn in W]


def train(trw, epochs=300):
    torch.manual_seed(0); np.random.seed(0)
    m = Net(4).to(DEV); opt = torch.optim.Adam(m.parameters(), lr=3e-3); base = 0.0
    for ep in range(epochs):
        opt.zero_grad(); loss = 0.0; rs = []; perm = np.random.permutation(len(trw))
        for ii in perm:
            f, ei, rn, n = trw[ii]
            lg = m(f, ei); idx, logp = sample_topk(lg, K); w = wfrom(lg, idx, n)
            r = float((w.detach() @ rn).item()) - cost * float(w.detach().abs().sum()); rs.append(r)
            loss = loss - (r - base) * logp - 0.01 * torch.distributions.Categorical(logits=lg).entropy()
        (loss / len(trw)).backward(); opt.step(); base = 0.9 * base + 0.1 * float(np.mean(rs))
    m.eval(); return m


def backtest(m, W):
    net, turns, prev, N = [], [], None, W[0][0].shape[0]
    for mu, S, rn in W:
        f, C = feats(mu, S)
        with torch.no_grad():
            lg = m(torch.tensor(f, device=DEV), knn(C)); idx, _ = sample_topk(lg, K, greedy=True); w = wfrom(lg, idx, N).cpu().numpy()
        turn = float(np.abs(w - prev).sum()) if prev is not None else 1.0; d = rn @ w; d[0] -= cost * turn
        net.extend(d.tolist()); turns.append(turn); prev = w
    mm = perf_metrics(np.asarray(net)); mm["turn"] = float(np.mean(turns)); return mm


def main():
    sp = load_prices(SP100, "2005-01-01", "2024-12-31").pct_change().dropna().values
    nq = get_returns("nasdaq100").values; fr = get_returns("french49").values
    dow = load_prices(DOW, "2005-01-01", "2024-12-31").pct_change().dropna().values
    pooled = to_train(windows(sp)) + to_train(windows(nq)) + to_train(windows(fr))
    print(f"E38 multi-market investor: pooled train {len(pooled)} windows (SP100+NASDAQ+French); held-out test DOW", flush=True)
    m_multi = train(pooled)
    m_single = train(to_train(windows(sp)))  # single-market baseline (S&P100 only)
    Wdow = windows(dow); ewd = perf_metrics(np.concatenate([rn @ (np.ones(rn.shape[1]) / rn.shape[1]) for _, _, rn in Wdow]))["sharpe"]
    bm = backtest(m_multi, Wdow); bs = backtest(m_single, Wdow)
    print(f"  held-out DOW: multi-market DF Sharpe {bm['sharpe']:.3f} (turn {bm['turn']:.2f}) | single-market(SP100) DF {bs['sharpe']:.3f} | EqualWeight {ewd:.3f}", flush=True)
    out = {"dow_multimarket": bm["sharpe"], "dow_singlemarket": bs["sharpe"], "dow_equalweight": ewd, "multi_turn": bm["turn"]}
    json.dump(out, open(os.path.join(HERE, "results", "e38_multimarket_investor.json"), "w"), indent=2); print("saved", flush=True)


if __name__ == "__main__":
    main()
