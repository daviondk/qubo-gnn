"""E37 (loop): OOD transfer of the INVESTOR capstone. Train the decision-focused amortized GNN policy
(REINFORCE on realized net return) on S&P100, then deploy WITHOUT retraining on NASDAQ100 and DOW OOS
backtests. Does the investor-Sharpe advantage (vs equal-weight / per-instance optimizer) transfer across
markets? Run in .venv.
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


def backtest(m, W):  # decision-focused greedy
    net, turns, prev, N = [], [], None, W[0][0].shape[0]
    for mu, S, rn in W:
        f, C = feats(mu, S)
        with torch.no_grad():
            lg = m(torch.tensor(f, device=DEV), knn(C)); idx, _ = sample_topk(lg, K, greedy=True); w = wfrom(lg, idx, N).cpu().numpy()
        turn = float(np.abs(w - prev).sum()) if prev is not None else 1.0; d = rn @ w; d[0] -= cost * turn
        net.extend(d.tolist()); turns.append(turn); prev = w
    m_ = perf_metrics(np.asarray(net)); m_["turn"] = float(np.mean(turns)); return m_


def ew_backtest(W):
    net, N = [], W[0][0].shape[0]
    for mu, S, rn in W: net.extend((rn @ (np.ones(N) / N)).tolist())
    return perf_metrics(np.asarray(net))["sharpe"]


def main():
    R = load_prices(SP100, "2005-01-01", "2024-12-31").pct_change().dropna().values; N = R.shape[1]
    Wsp = windows(R); sp = int(0.55 * len(Wsp)); trW = Wsp[:sp]
    print(f"E37 investor-capstone OOD: train decision-focused on S&P100 ({len(trW)} windows)", flush=True)
    trw = [(torch.tensor(feats(mu, S)[0], device=DEV), knn(feats(mu, S)[1]), torch.tensor(rn.sum(0), dtype=torch.float32, device=DEV)) for mu, S, rn in trW]
    torch.manual_seed(0); np.random.seed(0)
    m = Net(4).to(DEV); opt = torch.optim.Adam(m.parameters(), lr=3e-3); base = 0.0
    for ep in range(300):
        opt.zero_grad(); loss = 0.0; rs = []
        for f, ei, rn in trw:
            lg = m(f, ei); idx, logp = sample_topk(lg, K); w = wfrom(lg, idx, N)
            r = float((w.detach() @ rn).item()) - cost * float(w.detach().abs().sum()); rs.append(r)
            loss = loss - (r - base) * logp - 0.01 * torch.distributions.Categorical(logits=lg).entropy()
        (loss / len(trw)).backward(); opt.step(); base = 0.9 * base + 0.1 * float(np.mean(rs))
    m.eval(); print("  trained on S&P100", flush=True)
    out = {}
    # in-dist S&P100 test + OOD NASDAQ, DOW (full series as OOS proxy, no retrain)
    tests = {"SP100-test": Wsp[sp:],
             "NASDAQ(OOD)": windows(get_returns("nasdaq100").values),
             "DOW(OOD)": windows(load_prices(DOW, "2005-01-01", "2024-12-31").pct_change().dropna().values)}
    for nm, W in tests.items():
        bt = backtest(m, W); ew = ew_backtest(W)
        out[nm] = {**bt, "equalweight_sharpe": ew}
        print(f"  {nm:<12}: decision-focused Sharpe {bt['sharpe']:.3f} (Sortino {bt['sortino']:.3f}, turn {bt['turn']:.2f}) vs EqualWeight {ew:.3f}", flush=True)
    json.dump(out, open(os.path.join(HERE, "results", "e37_decision_ood.json"), "w"), indent=2); print("saved", flush=True)


if __name__ == "__main__":
    main()
