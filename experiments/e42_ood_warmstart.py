"""E42 (loop): OOD warm-start. Train the optimizer-mode (imitation) amortized GNN on S&P100, then use it
WITHOUT retraining as the tabu warm-start initial state on NASDAQ100 and DOW instances. Does the universal
init accelerate tabu on a NEW market (warm vs cold)? Combines two robust optimizer-mode properties
(OOD transfer + warm-start). Run in .venv.
"""
import os, sys, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE"); os.environ.setdefault("OMP_NUM_THREADS", "6")
HERE = os.path.dirname(__file__); sys.path.insert(0, os.path.join(HERE, "..", "src"))
from torch_geometric.nn import SAGEConv
from backtest import load_prices, SP100
from datasets import get_returns
from baselines import tabu_qubo
from qubo_portfolio import selection_qubo, decode_selection
from tabu import TabuSampler
DEV = "cuda" if torch.cuda.is_available() else "cpu"; LAM, K = 0.5, 15
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


def windows(R, lb=252, step=21, mx=30):
    idx = list(range(lb, len(R), step))[-mx:]; out = []
    for t in idx:
        w = R[t - lb:t]; mu = w.mean(0); S = np.cov(w, rowvar=False); out.append((mu, 0.5 * (S + S.T) + 1e-8 * np.eye(R.shape[1])))
    return out


def warm(q, x0, k):
    init = [{i: int(x0[i]) for i in range(q.n)} for _ in range(k)]
    res = TabuSampler().sample(q.to_dimod(), num_reads=k, seed=0, initial_states=init); b = res.first
    return q.energy(np.array([b.sample[i] for i in range(q.n)], np.int8))


def main():
    Rsp = load_prices(SP100, "2005-01-01", "2024-12-31").pct_change().dropna().values; N = Rsp.shape[1]
    Wsp = windows(Rsp, mx=120)
    print(f"E42 OOD warm-start: train optimizer-amortized on S&P100 ({len(Wsp)} windows)", flush=True)
    TR = []
    for mu, S in Wsp:
        q = selection_qubo(mu, S, K, risk_aversion=LAM, return_weight=1 - LAM)
        lab = decode_selection(tabu_qubo(q, num_reads=80, seed=0)["x"]); f, C = feats(mu, S)
        TR.append((torch.tensor(f, device=DEV), knn(C), torch.tensor(np.isin(np.arange(N), lab).astype(np.float32), device=DEV)))
    torch.manual_seed(0); np.random.seed(0)
    m = Net(4).to(DEV); opt = torch.optim.Adam(m.parameters(), lr=1.3e-3); pw = torch.tensor([(N - K) / K], device=DEV)
    for ep in range(400):
        m.train(); perm = np.random.permutation(len(TR))
        for bi in range(0, len(TR), 16):
            opt.zero_grad(); bl = 0.0
            for ii in perm[bi:bi + 16]: bl = bl + F.binary_cross_entropy_with_logits(m(TR[ii][0], TR[ii][1]), TR[ii][2], pos_weight=pw)
            (bl / max(1, len(perm[bi:bi + 16]))).backward(); opt.step()
    m.eval(); print("  trained on S&P100", flush=True)
    out = {}
    markets = {"NASDAQ": get_returns("nasdaq100").values, "DOW": load_prices(DOW, "2005-01-01", "2024-12-31").pct_change().dropna().values}
    for nm, R in markets.items():
        W = windows(R, mx=25); nn_ = R.shape[1]; cold, warm4, amort = [], [], []
        for mu, S in W:
            q = selection_qubo(mu, S, K, risk_aversion=LAM, return_weight=1 - LAM)
            best = tabu_qubo(q, num_reads=120, seed=0)["energy"]
            f, C = feats(mu, S)
            with torch.no_grad(): p = m(torch.tensor(f, device=DEV), knn(C)).cpu().numpy()
            xa = np.zeros(nn_, np.int8); xa[np.argsort(-p)[:K]] = 1
            ea = q.energy(xa); ec = tabu_qubo(q, num_reads=4, seed=0)["energy"]; ew = warm(q, xa, 4)
            g = lambda e: (e - best) / abs(best) * 100 if abs(best) > 1e-12 else 0.0
            amort.append(g(ea)); cold.append(g(ec)); warm4.append(g(ew))
        out[nm] = {"amortized_alone": float(np.mean(amort)), "cold_tabu4": float(np.mean(cold)), "warm_tabu4": float(np.mean(warm4))}
        print(f"  {nm} (OOD, no retrain): amortized-alone {np.mean(amort):.3f}% | cold-tabu4 {np.mean(cold):.3f}% | WARM-tabu4 {np.mean(warm4):.3f}%", flush=True)
    json.dump(out, open(os.path.join(HERE, "results", "e42_ood_warmstart.json"), "w"), indent=2); print("saved", flush=True)


if __name__ == "__main__":
    main()
