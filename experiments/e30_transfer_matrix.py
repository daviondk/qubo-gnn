"""E30 (loop): cross-market TRANSFER MATRIX for the amortized GNN. Train on market X, evaluate (gap vs
per-instance tabu) on market Y, for X,Y in {S&P100, NASDAQ100, DOW, French49}. A clean generality result:
does the amortized selection heuristic transfer across markets of different size/composition? K=8 (< min N).
Run in .venv. Saves a 4x4 matrix + figure.
"""
import os, sys, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE"); os.environ.setdefault("OMP_NUM_THREADS", "6")
HERE = os.path.dirname(__file__); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from torch_geometric.nn import SAGEConv
from backtest import load_prices, SP100
from datasets import get_returns
from baselines import tabu_qubo
from qubo_portfolio import selection_qubo
from amortized import sel_obj
DEV = "cuda" if torch.cuda.is_available() else "cpu"; LAM, K = 0.5, 8
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


def windows(R, lookback=252, step=21, mx=50):
    idx = list(range(lookback, len(R), step))[-mx:]
    out = []
    for t in idx:
        w = R[t - lookback:t]; mu = w.mean(0); S = np.cov(w, rowvar=False); out.append((mu, 0.5 * (S + S.T) + 1e-8 * np.eye(R.shape[1])))
    return out


def prep(W):
    items = []
    for mu, S in W:
        q = selection_qubo(mu, S, K, risk_aversion=LAM, return_weight=1 - LAM)
        r = tabu_qubo(q, num_reads=80, seed=0); xi = np.flatnonzero(np.asarray(r["x"]) > 0.5)
        sel = xi if len(xi) == K else np.argsort(-np.asarray(r["x"]))[:K]
        f, C = feats(mu, S)
        items.append({"feats": torch.tensor(f, device=DEV), "ei": knn(C), "mu": mu, "S": S, "sel": sel,
                      "ref": sel_obj(sel, mu, S, K)})
    return items


def train_on(items, epochs=400):
    torch.manual_seed(0); np.random.seed(0)
    m = Net(4).to(DEV); opt = torch.optim.Adam(m.parameters(), lr=1.3e-3)
    labs = [torch.tensor(np.isin(np.arange(len(it["mu"])), it["sel"]).astype(np.float32), device=DEV) for it in items]
    N = len(items[0]["mu"]); pw = torch.tensor([(N - K) / K], device=DEV)
    for ep in range(epochs):
        m.train(); perm = np.random.permutation(len(items))
        for bi in range(0, len(items), 16):
            opt.zero_grad(); bl = 0.0
            for ii in perm[bi:bi + 16]:
                bl = bl + F.binary_cross_entropy_with_logits(m(items[ii]["feats"], items[ii]["ei"]), labs[ii], pos_weight=pw)
            (bl / max(1, len(perm[bi:bi + 16]))).backward(); opt.step()
    m.eval(); return m


def evalgap(m, items):
    g = []
    for it in items:
        with torch.no_grad(): p = m(it["feats"], it["ei"]).cpu().numpy()
        oa = sel_obj(np.argsort(-p)[:K], it["mu"], it["S"], K)
        g.append((oa - it["ref"]) / abs(it["ref"]) * 100 if abs(it["ref"]) > 1e-12 else 0.0)
    return float(np.mean(g))


def main():
    mkts = {}
    mkts["SP100"] = load_prices(SP100, "2005-01-01", "2024-12-31").pct_change().dropna().values
    mkts["NASDAQ"] = get_returns("nasdaq100").values
    mkts["DOW"] = load_prices(DOW, "2005-01-01", "2024-12-31").pct_change().dropna().values
    mkts["French49"] = get_returns("french49").values
    print("E30 transfer matrix; markets:", {k: v.shape[1] for k, v in mkts.items()}, flush=True)
    names = list(mkts); data = {}
    for nm in names:
        W = windows(mkts[nm]); sp = int(0.6 * len(W)); data[nm] = (prep(W[:sp]), prep(W[sp:]))
        print(f"  prepped {nm} ({mkts[nm].shape[1]} assets, {len(W)} windows)", flush=True)
    M = np.zeros((len(names), len(names)))
    models = {nm: train_on(data[nm][0]) for nm in names}
    for i, tr in enumerate(names):
        for j, te in enumerate(names):
            M[i, j] = evalgap(models[tr], data[te][1])
    print("\n=== Transfer matrix: gap%% vs per-instance tabu (rows=train, cols=test) ===")
    print("train\\test  " + "".join(f"{n:>10}" for n in names))
    for i, tr in enumerate(names):
        print(f"{tr:<11}" + "".join(f"{M[i,j]:>10.3f}" for j in range(len(names))), flush=True)
    json.dump({"names": names, "matrix": M.tolist()}, open(os.path.join(HERE, "results", "e30_transfer_matrix.json"), "w"), indent=2)
    plt.figure(figsize=(6, 5)); plt.imshow(M, cmap="RdYlGn_r", vmin=0, vmax=3)
    plt.colorbar(label="gap % vs tabu"); plt.xticks(range(len(names)), names, rotation=30); plt.yticks(range(len(names)), names)
    for i in range(len(names)):
        for j in range(len(names)): plt.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=9)
    plt.xlabel("test market"); plt.ylabel("train market"); plt.title("Amortized GNN cross-market transfer (gap% vs tabu)")
    plt.tight_layout(); plt.savefig(os.path.join(HERE, "..", "results", "figures", "fig_e30_transfer.png"), dpi=130)
    print("saved fig_e30_transfer.png", flush=True)


if __name__ == "__main__":
    main()
