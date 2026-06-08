"""Exp 4: improve the SUPERVISED amortized model (the one that works, 1.05%).
Two levers: (1) richer node features, (2) edge-weighted GraphConv (covariance |corr| on edges enters
message passing) + pos-weighted BCE. Compare to the baseline (SAGE + basic features) on the same
S&P100 windows + OOD (NASDAQ100). Teacher = tabu (=exact here). Run in .venv.
"""
from __future__ import annotations
import os, sys, json, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE"); os.environ.setdefault("OMP_NUM_THREADS", "8")
HERE = os.path.dirname(__file__); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GraphConv
from amortized import sel_obj, DEVICE
from amortized_transfer import windows_from_returns
from backtest import load_prices, SP100
from datasets import get_returns
from baselines import tabu_qubo
from qubo_portfolio import selection_qubo
K = 15


def rich_feats(mu, Sigma, k=10):
    sig = np.sqrt(np.clip(np.diag(Sigma), 1e-12, None)); C = Sigma / np.outer(sig, sig)
    absC = np.abs(C); np.fill_diagonal(absC, 0.0)
    avgc = absC.sum(1) / (len(mu) - 1)
    topc = np.sort(absC, 1)[:, -k:].sum(1)
    def z(x): return (x - x.mean()) / (x.std() + 1e-9)
    def rank(x): return (np.argsort(np.argsort(x)) / (len(x) - 1))
    F_ = np.column_stack([z(mu), z(sig), z(avgc), rank(mu), rank(sig), z(topc), np.ones_like(mu)])
    return F_.astype(np.float32), C


def edges_w(C, k=10):
    n = C.shape[0]; A = np.abs(C.copy()); np.fill_diagonal(A, -1)
    r, c, w = [], [], []
    for i in range(n):
        for j in np.argsort(-A[i])[:k]:
            r += [i, int(j)]; c += [int(j), i]; w += [abs(C[i, int(j)]), abs(C[i, int(j)])]
    return np.array([r, c], np.int64), np.array(w, np.float32)


def build(mu, Sigma, rich):
    if rich:
        f, C = rich_feats(mu, Sigma); ei, ew = edges_w(C)
        return {"feats": torch.tensor(f, device=DEVICE),
                "ei": torch.tensor(ei, device=DEVICE), "ew": torch.tensor(ew, device=DEVICE),
                "mu": mu, "Sigma": Sigma}
    else:
        from amortized import build_instance
        d = build_instance(mu, Sigma, K)
        return {"feats": d["feats"], "ei": d["edge_index"], "ew": None, "mu": mu, "Sigma": Sigma}


class SageNet(nn.Module):
    def __init__(self, din, h=64, L=3):
        super().__init__(); self.cs = nn.ModuleList(); c = din
        for _ in range(L):
            self.cs.append(SAGEConv(c, h)); c = h
        self.o = nn.Linear(h, 1)

    def forward(self, x, ei, ew):
        for c in self.cs:
            x = F.relu(c(x, ei))
        return self.o(x).squeeze(-1)


class EdgeNet(nn.Module):
    def __init__(self, din, h=64, L=3):
        super().__init__(); self.cs = nn.ModuleList(); c = din
        for _ in range(L):
            self.cs.append(GraphConv(c, h)); c = h
        self.o = nn.Linear(h, 1)

    def forward(self, x, ei, ew):
        for c in self.cs:
            x = F.relu(c(x, ei, ew))
        return self.o(x).squeeze(-1)


def train_eval(make_model, rich, train_raw, labels, test_raw, ood_raw, epochs=600):
    train = [build(mu, S, rich) for mu, S in train_raw]
    torch.manual_seed(0); np.random.seed(0)
    m = make_model(train[0]["feats"].shape[1]).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    pw = torch.tensor([(71 - K) / K], device=DEVICE)
    for ep in range(epochs):
        m.train(); perm = np.random.permutation(len(train))
        for bi in range(0, len(train), 16):
            opt.zero_grad(); bl = 0.0
            for ii in perm[bi:bi + 16]:
                d = train[ii]; logit = m(d["feats"], d["ei"], d["ew"])
                bl = bl + F.binary_cross_entropy_with_logits(logit, labels[ii], pos_weight=pw)
            (bl / max(1, len(perm[bi:bi + 16]))).backward(); opt.step()
    m.eval()
    def ev(raw):
        gs = []
        for mu, S in raw:
            q = selection_qubo(mu, S, K, risk_aversion=0.5, return_weight=0.5)
            rt = tabu_qubo(q, num_reads=100, seed=0); St = np.flatnonzero(np.asarray(rt["x"]) > 0.5)
            ref = sel_obj(St, mu, S, K) if len(St) == K else sel_obj(np.argsort(-np.asarray(rt["x"]))[:K], mu, S, K)
            d = build(mu, S, rich)
            with torch.no_grad():
                p = m(d["feats"], d["ei"], d["ew"]).cpu().numpy()
            oa = sel_obj(np.argsort(-p)[:K], mu, S, K)
            gs.append((oa - ref) / abs(ref) * 100 if abs(ref) > 1e-12 else 0.0)
        return float(np.mean(gs)), float(np.median(gs))
    return ev(test_raw), ev(ood_raw)


def main():
    px = load_prices(SP100, "2005-01-01", "2024-12-31"); R = px.pct_change().dropna().values
    w = windows_from_returns(R); split = int(0.7 * len(w))
    train_raw, test_raw = w[:split], w[split:]
    ood_raw = windows_from_returns(get_returns("nasdaq100").values, max_w=40)
    print("computing tabu labels (once)...", flush=True)
    labels = []
    for mu, S in train_raw:
        q = selection_qubo(mu, S, K, risk_aversion=0.5, return_weight=0.5)
        r = tabu_qubo(q, num_reads=100, seed=0); lab = np.zeros(len(mu), np.float32)
        lab[np.flatnonzero(np.asarray(r["x"]) > 0.5)] = 1.0
        labels.append(torch.tensor(lab, device=DEVICE))
    out = {}
    print("\nmodel                    test mean/med   OOD(NASDAQ) mean/med", flush=True)
    for name, mk, rich in [("baseline SAGE+basic", lambda d: SageNet(d), False),
                           ("SAGE+rich-feats", lambda d: SageNet(d), True),
                           ("EdgeConv+rich+covW", lambda d: EdgeNet(d), True)]:
        (tm, tmed), (om, omed) = train_eval(mk, rich, train_raw, labels, test_raw, ood_raw)
        out[name] = {"test_mean%": tm, "test_med%": tmed, "ood_mean%": om, "ood_med%": omed}
        print(f"{name:<24} {tm:6.2f}/{tmed:5.2f}%     {om:6.2f}/{omed:5.2f}%", flush=True)
    json.dump(out, open(os.path.join(HERE, "results", "arch_lab4.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
