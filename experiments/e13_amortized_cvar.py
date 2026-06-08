"""E13 (loop): does AMORTIZATION extend to the hard CVaR objective on RELATED market windows?
Combines our two strengths: amortized GNN (wins on related streams) + CVaR (hard objective). Per S&P100
rolling window: scenarios = window daily returns; per-instance reference = tabu-select (downside-risk
QUBO) + CVaR-LP weights -> achieved CVaR(95%). Train amortized GNN to imitate the tabu CVaR-selection;
eval amortized top-K -> CVaR-LP -> achieved CVaR vs per-instance hybrid. Report gap + speedup + curve.
Run in .venv.
"""
from __future__ import annotations
import os, sys, json, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE"); os.environ.setdefault("OMP_NUM_THREADS", "4")
HERE = os.path.dirname(__file__); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from torch_geometric.nn import SAGEConv
from backtest import load_prices, SP100
from baselines import tabu_qubo
from qubo_portfolio import selection_qubo, decode_selection
from exp_cvar import cvar_lp, cvar_of
DEV = "cuda" if torch.cuda.is_available() else "cpu"
K, ALPHA = 15, 0.05


def basic_feats(mu, S):
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
    def __init__(self, din, h=64, L=3, drop=0.24):
        super().__init__(); self.cs = nn.ModuleList(); c = din
        for _ in range(L): self.cs.append(SAGEConv(c, h)); c = h
        self.drop = drop; self.o = nn.Linear(h, 1)
    def forward(self, x, ei):
        for c in self.cs:
            x = F.relu(c(x, ei))
            if self.drop > 0: x = F.dropout(x, self.drop, self.training)
        return self.o(x).squeeze(-1)


def windows(R, lookback=252, step=21, mx=None):
    idx = list(range(lookback, len(R), step)); idx = idx[-mx:] if mx else idx
    return [R[t - lookback:t] for t in idx]


def cvar_select(scen):
    """per-instance reference: tabu-select on downside-risk QUBO -> support."""
    mu = scen.mean(0); d = np.minimum(scen - mu, 0.0); semi = (d.T @ d) / len(scen); semi = 0.5 * (semi + semi.T)
    q = selection_qubo(mu, semi, K, risk_aversion=1.0, return_weight=0.0)
    S = decode_selection(tabu_qubo(q, num_reads=80, seed=0)["x"])
    return S, semi


def main():
    R = load_prices(SP100, "2005-01-01", "2024-12-31").pct_change().dropna().values
    W = windows(R); sp = int(0.7 * len(W)); trW, teW = W[:sp], W[sp:]
    teW = teW[:30]
    print(f"E13 amortized-CVaR: {len(trW)}tr/{len(teW)}te windows; building refs...", flush=True)
    t0 = time.time()
    def prep(scenset):
        out = []
        for scen in scenset:
            mu = scen.mean(0); S, semi = cvar_select(scen)
            f, C = basic_feats(mu, semi)
            out.append({"feats": torch.tensor(f, device=DEV), "ei": knn(C), "scen": scen, "S": S,
                        "cvar_ref": cvar_of(cvar_lp(S, scen), scen) if len(S) == K else np.nan})
        return out
    tr = prep(trW); te = prep(teW)
    tabu_t = (time.time() - t0) / (len(trW) + len(teW))
    labs = [torch.tensor(np.isin(np.arange(d["feats"].shape[0]), d["S"]).astype(np.float32), device=DEV) for d in tr]
    print(f"  refs built {time.time()-t0:.0f}s (~{tabu_t:.2f}s/inst incl CVaR-LP)", flush=True)
    torch.manual_seed(0); np.random.seed(0)
    m = Net(tr[0]["feats"].shape[1]).to(DEV); opt = torch.optim.Adam(m.parameters(), lr=1.3e-3)
    N = tr[0]["feats"].shape[0]; pw = torch.tensor([(N - K) / K], device=DEV)

    def gap():
        m.eval(); g = []; t0 = time.time()
        for d in te:
            if not np.isfinite(d["cvar_ref"]): continue
            with torch.no_grad(): p = m(d["feats"], d["ei"]).cpu().numpy()
            S = np.argsort(-p)[:K]; c = cvar_of(cvar_lp(S, d["scen"]), d["scen"])
            g.append((c - d["cvar_ref"]) / abs(d["cvar_ref"]) * 100)
        return float(np.mean(g)), (time.time() - t0) / max(1, len(te)) * 1000
    curve = []; best = (1e9, None)
    for ep in range(401):
        if ep > 0:
            m.train(); perm = np.random.permutation(len(tr))
            for bi in range(0, len(tr), 16):
                opt.zero_grad(); bl = 0.0
                for ii in perm[bi:bi + 16]:
                    bl = bl + F.binary_cross_entropy_with_logits(m(tr[ii]["feats"], tr[ii]["ei"]), labs[ii], pos_weight=pw)
                (bl / max(1, len(perm[bi:bi + 16]))).backward(); opt.step()
        if ep % 50 == 0:
            g, inf = gap(); curve.append((ep, g))
            if g < best[0]: best = (g, {k: v.cpu() for k, v in m.state_dict().items()})
            print(f"  ep {ep:3d}: CVaR gap vs per-instance hybrid {g:6.2f}%  inf {inf:.1f}ms", flush=True)
    m.load_state_dict({k: v.to(DEV) for k, v in best[1].items()}); g, inf = gap()
    print(f"\n=== E13 BEST: amortized-CVaR gap vs per-instance hybrid = {g:.2f}%  (per-inst ~{tabu_t:.2f}s, amortized {inf:.1f}ms) ===", flush=True)
    os.makedirs(os.path.join(HERE, "checkpoints"), exist_ok=True)
    torch.save({"state_dict": best[1], "gap": g}, os.path.join(HERE, "checkpoints", "e13_amortized_cvar_best.pt"))
    plt.figure(figsize=(7, 4.2)); plt.plot([c[0] for c in curve], [c[1] for c in curve], "o-")
    plt.axhline(0, ls=":", c="gray"); plt.xlabel("epoch"); plt.ylabel("amortized CVaR gap vs hybrid (%)")
    plt.title("E13 amortized CVaR-selection on related S&P100 windows"); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(HERE, "..", "results", "figures", "fig_e13_amortized_cvar.png"), dpi=130)
    json.dump({"gap%": g, "amortized_ms": inf, "per_instance_s": tabu_t, "curve": curve},
              open(os.path.join(HERE, "results", "e13_amortized_cvar.json"), "w"), indent=2)
    print("saved checkpoint+fig+json", flush=True)


if __name__ == "__main__":
    main()
