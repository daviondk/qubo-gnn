"""Amortized GNN for portfolio cardinality selection.

Train ONE shared GNN on a distribution of portfolio QUBO instances (rolling windows of real S&P100
returns). At test time, selection is a single forward pass (milliseconds) -- no per-instance
optimization. Compare to per-instance tabu (strong baseline) on a STREAM of held-out instances:
the win = amortized matches tabu quality at far lower per-instance time.

Unsupervised amortized loss: mean over the instance distribution of the relaxed selection-QUBO energy
p^T Q p  (Q = cardinality selection QUBO, per-instance, globally scaled). Inference: top-K of the GNN
probabilities. Run in .venv.
"""
from __future__ import annotations

import sys, os, json, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "8")
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

from backtest import load_prices, SP100
from qubo_portfolio import selection_qubo
from baselines import tabu_qubo, convex_reweight

LAM = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def corr_graph(Sigma, k=8):
    d = np.sqrt(np.clip(np.diag(Sigma), 1e-12, None))
    C = Sigma / np.outer(d, d)
    n = C.shape[0]
    A = np.abs(C.copy()); np.fill_diagonal(A, -1)
    rows, cols = [], []
    for i in range(n):
        nb = np.argsort(-A[i])[:k]
        for j in nb:
            rows += [i, j]; cols += [j, i]
    ei = np.array([rows, cols], dtype=np.int64)
    return ei


def node_feats(mu, Sigma):
    sig = np.sqrt(np.clip(np.diag(Sigma), 1e-12, None))
    d = sig
    C = Sigma / np.outer(d, d)
    avgcorr = (np.abs(C).sum(1) - 1) / (len(mu) - 1)
    def z(x): return (x - x.mean()) / (x.std() + 1e-9)
    return np.column_stack([z(mu), z(sig), z(avgcorr), np.ones_like(mu)]).astype(np.float32)


def build_instance(mu, Sigma, K, k=8):
    q = selection_qubo(mu, Sigma, K, risk_aversion=LAM, return_weight=1 - LAM)
    Q = q.Q.copy()
    off = Q - np.diag(np.diag(Q)); scale = np.mean(np.abs(off[off != 0])) or 1.0
    return {
        "Qn": torch.tensor(Q / scale, dtype=torch.float32, device=DEVICE),
        "edge_index": torch.tensor(corr_graph(Sigma, k), dtype=torch.long, device=DEVICE),
        "feats": torch.tensor(node_feats(mu, Sigma), dtype=torch.float32, device=DEVICE),
        "mu": mu, "Sigma": Sigma, "q": q, "K": K,
    }


class AmortizedGNN(nn.Module):
    def __init__(self, in_dim, hidden=64, layers=3):
        super().__init__()
        self.convs = nn.ModuleList()
        cur = in_dim
        for _ in range(layers):
            self.convs.append(SAGEConv(cur, hidden, aggr="mean")); cur = hidden
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, edge_index):
        for c in self.convs:
            x = F.relu(c(x, edge_index))
        return torch.sigmoid(self.head(x)).squeeze(-1)


def sel_obj(S, mu, Sigma, K):
    z = np.zeros(len(mu)); z[list(S)] = 1
    return float((LAM / K**2) * (z @ Sigma @ z) - ((1 - LAM) / K) * (mu @ z))


def main():
    K = int(os.environ.get("K", "15"))
    px = load_prices(SP100, "2005-01-01", "2024-12-31")
    R = px.pct_change().dropna().values
    dates = px.pct_change().dropna().index
    N = R.shape[1]
    lookback, step = 252, 21
    idxs = list(range(lookback, len(R), step))
    insts = []
    for t in idxs:
        w = R[t - lookback:t]
        mu = w.mean(0); Sig = np.cov(w, rowvar=False); Sig = 0.5 * (Sig + Sig.T) + 1e-8 * np.eye(N)
        insts.append((t, mu, Sig))
    split = int(0.7 * len(insts))
    train_raw, test_raw = insts[:split], insts[split:]
    print(f"N={N} K={K} | {len(train_raw)} train windows, {len(test_raw)} test windows | "
          f"test {dates[test_raw[0][0]].date()}->{dates[test_raw[-1][0]].date()}", flush=True)

    train = [build_instance(mu, Sig, K) for _, mu, Sig in train_raw]
    test = [build_instance(mu, Sig, K) for _, mu, Sig in test_raw]

    mode = os.environ.get("MODE", "sup")   # sup = imitate tabu (robust) | unsup = relaxed-energy
    torch.manual_seed(0); np.random.seed(0)
    model = AmortizedGNN(train[0]["feats"].shape[1], hidden=64, layers=3).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    if mode == "sup":
        print("generating tabu labels for train windows (one-time)...", flush=True)
        tl0 = time.time()
        for d in train:
            r = tabu_qubo(d["q"], num_reads=100, seed=0)
            lab = np.zeros(N, dtype=np.float32); S = np.flatnonzero(np.asarray(r["x"]) > 0.5)
            lab[S] = 1.0
            d["label"] = torch.tensor(lab, device=DEVICE)
        print(f"  labels in {time.time()-tl0:.0f}s", flush=True)

    t0 = time.time()
    for epoch in range(400):
        model.train(); perm = np.random.permutation(len(train)); loss_acc = 0.0
        for bi in range(0, len(train), 16):
            opt.zero_grad(); batch_loss = 0.0
            for ii in perm[bi:bi + 16]:
                d = train[ii]; p = model(d["feats"], d["edge_index"])
                if mode == "sup":
                    batch_loss = batch_loss + F.binary_cross_entropy(p.clamp(1e-6, 1 - 1e-6), d["label"])
                else:
                    batch_loss = batch_loss + p @ (d["Qn"] @ p)
            batch_loss = batch_loss / max(1, len(perm[bi:bi + 16]))
            batch_loss.backward(); opt.step(); loss_acc += float(batch_loss)
        if epoch % 50 == 0:
            print(f"  epoch {epoch} train_loss {loss_acc:.3f}", flush=True)
    train_time = time.time() - t0
    print(f"amortized training ({mode}): {train_time:.0f}s (one-time)", flush=True)

    # ---- eval on held-out stream ----
    model.eval()
    amort_gap, amort_inf_t, tabu_gap, tabu_t = [], [], [], []
    ref_objs = []
    for d in test:
        mu, Sig, q, Kk = d["mu"], d["Sigma"], d["q"], d["K"]
        # reference = per-instance tabu high-budget (optimal at N~71 per ablation)
        rt = tabu_qubo(q, num_reads=200, seed=0); St = np.flatnonzero(np.asarray(rt["x"]) > 0.5)
        ref = sel_obj(St, mu, Sig, Kk) if len(St) == Kk else min(
            sel_obj(np.argsort(-np.asarray(rt["x"]))[:Kk], mu, Sig, Kk), 0)
        ref_objs.append(ref); tabu_t.append(rt["time"])
        # amortized: one forward pass
        ti = time.time()
        with torch.no_grad():
            p = model(d["feats"], d["edge_index"]).cpu().numpy()
        S = np.argsort(-p)[:Kk]
        amort_inf_t.append(time.time() - ti)
        oa = sel_obj(S, mu, Sig, Kk)
        amort_gap.append((oa - ref) / abs(ref) * 100 if abs(ref) > 1e-12 else 0.0)
    res = {
        "amort_mean_gap_vs_tabu_%": float(np.mean(amort_gap)),
        "amort_median_gap_%": float(np.median(amort_gap)),
        "amort_inference_ms": float(np.mean(amort_inf_t) * 1000),
        "tabu_solve_s": float(np.mean(tabu_t)),
        "speedup_x": float(np.mean(tabu_t) / (np.mean(amort_inf_t) + 1e-9)),
        "train_time_s": train_time, "n_test": len(test),
    }
    print("\n=== AMORTIZED vs per-instance Tabu (held-out S&P100 windows) ===")
    for k, v in res.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    os.makedirs("results/amortized", exist_ok=True)
    json.dump(res, open(f"results/amortized/sp100_K{K}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
