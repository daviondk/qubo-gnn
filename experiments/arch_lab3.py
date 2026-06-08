"""Exp 3: can a CARDINALITY-AWARE unsupervised loss fix the amortized collapse?
Train ONE shared GNN across S&P100 windows with an UNSUPERVISED objective and compare:
  - unsup-plain : loss = mean_i p^T Q_i p                (the collapsing one, docs/15 -> 71% gap)
  - unsup-knorm : loss = mean_i p~^T Q_i p~, p~=clip(K p/sum p,0,1)   (cardinality-aware, label-free)
  - supervised  : imitate tabu (docs/15 -> ~1.05% gap) — reference number, not retrained here.
Eval: top-K of the GNN probs on held-out windows, gap vs per-instance tabu. If unsup-knorm << 71%, we
have a LABEL-FREE amortized solver (a real architecture improvement). Run in .venv.
"""
from __future__ import annotations
import os, sys, json, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE"); os.environ.setdefault("OMP_NUM_THREADS", "8")
HERE = os.path.dirname(__file__); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, torch, torch.nn.functional as F
from amortized import AmortizedGNN, build_instance, sel_obj, DEVICE
from amortized_transfer import windows_from_returns
from backtest import load_prices, SP100
from baselines import tabu_qubo
from qubo_portfolio import selection_qubo
K = 15


def train_model(model, train, mode, epochs=400):
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for ep in range(epochs):
        model.train(); perm = np.random.permutation(len(train))
        for bi in range(0, len(train), 16):
            opt.zero_grad(); bl = 0.0
            for ii in perm[bi:bi + 16]:
                d = train[ii]; p = model(d["feats"], d["edge_index"])
                pe = torch.clamp(K * p / (p.sum() + 1e-9), 0, 1) if mode == "knorm" else p
                bl = bl + pe @ (d["Qn"] @ pe)
            (bl / max(1, len(perm[bi:bi + 16]))).backward(); opt.step()


def evaluate(model, test_raw, test):
    gaps = []
    for (mu, Sig), d in zip(test_raw, test):
        q = selection_qubo(mu, Sig, K, risk_aversion=0.5, return_weight=0.5)
        rt = tabu_qubo(q, num_reads=100, seed=0); St = np.flatnonzero(np.asarray(rt["x"]) > 0.5)
        ref = sel_obj(St, mu, Sig, K) if len(St) == K else sel_obj(np.argsort(-np.asarray(rt["x"]))[:K], mu, Sig, K)
        with torch.no_grad():
            p = model(d["feats"], d["edge_index"]).cpu().numpy()
        oa = sel_obj(np.argsort(-p)[:K], mu, Sig, K)
        gaps.append((oa - ref) / abs(ref) * 100 if abs(ref) > 1e-12 else 0.0)
    return float(np.mean(gaps)), float(np.median(gaps))


def main():
    px = load_prices(SP100, "2005-01-01", "2024-12-31"); R = px.pct_change().dropna().values
    w = windows_from_returns(R); split = int(0.7 * len(w))
    train_raw, test_raw = w[:split], w[split:]
    train = [build_instance(mu, Sig, K) for mu, Sig in train_raw]
    test = [build_instance(mu, Sig, K) for mu, Sig in test_raw]
    print(f"S&P100: {len(train)} train, {len(test)} test windows", flush=True)
    out = {}
    for mode in ["plain", "knorm"]:
        torch.manual_seed(0); np.random.seed(0)
        m = AmortizedGNN(train[0]["feats"].shape[1], hidden=64, layers=3).to(DEVICE)
        t0 = time.time(); train_model(m, train, mode); dt = time.time() - t0
        mean, med = evaluate(m, test_raw, test)
        out[mode] = {"mean_gap%": mean, "median_gap%": med, "train_s": dt}
        print(f"unsup-{mode:<6}: mean gap {mean:.2f}%  median {med:.2f}%  (train {dt:.0f}s)", flush=True)
    out["supervised(ref docs/15)"] = {"mean_gap%": 1.05, "median_gap%": 0.67}
    print("supervised (ref): mean 1.05% / median 0.67%")
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "results", "arch_lab3.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
