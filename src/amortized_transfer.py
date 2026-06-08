"""Harden the amortized win: OUT-OF-DISTRIBUTION transfer. Train the amortized GNN (imitate tabu) on
S&P 100 rolling windows, then evaluate -- WITHOUT retraining -- on held-out S&P100 AND on entirely
different universes (NASDAQ-100, French 49-Industry; different N, different market). The model is
N-agnostic (GraphSAGE over per-asset features), so it can score any universe. A small OOD gap vs
per-instance tabu = genuine amortized generalization. Run in .venv.
"""
from __future__ import annotations
import sys, os, json, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE"); os.environ.setdefault("OMP_NUM_THREADS", "8")
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np, torch, torch.nn.functional as F

from amortized import AmortizedGNN, build_instance, sel_obj, DEVICE
from backtest import load_prices, SP100
from datasets import get_returns
from baselines import tabu_qubo
from qubo_portfolio import selection_qubo

K = 15


def windows_from_returns(R, lookback=252, step=21, max_w=None):
    idx = list(range(lookback, len(R), step))
    if max_w:
        idx = idx[-max_w:]
    out = []
    for t in idx:
        w = R[t - lookback:t]
        mu = w.mean(0); Sig = np.cov(w, rowvar=False); Sig = 0.5 * (Sig + Sig.T) + 1e-8 * np.eye(R.shape[1])
        out.append((mu, Sig))
    return out


def eval_universe(model, insts, K):
    gaps, inf_t, tabu_t = [], [], []
    for mu, Sig in insts:
        d = build_instance(mu, Sig, K)
        q = selection_qubo(mu, Sig, K, risk_aversion=0.5, return_weight=0.5)
        rt = tabu_qubo(q, num_reads=100, seed=0); St = np.flatnonzero(np.asarray(rt["x"]) > 0.5)
        ref = sel_obj(St, mu, Sig, K) if len(St) == K else sel_obj(np.argsort(-np.asarray(rt["x"]))[:K], mu, Sig, K)
        tabu_t.append(rt["time"])
        t0 = time.time()
        with torch.no_grad():
            p = model(d["feats"], d["edge_index"]).cpu().numpy()
        inf_t.append(time.time() - t0)
        S = np.argsort(-p)[:K]; oa = sel_obj(S, mu, Sig, K)
        gaps.append((oa - ref) / abs(ref) * 100 if abs(ref) > 1e-12 else 0.0)
    return {"mean_gap%": float(np.mean(gaps)), "median_gap%": float(np.median(gaps)),
            "inf_ms": float(np.mean(inf_t) * 1000), "tabu_s": float(np.mean(tabu_t)),
            "speedup": float(np.mean(tabu_t) / (np.mean(inf_t) + 1e-9)), "n": len(insts)}


def main():
    # ---- train on S&P100 ----
    px = load_prices(SP100, "2005-01-01", "2024-12-31"); R = px.pct_change().dropna().values
    N = R.shape[1]
    w_all = windows_from_returns(R)
    split = int(0.7 * len(w_all)); train_raw, sp_test = w_all[:split], w_all[split:]
    print(f"train S&P100: N={N}, {len(train_raw)} windows; labels via tabu...", flush=True)
    train = [build_instance(mu, Sig, K) for mu, Sig in train_raw]
    t0 = time.time()
    for (mu, Sig), d in zip(train_raw, train):
        q = selection_qubo(mu, Sig, K, risk_aversion=0.5, return_weight=0.5)
        r = tabu_qubo(q, num_reads=100, seed=0); lab = np.zeros(N, dtype=np.float32)
        lab[np.flatnonzero(np.asarray(r["x"]) > 0.5)] = 1.0
        d["label"] = torch.tensor(lab, device=DEVICE)
    print(f"  labels {time.time()-t0:.0f}s", flush=True)
    torch.manual_seed(0); np.random.seed(0)
    model = AmortizedGNN(train[0]["feats"].shape[1], hidden=64, layers=3).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for epoch in range(400):
        model.train(); perm = np.random.permutation(len(train))
        for bi in range(0, len(train), 16):
            opt.zero_grad(); bl = 0.0
            for ii in perm[bi:bi + 16]:
                p = model(train[ii]["feats"], train[ii]["edge_index"])
                bl = bl + F.binary_cross_entropy(p.clamp(1e-6, 1 - 1e-6), train[ii]["label"])
            (bl / max(1, len(perm[bi:bi + 16]))).backward(); opt.step()
    print("trained.", flush=True)

    # ---- eval in-distribution + OOD ----
    model.eval()
    res = {}
    res["SP100 (in-dist, held-out)"] = eval_universe(model, sp_test, K)
    res["NASDAQ100 (OOD universe)"] = eval_universe(model, windows_from_returns(get_returns("nasdaq100").values, max_w=40), K)
    res["French49 (OOD, diff N)"] = eval_universe(model, windows_from_returns(get_returns("french49").values, max_w=40), K)
    print("\n=== Amortized OOD transfer (gap vs per-instance tabu; trained ONLY on S&P100) ===")
    print(f"{'test universe':<28}{'mean gap%':>10}{'median%':>9}{'inf ms':>8}{'tabu s':>8}{'speedup':>9}{'n':>5}")
    for k, v in res.items():
        print(f"{k:<28}{v['mean_gap%']:>10.3f}{v['median_gap%']:>9.3f}{v['inf_ms']:>8.2f}{v['tabu_s']:>8.2f}{v['speedup']:>9.0f}{v['n']:>5}")
    os.makedirs("results/amortized", exist_ok=True)
    json.dump(res, open("results/amortized/transfer.json", "w"), indent=2)


if __name__ == "__main__":
    main()
