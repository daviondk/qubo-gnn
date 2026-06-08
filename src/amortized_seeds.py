"""Robustness/error-bars for the headline amortized result: reuse one tabu-labelled training set,
retrain the amortized GNN over several seeds, report mean+/-std gap vs per-instance tabu on the
held-out S&P100 stream. Run in .venv.
"""
from __future__ import annotations
import sys, os, json, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE"); os.environ.setdefault("OMP_NUM_THREADS", "8")
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch, torch.nn.functional as F
from amortized import AmortizedGNN, build_instance, sel_obj, DEVICE
from amortized_transfer import windows_from_returns, eval_universe
from backtest import load_prices, SP100
from baselines import tabu_qubo
from qubo_portfolio import selection_qubo

K = 15


def main():
    px = load_prices(SP100, "2005-01-01", "2024-12-31"); R = px.pct_change().dropna().values; N = R.shape[1]
    w_all = windows_from_returns(R); split = int(0.7 * len(w_all))
    train_raw, test_raw = w_all[:split], w_all[split:]
    train = [build_instance(mu, Sig, K) for mu, Sig in train_raw]
    print(f"labelling {len(train)} train windows (tabu, once)...", flush=True)
    for (mu, Sig), d in zip(train_raw, train):
        q = selection_qubo(mu, Sig, K, risk_aversion=0.5, return_weight=0.5)
        r = tabu_qubo(q, num_reads=100, seed=0); lab = np.zeros(N, dtype=np.float32)
        lab[np.flatnonzero(np.asarray(r["x"]) > 0.5)] = 1.0
        d["label"] = torch.tensor(lab, device=DEVICE)
    test = [build_instance(mu, Sig, K) for mu, Sig in test_raw]
    gaps = []
    for seed in range(5):
        torch.manual_seed(seed); np.random.seed(seed)
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
        model.eval()
        # quick eval: gap vs tabu on test (reuse sel_obj; tabu ref per instance)
        g = []
        for (mu, Sig), d in zip(test_raw, test):
            q = selection_qubo(mu, Sig, K, risk_aversion=0.5, return_weight=0.5)
            rt = tabu_qubo(q, num_reads=100, seed=0); St = np.flatnonzero(np.asarray(rt["x"]) > 0.5)
            ref = sel_obj(St, mu, Sig, K) if len(St) == K else sel_obj(np.argsort(-np.asarray(rt["x"]))[:K], mu, Sig, K)
            with torch.no_grad():
                p = model(d["feats"], d["edge_index"]).cpu().numpy()
            oa = sel_obj(np.argsort(-p)[:K], mu, Sig, K)
            g.append((oa - ref) / abs(ref) * 100 if abs(ref) > 1e-12 else 0.0)
        mg = float(np.mean(g)); gaps.append(mg); print(f"  seed {seed}: mean gap {mg:.3f}%", flush=True)
    print(f"\n=== Amortized robustness (5 seeds, S&P100 held-out) ===")
    print(f"mean gap vs tabu = {np.mean(gaps):.3f}% +/- {np.std(gaps):.3f}%  (seeds: {[round(x,2) for x in gaps]})")
    os.makedirs("results/amortized", exist_ok=True)
    json.dump({"per_seed_mean_gap%": gaps, "mean": float(np.mean(gaps)), "std": float(np.std(gaps))},
              open("results/amortized/seeds.json", "w"), indent=2)


if __name__ == "__main__":
    main()
