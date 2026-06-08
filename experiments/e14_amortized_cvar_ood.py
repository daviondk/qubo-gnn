"""E14 (loop): does the amortized CVaR-selection (E13) transfer OUT-OF-DISTRIBUTION? Load the E13
checkpoint (trained on S&P100 CVaR windows) and evaluate WITHOUT retraining on NASDAQ100 CVaR windows
(per-instance hybrid reference). If gap stays small => amortized CVaR generalizes across markets too.
Run in .venv.
"""
from __future__ import annotations
import os, sys, json, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE"); os.environ.setdefault("OMP_NUM_THREADS", "4")
HERE = os.path.dirname(__file__); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, torch
from datasets import get_returns
from e13_amortized_cvar import Net, basic_feats, knn, windows, cvar_select, K
from exp_cvar import cvar_lp, cvar_of
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    ckpt = torch.load(os.path.join(HERE, "checkpoints", "e13_amortized_cvar_best.pt"), map_location=DEV)
    m = Net(4).to(DEV); m.load_state_dict({k: v.to(DEV) for k, v in ckpt["state_dict"].items()}); m.eval()
    print(f"loaded E13 ckpt (in-dist gap {ckpt['gap']:.2f}%). Building NASDAQ100 CVaR refs (OOD)...", flush=True)
    R = get_returns("nasdaq100").values
    teW = windows(R, mx=30)
    g = []; t0 = time.time()
    for scen in teW:
        mu = scen.mean(0); S, semi = cvar_select(scen)
        if len(S) != K: continue
        ref = cvar_of(cvar_lp(S, scen), scen)
        f, C = basic_feats(mu, semi)
        with torch.no_grad():
            p = m(torch.tensor(f, device=DEV), knn(C)).cpu().numpy()
        c = cvar_of(cvar_lp(np.argsort(-p)[:K], scen), scen)
        g.append((c - ref) / abs(ref) * 100 if abs(ref) > 1e-12 else 0.0)
    print(f"\n=== E14 amortized-CVaR OOD (NASDAQ100, no retrain): mean gap {np.mean(g):.2f}%  median {np.median(g):.2f}%  (n={len(g)}) ===", flush=True)
    json.dump({"ood_mean_gap%": float(np.mean(g)), "ood_median_gap%": float(np.median(g)), "n": len(g)},
              open(os.path.join(HERE, "results", "e14_amortized_cvar_ood.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
