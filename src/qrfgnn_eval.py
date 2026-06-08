"""STAGE 2 (run in .venv): load the original-QRF-GNN supports from stage 1, convex-reweight each
(eps=0.01, delta=1), compute the exact Cura MED/VRE/MRE, and compare to published methods.
"""
from __future__ import annotations

import sys, os, json
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from portfolio_data import download_orlib, load_orlib
from baselines import convex_reweight
from orlib_metrics import unconstrained_frontier, cura_metrics

EPS, DELTA = 0.01, 1.0
PUBLISHED = {  # Cura 2009 MED + best published
    "port1": {"GA": 0.0040, "TS": 0.0040, "SA": 0.0040, "PSO": 0.0049, "IPSO-SA": 0.0001, "Firefly": 0.0003},
    "port2": {"GA": 0.0076, "TS": 0.0082, "SA": 0.0078, "PSO": 0.0090, "IPSO-SA": 0.0001, "Firefly": 0.0009},
    "port3": {"GA": 0.0020, "TS": 0.0021, "SA": 0.0021, "PSO": 0.0022, "IPSO-SA": 0.0000, "Firefly": 0.0004},
    "port4": {"GA": 0.0041, "TS": 0.0041, "SA": 0.0041, "PSO": 0.0052, "IPSO-SA": 0.0001, "Firefly": 0.0003},
    "port5": {"GA": 0.0093, "TS": 0.0010, "SA": 0.0010, "PSO": 0.0019, "IPSO-SA": 0.0000, "Firefly": 0.0000},
}


def eval_instance(name):
    f = f"results/qrfgnn_portfolio/{name}_supports.json"
    if not os.path.exists(f):
        return None
    d = json.load(open(f))
    mu, Sigma, _ = load_orlib(download_orlib("data/orlib")[name])
    lams = np.array(d["lambdas"])
    sv, sr = unconstrained_frontier(mu, Sigma, n_points=2000)
    V, R = [], []
    for lam, sup in zip(lams, d["supports"]):
        sup = np.array(sup, int)
        w = convex_reweight(mu, Sigma, sup, risk_aversion=float(lam), return_weight=float(1 - lam),
                            eps=EPS, delta=DELTA)
        V.append(float(w @ Sigma @ w)); R.append(float(mu @ w))
    met = cura_metrics(V, R, sv, sr)
    return met


def main():
    names = sys.argv[1:] if len(sys.argv) > 1 else ["port1", "port2", "port3", "port4", "port5"]
    results = {}
    for name in names:
        met = eval_instance(name)
        if met is None:
            print(f"{name}: no supports file (run qrfgnn_select.py in .venv-dgl first)")
            continue
        results[name] = met
        print(f"{name}: QRF-GNN(original) MED={met['MED']:.4f} VRE={met['VRE']:.3f}% MRE={met['MRE']:.3f}%")
    os.makedirs("results/qrfgnn_portfolio", exist_ok=True)
    json.dump(results, open("results/qrfgnn_portfolio/med_eval.json", "w"), indent=2)
    # comparison
    print("\n=== MED: original QRF-GNN vs published ===")
    cols = ["GA", "TS", "SA", "PSO", "IPSO-SA", "Firefly"]
    print(f"{'instance':<8}{'QRF-GNN':>9}" + "".join(f"{c:>9}" for c in cols))
    for name in names:
        if name not in results:
            continue
        row = f"{name:<8}{results[name]['MED']:>9.4f}"
        for c in cols:
            row += f"{PUBLISHED[name].get(c, float('nan')):>9.4f}"
        print(row)


if __name__ == "__main__":
    main()
