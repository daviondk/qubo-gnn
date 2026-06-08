"""Exact cardinality-frontier MED via SCIP (free, no size limit) for OR-Library port1-5.
Gives the TRUE MED floor (incl. port5 N=225, where free Gurobi can't run). Same Cura metric.
"""
from __future__ import annotations

import sys, os, json
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from portfolio_data import download_orlib, load_orlib, ORLIB_FILES
from baselines import scip_cardinality
from orlib_metrics import unconstrained_frontier, cura_metrics, lambda_grid

EPS, DELTA = 0.01, 1.0


def main():
    names = sys.argv[1:] if len(sys.argv) > 1 else ["port1", "port2", "port3", "port4", "port5"]
    tl = float(os.environ.get("SCIP_TL", "60"))
    lams = lambda_grid(51)
    paths = download_orlib("data/orlib")
    results = {}
    for name in names:
        mu, Sigma, _ = load_orlib(paths[name])
        sv, sr = unconstrained_frontier(mu, Sigma, n_points=2000)
        V, R, gaps, maxt = [], [], [], 0.0
        for lam in lams:
            r = scip_cardinality(mu, Sigma, 10, risk_aversion=float(lam), return_weight=float(1 - lam),
                                 eps=EPS, delta=DELTA, time_limit=tl)
            w = r["weights"]
            V.append(float(w @ Sigma @ w)); R.append(float(mu @ w))
            gaps.append(float(r["gap"])); maxt = max(maxt, r["time"])
        met = cura_metrics(V, R, sv, sr)
        met["max_gap"] = max(gaps); met["max_time_per_point"] = maxt
        results[name] = met
        print(f"{name} (N={len(mu)}): EXACT(SCIP) MED={met['MED']:.4f} VRE={met['VRE']:.3f}% "
              f"MRE={met['MRE']:.3f}% max_gap={max(gaps):.4f} max_t/pt={maxt:.1f}s", flush=True)
    os.makedirs("results/orlib_med", exist_ok=True)
    json.dump(results, open("results/orlib_med/scip_exact.json", "w"), indent=2)
    print("saved results/orlib_med/scip_exact.json")


if __name__ == "__main__":
    main()
