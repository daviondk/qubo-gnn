"""Axis A: run the cardinality QUBO solver comparison on NEW public datasets (French 49-Industry,
NASDAQ-100, crypto), so dataset coverage matches the literature. French49 enables a direct 2026
head-to-head with Lozano (arXiv:2605.17628, penalty-free QUBO, regret metric, same dataset).

Metric: optimality gap of GNN / SA / tabu / greedy vs SCIP-exact cardinality MIQP (the 'regret').
Run in .venv.
"""
from __future__ import annotations

import sys, os, json
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "8")
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from datasets import get_returns
from exp_cardinality import run_point


def main():
    specs = [("french49", 10), ("nasdaq100", 10), ("crypto", 6)]
    if len(sys.argv) > 1:
        specs = [(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 10)]
    ra, rw = 1.0, 0.5
    allres = {}
    for name, K in specs:
        R = get_returns(name)
        mu = R.mean().values; Sigma = R.cov().values; Sigma = 0.5 * (Sigma + Sigma.T)
        n = len(mu)
        print(f"\n=== {name}: N={n} K={K} (ra={ra} rw={rw}) ===", flush=True)
        res = run_point(mu, Sigma, K, ra, rw, gnn_epochs=1200)
        ref = res["_ref"]
        print(f"reference: {'exact MIQP' if ref['exact'] else 'best-found'}")
        hdr = f"{'method':<14}{'obj':>12}{'gap%(regret)':>14}{'sharpe':>9}{'k':>4}{'feas':>6}{'t(s)':>8}"
        print(hdr); print("-" * len(hdr))
        for m, d in res.items():
            if m == "_ref":
                continue
            print(f"{m:<14}{d.get('obj',float('nan')):>12.6f}{d.get('gap',float('nan'))*100:>14.3f}"
                  f"{d.get('sharpe',float('nan')):>9.4f}{d['k']:>4}{str(d['feasible']):>6}{d['time']:>8.2f}")
        allres[name] = {m: {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                            for k, v in d.items() if k != 'mu' and k != 'Sigma'}
                        for m, d in res.items() if m != "_ref"}
    os.makedirs("results/datasets", exist_ok=True)
    json.dump(allres, open("results/datasets/cardinality_extra.json", "w"), indent=2, default=str)


if __name__ == "__main__":
    main()
