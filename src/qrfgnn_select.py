"""STAGE 1 (run in .venv-dgl): use the EXACT original QRF-GNN to pick K-asset supports across the
risk-return frontier for OR-Library instances. Saves supports to results/qrfgnn_portfolio/.

Stage 2 (qrfgnn_eval.py, run in .venv) reweights + computes the Cura MED vs published methods.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "8")
import sys, json, time
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from portfolio_data import download_orlib, load_orlib, ORLIB_FILES
from qrfgnn_portfolio import solve_selection_original

K = 10


def main():
    names = sys.argv[1:] if len(sys.argv) > 1 else ["port1", "port2", "port3", "port4", "port5"]
    n_lam = 51
    epochs = int(os.environ.get("QRF_EPOCHS", "3000"))
    seeds = int(os.environ.get("QRF_SEEDS", "3"))
    lams = np.linspace(0.0, 1.0, n_lam)
    paths = download_orlib("data/orlib")
    os.makedirs("results/qrfgnn_portfolio", exist_ok=True)
    for name in names:
        mu, Sigma, _ = load_orlib(paths[name])
        t0 = time.time()
        supports = []
        for i, lam in enumerate(lams):
            sup = solve_selection_original(mu, Sigma, K, float(lam), epochs=epochs, seeds=seeds)
            supports.append([int(x) for x in sup])
            if i % 10 == 0:
                print(f"  {name} lam={lam:.2f} support={sorted(sup)[:5]}... t={time.time()-t0:.0f}s", flush=True)
        out = {"instance": name, "N": len(mu), "K": K, "lambdas": lams.tolist(),
               "supports": supports, "epochs": epochs, "seeds": seeds}
        with open(f"results/qrfgnn_portfolio/{name}_supports.json", "w") as f:
            json.dump(out, f)
        print(f">>> {name}: saved {len(supports)} supports, total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
