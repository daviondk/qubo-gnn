"""Gset MaxCut reproduction check: does our GNN reproduce the QRF-GNN paper cut values?
Reports GNN-alone and GNN+SA-refine vs paper (QRF-GNN) and best-known, plus SA-from-scratch control.
"""
from __future__ import annotations

import sys, os, json, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import glob

from maxcut import maxcut_qubo, cut_value, GSET_BEST_KNOWN, QRF_GNN_REPORTED
from gnn_solver import solve_qubo_gnn, GNNHypers
from baselines import sa_qubo


def main():
    files = sorted(glob.glob("Gset/G*.txt"))
    seeds = [1, 2, 3]
    rows = []
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        if name not in GSET_BEST_KNOWN:
            continue
        q = maxcut_qubo(f)
        bk = GSET_BEST_KNOWN[name]
        # GNN alone (no SA refine) and GNN+SA refine, best over seeds
        best_alone, best_ref = -1, -1
        for s in seeds:
            h = GNNHypers(model="qrf", epochs=2500, hidden=128, dim_embedding=20, n_layers=3,
                          lr=1e-3, anneal_rate=0.0, eval_every=50, patience=800, ls_passes=40,
                          n_round_samples=12, refine_sa=False)
            r = solve_qubo_gnn(q, h, device="cuda", seed=s)
            best_alone = max(best_alone, cut_value(q, r["x"]))
            h.refine_sa = True; h.refine_reads = 30
            r2 = solve_qubo_gnn(q, h, device="cuda", seed=s)
            best_ref = max(best_ref, cut_value(q, r2["x"]))
        rsa = sa_qubo(q, num_reads=100, seed=0); sa_cut = -rsa["energy"]
        paper = QRF_GNN_REPORTED.get(name, float("nan"))
        rows.append({"name": name, "n": q.meta["n"], "best_known": bk, "paper_qrf": paper,
                     "gnn_alone": best_alone, "gnn_sa": best_ref, "sa_scratch": sa_cut})
        print(f"{name} (n={q.meta['n']}): best-known={bk} paper-QRF={paper} | "
              f"GNN-alone={best_alone:.0f} ({best_alone/bk:.3f}) "
              f"GNN+SA={best_ref:.0f} ({best_ref/bk:.3f}) SA={sa_cut:.0f} ({sa_cut/bk:.3f})")
    os.makedirs("results/maxcut", exist_ok=True)
    with open("results/maxcut/gset_repro.json", "w") as fp:
        json.dump(rows, fp, indent=2)


if __name__ == "__main__":
    main()
