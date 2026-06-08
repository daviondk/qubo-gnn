"""HARD benchmark where the optimum is UNKNOWN: Gset MaxCut (the standard benchmark for PI-GNN/QRF-GNN).
Compare our GNN-solver (explore->exploit + LS + seeded-SA) vs SA vs tabu, reported as % of BEST-KNOWN
(a heuristic record, NOT a proven optimum). Includes a large graph (G55, 5000 nodes) where no exact
optimum exists. Run in .venv.
"""
from __future__ import annotations
import os, sys, json, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE"); os.environ.setdefault("OMP_NUM_THREADS", "8")
HERE = os.path.dirname(__file__); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np
from maxcut import maxcut_qubo, cut_value, GSET_BEST_KNOWN
from baselines import sa_qubo, tabu_qubo
from gnn_solver import solve_qubo_gnn, GNNHypers
import glob


def main():
    graphs = sys.argv[1:] if len(sys.argv) > 1 else ["G14", "G22", "G55"]
    rows = []
    for g in graphs:
        files = glob.glob(os.path.join(HERE, "..", "Gset", f"{g}.txt")) or glob.glob(os.path.join(HERE, "..", "Gset", f"{g}*"))
        if not files:
            print(f"{g}: file not found"); continue
        q = maxcut_qubo(files[0]); bk = GSET_BEST_KNOWN.get(g, 0); n = q.meta["n"]
        print(f"\n=== {g}: n={n} best-known={bk} (optimum UNKNOWN) ===", flush=True)
        res = {}
        r = sa_qubo(q, num_reads=50, seed=0); res["SA"] = (cut_value(q, r["x"]), r["time"])
        r = tabu_qubo(q, num_reads=20, seed=0); res["Tabu"] = (cut_value(q, r["x"]), r["time"])
        h = GNNHypers(model="qrf", epochs=2500, hidden=128, dim_embedding=20, n_layers=3, lr=1e-3,
                      anneal_rate=0.0, eval_every=50, patience=800, ls_passes=40, n_round_samples=12,
                      refine_sa=True, refine_reads=30)
        r = solve_qubo_gnn(q, h, device="cuda", seed=1); res["GNN"] = (cut_value(q, r["x"]), r["time"])
        print(f"{'method':<8}{'cut':>10}{'%best-known':>13}{'t(s)':>8}")
        for m, (c, t) in res.items():
            print(f"{m:<8}{c:>10.0f}{100*c/bk:>12.2f}%{t:>8.1f}", flush=True)
        rows.append({"graph": g, "n": n, "best_known": bk,
                     **{m: {"cut": c, "pct": 100 * c / bk, "t": t} for m, (c, t) in res.items()}})
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(rows, open(os.path.join(HERE, "results", "gset_hard.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
