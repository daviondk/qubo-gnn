"""Scaling QUBO-solver benchmark on dense portfolio cardinality QUBOs.
All solvers attack the SAME selection_qubo; compare on QUBO energy (lower=better), gap-to-best, time,
feasibility. As N grows: exact SCIP times out, SA/tabu degrade, and the GNN-QUBO (with local-search
polish) should give the best feasible solution -> the regime where GNN-QUBO wins.

Run in .venv (torch+pyg+scip+neal+tabu).
"""
from __future__ import annotations

import sys, os, json, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "8")
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from qubo_portfolio import selection_qubo
from baselines import sa_qubo, tabu_qubo, scip_qubo
from gnn_solver import solve_qubo_gnn, GNNHypers

LAM = 0.5


def synth(N, seed=11, nf=10):
    rng = np.random.default_rng(seed)
    B = rng.standard_normal((N, nf)) * 0.02; d = np.abs(rng.standard_normal(N)) * 0.01 + 0.005
    Sig = B @ B.T + np.diag(d ** 2)
    mu = rng.standard_normal(N) * 0.004 + 0.003
    return mu, 0.5 * (Sig + Sig.T)


def main():
    Ns = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["500", "1000"])]
    tl = float(os.environ.get("TL", "120"))
    allres = {}
    for N in Ns:
        K = max(5, N // 20)
        mu, Sig = synth(N)
        q = selection_qubo(mu, Sig, K, risk_aversion=LAM, return_weight=1 - LAM)
        print(f"\n=== N={N} K={K} (dense selection QUBO, {N} binary vars) ===", flush=True)
        res = {}
        # exact-ish SCIP (global, time-limited)
        try:
            r = scip_qubo(q, time_limit=tl)
            res["SCIP"] = {"E": r["energy"], "k": int(r["x"].sum()), "t": r["time"], "gap_mip": r["gap"], "status": r["status"]}
        except Exception as e:
            res["SCIP"] = {"E": float("inf"), "note": str(e)[:50]}
        r = sa_qubo(q, num_reads=100, seed=0); res["SA"] = {"E": r["energy"], "k": int(np.asarray(r["x"]).sum()), "t": r["time"]}
        r = tabu_qubo(q, num_reads=50, seed=0); res["Tabu"] = {"E": r["energy"], "k": int(np.asarray(r["x"]).sum()), "t": r["time"]}
        h = GNNHypers(model="qrf", epochs=1500, hidden=128, dim_embedding=24, n_layers=3, lr=1e-3,
                      anneal_rate=0.0, eval_every=50, patience=400, ls_passes=120, n_round_samples=16,
                      refine_sa=True, refine_reads=30)
        r = solve_qubo_gnn(q, h, device="cuda", seed=0); res["GNN"] = {"E": r["energy"], "k": int(np.asarray(r["x"]).sum()), "t": r["time"]}

        best = min(v["E"] for v in res.values() if np.isfinite(v.get("E", np.inf)))
        print(f"{'method':<8}{'energy':>14}{'gap%':>9}{'k(=K?)':>9}{'t(s)':>9}  note")
        for m_, v in res.items():
            E = v.get("E", float("inf"))
            gap = (E - best) / abs(best) * 100 if np.isfinite(E) and abs(best) > 1e-12 else float("nan")
            note = v.get("status", "") or v.get("note", "")
            print(f"{m_:<8}{E:>14.4f}{gap:>9.3f}{v.get('k','?'):>9}{v.get('t',0):>9.1f}  {note}")
            v["gap%"] = gap
        allres[N] = res
    os.makedirs("results/scaling_qubo", exist_ok=True)
    json.dump(allres, open("results/scaling_qubo/scaling.json", "w"), indent=2, default=str)


if __name__ == "__main__":
    main()
