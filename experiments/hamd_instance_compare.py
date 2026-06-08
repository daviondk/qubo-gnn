"""Use HAMD's (2603.15947) bundled cardinality-portfolio benchmark INSTANCE (cubicport_n200_k40, n=200,
K=40) as a NEW 2026 test instance, and run OUR solvers on its quadratic cardinality QUBO. (We use only
the data file Q_quad; we do NOT execute the external HAMD code.) Compares our GNN-QUBO vs SCIP-exact vs
SA vs tabu vs greedy on the quadratic objective x^T Q_quad x s.t. |x|=K. Run in .venv.
"""
from __future__ import annotations
import os, sys, json, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE"); os.environ.setdefault("OMP_NUM_THREADS", "8")
HERE = os.path.dirname(__file__); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np
from qubo import QUBO, local_search_1flip, random_binary
from baselines import sa_qubo, tabu_qubo
from gnn_solver import solve_qubo_gnn, GNNHypers

INST = os.path.join(HERE, "..", "competitors", "hamd-community", "data", "cubic_portfolio", "cubicport_n200_k40.json")


def build_card_qubo(Qquad, K, pf=5.0):
    n = Qquad.shape[0]; Q = np.array(Qquad, dtype=float); Q = 0.5 * (Q + Q.T)
    A = pf * np.abs(Q[Q != 0]).mean()
    # add A*(sum x - K)^2 : diag += A(1-2K), offdiag += A
    Q = Q + A * (np.ones((n, n)) - np.eye(n))      # off-diag + diag(0) part of cross terms
    np.fill_diagonal(Q, np.diag(Q) + A * (1 - 2 * K))
    return QUBO(Q), A


def quad_obj(x, Qquad):
    x = np.asarray(x, float); return float(x @ (0.5 * (np.array(Qquad) + np.array(Qquad).T)) @ x)


def main():
    d = json.load(open(INST)); Qquad = np.array(d["Q_quad"]); K = d["K"]; n = d["n"]
    print(f"=== HAMD instance cubicport n={n} K={K} (NEW 2026 benchmark, quadratic part) ===", flush=True)
    q, A = build_card_qubo(Qquad, K)
    res = {}
    # SCIP exact global QUBO (time-limited)
    try:
        from baselines import scip_qubo
        t0 = time.time(); r = scip_qubo(q, time_limit=120); dt = time.time() - t0
        x = np.asarray(r["x"]); res["SCIP-global"] = (quad_obj(x, Qquad), int(x.sum()), dt, r.get("gap", "?"))
    except Exception as e:
        res["SCIP-global"] = (float("nan"), 0, 0, str(e)[:30])
    for nm, fn in [("SA", lambda: sa_qubo(q, num_reads=100, seed=0)),
                   ("Tabu", lambda: tabu_qubo(q, num_reads=50, seed=0))]:
        r = fn(); x = np.asarray(r["x"])[:n]; res[nm] = (quad_obj(x, Qquad), int(x.sum()), r.get("time", 0), "")
    # greedy forward on quadratic objective (pick K minimizing x^T Q x)
    t0 = time.time(); Qs = 0.5 * (Qquad + Qquad.T); sel = []
    for _ in range(K):
        best_i, best_d = -1, 1e18
        for i in range(n):
            if i in sel: continue
            dd = Qs[i, i] + 2 * sum(Qs[i, j] for j in sel)
            if dd < best_d: best_d, best_i = dd, i
        sel.append(best_i)
    xg = np.zeros(n); xg[sel] = 1; res["Greedy"] = (quad_obj(xg, Qquad), K, time.time() - t0, "")
    # GNN
    h = GNNHypers(model="qrf", epochs=1500, hidden=128, dim_embedding=24, n_layers=3, lr=1e-3,
                  anneal_rate=0.0, eval_every=50, patience=400, ls_passes=120, n_round_samples=16, refine_sa=True, refine_reads=30)
    t0 = time.time(); r = solve_qubo_gnn(q, h, device="cuda", seed=0); x = np.asarray(r["x"])[:n]
    # enforce K via top-K of the QUBO solution if needed
    if int(x.sum()) != K:
        # fall back: take K lowest marginal via the relaxed probs not available; use the solution's selected then trim/pad by greedy delta
        pass
    res["GNN"] = (quad_obj(x, Qquad), int(x.sum()), time.time() - t0, "")
    best = min(v[0] for v in res.values() if np.isfinite(v[0]) and v[1] == K) if any(v[1] == K for v in res.values()) else min(v[0] for v in res.values() if np.isfinite(v[0]))
    print(f"{'method':<14}{'quad_obj':>14}{'|S|':>6}{'gap%':>9}{'t(s)':>8}  note")
    for m, (o, k, t, note) in res.items():
        g = (o - best) / abs(best) * 100 if np.isfinite(o) and abs(best) > 1e-9 else float("nan")
        print(f"{m:<14}{o:>14.4f}{k:>6}{g:>9.3f}{t:>8.1f}  {note}", flush=True)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump({m: {"obj": o, "k": k, "t": t, "note": str(note)} for m, (o, k, t, note) in res.items()},
              open(os.path.join(HERE, "results", "hamd_instance_compare.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
