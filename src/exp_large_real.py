"""BIG real-portfolio test (addresses 'are exact solutions really achievable at scale?').
Large universe (S&P 500, ~480 assets), EMPIRICAL covariance, cardinality K. Compare GNN / SA / tabu /
greedy and SCIP-exact MIQP (time-limited). At this size the exact MIQP is expected to struggle/time out;
we report best-found gap, whether SCIP proved optimality, and wall-clock. Run in .venv.
"""
from __future__ import annotations
import sys, os, json, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE"); os.environ.setdefault("OMP_NUM_THREADS", "8")
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from datasets import get_returns
from qubo_portfolio import selection_qubo, decode_selection
from baselines import sa_qubo, tabu_qubo, greedy_selection, scip_cardinality, convex_reweight

LAM = 0.5


def fin_obj(w, mu, Sig):
    """True financial objective on continuous weights: ra w'Sigma w - rw mu'w (ra=LAM, rw=1-LAM)."""
    w = np.asarray(w, float)
    return float(LAM * (w @ Sig @ w) - (1 - LAM) * (mu @ w))


def main():
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    window = int(sys.argv[2]) if len(sys.argv) > 2 else 750   # empirical-cov window (days)
    R = get_returns("sp500").values
    R = R[-window:]
    mu = R.mean(0); Sig = np.cov(R, rowvar=False); Sig = 0.5 * (Sig + Sig.T)
    N = len(mu)
    print(f"=== S&P500 BIG: N={N} K={K} window={window}d, empirical covariance ===", flush=True)
    q = selection_qubo(mu, Sig, K, risk_aversion=LAM, return_weight=1 - LAM)
    res = {}
    # exact MIQP (continuous weights) — the TRUE optimum of the financial objective. Does it solve at N~460?
    t0 = time.time()
    try:
        mi = scip_cardinality(mu, Sig, K, risk_aversion=LAM, return_weight=1 - LAM, eps=0.0, delta=1.0, time_limit=180)
        res["SCIP-MIQP(exact)"] = {"obj": fin_obj(mi["weights"], mu, Sig), "t": mi["time"],
                                   "gap_proved": mi["gap"], "status": mi["status"], "k": len(mi["support"])}
    except Exception as e:
        res["SCIP-MIQP(exact)"] = {"obj": float("inf"), "t": time.time() - t0, "note": str(e)[:50], "k": 0}

    def rec(nm, x, t):
        S = decode_selection(np.asarray(x)[:N])
        w = convex_reweight(mu, Sig, S, risk_aversion=LAM, return_weight=1 - LAM) if len(S) == K else np.ones(N) / N
        res[nm] = {"obj": fin_obj(w, mu, Sig), "t": t, "k": len(S)}
    r = sa_qubo(q, num_reads=100, seed=0); rec("SA+reweight", r["x"], r["time"])
    r = tabu_qubo(q, num_reads=50, seed=0); rec("Tabu+reweight", r["x"], r["time"])
    r = greedy_selection(mu, Sig, K, risk_aversion=LAM, return_weight=1 - LAM); rec("Greedy+reweight", r["x"], r["time"])
    from gnn_solver import solve_qubo_gnn, GNNHypers
    h = GNNHypers(model="qrf", epochs=1500, hidden=128, dim_embedding=24, n_layers=3, lr=1e-3,
                  anneal_rate=0.0, eval_every=50, patience=400, ls_passes=120, n_round_samples=16,
                  refine_sa=True, refine_reads=30)
    r = solve_qubo_gnn(q, h, device="cuda", seed=0); rec("GNN+reweight", r["x"], r["time"])

    best = min(v["obj"] for v in res.values() if np.isfinite(v["obj"]))
    print(f"{'method':<12}{'obj':>12}{'gap%':>9}{'k':>5}{'t(s)':>8}  note")
    for m, v in res.items():
        g = (v["obj"] - best) / abs(best) * 100 if np.isfinite(v["obj"]) else float("nan")
        note = v.get("status", "") or v.get("note", "")
        if m.startswith("SCIP-MIQP") and "gap_proved" in v:
            note = f"proved_gap={v['gap_proved']:.3f} {note}"
        print(f"{m:<12}{v['obj']:>12.6f}{g:>9.3f}{v['k']:>5}{v['t']:>8.1f}  {note}")
    os.makedirs("results/large_real", exist_ok=True)
    json.dump(res, open(f"results/large_real/sp500_K{K}_w{window}.json", "w"), indent=2, default=str)


if __name__ == "__main__":
    main()
