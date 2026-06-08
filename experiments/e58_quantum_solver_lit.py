"""E58 (USER DIRECTIVE): benchmark our solver in the QUANTUM-SOLVER literature framing (Phillipson 2020
2012.01121; Palmer/Orus 2021 2106.06735). Classical Markowitz as a WEIGHT-ENCODED QUBO with budget penalty
+ investment band (w_max), solved as a SOLVER benchmark: objective value / optimality-gap vs EXACT (SCIP
global-QUBO) / time. Solvers: SCIP(exact) vs tabu vs SA vs our GNN-QUBO. Real S&P100, varying N. Shows our
GNN reaches the optimum where exact is tractable and scales where exact slows -- the same metric these
quantum papers report (objective/optimality/time vs classical). Run in .venv.
"""
import os, sys, json, time, numpy as np
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE"); os.environ.setdefault("OMP_NUM_THREADS", "6")
HERE = os.path.dirname(__file__); sys.path.insert(0, os.path.join(HERE, "..", "src"))
from backtest import load_prices, SP100
from qubo_portfolio import weight_qubo
from baselines import tabu_qubo, sa_qubo
from gnn_solver import solve_qubo_gnn, GNNHypers


def main():
    R = load_prices(SP100, "2018-01-01", "2024-12-31").pct_change().dropna().values
    Nfull = R.shape[1]
    h = GNNHypers(model="qrf", epochs=2000, hidden=128, dim_embedding=20, n_layers=3, lr=1e-3, anneal_rate=0.0,
                  eval_every=50, patience=400, ls_passes=150, n_round_samples=20, refine_sa=True, refine_reads=40)
    print("E58 quantum-solver-literature framing: weight-encoded Markowitz QUBO (budget + investment band)", flush=True)
    out = {}
    for N in [20, 40, 60]:
        sub = np.arange(N)
        w = R[-504:, sub]; mu = w.mean(0) * 252; S = np.cov(w, rowvar=False) * 252; S = 0.5 * (S + S.T) + 1e-8 * np.eye(N)
        q, spec = weight_qubo(mu, S, n_bits=3, risk_aversion=0.5, return_weight=0.5, w_max=0.20)  # band: max 20%/asset
        nv = q.n
        res = {}
        t0 = time.time()
        try:
            from baselines import scip_qubo; r = scip_qubo(q, time_limit=60); res["SCIP(exact)"] = (r["energy"], time.time() - t0, r.get("status", "?"))
        except Exception as e:
            res["SCIP(exact)"] = (float("nan"), time.time() - t0, "err")
        r = tabu_qubo(q, num_reads=150, seed=0); res["tabu"] = (r["energy"], r["time"], "")
        r = sa_qubo(q, num_reads=150, seed=0); res["SA"] = (r["energy"], r["time"], "")
        r = solve_qubo_gnn(q, h, device="cuda", seed=0); res["GNN-QUBO"] = (r["energy"], r["time"], "")
        best = min(v[0] for v in res.values() if np.isfinite(v[0]))
        print(f"\n=== N={N} assets, nv={nv} binary vars (n_bits=3, band w_max=0.20) ===", flush=True)
        print(f"{'solver':<14}{'objective':>12}{'opt-gap%':>10}{'time(s)':>9}", flush=True)
        row = {}
        for m, (e, t, st) in res.items():
            g = (e - best) / abs(best) * 100 if np.isfinite(e) and abs(best) > 1e-12 else float("nan")
            row[m] = {"obj": e, "gap%": g, "t": t, "status": st}
            print(f"{m:<14}{e:>12.5f}{g:>10.3f}{t:>9.1f}  {st}", flush=True)
        out[N] = row
    json.dump(out, open(os.path.join(HERE, "results", "e58_quantum_solver_lit.json"), "w"), indent=2); print("saved", flush=True)


if __name__ == "__main__":
    main()
