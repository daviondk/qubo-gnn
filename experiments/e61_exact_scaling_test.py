"""E61 (USER DIRECTIVE): decisive test of Stopfer's claim 'MIP solves cardinality to optimality in seconds'
at scale, and whether QUBO/our-solver has any per-instance niche. On Stopfer's nasdaq mu/Sigma, cardinality
MIQP at N=200,500,1000 (K=N/10): exact Gurobi MIQP (time-limited) -- does it solve fast + prove optimality?
-- vs our selection-QUBO solvers (tabu, GNN-QUBO) on objective-gap + time. If Gurobi solves fast, the only
QUBO value is amortization; if Gurobi struggles, there is a per-instance niche. Run in .venv.
"""
import os, sys, json, time, numpy as np, pandas as pd
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE"); os.environ.setdefault("OMP_NUM_THREADS", "6")
HERE = os.path.dirname(__file__); sys.path.insert(0, os.path.join(HERE, "..", "src"))
from baselines import miqp_cardinality, tabu_qubo
from qubo_portfolio import selection_qubo, decode_selection
from amortized import sel_obj
from gnn_solver import solve_qubo_gnn, GNNHypers
DATA = os.path.join(HERE, "..", "competitors", "portfolio_opt_benchmark", "src", "problems", "MarkowitzPortfolio")


def main():
    mu_all = pd.read_csv(os.path.join(DATA, "nasdaq_annual_returns.csv"), sep="\t").iloc[0]
    cov_all = pd.read_csv(os.path.join(DATA, "nasdaq_annualized_covariance_matrix.csv"), sep="\t", index_col=0)
    tickers = [t for t in mu_all.index if t in cov_all.columns]
    rng = np.random.default_rng(0)
    h = GNNHypers(model="qrf", epochs=1200, hidden=128, dim_embedding=20, n_layers=3, lr=1e-3, anneal_rate=0.0,
                  eval_every=50, patience=300, ls_passes=120, n_round_samples=16, refine_sa=True, refine_reads=30)
    LAM = 0.5
    print("E61 exact-scaling test (Stopfer nasdaq, cardinality MIQP) -- does Gurobi solve fast at scale?", flush=True)
    out = {}
    for N in [200, 500, 1000]:
        sel_t = sorted(rng.choice(tickers, size=N, replace=False).tolist())
        mu = mu_all[sel_t].values.astype(float); S = cov_all.loc[sel_t, sel_t].values.astype(float); S = 0.5 * (S + S.T) + 1e-10 * np.eye(N)
        K = max(5, N // 10)
        # exact Gurobi MIQP, 60s limit
        t0 = time.time(); r = miqp_cardinality(mu, S, K, risk_aversion=LAM, return_weight=1 - LAM, time_limit=60.0); tg = time.time() - t0
        gstat = {1: "OPTIMAL", 2: "OPTIMAL", 9: "TIMELIMIT"}.get(r["status"], str(r["status"]))
        obj_exact = r["objective"]; gap_exact = r["gap"]
        # our QUBO solvers (selection-QUBO -> objective via sel_obj on K-best for fair compare)
        q = selection_qubo(mu, S, K, risk_aversion=LAM, return_weight=1 - LAM)
        t0 = time.time(); xt = tabu_qubo(q, num_reads=120, seed=0)["x"]; tt = time.time() - t0
        st = np.flatnonzero(np.asarray(xt) > 0.5); st = st if len(st) == K else np.argsort(-np.asarray(xt))[:K]
        obj_tabu = sel_obj(st, mu, S, K)
        t0 = time.time(); xg = solve_qubo_gnn(q, h, device="cuda", seed=0)["x"]; tgn = time.time() - t0
        sg = decode_selection(xg); sg = sg if len(sg) == K else np.argsort(-np.asarray(xg))[:K]
        obj_gnn = sel_obj(sg, mu, S, K)
        best = min(obj_exact, obj_tabu, obj_gnn)
        out[N] = {"K": K, "gurobi": {"obj": obj_exact, "gap_vs_best%": (obj_exact - best) / abs(best) * 100, "time": tg, "status": gstat, "mipgap": gap_exact},
                  "tabu": {"obj": obj_tabu, "gap_vs_best%": (obj_tabu - best) / abs(best) * 100, "time": tt},
                  "GNN-QUBO": {"obj": obj_gnn, "gap_vs_best%": (obj_gnn - best) / abs(best) * 100, "time": tgn}}
        print(f"  N={N:>4} K={K}: Gurobi {gstat} obj{obj_exact:.5f} gap{(obj_exact-best)/abs(best)*100:.2f}% t{tg:.1f}s (mipgap {gap_exact*100:.1f}%) | "
              f"tabu gap{(obj_tabu-best)/abs(best)*100:.2f}% t{tt:.1f}s | GNN gap{(obj_gnn-best)/abs(best)*100:.2f}% t{tgn:.1f}s", flush=True)
    json.dump(out, open(os.path.join(HERE, "results", "e61_exact_scaling.json"), "w"), indent=2); print("saved", flush=True)


if __name__ == "__main__":
    main()
