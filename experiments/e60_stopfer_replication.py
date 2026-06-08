"""E60 (USER DIRECTIVE): run OUR solver on Stopfer & Wagner 2025 (2509.17876) EXACT instances + metric.
Their public data/instances: competitors/portfolio_opt_benchmark. MinVola QUBO: min w'Sigma w s.t. w'mu>=eps,
sum w=1, 0<=w<=u; 4-bit weight encoding (coeffs [.5,.25,.125,.125]*u_i), penalties phi=psi=1000.
We load their nasdaq mu/Sigma + instance configs (asset lists, eps=min_return, u=asset_limits), build the
SAME QUBO, solve with our GNN-QUBO / SA / tabu, decode weights, compute approximation ratio
Theta = achieved_volatility / continuous_QP_optimum (cvxpy exact) + feasibility, and compare to their
reported curves (Gurobi 1.0; heuristic 1.2-1.5; their SA/tabu >=2.0). Run in .venv.
"""
import os, sys, json, glob, time, numpy as np, pandas as pd
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE"); os.environ.setdefault("OMP_NUM_THREADS", "6")
HERE = os.path.dirname(__file__); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import cvxpy as cp
from qubo import QUBO
from baselines import tabu_qubo, sa_qubo
from gnn_solver import solve_qubo_gnn, GNNHypers
BENCH = os.path.join(HERE, "..", "competitors", "portfolio_opt_benchmark")
DATA = os.path.join(BENCH, "src", "problems", "MarkowitzPortfolio")
PHI = PSI = 1000.0
COEF = np.array([0.5, 0.25, 0.125, 0.125])  # 4-bit weight encoding (d=3)


def load_nasdaq():
    mu = pd.read_csv(os.path.join(DATA, "nasdaq_annual_returns.csv"), sep="\t")
    mu = mu.iloc[0]  # single row -> Series indexed by ticker
    cov = pd.read_csv(os.path.join(DATA, "nasdaq_annualized_covariance_matrix.csv"), sep="\t", index_col=0)
    return mu, cov


def cont_opt(mu, S, u, eps):
    n = len(mu); w = cp.Variable(n)
    cons = [cp.sum(w) == 1, w >= 0, w <= u, mu @ w >= eps]
    try:
        cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(S))), cons).solve(solver=cp.CLARABEL)
        if w.value is None: return None, None
        return float(w.value @ S @ w.value), w.value
    except Exception:
        return None, None


def build_qubo(mu, S, u, eps):
    n = len(mu); nv = 4 * n
    B = np.zeros((n, nv))
    for i in range(n):
        B[i, 4 * i:4 * i + 4] = COEF * u[i]
    Q = B.T @ S @ B + PSI * (B.T @ np.ones((n, n)) @ B) + PHI * (B.T @ np.outer(mu, mu) @ B)
    lin = -2 * PSI * (np.ones(n) @ B) - 2 * PHI * eps * (mu @ B)
    Q = Q + np.diag(lin)
    Q = 0.5 * (Q + Q.T)
    return QUBO(Q), B


def decode(x, B):
    w = B @ np.asarray(x)[:B.shape[1]]; return w


def main():
    mu_all, cov_all = load_nasdaq()
    tickers = list(mu_all.index)
    cfgs = sorted(glob.glob(os.path.join(BENCH, "config", "config_files", "problem_configs",
                                         "MarkowitzPortfolio", "from_nasdaq", "*assets", "*minvola*.json")))
    by_size = {}
    for c in cfgs:
        d = json.load(open(c)); pc = d["problem_config"]
        if not pc.get("choose_nasdaq_assets"): continue
        n = len(pc["nasdaq_assets"]); by_size.setdefault(n, []).append((c, pc))
    h = GNNHypers(model="qrf", epochs=1500, hidden=128, dim_embedding=20, n_layers=3, lr=1e-3, anneal_rate=0.0,
                  eval_every=50, patience=400, ls_passes=150, n_round_samples=20, refine_sa=True, refine_reads=40)
    print("E60 Stopfer&Wagner replication: OUR solver on THEIR exact minvola instances", flush=True)
    print(f"  nasdaq universe: {len(tickers)} tickers; sizes available: {sorted(by_size)}", flush=True)
    out = {}
    for n in [10, 20, 50, 100]:
        if n not in by_size: continue
        rows = {"GNN-QUBO": [], "SA": [], "tabu": []}; feas = {"GNN-QUBO": 0, "SA": 0, "tabu": 0}; ninst = 0
        for c, pc in by_size[n][:3]:  # 3 instances per size
            assets = pc["nasdaq_assets"];
            if any(a not in mu_all.index for a in assets): continue
            mu = mu_all[assets].values.astype(float)
            S = cov_all.loc[assets, assets].values.astype(float); S = 0.5 * (S + S.T)
            u = np.asarray(pc["asset_limits"], float) if pc.get("asset_limits") else np.full(n, max(1/10, 3/n))
            if u.shape[0] != n: u = np.full(n, max(1/10, 3/n))
            eps = float(pc["min_return"])
            fopt, _ = cont_opt(mu, S, u, eps)
            if fopt is None or fopt <= 0: continue
            q, B = build_qubo(mu, S, u, eps); ninst += 1
            for m, solve in [("GNN-QUBO", lambda: solve_qubo_gnn(q, h, device="cuda", seed=0)["x"]),
                             ("SA", lambda: sa_qubo(q, num_reads=150, seed=0)["x"]),
                             ("tabu", lambda: tabu_qubo(q, num_reads=150, seed=0)["x"])]:
                x = solve(); w = decode(x, B)
                vol = float(w @ S @ w); budget_ok = abs(w.sum() - 1) < 0.05; ret_ok = (mu @ w) >= eps - 1e-4
                if budget_ok and ret_ok: feas[m] += 1
                rows[m].append(vol / fopt)
        out[n] = {m: {"Theta_mean": float(np.mean(rows[m])) if rows[m] else None,
                      "feasible": f"{feas[m]}/{ninst}"} for m in rows}
        print(f"  n={n:>3} ({ninst} inst): " + " | ".join(
            f"{m} Theta~{np.mean(rows[m]):.2f} feas {feas[m]}/{ninst}" for m in rows if rows[m]), flush=True)
    print("  (their paper: Gurobi 1.0 | tailored-heuristic 1.2-1.5 | their SA/tabu >=2.0)", flush=True)
    json.dump(out, open(os.path.join(HERE, "results", "e60_stopfer.json"), "w"), indent=2); print("saved", flush=True)


if __name__ == "__main__":
    main()
