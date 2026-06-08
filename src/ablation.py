"""Systematic ablation of the GNN-QUBO solver on portfolio cardinality QUBOs:
hyperparameters x penalty functions x formulation, measured as optimality gap vs SCIP-exact.
Also SA / tabu / PI-GNN-style references. Goal: lock the best config + show it dominates the field.

Run in .venv (torch+pyg+scip+cvxpy+neal+tabu).
"""
from __future__ import annotations

import sys, os, json, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "8")
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from portfolio_data import download_orlib, load_orlib
from qubo_portfolio import selection_qubo, decode_selection
from baselines import scip_cardinality, convex_reweight, sa_qubo, tabu_qubo
from gnn_solver import solve_qubo_gnn, solve_qubo_gnn_multi, GNNHypers

LAM = 0.5


def synth(N, seed=5, nf=6):
    rng = np.random.default_rng(seed)
    B = rng.standard_normal((N, nf)) * 0.02; d = np.abs(rng.standard_normal(N)) * 0.01 + 0.005
    Sig = B @ B.T + np.diag(d ** 2)
    mu = rng.standard_normal(N) * 0.004 + 0.003
    return mu, 0.5 * (Sig + Sig.T)


def obj(w, mu, Sigma):
    return float(LAM * (w @ Sigma @ w) - (1 - LAM) * (mu @ w))


def make_instances():
    p = download_orlib("data/orlib")
    inst = {}
    for name, K in [("port2", 10), ("port4", 10)]:
        mu, Sig, _ = load_orlib(p[name]); inst[name] = (mu, Sig, K)
    mu, Sig = synth(200); inst["synth200"] = (mu, Sig, 20)
    return inst


# baseline GNN config; each ablation overrides one field
BASE = dict(model="qrf", dim_embedding=32, hidden=128, n_layers=3, dropout=0.1, lr=1e-3,
            epochs=1200, patience=400, anneal_rate=0.0, recurrent=True, use_pagerank=True,
            eval_every=50, n_round_samples=12, local_search=True, ls_passes=80,
            refine_sa=True, refine_reads=20)

CONFIGS = [
    ("base", {}, 10.0),
    ("model=pignn", {"model": "pignn"}, 10.0),
    ("no_recurrent", {"recurrent": False}, 10.0),
    ("no_refine", {"refine_sa": False}, 10.0),
    ("no_pagerank", {"use_pagerank": False}, 10.0),
    ("no_localsearch", {"local_search": False}, 10.0),
    ("lr=3e-3", {"lr": 3e-3}, 10.0),
    ("lr=5e-4", {"lr": 5e-4}, 10.0),
    ("epochs=2500", {"epochs": 2500}, 10.0),
    ("hidden=64", {"hidden": 64}, 10.0),
    ("hidden=256", {"hidden": 256}, 10.0),
    ("layers=2", {"n_layers": 2}, 10.0),
    ("layers=4", {"n_layers": 4}, 10.0),
    ("embed=16", {"dim_embedding": 16}, 10.0),
    ("embed=64", {"dim_embedding": 64}, 10.0),
    ("anneal=2e-4", {"anneal_rate": 2e-4}, 10.0),     # binarization penalty ON
    ("penalty=2x", {}, 2.0),                           # weaker cardinality penalty
    ("penalty=5x", {}, 5.0),
    ("penalty=20x", {}, 20.0),                         # stronger
    ("rounds=24", {"n_round_samples": 24}, 10.0),
]


def run_config(name, overrides, pf, instances, exact):
    gaps, feas = [], []
    h = GNNHypers(**{**BASE, **overrides})
    for inst, (mu, Sig, K) in instances.items():
        q = selection_qubo(mu, Sig, K, risk_aversion=LAM, return_weight=1 - LAM, penalty_factor=pf)
        r = solve_qubo_gnn(q, h, device="cuda", seed=0)
        S = list(decode_selection(r["x"]))
        ok = len(S) == K
        if not ok:
            S = list(np.argsort(-r["x"])[:K])
        w = convex_reweight(mu, Sig, S, risk_aversion=LAM, return_weight=1 - LAM)
        gaps.append((obj(w, mu, Sig) - exact[inst]) / abs(exact[inst]) * 100)
        feas.append(ok)
    return float(np.mean(gaps)), float(np.mean(feas))


def main():
    inst = make_instances()
    # exact references
    exact = {}
    for name, (mu, Sig, K) in inst.items():
        r = scip_cardinality(mu, Sig, K, risk_aversion=LAM, return_weight=1 - LAM, time_limit=120)
        exact[name] = obj(r["weights"], mu, Sig)
        print(f"exact {name}: obj={exact[name]:.6f} gap={r['gap']:.3f}", flush=True)

    rows = []
    # classical refs
    for cname, fn in [("SA(100)", lambda q: sa_qubo(q, num_reads=100, seed=0)),
                      ("Tabu(50)", lambda q: tabu_qubo(q, num_reads=50, seed=0))]:
        gaps = []
        for name, (mu, Sig, K) in inst.items():
            q = selection_qubo(mu, Sig, K, risk_aversion=LAM, return_weight=1 - LAM)
            S = list(decode_selection(fn(q)["x"]))
            w = convex_reweight(mu, Sig, S, risk_aversion=LAM, return_weight=1 - LAM) if len(S) == K else np.ones(len(mu))/len(mu)
            gaps.append((obj(w, mu, Sig) - exact[name]) / abs(exact[name]) * 100)
        rows.append((cname, float(np.mean(gaps)), 1.0, 0.0)); print(f"{cname:<16} gap={rows[-1][1]:.3f}%", flush=True)

    for name, ov, pf in CONFIGS:
        t0 = time.time(); g, f = run_config(name, ov, pf, inst, exact); dt = time.time() - t0
        rows.append((name, g, f, dt)); print(f"{name:<16} gap={g:.3f}%  feas={f:.2f}  ({dt:.0f}s)", flush=True)

    rows.sort(key=lambda r: r[1])
    print("\n=== ABLATION (mean gap vs SCIP-exact over port2,port4,synth200; lower=better) ===")
    print(f"{'config':<16}{'mean_gap%':>10}{'feasible':>10}")
    for nm, g, f, dt in rows:
        print(f"{nm:<16}{g:>10.3f}{f:>10.2f}")
    os.makedirs("results/ablation", exist_ok=True)
    json.dump([{"config": nm, "mean_gap_pct": g, "feasible": f} for nm, g, f, dt in rows],
              open("results/ablation/ablation.json", "w"), indent=2)


if __name__ == "__main__":
    main()
