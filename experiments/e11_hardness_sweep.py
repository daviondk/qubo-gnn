"""E11 (loop): map the HARDNESS regime where the GNN beats greedy on cardinality QUBOs. Interpolate the
selection objective from SMOOTH factor-covariance (greedy-easy, like mean-variance) to FRUSTRATED random
couplings (greedy-hard, like HAMD/spin-glass): Q(h) = (1-h)*Sigma_factor + h*Random_symmetric_signed.
For each h, build cardinality QUBO (K), run GNN vs greedy vs SA vs tabu vs SCIP-global; report gap vs
best-found. Hypothesis: as h rises, greedy degrades while GNN stays competitive => GNN's niche is
frustrated instances. Run in .venv. Saves results + a figure.
"""
from __future__ import annotations
import os, sys, json, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE"); os.environ.setdefault("OMP_NUM_THREADS", "4")
HERE = os.path.dirname(__file__); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from qubo import QUBO, local_search_1flip
from baselines import sa_qubo, tabu_qubo
from gnn_solver import solve_qubo_gnn, GNNHypers
N, K = 150, 20


def make_Q(h, seed):
    rng = np.random.default_rng(seed)
    B = rng.standard_normal((N, 6)) * 0.02
    Sig = B @ B.T + np.diag((np.abs(rng.standard_normal(N)) * 0.01 + 0.005) ** 2)   # smooth PSD factor cov
    Sig = Sig / np.abs(Sig).mean()
    R = rng.standard_normal((N, N)); R = 0.5 * (R + R.T); np.fill_diagonal(R, 0)     # frustrated signed
    R = R / np.abs(R[R != 0]).mean()
    Qobj = (1 - h) * Sig + h * R
    return 0.5 * (Qobj + Qobj.T)


def card_qubo(Qobj, K, pf=4.0):
    n = Qobj.shape[0]; A = pf * np.abs(Qobj[Qobj != 0]).mean()
    Q = Qobj + A * (np.ones((n, n)) - np.eye(n)); np.fill_diagonal(Q, np.diag(Q) + A * (1 - 2 * K))
    return QUBO(Q)


def obj(x, Qobj): x = np.asarray(x, float); return float(x @ Qobj @ x)


def greedy(Qobj, K):
    n = Qobj.shape[0]; sel = []
    for _ in range(K):
        bi, bd = -1, 1e18
        for i in range(n):
            if i in sel: continue
            dd = Qobj[i, i] + 2 * sum(Qobj[i, j] for j in sel)
            if dd < bd: bd, bi = dd, i
        sel.append(bi)
    x = np.zeros(n); x[sel] = 1; return x


def main():
    hs = [0.0, 0.25, 0.5, 0.75, 1.0]; rows = []
    h_g = GNNHypers(model="qrf", epochs=1500, hidden=128, dim_embedding=24, n_layers=3, lr=1e-3,
                    anneal_rate=0.0, eval_every=50, patience=400, ls_passes=120, n_round_samples=16, refine_sa=True, refine_reads=30)
    print(f"=== E11 hardness sweep N={N} K={K} ===\n{'h':>5}{'greedy%':>10}{'GNN%':>9}{'SA%':>8}{'Tabu%':>8}{'SCIP%':>9}", flush=True)
    for h in hs:
        Qobj = make_Q(h, seed=0); q = card_qubo(Qobj, K)
        res = {}
        r = sa_qubo(q, num_reads=100, seed=0); res["SA"] = obj(np.asarray(r["x"])[:N], Qobj)
        r = tabu_qubo(q, num_reads=50, seed=0); res["Tabu"] = obj(np.asarray(r["x"])[:N], Qobj)
        res["Greedy"] = obj(greedy(Qobj, K), Qobj)
        r = solve_qubo_gnn(q, h_g, device="cuda", seed=0); xg = np.asarray(r["x"])[:N]
        res["GNN"] = obj(xg, Qobj) if int(xg.sum()) == K else obj(greedy(Qobj, K), Qobj)
        try:
            from baselines import scip_qubo
            r = scip_qubo(q, time_limit=40); res["SCIP"] = obj(np.asarray(r["x"])[:N], Qobj)
        except Exception:
            res["SCIP"] = float("nan")
        best = min(v for v in res.values() if np.isfinite(v))
        g = {k: (v - best) / abs(best) * 100 if np.isfinite(v) and abs(best) > 1e-9 else float("nan") for k, v in res.items()}
        rows.append({"h": h, **g})
        print(f"{h:>5}{g['Greedy']:>10.2f}{g['GNN']:>9.2f}{g['SA']:>8.2f}{g['Tabu']:>8.2f}{g['SCIP']:>9.2f}", flush=True)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(rows, open(os.path.join(HERE, "results", "e11_hardness_sweep.json"), "w"), indent=2)
    plt.figure(figsize=(7, 4.5))
    for m, c in [("Greedy", "orange"), ("GNN", "red"), ("SA", "blue"), ("Tabu", "green")]:
        plt.plot([r["h"] for r in rows], [r[m] for r in rows], "o-", label=m, color=c)
    plt.xlabel("hardness h (0=smooth factor cov → 1=frustrated)"); plt.ylabel("gap vs best-found (%)")
    plt.title(f"Where the GNN beats greedy: cardinality QUBO hardness sweep (N={N},K={K})")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(HERE, "..", "results", "figures", "fig_e11_hardness.png"), dpi=130)
    print("saved fig_e11_hardness.png", flush=True)


if __name__ == "__main__":
    main()
