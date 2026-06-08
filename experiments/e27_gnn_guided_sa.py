"""E27 (loop): GNN-guided SA hybrid (the 'learning augments a metaheuristic' direction, IsingFormer-style)
on HARD instances where SA is strong. Compare cold SA (k reads) vs GNN-initialized SA (per-instance GNN
selection as initial state + k reads) on frustrated and weight-encoded QUBOs. Does GNN-init accelerate/
improve SA? Run in .venv.
"""
import os, sys, json, time, numpy as np
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE"); os.environ.setdefault("OMP_NUM_THREADS", "6")
HERE = os.path.dirname(__file__); sys.path.insert(0, os.path.join(HERE, "..", "src"))
from qubo import QUBO
from baselines import sa_qubo, tabu_qubo
from qubo_portfolio import selection_qubo, weight_qubo
from gnn_solver import solve_qubo_gnn, GNNHypers
import neal


def gnn_init(q):
    h = GNNHypers(model="qrf", epochs=1500, hidden=128, dim_embedding=20, n_layers=3, lr=1e-3, anneal_rate=0.0,
                  eval_every=50, patience=400, ls_passes=0, n_round_samples=8, refine_sa=False, refine_reads=0)
    r = solve_qubo_gnn(q, h, device="cuda", seed=0); return np.asarray(r["x"])[:q.n], r["time"]


def sa_warm(q, x0, k):
    t0 = time.time(); bqm = q.to_dimod()
    init = [{i: int(x0[i]) for i in range(q.n)} for _ in range(k)]
    res = neal.SimulatedAnnealingSampler().sample(bqm, num_reads=k, seed=0, initial_states=init)
    b = res.first; x = np.array([b.sample[i] for i in range(q.n)], np.int8); return q.energy(x), time.time() - t0


def build_frustrated(N=150, K=20, seed=0):
    rng = np.random.default_rng(seed); B = rng.standard_normal((N, 6)) * 0.02
    Sig = B @ B.T + np.diag((np.abs(rng.standard_normal(N)) * 0.01 + 0.005) ** 2); Sig /= np.abs(Sig).mean()
    Rm = rng.standard_normal((N, N)); Rm = 0.5 * (Rm + Rm.T); np.fill_diagonal(Rm, 0); Rm /= np.abs(Rm[Rm != 0]).mean()
    Qo = 0.7 * Rm + 0.3 * Sig; A = 4 * np.abs(Qo[Qo != 0]).mean()
    Q = Qo + A * (np.ones((N, N)) - np.eye(N)); np.fill_diagonal(Q, np.diag(Q) + A * (1 - 2 * K)); return QUBO(Q)


def main():
    insts = {}
    insts["frustr150"] = build_frustrated()
    rng = np.random.default_rng(3); N = 40; B = rng.standard_normal((N, 6)) * 0.02
    Sig = B @ B.T + np.diag((np.abs(rng.standard_normal(N)) * 0.01 + 0.005) ** 2)
    q4, _ = weight_qubo(rng.standard_normal(N) * 0.004 + 0.003, 0.5 * (Sig + Sig.T), n_bits=4, risk_aversion=0.5, return_weight=0.5)
    insts["weight160"] = q4
    out = {}
    for nm, q in insts.items():
        best = tabu_qubo(q, num_reads=200, seed=0)["energy"]
        x0, tg = gnn_init(q)
        print(f"\n=== {nm} (n={q.n}) best(tabu200)={best:.5f}; GNN-init {tg:.1f}s ===", flush=True)
        print(f"{'k':>4}{'coldSA gap%':>13}{'coldSA t':>10}{'warmSA gap%':>13}{'warmSA t':>10}")
        rows = []
        for k in [1, 4, 16, 64]:
            r = sa_qubo(q, num_reads=k, seed=0); cg = (r["energy"] - best) / abs(best) * 100; ct = r["time"]
            ew, wt = sa_warm(q, x0, k); wg = (ew - best) / abs(best) * 100
            print(f"{k:>4}{cg:>13.3f}{ct:>10.3f}{wg:>13.3f}{wt:>10.3f}", flush=True)
            rows.append({"k": k, "cold_gap%": cg, "cold_t": ct, "warm_gap%": wg, "warm_t": wt})
        out[nm] = {"best": best, "gnn_init_t": tg, "rows": rows}
    json.dump(out, open(os.path.join(HERE, "results", "e27_gnn_guided_sa.json"), "w"), indent=2)
    print("saved", flush=True)


if __name__ == "__main__":
    main()
