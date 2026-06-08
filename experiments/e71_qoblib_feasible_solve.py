"""E71 (USER DIRECTIVE): make our solver actually SOLVE QOBLIB feasibly. QOBLIB's penalty7=1e7 is too weak
for generic samplers (E70: all infeasible). We BOOST the constraint penalty (constraint-aware), solve with
GNN/SB/tabu, then evaluate on the ORIGINAL .qs energy + feasibility, vs their best-known solution's energy.
Run in .venv."""
import sys, os, json, time, numpy as np
sys.path.insert(0, "experiments"); sys.path.insert(0, "src")
from e70b_qoblib_feasible import load_qs, load_tbl, feasible, load_sol
from qubo import QUBO
from baselines import tabu_qubo, sa_qubo
from gnn_solver import solve_qubo_gnn, GNNHypers
import simulated_bifurcation as sbif, torch
T, B_TOT, B_CSH, PBOOST = 10, 4, 10, 5e8
h = GNNHypers(model="qrf", epochs=2500, hidden=128, dim_embedding=20, n_layers=3, lr=1e-3, anneal_rate=0.0,
              eval_every=50, patience=500, ls_passes=200, n_round_samples=24, refine_sa=True, refine_reads=40)


def boost(Q, mp):
    Q2 = Q.copy()
    for t in range(T):
        for rhs, sel in [(B_TOT, [(i, 1) for i, v in mp.items() if v[0] == "x" and v[4] == t] + [(i, 2 ** v[1]) for i, v in mp.items() if v[0] == "s2" and v[2] == t]),
                         (B_CSH, [(i, v[3]) for i, v in mp.items() if v[0] == "x" and v[4] == t] + [(i, 2 ** v[1]) for i, v in mp.items() if v[0] == "s1" and v[2] == t])]:
            for a, ca in sel:
                Q2[a, a] += PBOOST * (ca * ca - 2 * rhs * ca)
                for b, cb in sel:
                    if a < b: Q2[a, b] += PBOOST * ca * cb; Q2[b, a] += PBOOST * ca * cb
    return QUBO(Q2)


def main():
    mp = load_tbl("x"); out = {}
    for qtag in ["0", "0.00001", "0.001"]:
        qs = f"experiments/results/qoblib_qs/a010_q{qtag}.qs"
        if not os.path.exists(qs): continue
        q, n = load_qs(qs); qb = boost(q.Q, mp)
        bkx = load_sol(qtag)[:n]; bk_e = q.energy(bkx); bk_f, _ = feasible(bkx, mp)
        print(f"[q={qtag}] their best-known: feasible={bk_f} energy={bk_e:.0f}", flush=True)
        res = {"their_bk_energy": float(bk_e)}
        solvers = [("GNN", lambda: np.asarray(solve_qubo_gnn(qb, h, device="cuda", seed=0)["x"])),
                   ("tabu", lambda: np.asarray(tabu_qubo(qb, num_reads=300, seed=0)["x"])),
                   ("SB", lambda: np.asarray(sbif.minimize(torch.tensor(qb.Q.astype(np.float64)), domain="binary", agents=128, max_steps=40000, best_only=True)[0].cpu()).reshape(-1))]
        for name, getx in solvers:
            x = getx().astype(np.int8)[:n]; f_, v_ = feasible(x, mp); e_orig = q.energy(x)
            gap = (e_orig - bk_e) / abs(bk_e) * 100 if f_ else None
            res[name] = {"feasible": bool(f_), "viol": int(v_), "energy": float(e_orig), "gap_vs_bk%": gap}
            tag = f"gap {gap:+.2f}% vs best-known" if f_ else "INFEASIBLE"
            print(f"   {name}: feasible={f_} viol={v_} energy={e_orig:.0f} -> {tag}", flush=True)
        out[qtag] = res
    json.dump(out, open("experiments/results/e71_qoblib_feasible.json", "w"), indent=2)
    print("saved", flush=True)


if __name__ == "__main__":
    main()
