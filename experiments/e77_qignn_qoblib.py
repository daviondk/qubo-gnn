"""E77 (TUNE our GNN on QOBLIB portfolio benchmark via QIGNN iterative refinement). Apply hidden-state
feedback iterative-refinement GNN (ICLR-2026 QIGNN idea) to the QOBLIB QUBO, then constraint repair, eval
with THEIR objective vs best-known. Goal: beat our prior feasible -64395 (gap 42%) toward best-known -110525.
Run in .venv."""
import sys, os, json, numpy as np, torch
sys.path.insert(0, "experiments"); sys.path.insert(0, "src")
from e70b_qoblib_feasible import load_qs, load_tbl, feasible
from e72_qoblib_evalobj import eval_obj, load_data
import e72_qoblib_evalobj as E
from e74_qoblib_repair import repair
from e76_qignn import solve_qignn
from qubo import QUBO
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    mp = load_tbl("x"); S, p, cov = load_data()
    q, n = load_qs("experiments/results/qoblib_qs/a010_q0.qs"); BK = -110525; qv = 0.0
    out = {}
    for Pboost in [1e7, 3e7]:
        E.PBOOST = Pboost - 1e7; qb = E.boost(q.Q, mp) if Pboost > 1e7 else q
        Qd = np.asarray(qb.Q); diag = np.diag(Qd).reshape(-1, 1)
        sf = np.column_stack([(diag - diag.mean()) / (diag.std() + 1e-9), np.ones((n, 1))]).astype(np.float32)
        A = Qd - np.diag(np.diag(Qd)); rr, cc = np.nonzero(A); ei = torch.tensor(np.vstack([rr, cc]), dtype=torch.long, device=DEV)
        best = None
        for seed in range(4):  # restarts
            r = solve_qignn(qb, sf, ei, epochs=4000, hidden=128, n_layers=3, anneal=2e-4, eval_every=300, ls_passes=300, n_round=40, seed=seed)
            x = np.asarray(r["x"]).astype(np.int8)[:n]
            xr = repair(x, mp); f = feasible(xr, mp)[0]; o = eval_obj(xr, mp, S, p, cov, qv)
            rawo = eval_obj(x, mp, S, p, cov, qv); rawf = feasible(x, mp)[0]
            if f and (best is None or o < best[0]): best = (o, seed, rawo)
            print(f"  Pboost={Pboost:.0e} seed{seed}: raw obj={rawo:.0f}(feas {rawf}) | repaired obj={o:.0f}(feas {f})", flush=True)
        if best:
            bo, bs, braw = best; print(f"  => BEST QIGNN+repair: obj={bo:.0f} gap={(bo-BK)/abs(BK)*100:+.1f}% (BK {BK}; prior -64395=+41.7%)", flush=True)
            out[f"{Pboost:.0e}"] = {"best_obj": bo, "gap%": (bo - BK) / abs(BK) * 100}
    json.dump(out, open("experiments/results/e77_qignn_qoblib.json", "w"), indent=2); print("saved", flush=True)


if __name__ == "__main__":
    main()
