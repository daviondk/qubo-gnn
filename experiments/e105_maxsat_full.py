"""E105: FULL Weighted Max-3-SAT comparison vs HyperSAT (2025) on ALL 6 SATLIB datasets. Same data, weights
U[1,10], metric avg weighted-unsat. GNN+1flip. Run in .venv."""
import sys, glob, numpy as np, torch
sys.path.insert(0, "experiments"); sys.path.insert(0, "src")
from e104b_maxsat_fast import parse_cnf, solve
HS = {"uf100": 15.64, "uuf100": 20.46, "uf200": 28.98, "uuf200": 35.55, "uf250": 33.24, "uuf250": 41.64}
BASE = {"uf100": 32.48, "uuf100": 41.65, "uf200": 67.38, "uuf200": 81.68, "uf250": 79.06, "uuf250": 100.04}


def files_for(p):
    fs = sorted(glob.glob(f"competitors/satlib/**/{p}*.cnf", recursive=True))
    return [f for f in fs if "/." not in f or True][:25]


def main():
    print("=== Weighted Max-3-SAT: OUR GNN+1flip vs HyperSAT (avg weighted-unsat, lower=better) ===", flush=True)
    rng = np.random.default_rng(0)
    for p in ["uf100", "uuf100", "uf200", "uuf200", "uf250", "uuf250"]:
        fs = files_for(p)
        if not fs: print(f"  {p}: NO FILES", flush=True); continue
        res = []
        for f in fs:
            nv, clauses = parse_cnf(f)
            if len(clauses) == 0: continue
            w = rng.integers(1, 11, len(clauses)).astype(float)
            res.append(solve(nv, clauses, w, epochs=2000, restarts=3))
        m = np.mean(res); hs = HS[p]; bl = BASE[p]
        verd = "BEATS HyperSAT" if m < hs else ("beats Liu-baseline" if m < bl else "behind")
        print(f"  {p}-{'430' if '100' in p else '860' if '200' in p else '1065'} ({len(res)} inst): OURS {m:.2f} | HyperSAT {hs} | Liu {bl} -> {verd}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
