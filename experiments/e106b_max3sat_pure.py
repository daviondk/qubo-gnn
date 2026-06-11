"""E106b: Max-3-SAT vs OptGNN Table 2 -- PURE GNN only (no LS), fast. Fair vs OptGNN (learned, rand-rounding).
N=100, r in {4.00,4.15,4.30}, metric avg #UNSAT. Run in .venv."""
import sys, numpy as np
sys.path.insert(0, "experiments")
from e106_max3sat_optgnn import gen_3sat, solve
def main():
    N = 100; ratios = [4.00, 4.15, 4.30]; K = 60
    OPT = {4.00: 4.46, 4.15: 5.15, 4.30: 5.84}; ERD = {4.00: 5.46, 4.15: 6.14, 4.30: 6.79}
    SP = {4.00: 3.32, 4.15: 3.87, 4.30: 3.94}; WS = {4.00: 0.14, 4.15: 0.36, 4.30: 0.68}
    print(f"=== Max-3-SAT N={N}, {K} inst/ratio | PURE GNN (no LS) | avg #UNSAT (lower=better) ===", flush=True)
    print(f"{'r':>5} {'OUR pure':>9} {'OptGNN':>7} {'ErdosGNN':>8} {'SurvProp':>8} {'WalkSAT':>8}  verdict", flush=True)
    for r in ratios:
        M = round(r * N); rng = np.random.default_rng(int(r * 100)); pures = []
        for k in range(K):
            cl = gen_3sat(N, M, rng); p, _ = solve(N, cl, epochs=2000, restarts=5, do_ls=False); pures.append(p)
        mp = float(np.mean(pures)); se = float(np.std(pures) / np.sqrt(K))
        v = "BEATS OptGNN+ErdosGNN" if mp < OPT[r] else ("beats ErdosGNN" if mp < ERD[r] else "~")
        print(f"{r:>5} {mp:>7.2f}±{se:.2f} {OPT[r]:>7} {ERD[r]:>8} {SP[r]:>8} {WS[r]:>8}  {v}", flush=True)
    print("done", flush=True)
if __name__ == "__main__":
    main()
