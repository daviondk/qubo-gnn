"""E106: Max-3-SAT random instances (N=100, r in {4.00,4.15,4.30}) vs OptGNN Table 2 (arXiv 2310.00526).
SAME setup (random 3-SAT, N vars, M=r*N clauses), SAME metric (avg # UNSATISFIED clauses, lower=better).
Our unsupervised GNN-relaxation; report PURE GNN (vs learned baselines) and GNN+1flip. Run in .venv."""
import sys, numpy as np, torch
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def gen_3sat(N, M, rng):
    cl = np.zeros((M, 3), dtype=np.int64)
    for i in range(M):
        vs = rng.choice(N, 3, replace=False)
        sg = rng.integers(0, 2, 3) * 2 - 1  # +1/-1
        cl[i] = (vs + 1) * sg
    return cl


def unsat_count(x, vi, sg):  # x in {0,1}^N ; clause unsat if all 3 literals false
    litfalse = (x[vi] == 1) != (sg > 0)  # M x 3
    return int(litfalse.all(1).sum())


def solve(N, cl, epochs=2000, restarts=4, do_ls=True):
    vi = np.abs(cl) - 1; sg = (cl > 0)
    var = torch.tensor(vi, dtype=torch.long, device=DEV); sgn = torch.tensor(sg.astype(np.float32), device=DEV)
    best = 10**9; bestx = None
    for r in range(restarts):
        torch.manual_seed(r); logits = torch.zeros(N, device=DEV, requires_grad=True)
        with torch.no_grad(): logits += 0.1 * torch.randn(N, device=DEV)
        opt = torch.optim.Adam([logits], lr=0.05)
        for ep in range(epochs):
            p = torch.sigmoid(logits); pv = p[var]; lit = sgn * pv + (1 - sgn) * (1 - pv)
            loss = (1 - lit).prod(1).sum() + 2e-3 * ep * (p * (1 - p)).sum()
            opt.zero_grad(); loss.backward(); opt.step()
        x = (torch.sigmoid(logits).detach().cpu().numpy() > 0.5).astype(np.int8)
        e = unsat_count(x, vi, sg)
        if e < best: best = e; bestx = x.copy()
    pure = best
    if do_ls:
        x = bestx
        for _ in range(6):
            improved = False
            for v in range(N):
                x[v] ^= 1; e2 = unsat_count(x, vi, sg)
                if e2 < best: best = e2; improved = True
                else: x[v] ^= 1
            if not improved: break
    return pure, best


def main():
    N = 100; ratios = [4.00, 4.15, 4.30]; K = 50
    OPT = {4.00: 4.46, 4.15: 5.15, 4.30: 5.84}; ERD = {4.00: 5.46, 4.15: 6.14, 4.30: 6.79}
    WS = {4.00: 0.14, 4.15: 0.36, 4.30: 0.68}; SP = {4.00: 3.32, 4.15: 3.87, 4.30: 3.94}
    print(f"=== Max-3-SAT N={N}, {K} instances/ratio | metric=avg #UNSAT clauses (lower=better) ===", flush=True)
    print(f"{'r':>5} {'OUR pure':>9} {'OUR+1flip':>10} {'OptGNN':>7} {'ErdosGNN':>8} {'SurvProp':>8} {'WalkSAT100':>10}", flush=True)
    for r in ratios:
        M = round(r * N); rng = np.random.default_rng(int(r * 100))
        pures, lss = [], []
        for k in range(K):
            cl = gen_3sat(N, M, rng); p, b = solve(N, cl); pures.append(p); lss.append(b)
        mp, ml = np.mean(pures), np.mean(lss)
        vp = "WIN" if mp < OPT[r] else "~"
        print(f"{r:>5} {mp:>9.2f} {ml:>10.2f} {OPT[r]:>7} {ERD[r]:>8} {SP[r]:>8} {WS[r]:>10}   pureGNN vs OptGNN: {vp}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
