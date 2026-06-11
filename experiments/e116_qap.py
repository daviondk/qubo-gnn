"""E116: Quadratic Assignment Problem (QAP) on QAPLIB -- NOT in QIGNN (assignment/permutation).
min_pi sum_ij F[i,j] D[pi[i],pi[j]]. Our QUBO-QAP relaxation (softmax assignment + column penalty) +
Hungarian decode + 2-opt LS. Metric: gap to best-known (%). DL baselines: Two-Stage GPN 9-30% (2024).
Run in .venv."""
import sys, numpy as np, torch
from scipy.optimize import linear_sum_assignment
DEV = "cuda" if torch.cuda.is_available() else "cpu"
BEST = {"nug12": 578, "nug20": 2570, "tai15a": 388214, "chr12a": 9552, "had20": 6922, "rou12": 235528}


def load(path):
    nums = open(path).read().split(); n = int(nums[0]); rest = list(map(float, nums[1:]))
    F = np.array(rest[:n * n]).reshape(n, n); D = np.array(rest[n * n:2 * n * n]).reshape(n, n)
    return n, F, D


def cost(perm, F, D):
    return float((F * D[np.ix_(perm, perm)]).sum())


def two_opt(perm, F, D):
    n = len(perm); improved = True
    while improved:
        improved = False; c = cost(perm, F, D)
        for i in range(n):
            for j in range(i + 1, n):
                perm[i], perm[j] = perm[j], perm[i]
                c2 = cost(perm, F, D)
                if c2 < c - 1e-9: c = c2; improved = True
                else: perm[i], perm[j] = perm[j], perm[i]
    return perm


def solve(n, F, D, restarts=8, epochs=2000):
    Ft = torch.tensor(F, dtype=torch.float32, device=DEV); Dt = torch.tensor(D, dtype=torch.float32, device=DEV)
    best = 1e18; bestp = None
    for r in range(restarts):
        torch.manual_seed(r); logits = torch.randn(n, n, device=DEV, requires_grad=True)
        opt = torch.optim.Adam([logits], lr=0.05)
        for ep in range(epochs):
            T = max(0.1, 1.0 - ep / epochs)
            P = torch.softmax(logits / T, 1)  # row-stochastic (each facility -> location)
            energy = (P * (Ft @ P @ Dt.t())).sum()  # sum_ik P[i,k] (F P D^T)[i,k]
            colpen = ((P.sum(0) - 1) ** 2).sum()
            loss = energy + (0.5 + 5.0 * ep / epochs) * (F.max() * D.max()) * colpen / n
            opt.zero_grad(); loss.backward(); opt.step()
        P = torch.softmax(logits, 1).detach().cpu().numpy()
        ri, ci = linear_sum_assignment(-P)  # facility ri -> location ci
        perm = np.empty(n, dtype=int); perm[ri] = ci
        perm = two_opt(list(perm), F, D)
        c = cost(perm, F, D)
        if c < best: best = c; bestp = perm
    return best


def main():
    print("=== QAP on QAPLIB: gap to best-known (%, lower=better) | DL: Two-Stage GPN 9-30% ===", flush=True)
    for inst in ["chr12a", "nug12", "rou12", "nug20", "had20", "tai15a"]:
        path = f"competitors/qaplib/{inst}.dat"
        try:
            n, F, D = load(path)
        except Exception: print(f"  {inst}: no file"); continue
        c = solve(n, F, D); bk = BEST[inst]; gap = 100 * (c - bk) / bk
        v = "BEAT DL-GNN(<9%)" if gap < 9 else ("~DL-GNN(9-30%)" if gap <= 30 else "behind (>30%)")
        print(f"  {inst} (n={n}): OUR cost {int(c)} | best-known {bk} | gap {gap:.1f}% -> {v}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
