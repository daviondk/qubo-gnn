"""E115: K-SAT (RandCSPBench, arXiv:2602.18419, 2026, Angelini/Bocconi) -- K-SAT is NOT in QIGNN.
Metric: Score = % of satisfiable instances solved (find 0-unsat assignment), higher=better. N=256,
alpha in [3,5] for 3-SAT (matching their range). Our unsupervised relaxation + greedy 1-flip vs WalkSAT
(classical oracle for satisfiability + comparison). Their GNN baselines: NeuroSAT 84.48, QuerySAT 92.38;
classical FMS 99.98 (on 3-SAT). Run in .venv."""
import sys, numpy as np, torch
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def gen_ksat(K, N, M, rng):
    cl = np.zeros((M, K), dtype=np.int64)
    for i in range(M):
        vs = rng.choice(N, K, replace=False); sg = rng.integers(0, 2, K) * 2 - 1
        cl[i] = (vs + 1) * sg
    return cl


def nunsat(x, vi, sg):
    return int(((x[vi] == 1) != (sg > 0)).all(1).sum())


def walksat(cl, N, rng, max_flips, restarts, p=0.5):
    vi = np.abs(cl) - 1; sg = (cl > 0); M = len(cl)
    # var -> clause membership
    incl = [[] for _ in range(N)]
    for ci in range(M):
        for v in vi[ci]: incl[v].append(ci)
    for r in range(restarts):
        x = rng.integers(0, 2, N).astype(np.int8)
        for f in range(max_flips):
            unsat_mask = ((x[vi] == 1) != (sg > 0)).all(1)
            uc = np.where(unsat_mask)[0]
            if len(uc) == 0: return True
            c = uc[rng.integers(len(uc))]
            vars_c = vi[c]
            if rng.random() < p:
                v = vars_c[rng.integers(len(vars_c))]
            else:
                # min break-count
                best, bv = 10**9, vars_c[0]
                for v in vars_c:
                    x[v] ^= 1
                    bc = ((x[vi[incl[v]]] == 1) != (sg[incl[v]] > 0)).all(1).sum()
                    x[v] ^= 1
                    if bc < best: best = bc; bv = v
                v = bv
            x[v] ^= 1
    return False


def our_solve(cl, N, restarts=8, epochs=2000):
    vi = np.abs(cl) - 1; sg = (cl > 0)
    var = torch.tensor(vi, dtype=torch.long, device=DEV); sgn = torch.tensor(sg.astype(np.float32), device=DEV)
    M = len(cl)
    inc = [[] for _ in range(N)]
    for ci in range(M):
        for v in vi[ci]: inc[v].append(ci)
    for r in range(restarts):
        torch.manual_seed(r); logits = torch.zeros(N, device=DEV, requires_grad=True)
        with torch.no_grad(): logits += 0.1 * torch.randn(N, device=DEV)
        opt = torch.optim.Adam([logits], lr=0.05)
        for ep in range(epochs):
            p = torch.sigmoid(logits); pv = p[var]; lit = sgn * pv + (1 - sgn) * (1 - pv)
            loss = (1 - lit).prod(1).sum() + 2e-3 * ep * (p * (1 - p)).sum()
            opt.zero_grad(); loss.backward(); opt.step()
        x = (torch.sigmoid(logits).detach().cpu().numpy() > 0.5).astype(np.int8)
        # greedy 1-flip to local min
        u = nunsat(x, vi, sg); improved = True
        while improved and u > 0:
            improved = False
            for v in range(N):
                x[v] ^= 1; u2 = nunsat(x, vi, sg)
                if u2 < u: u = u2; improved = True
                else: x[v] ^= 1
        if u == 0: return True
    return False


def main():
    K, N = 3, 256; n_inst = 80
    rng = np.random.default_rng(2026)
    sat_cnt = our_cnt = ws_cnt = 0
    print(f"=== 3-SAT N={N} RandCSPBench-style (alpha~U[3,5]) | Score=%solved ===", flush=True)
    for t in range(n_inst):
        alpha = rng.uniform(3.0, 5.0); M = round(alpha * N)
        cl = gen_ksat(K, N, M, rng)
        ws = walksat(cl, N, rng, max_flips=40 * N, restarts=10)
        ours = our_solve(cl, N)
        if ws or ours: sat_cnt += 1  # satisfiable = solved by either
        if ws: ws_cnt += 1
        if ours: our_cnt += 1
        if t % 20 == 19:
            print(f"  [{t+1}/{n_inst}] sat={sat_cnt} ourSolved={our_cnt} wsSolved={ws_cnt}", flush=True)
    s_our = 100 * our_cnt / sat_cnt; s_ws = 100 * ws_cnt / sat_cnt
    print(f"OUR Score {s_our:.2f}% | WalkSAT {s_ws:.2f}% | (paper: NeuroSAT 84.48, QuerySAT 92.38, FMS 99.98)", flush=True)
    v = "BEAT both GNNs" if s_our > 92.38 else ("beat NeuroSAT" if s_our > 84.48 else "behind GNNs")
    print(f"verdict vs GNN baselines: {v}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
