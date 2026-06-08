"""E104 (VALID NEW comparison vs SOTA HyperSAT 2025): Weighted Max-3-SAT on SATLIB uf100-430 (SAME data),
weights ~U[1,10] per clause (SAME scheme), metric = avg weighted UNSAT clauses (SAME metric, lower=better).
Our unsupervised relaxation (trainable logits + cubic weighted-unsat loss + anneal + 1-flip polish).
HyperSAT ~15.64, baselines 32.48/99.15. Run in .venv."""
import sys, glob, numpy as np, torch
sys.path.insert(0, "src")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def parse_cnf(path):
    clauses = []; nv = 0
    for ln in open(path):
        if ln.startswith("p"): nv = int(ln.split()[2]); continue
        if ln.startswith("c") or ln.startswith("%") or ln.startswith("0") or not ln.strip(): continue
        lits = [int(x) for x in ln.split() if x != "0" and x.lstrip("-").isdigit()]
        if len(lits) >= 2: clauses.append(lits[:3])
    return nv, clauses


def w_unsat_discrete(x, clauses, w):
    tot = 0.0
    for c, wc in zip(clauses, w):
        if all((x[abs(l) - 1] == 1) != (l > 0) for l in c): tot += wc  # all literals false
    return tot


def solve(nv, clauses, w, epochs=2500, restarts=5):
    # precompute clause tensors
    C = len(clauses); var = torch.zeros(C, 3, dtype=torch.long, device=DEV); sgn = torch.zeros(C, 3, device=DEV)
    for ci, c in enumerate(clauses):
        for k in range(3):
            l = c[k] if k < len(c) else c[-1]; var[ci, k] = abs(l) - 1; sgn[ci, k] = 1.0 if l > 0 else 0.0
    wt = torch.tensor(w, dtype=torch.float32, device=DEV)
    best = 1e18; bestx = None
    for r in range(restarts):
        torch.manual_seed(r); logits = torch.zeros(nv, device=DEV, requires_grad=True)
        with torch.no_grad(): logits += 0.1 * torch.randn(nv, device=DEV)
        opt = torch.optim.Adam([logits], lr=0.05)
        for ep in range(epochs):
            p = torch.sigmoid(logits); pv = p[var]  # C x 3
            lit = sgn * pv + (1 - sgn) * (1 - pv)  # literal-true prob
            unsat = (1 - lit).prod(1)  # C
            loss = (wt * unsat).sum() + 2e-3 * ep * (p * (1 - p)).sum()
            opt.zero_grad(); loss.backward(); opt.step()
        x = (torch.sigmoid(logits).detach().cpu().numpy() > 0.5).astype(np.int8)
        e = w_unsat_discrete(x, clauses, w)
        if e < best: best = e; bestx = x.copy()
    # 1-flip polish (WalkSAT-style local search on best)
    x = bestx; improved = True
    while improved:
        improved = False
        for v in range(nv):
            x[v] ^= 1; e2 = w_unsat_discrete(x, clauses, w)
            if e2 < best: best = e2; improved = True
            else: x[v] ^= 1
    return best


def main():
    files = sorted(glob.glob("competitors/satlib/uf100-*.cnf"))[:50]
    rng = np.random.default_rng(0); res = []
    for f in files:
        nv, clauses = parse_cnf(f); w = rng.integers(1, 11, len(clauses)).astype(float)
        res.append(solve(nv, clauses, w))
    m = np.mean(res)
    print(f"=== Weighted Max-3-SAT UF100-430 ({len(files)} instances) ===", flush=True)
    print(f"OUR avg weighted-UNSAT = {m:.2f} +- {np.std(res)/np.sqrt(len(res)):.2f} | HyperSAT ~15.64, baselines 32.48/99.15", flush=True)
    print(f"verdict: {'BEATS HyperSAT' if m < 15.64 else 'beats baselines' if m < 32.48 else 'behind'}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
