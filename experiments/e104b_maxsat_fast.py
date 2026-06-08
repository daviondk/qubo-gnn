"""E104b (fast vectorized): Weighted Max-3-SAT on SATLIB uf100-430 vs HyperSAT ~15.64. Vectorized weighted-
unsat (numpy), capped 1-flip polish. Run in .venv."""
import sys, glob, numpy as np, torch
sys.path.insert(0, "src")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def parse_cnf(path):
    clauses = []; nv = 0
    for ln in open(path):
        if ln.startswith("p"): nv = int(ln.split()[2]); continue
        if ln[0] in "c%0\n" or not ln.strip(): continue
        lits = [int(x) for x in ln.split() if x.lstrip("-").isdigit() and x != "0"]
        if len(lits) >= 3: clauses.append(lits[:3])
    return nv, np.array(clauses)


def solve(nv, clauses, w, epochs=2000, restarts=4):
    C = len(clauses); vi = np.abs(clauses) - 1; sg = (clauses > 0).astype(np.float32)  # C x 3
    var = torch.tensor(vi, dtype=torch.long, device=DEV); sgn = torch.tensor(sg, device=DEV)
    wt = torch.tensor(w, dtype=torch.float32, device=DEV)
    wnp = w.astype(np.float64)
    def wunsat(x):  # vectorized discrete weighted-unsat
        litfalse = (x[vi] == 1) != (sg > 0)  # C x 3 : literal false
        return float((wnp * litfalse.all(1)).sum())
    best = 1e18; bestx = None
    for r in range(restarts):
        torch.manual_seed(r); logits = torch.zeros(nv, device=DEV, requires_grad=True)
        with torch.no_grad(): logits += 0.1 * torch.randn(nv, device=DEV)
        opt = torch.optim.Adam([logits], lr=0.05)
        for ep in range(epochs):
            p = torch.sigmoid(logits); pv = p[var]; lit = sgn * pv + (1 - sgn) * (1 - pv)
            loss = (wt * (1 - lit).prod(1)).sum() + 2e-3 * ep * (p * (1 - p)).sum()
            opt.zero_grad(); loss.backward(); opt.step()
        x = (torch.sigmoid(logits).detach().cpu().numpy() > 0.5).astype(np.int8)
        e = wunsat(x)
        if e < best: best = e; bestx = x.copy()
    # capped vectorized 1-flip polish
    x = bestx
    for _ in range(4):
        improved = False
        for v in range(nv):
            x[v] ^= 1; e2 = wunsat(x)
            if e2 < best - 1e-9: best = e2; improved = True
            else: x[v] ^= 1
        if not improved: break
    return best


def main():
    files = sorted(glob.glob("competitors/satlib/uf100-*.cnf"))[:30]
    rng = np.random.default_rng(0); res = []
    for i, f in enumerate(files):
        nv, clauses = parse_cnf(f); w = rng.integers(1, 11, len(clauses)).astype(float)
        res.append(solve(nv, clauses, w))
        if i % 10 == 9: print(f"  [{i+1}/30] mean={np.mean(res):.2f}", flush=True)
    m = np.mean(res)
    print(f"=== Weighted Max-3-SAT UF100-430 ({len(files)} inst) ===", flush=True)
    print(f"OUR avg weighted-UNSAT = {m:.2f} | HyperSAT ~15.64, baselines 32.48/99.15 -> {'BEATS HyperSAT' if m<15.64 else 'beats baselines' if m<32.48 else 'behind'}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
