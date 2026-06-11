"""E107: weighted Max-k-SAT vs SplitGNN (arXiv 2511.19544, Table 2, Nov 2025 supervised SOTA).
Setup: uniform-random weighted k-SAT WUF(k,n=60,m=600), weights U[1,100]. Metric: dObj = sum of weights of
UNSAT clauses (lower=better). SplitGNN: WUF(2,60,600)=26.493, WUF(3,60,600)=134.746. Our unsupervised GNN
relaxation + 1-flip. k=2 is purely quadratic (our ideal). Run in .venv."""
import sys, numpy as np, torch
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def gen_wksat(k, n, m, rng):
    cl = np.zeros((m, k), dtype=np.int64)
    for i in range(m):
        vs = rng.choice(n, k, replace=False); sg = rng.integers(0, 2, k) * 2 - 1
        cl[i] = (vs + 1) * sg
    w = rng.integers(1, 101, m).astype(np.float64)
    return cl, w


def wunsat(x, vi, sg, w):  # weighted sum of unsat clauses (all literals false)
    return float((w * ((x[vi] == 1) != (sg > 0)).all(1)).sum())


def solve(n, cl, w, epochs=2500, restarts=6):
    vi = np.abs(cl) - 1; sg = (cl > 0); k = cl.shape[1]
    var = torch.tensor(vi, dtype=torch.long, device=DEV); sgn = torch.tensor(sg.astype(np.float32), device=DEV)
    wt = torch.tensor(w, dtype=torch.float32, device=DEV)
    best = 1e18; bestx = None
    for r in range(restarts):
        torch.manual_seed(r); logits = torch.zeros(n, device=DEV, requires_grad=True)
        with torch.no_grad(): logits += 0.1 * torch.randn(n, device=DEV)
        opt = torch.optim.Adam([logits], lr=0.05)
        for ep in range(epochs):
            p = torch.sigmoid(logits); pv = p[var]; lit = sgn * pv + (1 - sgn) * (1 - pv)
            loss = (wt * (1 - lit).prod(1)).sum() + 2e-3 * ep * (p * (1 - p)).sum()
            opt.zero_grad(); loss.backward(); opt.step()
        x = (torch.sigmoid(logits).detach().cpu().numpy() > 0.5).astype(np.int8)
        e = wunsat(x, vi, sg, w)
        if e < best: best = e; bestx = x.copy()
    x = bestx  # 1-flip polish (n=60 small, fast)
    for _ in range(8):
        imp = False
        for v in range(n):
            x[v] ^= 1; e2 = wunsat(x, vi, sg, w)
            if e2 < best - 1e-9: best = e2; imp = True
            else: x[v] ^= 1
        if not imp: break
    return best


def main():
    SG = {2: 26.493, 3: 134.746}; n, m, K = 60, 600, 40
    print(f"=== weighted Max-k-SAT WUF(k,{n},{m}) vs SplitGNN Table 2 | dObj=sum w of UNSAT (lower=better) ===", flush=True)
    for k in [2, 3]:
        rng = np.random.default_rng(1000 + k); res = []
        for i in range(K):
            cl, w = gen_wksat(k, n, m, rng); res.append(solve(n, cl, w))
        mo = float(np.mean(res)); se = float(np.std(res) / np.sqrt(K))
        v = "BEATS SplitGNN" if mo < SG[k] else "behind SplitGNN"
        print(f"WUF({k},{n},{m}): OURS {mo:.2f}±{se:.2f} | SplitGNN {SG[k]} -> {v}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
