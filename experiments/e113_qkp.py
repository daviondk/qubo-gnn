"""E113: Quadratic Knapsack (QKP) -- NOT in QIGNN (packing w/ capacity, not graph cut/selection).
max sum p_i x_i + sum_{i<j} p_ij x_i x_j  s.t. sum w_i x_i <= C. Billionnet-Soutif-style random instances.
Metric: objective value (higher=better), gap to exact optimum (SCIP). Our QUBO-GNN relaxation + LS vs
greedy + exact. Run in .venv."""
import sys, numpy as np, torch
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def gen_qkp(n, density, rng):
    p = rng.integers(1, 101, n).astype(float)
    P = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < density:
                v = rng.integers(1, 101); P[i, j] = P[j, i] = v
    w = rng.integers(1, 51, n).astype(float); C = float(int(0.5 * w.sum()))
    return p, P, w, C


def obj(x, p, P):
    return float(p[x].sum() + P[np.ix_(x, x)].sum() / 2)


def greedy(p, P, w, C):
    n = len(p); sel = []; cap = C; remain = set(range(n))
    while True:
        best, bg = -1, -1
        for i in remain:
            if w[i] <= cap:
                g = p[i] + sum(P[i, j] for j in sel)
                ratio = g / w[i]
                if ratio > bg: bg = ratio; best = i
        if best < 0: break
        sel.append(best); remain.discard(best); cap -= w[best]
    return obj(sel, p, P)


def exact_scip(p, P, w, C):
    try:
        from pyscipopt import Model, quicksum
        n = len(p); m = Model(); m.hideOutput()
        x = [m.addVar(vtype="B") for _ in range(n)]
        m.addCons(quicksum(w[i] * x[i] for i in range(n)) <= C)
        obj = quicksum(p[i] * x[i] for i in range(n))
        for i in range(n):  # linearize y=x_i*x_j (P>0, max -> y<=x_i, y<=x_j suffices)
            for j in range(i + 1, n):
                if P[i, j] > 0:
                    y = m.addVar(vtype="C", lb=0, ub=1)
                    m.addCons(y <= x[i]); m.addCons(y <= x[j]); obj += float(P[i, j]) * y
        m.setObjective(obj, "maximize"); m.setParam("limits/time", 60); m.optimize()
        return m.getObjVal() if m.getNSols() > 0 else None
    except Exception:
        return None


def solve_qubo(p, P, w, C, restarts=6, epochs=1500):
    n = len(p); pv = torch.tensor(p, dtype=torch.float32, device=DEV)
    Pt = torch.tensor(P, dtype=torch.float32, device=DEV); wt = torch.tensor(w, dtype=torch.float32, device=DEV)
    scale = p.max() + P.max()
    best = -1
    for r in range(restarts):
        torch.manual_seed(r); logits = torch.zeros(n, device=DEV, requires_grad=True)
        with torch.no_grad(): logits += 0.1 * torch.randn(n, device=DEV)
        opt = torch.optim.Adam([logits], lr=0.05)
        for ep in range(epochs):
            q = torch.sigmoid(logits)
            val = (pv * q).sum() + 0.5 * (q @ Pt @ q)
            over = torch.relu((wt * q).sum() - C)
            loss = -val + (0.2 + 2.0 * ep / epochs) * scale * over ** 2 / n
            opt.zero_grad(); loss.backward(); opt.step()
        q = torch.sigmoid(logits).detach().cpu().numpy()
        # greedy feasible decode by q-ratio, then LS
        order = np.argsort(-q); sel = []; cap = C
        for i in order:
            if w[i] <= cap and q[i] > 0.3: sel.append(int(i)); cap -= w[i]
        # add LS: try adding any feasible item with positive gain
        improved = True
        while improved:
            improved = False; cap = C - sum(w[i] for i in sel); sset = set(sel)
            for i in range(n):
                if i not in sset and w[i] <= cap:
                    g = p[i] + sum(P[i, j] for j in sel)
                    if g > 0: sel.append(i); sset.add(i); cap -= w[i]; improved = True
        best = max(best, obj(sel, p, P))
    return best


def main():
    print("=== Quadratic Knapsack (objective, higher=better; gap to exact optimum) ===", flush=True)
    for n, dens in [(60, 0.25), (60, 0.5), (80, 0.5)]:
        rng = np.random.default_rng(int(n * 100 + dens * 10)); ours_g, gr_g = [], []
        for t in range(8):
            p, P, w, C = gen_qkp(n, dens, rng)
            o = solve_qubo(p, P, w, C); g = greedy(p, P, w, C); opt = exact_scip(p, P, w, C)
            if opt and opt > 0:
                ours_g.append(100 * (opt - o) / opt); gr_g.append(100 * (opt - g) / opt)
        if ours_g:
            mo, mg = np.mean(ours_g), np.mean(gr_g)
            v = "BEAT greedy" if mo < mg else ("=greedy" if abs(mo - mg) < 0.05 else "behind greedy")
            print(f"n={n} d={dens}: OUR gap-to-opt {mo:.2f}% | greedy gap {mg:.2f}% -> {v}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
