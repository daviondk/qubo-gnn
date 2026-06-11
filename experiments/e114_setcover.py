"""E114: Set Cover -- min #sets covering all elements (NOT in QIGNN; covering, not graph cut/selection).
Our QUBO relaxation (min sum y_j + penalty*uncovered) + greedy repair, vs greedy (ln n) + exact (SCIP).
Metric: #sets used (lower=better), gap to optimum. Run in .venv."""
import sys, numpy as np, torch
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def gen_sc(n_elem, m_sets, density, rng):
    cover = [set(np.where(rng.random(n_elem) < density)[0]) for _ in range(m_sets)]
    # ensure every element coverable
    for e in range(n_elem):
        if not any(e in c for c in cover): cover[rng.integers(m_sets)].add(e)
    return n_elem, cover


def greedy(n_elem, cover):
    uncov = set(range(n_elem)); used = 0
    while uncov:
        best = max(range(len(cover)), key=lambda j: len(cover[j] & uncov))
        uncov -= cover[best]; used += 1
    return used


def exact(n_elem, cover):
    try:
        from pyscipopt import Model, quicksum
        m = Model(); m.hideOutput(); y = [m.addVar(vtype="B") for _ in cover]
        for e in range(n_elem):
            m.addCons(quicksum(y[j] for j in range(len(cover)) if e in cover[j]) >= 1)
        m.setObjective(quicksum(y), "minimize"); m.setParam("limits/time", 30); m.optimize()
        return m.getObjVal() if m.getNSols() > 0 else None
    except Exception:
        return None


def solve_qubo(n_elem, cover, restarts=6, epochs=1500):
    M = len(cover)
    # element-set incidence
    inc = torch.zeros(n_elem, M, device=DEV)
    for j, c in enumerate(cover):
        for e in c: inc[e, j] = 1.0
    best = 10**9
    for r in range(restarts):
        torch.manual_seed(r); logits = torch.full((M,), -1.0, device=DEV, requires_grad=True)
        opt = torch.optim.Adam([logits], lr=0.05)
        for ep in range(epochs):
            y = torch.sigmoid(logits)
            covered = (inc * y).sum(1)  # coverage count per element
            uncov_pen = torch.relu(1.0 - covered).sum()
            loss = y.sum() + (1.0 + 5.0 * ep / epochs) * uncov_pen
            opt.zero_grad(); loss.backward(); opt.step()
        yb = (torch.sigmoid(logits).detach().cpu().numpy() > 0.5)
        sel = [j for j in range(M) if yb[j]]
        # repair: cover uncovered greedily
        covset = set().union(*[cover[j] for j in sel]) if sel else set()
        uncov = set(range(n_elem)) - covset
        while uncov:
            bj = max(range(M), key=lambda j: len(cover[j] & uncov)); sel.append(bj); uncov -= cover[bj]
        # prune redundant sets
        sel = list(dict.fromkeys(sel))
        for j in sorted(sel, key=lambda j: len(cover[j])):
            others = set().union(*[cover[k] for k in sel if k != j]) if len(sel) > 1 else set()
            if others >= set(range(n_elem)): sel.remove(j)
        best = min(best, len(sel))
    return best


def main():
    print("=== Set Cover: #sets used (lower=better), gap to exact optimum ===", flush=True)
    for ne, ms, d in [(100, 60, 0.1), (200, 100, 0.08), (150, 80, 0.12)]:
        rng = np.random.default_rng(ne + ms); og, gg = [], []
        for t in range(8):
            n, cov = gen_sc(ne, ms, d, rng)
            o = solve_qubo(n, cov); g = greedy(n, cov); opt = exact(n, cov)
            if opt: og.append(o - opt); gg.append(g - opt)
        if og:
            mo, mg = np.mean(og), np.mean(gg)
            v = "BEAT greedy" if mo < mg else ("=greedy" if mo == mg else "behind greedy")
            print(f"elem={ne} sets={ms} d={d}: OUR sets-over-opt {mo:+.2f} | greedy {mg:+.2f} -> {v}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
