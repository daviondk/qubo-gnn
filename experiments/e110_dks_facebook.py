"""E110: Densest-k-Subgraph on SNAP Facebook (n=4039, m=88234), k=20 -- benchmark from arXiv:2410.07388.
Metric: # edges in the best k-subset (higher=better). Our unsupervised QUBO-DkS relaxation (maximize edges -
cardinality penalty) + swap local search, vs standard greedy DkS heuristics (peeling, grow). Run in .venv."""
import sys, numpy as np, torch
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load(path):
    E = [tuple(map(int, l.split()[:2])) for l in open(path) if l.strip() and not l.startswith("#")]
    nodes = sorted({x for e in E for x in e}); idx = {u: i for i, u in enumerate(nodes)}
    ei = np.array([[idx[a], idx[b]] for a, b in E], dtype=np.int64)
    return len(nodes), ei


def edges_in(S, adj):
    Sset = set(S); return sum(1 for v in S for u in adj[v] if u in Sset and u > v)


def greedy_grow(N, adj, deg, k):
    # start from highest-degree node, add node with most links to current set
    cur = [int(np.argmax(deg))]; inset = np.zeros(N, bool); inset[cur[0]] = True
    gain = np.zeros(N)
    for u in adj[cur[0]]: gain[u] += 1
    while len(cur) < k:
        gain[inset] = -1; v = int(np.argmax(gain)); cur.append(v); inset[v] = True
        for u in adj[v]: gain[u] += 1
    return edges_in(cur, adj)


def greedy_peel(N, ei, adj, deg, k):
    # peel min-degree node until k remain (Charikar-style for fixed k)
    alive = np.ones(N, bool); d = deg.copy().astype(float); cnt = N
    import heapq
    while cnt > k:
        d2 = np.where(alive, d, np.inf); v = int(np.argmin(d2)); alive[v] = False; cnt -= 1
        for u in adj[v]:
            if alive[u]: d[u] -= 1
    S = [i for i in range(N) if alive[i]]
    return edges_in(S, adj)


def swap_ls(S, adj, N, deg):
    inset = np.zeros(N, bool); inset[S] = True; S = list(S)
    # internal degree of each node to set
    intd = np.zeros(N)
    for v in range(N):
        for u in adj[v]:
            if inset[u]: intd[v] += 1
    improved = True
    while improved:
        improved = False
        # node in S with least internal degree; node outside with most
        inn = [(intd[v], v) for v in S]; out = [(intd[v], v) for v in range(N) if not inset[v]]
        vmin = min(inn)[1]; vmax = max(out)[1]
        # swap gain: edges gained by vmax (to S\{vmin}) - edges lost by vmin
        gain = intd[vmax] - (1 if (vmin in adj[vmax]) else 0) - intd[vmin]
        if gain > 0:
            inset[vmin] = False; inset[vmax] = True; S.remove(vmin); S.append(vmax)
            for u in adj[vmin]: intd[u] -= 1
            for u in adj[vmax]: intd[u] += 1
            improved = True
    return edges_in(S, adj)


def solve_qubo(N, ei, adj, deg, k, restarts=5, epochs=1500):
    E0 = torch.tensor(ei[:, 0], device=DEV); E1 = torch.tensor(ei[:, 1], device=DEV)
    best = -1; bestS = None
    for r in range(restarts):
        torch.manual_seed(r); logits = torch.zeros(N, device=DEV, requires_grad=True)
        with torch.no_grad(): logits += 0.01 * torch.randn(N, device=DEV)
        opt = torch.optim.Adam([logits], lr=0.05)
        for ep in range(epochs):
            p = torch.sigmoid(logits)
            edges = (p[E0] * p[E1]).sum()
            card = (p.sum() - k) ** 2
            loss = -edges + (0.5 + 3.0 * ep / epochs) * card
            opt.zero_grad(); loss.backward(); opt.step()
        p = torch.sigmoid(logits).detach().cpu().numpy()
        S = list(np.argsort(-p)[:k])
        e = swap_ls(S, adj, N, deg)
        if e > best: best = e
    return best


def main():
    N, ei = load("competitors/snap/facebook.txt")
    adj = [[] for _ in range(N)]
    for a, b in ei: adj[a].append(b); adj[b].append(a)
    deg = np.array([len(a) for a in adj])
    print(f"=== Densest-k-Subgraph SNAP Facebook (N={N}, M={len(ei)}) | #edges in k-subset (higher=better) ===", flush=True)
    for k in [10, 20, 30]:
        gg = greedy_grow(N, adj, deg, k)
        gp = greedy_peel(N, ei, adj, deg, k)
        ours = solve_qubo(N, ei, adj, deg, k)
        maxposs = k * (k - 1) // 2
        best_base = max(gg, gp)
        v = "BEAT greedy" if ours > best_base else ("=greedy" if ours == best_base else "behind greedy")
        print(f"k={k}: OURS {ours} | greedy-grow {gg} | greedy-peel {gp} | max-clique {maxposs} -> {v}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
