"""E112: Graph Min-Bisection on Gset -- partition into 2 equal halves minimizing cut (lower=better).
Our QUBO relaxation (min cut + balance penalty) + swap LS, vs Kernighan-Lin (1970, standard) + spectral.
Run in .venv."""
import sys, numpy as np, torch, networkx as nx
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load_gset(path):
    rows = [l.split() for l in open(path) if l.strip()]
    E = rows[1:] if len(rows[0]) == 2 else rows
    N = max(max(int(a), int(b)) for a, b, *_ in E)
    ei = np.array([[int(a) - 1, int(b) - 1] for a, b, *_ in E], dtype=np.int64)
    return N, ei


def cut_of(part, ei):
    return int((part[ei[:, 0]] != part[ei[:, 1]]).sum())


def swap_ls(part, N, adj):
    # balanced swap LS: repeatedly swap the cross-pair with best gain
    part = part.copy()
    ext = np.zeros(N)  # external - internal degree (gain of moving)
    for v in range(N):
        for u in adj[v]:
            ext[v] += 1 if part[u] != part[v] else -1
    improved = True; it = 0
    while improved and it < 2000:
        it += 1; improved = False
        A = np.where(part == 0)[0]; B = np.where(part == 1)[0]
        ga = ext[A]; gb = ext[B]
        ia = A[np.argmax(ga)]; ib = B[np.argmax(gb)]
        conn = 1 if ib in adj[ia] else 0
        gain = ext[ia] + ext[ib] - 2 * conn
        if gain > 0:
            part[ia], part[ib] = 1, 0
            for u in adj[ia]: ext[u] += 2 if part[u] == 1 else -2
            for u in adj[ib]: ext[u] += 2 if part[u] == 0 else -2
            # recompute ext for swapped nodes
            for v in (ia, ib):
                ext[v] = sum(1 if part[u] != part[v] else -1 for u in adj[v])
            improved = True
    return part


def solve_qubo(N, ei, adj, restarts=5, epochs=1500):
    E0 = torch.tensor(ei[:, 0], device=DEV); E1 = torch.tensor(ei[:, 1], device=DEV)
    best = 10**9
    for r in range(restarts):
        torch.manual_seed(r); logits = torch.zeros(N, device=DEV, requires_grad=True)
        with torch.no_grad(): logits += 0.1 * torch.randn(N, device=DEV)
        opt = torch.optim.Adam([logits], lr=0.05)
        for ep in range(epochs):
            p = torch.sigmoid(logits)
            cut = (p[E0] * (1 - p[E1]) + (1 - p[E0]) * p[E1]).sum()
            bal = (p.sum() - N / 2) ** 2
            loss = cut + (0.3 + 2.0 * ep / epochs) * bal
            opt.zero_grad(); loss.backward(); opt.step()
        p = torch.sigmoid(logits).detach().cpu().numpy()
        part = np.zeros(N, dtype=int); part[np.argsort(-p)[:N // 2]] = 1  # top N/2 -> side 1 (balanced)
        part = swap_ls(part, N, adj)
        best = min(best, cut_of(part, ei))
    return best


def main():
    print("=== Min-Bisection on Gset (cut, lower=better) ===", flush=True)
    for g in ["G14", "G15", "G22", "G49", "G50"]:
        try:
            N, ei = load_gset(f"Gset/{g}.txt")
        except Exception: continue
        G = nx.Graph(); G.add_nodes_from(range(N)); G.add_edges_from(ei.tolist())
        adj = [list(G.neighbors(v)) for v in range(N)]
        ours = solve_qubo(N, ei, adj)
        a, b = nx.algorithms.community.kernighan_lin_bisection(G, seed=0)
        kl_part = np.zeros(N, int); 
        for v in b: kl_part[v] = 1
        kl = cut_of(kl_part, ei)
        # spectral
        try:
            import scipy.sparse.linalg as sla; from scipy.sparse import csgraph
            L = nx.laplacian_matrix(G).astype(float)
            vals, vecs = sla.eigsh(L, k=2, which="SM"); fied = vecs[:, 1]
            sp_part = np.zeros(N, int); sp_part[np.argsort(-fied)[:N // 2]] = 1
            sp = cut_of(sp_part, ei)
        except Exception: sp = -1
        best_base = min(kl, sp) if sp > 0 else kl
        v = "BEAT baselines" if ours < best_base else ("=KL" if ours == kl else "behind")
        print(f"{g} (N={N}): OURS {ours} | KL {kl} | spectral {sp} -> {v}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
