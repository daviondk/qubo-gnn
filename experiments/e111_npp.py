"""E111: Number Partitioning (NPP) -- discrepancy |sum +-a_i| (lower=better). Our QUBO-GNN relaxation on the
frustrated energy (sum s_i a_i)^2 vs Karmarkar-Karp (standard NPP baseline) + brute-force optimum (small N).
Run in .venv."""
import sys, numpy as np, torch, heapq
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def karmarkar_karp(a):
    h = [-x for x in a]; heapq.heapify(h)
    while len(h) > 1:
        x = -heapq.heappop(h); y = -heapq.heappop(h); heapq.heappush(h, -(x - y))
    return -h[0]


def brute_opt(a):
    n = len(a); best = sum(a)
    for m in range(1 << (n - 1)):
        s = sum(a[i] if (m >> i) & 1 else -a[i] for i in range(n - 1)) + a[n - 1]
        best = min(best, abs(s))
        if best == 0: break
    return best


def solve_qubo(a, restarts=8, epochs=3000):
    av = torch.tensor(a, dtype=torch.float32, device=DEV); n = len(a)
    best = 1e30
    for r in range(restarts):
        torch.manual_seed(r); logits = torch.zeros(n, device=DEV, requires_grad=True)
        with torch.no_grad(): logits += 0.1 * torch.randn(n, device=DEV)
        opt = torch.optim.Adam([logits], lr=0.03)
        for ep in range(epochs):
            p = torch.sigmoid(logits); s = ((2 * p - 1) * av).sum()
            loss = s * s + 3e-3 * ep * (p * (1 - p)).sum()
            opt.zero_grad(); loss.backward(); opt.step()
        x = (torch.sigmoid(logits).detach().cpu().numpy() > 0.5)
        disc = abs(sum(a[i] if x[i] else -a[i] for i in range(n)))
        # 1-flip polish
        improved = True
        while improved:
            improved = False
            for i in range(n):
                nd = abs((sum(a[j] if x[j] else -a[j] for j in range(n))) - 2 * (a[i] if x[i] else -a[i]))
                if nd < disc: x[i] = not x[i]; disc = nd; improved = True
        best = min(best, disc)
    return int(best)


def main():
    rng = np.random.default_rng(0)
    print("=== Number Partitioning: discrepancy |sum +-a_i| (lower=better) ===", flush=True)
    for n, B, label in [(20, 10**6, "vs OPT"), (40, 10**8, "vs KK"), (60, 10**10, "vs KK")]:
        ours_l, kk_l, opt_l = [], [], []
        for t in range(15):
            a = rng.integers(1, B, n).tolist()
            ours_l.append(solve_qubo(a)); kk_l.append(karmarkar_karp(a))
            if n <= 20: opt_l.append(brute_opt(a))
        mo, mk = np.mean(ours_l), np.mean(kk_l)
        extra = f" | OPT {np.mean(opt_l):.1f}" if opt_l else ""
        v = "BEAT KK" if mo < mk else ("=KK" if mo == mk else "behind KK")
        print(f"N={n} ({label}): OURS {mo:.1f} | KK {mk:.1f}{extra} -> {v}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
