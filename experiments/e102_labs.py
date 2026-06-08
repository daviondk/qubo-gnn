"""E102 (VALID comparison: QOBLIB LABS, same data=N, same metric=energy E). Low Autocorrelation Binary
Sequences: E(s)=sum_k (sum_i s_i s_{i+k})^2, s in {-1,+1}. Our unsupervised relaxation (trainable logits,
quartic LABS loss, annealing, restarts) vs QOBLIB best-known E. Run in .venv."""
import sys, numpy as np, torch
sys.path.insert(0, "src")
BEST = {20: 26, 30: 59, 40: 108, 50: 257, 60: 366}  # QOBLIB best-known (N<=40 optimal)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def labs_energy_np(s):
    N = len(s); return int(sum(int((s[:N - k] * s[k:]).sum()) ** 2 for k in range(1, N)))


def solve_labs(N, epochs=4000, restarts=30):
    best = 10 ** 9; rng = np.random.default_rng(0)
    for r in range(restarts):
        torch.manual_seed(r)
        logits = torch.zeros(N, device=DEV, requires_grad=True)
        with torch.no_grad(): logits += 0.1 * torch.randn(N, device=DEV)
        opt = torch.optim.Adam([logits], lr=0.05)
        for ep in range(epochs):
            p = torch.sigmoid(logits); s = 2 * p - 1
            E = sum(((s[:N - k] * s[k:]).sum()) ** 2 for k in range(1, N))
            lam = 2e-3 * ep; loss = E + lam * (p * (1 - p)).sum()
            opt.zero_grad(); loss.backward(); opt.step()
        sd = np.where(torch.sigmoid(logits).detach().cpu().numpy() > 0.5, 1, -1).astype(np.int64)
        e = labs_energy_np(sd); best = min(best, e)
        # local 1-flip polish
        improved = True
        while improved:
            improved = False
            for i in range(N):
                sd[i] *= -1; e2 = labs_energy_np(sd)
                if e2 < best: best = e2; improved = True
                else: sd[i] *= -1
    return best


def main():
    print("=== LABS: OUR relaxation vs QOBLIB best-known (energy E, lower=better) ===", flush=True)
    for N in [20, 30, 40, 50, 60]:
        e = solve_labs(N); bk = BEST[N]; gap = (e - bk) / bk * 100
        print(f"  N={N}: OURS E={e} | best-known {bk} -> {'OPTIMAL' if e==bk else f'+{gap:.1f}%'} (merit F={N*N/(2*e):.2f} vs {N*N/(2*bk):.2f})", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
