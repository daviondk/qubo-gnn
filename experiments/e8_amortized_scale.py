"""E8 (Phase A, Tier-2): strengthen the amortization win AT SCALE + save learning curves & checkpoints.
Trains the Optuna-best amortized GNN (SAGE + dropout + kNN-12 + basic feats) on a LARGE universe
(S&P 500, ~460 assets) where per-instance tabu is expensive -> the ms-inference speedup compounds.
Evaluates gap vs per-instance tabu on held-out S&P500 (test) + NASDAQ100 (OOD, different N).

SAVES (per user request 'save everything useful'):
 - best-val checkpoint  -> experiments/checkpoints/e8_amortized_best.pt
 - learning curve PNG    -> results/figures/fig_e8_learning_curve.png   (train loss + test/OOD gap vs epoch)
 - results JSON          -> experiments/results/e8_amortized_scale.json  (gaps, inference ms, tabu s, speedup)
Run in .venv.
"""
from __future__ import annotations
import os, sys, json, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE"); os.environ.setdefault("OMP_NUM_THREADS", "8")
HERE = os.path.dirname(__file__); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from torch_geometric.nn import SAGEConv
from amortized import sel_obj, DEVICE
from amortized_transfer import windows_from_returns
from datasets import get_returns
from baselines import tabu_qubo
from qubo_portfolio import selection_qubo
K = 30


def basic_feats(mu, S):
    sig = np.sqrt(np.clip(np.diag(S), 1e-12, None)); C = S / np.outer(sig, sig)
    ac = (np.abs(C).sum(1) - 1) / (len(mu) - 1)
    def z(x): return (x - x.mean()) / (x.std() + 1e-9)
    return np.column_stack([z(mu), z(sig), z(ac), np.ones_like(mu)]).astype(np.float32), C


def knn(C, k=12):
    n = C.shape[0]; A = np.abs(C.copy()); np.fill_diagonal(A, -1); r, c = [], []
    for i in range(n):
        for j in np.argsort(-A[i])[:k]:
            r += [i, int(j)]; c += [int(j), i]
    return torch.tensor(np.array([r, c], np.int64), device=DEVICE)


class Net(nn.Module):
    def __init__(self, din, h=64, L=3, drop=0.24):
        super().__init__(); self.cs = nn.ModuleList(); c = din
        for _ in range(L):
            self.cs.append(SAGEConv(c, h)); c = h
        self.drop = drop; self.o = nn.Linear(h, 1)

    def forward(self, x, ei):
        for c in self.cs:
            x = F.relu(c(x, ei))
            if self.drop > 0: x = F.dropout(x, self.drop, self.training)
        return self.o(x).squeeze(-1)


def tabu_ref(mu, S):
    q = selection_qubo(mu, S, K, risk_aversion=0.5, return_weight=0.5)
    t0 = time.time(); r = tabu_qubo(q, num_reads=50, seed=0); dt = time.time() - t0
    idx = np.flatnonzero(np.asarray(r["x"]) > 0.5)
    idx = idx if len(idx) == K else np.argsort(-np.asarray(r["x"]))[:K]
    return idx, sel_obj(idx, mu, S, K), dt


def main():
    R = get_returns("sp500").values; N = R.shape[1]
    w = windows_from_returns(R); sp = int(0.7 * len(w)); tr_raw, te_raw = w[:sp], w[sp:]
    oo_raw = windows_from_returns(get_returns("nasdaq100").values, max_w=40)
    print(f"E8 scale: S&P500 N={N}, K={K}, {len(tr_raw)} train / {len(te_raw)} test / {len(oo_raw)} OOD windows", flush=True)
    print("caching tabu labels/refs (N=460 is slow)...", flush=True)
    t0 = time.time()
    lab = [tabu_ref(mu, S)[0] for mu, S in tr_raw]
    te_ref = [(tabu_ref(mu, S)) for mu, S in te_raw]; oo_ref = [(tabu_ref(mu, S)) for mu, S in oo_raw]
    tabu_t = np.mean([x[2] for x in te_ref] + [x[2] for x in oo_ref])
    print(f"  cached in {time.time()-t0:.0f}s; mean tabu/instance = {tabu_t:.2f}s", flush=True)

    def build(raw):
        out = []
        for mu, S in raw:
            f, C = basic_feats(mu, S); out.append((torch.tensor(f, device=DEVICE), knn(C), mu, S))
        return out
    tr, te, oo = build(tr_raw), build(te_raw), build(oo_raw)
    labs = [torch.tensor(np.isin(np.arange(len(mu)), l).astype(np.float32), device=DEVICE) for (mu, _), l in zip(tr_raw, lab)]

    torch.manual_seed(0); np.random.seed(0)
    m = Net(tr[0][0].shape[1]).to(DEVICE); opt = torch.optim.Adam(m.parameters(), lr=1.3e-3)
    pw = torch.tensor([(N - K) / K], device=DEVICE)

    def gap(insts, refs):
        m.eval(); g = []; t0 = time.time()
        for (x, ei, mu, S), rf in zip(insts, refs):
            with torch.no_grad():
                p = m(x, ei).cpu().numpy()
            g.append((sel_obj(np.argsort(-p)[:K], mu, S, K) - rf[1]) / abs(rf[1]) * 100 if abs(rf[1]) > 1e-12 else 0.0)
        return float(np.mean(g)), (time.time() - t0) / len(insts) * 1000  # mean gap%, inference ms/instance

    curve = []; best = (1e9, None)
    EP = 500
    for ep in range(EP + 1):
        if ep > 0:
            m.train(); perm = np.random.permutation(len(tr)); tl = 0.0
            for bi in range(0, len(tr), 16):
                opt.zero_grad(); bl = 0.0
                for ii in perm[bi:bi + 16]:
                    bl = bl + F.binary_cross_entropy_with_logits(m(tr[ii][0], tr[ii][1]), labs[ii], pos_weight=pw)
                bl = bl / max(1, len(perm[bi:bi + 16])); bl.backward(); opt.step(); tl += float(bl)
        if ep % 25 == 0:
            tg, inf_ms = gap(te, te_ref); og, _ = gap(oo, oo_ref)
            curve.append((ep, tg, og));
            if tg < best[0]: best = (tg, {k: v.cpu() for k, v in m.state_dict().items()})
            print(f"  ep {ep:3d}: test {tg:6.3f}%  OOD {og:6.3f}%  inf {inf_ms:.2f}ms", flush=True)

    # restore best, final eval
    m.load_state_dict({k: v.to(DEVICE) for k, v in best[1].items()}); m.eval()
    tg, inf_ms = gap(te, te_ref); og, _ = gap(oo, oo_ref)
    speedup = tabu_t / (inf_ms / 1000 + 1e-9)
    print(f"\n=== E8 BEST: test {tg:.3f}%  OOD {og:.3f}%  | inf {inf_ms:.2f}ms vs tabu {tabu_t:.2f}s -> {speedup:.0f}x ===", flush=True)

    os.makedirs(os.path.join(HERE, "checkpoints"), exist_ok=True)
    torch.save({"state_dict": best[1], "config": {"hidden": 64, "layers": 3, "dropout": 0.24, "knn": 12, "K": K},
                "test_gap": tg, "ood_gap": og}, os.path.join(HERE, "checkpoints", "e8_amortized_best.pt"))
    ep_a = [c[0] for c in curve]; tgs = [c[1] for c in curve]; ogs = [c[2] for c in curve]
    plt.figure(figsize=(7, 4.5))
    plt.plot(ep_a, tgs, "o-", label="test gap % (S&P500)"); plt.plot(ep_a, ogs, "s-", label="OOD gap % (NASDAQ)")
    plt.axhline(0, ls=":", c="gray"); plt.xlabel("epoch"); plt.ylabel("mean gap vs tabu (%)")
    plt.title(f"E8 amortized @ scale (S&P500 N={N}, K={K}): learning curve"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(HERE, "..", "results", "figures", "fig_e8_learning_curve.png"), dpi=130)
    json.dump({"N": N, "K": K, "test_gap%": tg, "ood_gap%": og, "inference_ms": inf_ms, "tabu_s": tabu_t,
               "speedup": speedup, "curve": curve}, open(os.path.join(HERE, "results", "e8_amortized_scale.json"), "w"), indent=2)
    print("saved: checkpoint, fig_e8_learning_curve.png, e8_amortized_scale.json", flush=True)


if __name__ == "__main__":
    main()
