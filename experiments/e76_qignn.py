"""E76 (IMPROVE our GNN per ICLR-2026 QIGNN): iterative refinement with HIDDEN-STATE feedback.
Feed the GNN's penultimate hidden state h^t back as dynamic node features each iteration (vs our current
scalar-prob feedback) -> escapes poor local minima. Test on Gset MaxCut vs best-known + our prob-fb numbers
(G14 -0.13%, G22 -0.03%). Run in .venv."""
import sys, os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "experiments"); sys.path.insert(0, "src")
from torch_geometric.nn import SAGEConv
from qubo import QUBO, local_search_1flip
from maxcut import maxcut_qubo, cut_value, GSET_BEST_KNOWN as BK
DEV = "cuda" if torch.cuda.is_available() else "cpu"


class QIGNN(nn.Module):
    def __init__(self, in_static, hidden=128, n_layers=3, dropout=0.1):
        super().__init__(); self.hidden = hidden; self.dropout = dropout
        self.blocks = nn.ModuleList(); self.norms = nn.ModuleList(); self.proj = nn.ModuleList()
        cur = in_static + hidden  # static feats + hidden-state feedback
        for _ in range(n_layers):
            self.blocks.append(nn.ModuleList([SAGEConv(cur, hidden, aggr="mean"), SAGEConv(cur, hidden, aggr="max")]))
            self.norms.append(nn.LayerNorm(hidden)); self.proj.append(nn.Linear(cur, hidden) if cur != hidden else nn.Identity()); cur = hidden
        self.out = nn.Linear(cur, 1); self.act = nn.LeakyReLU()

    def forward(self, xs, ei, hdyn):
        x = torch.cat([xs, hdyn], dim=1)
        for (cm, cx), norm, proj in zip(self.blocks, self.norms, self.proj):
            h = norm(cm(x, ei) + cx(x, ei)); x = self.act(h + proj(x)); x = F.dropout(x, self.dropout, self.training)
        return torch.sigmoid(self.out(x)).squeeze(-1), x


def solve_qignn(qubo, sf, ei, epochs=3000, hidden=128, n_layers=3, lr=1e-3, anneal=2e-4,
                eval_every=200, ls_passes=300, n_round=30, seed=0):
    torch.manual_seed(seed); np.random.seed(seed); rng = np.random.default_rng(seed)
    n = qubo.n; Qt = torch.tensor(qubo.Q / (np.abs(qubo.Q[qubo.Q != 0]).mean() + 1e-12), dtype=torch.float32, device=DEV)
    xs = torch.tensor(sf, dtype=torch.float32, device=DEV)
    net = QIGNN(xs.shape[1], hidden, n_layers).to(DEV); opt = torch.optim.Adam(net.parameters(), lr=lr)
    hdyn = torch.zeros((n, hidden), device=DEV)
    best_x = (sf[:, 0] > sf[:, 0].mean()).astype(np.int8); best_e = qubo.energy(best_x)
    for ep in range(epochs):
        net.train(); probs, hnew = net(xs, ei, hdyn)
        cost = probs @ (Qt @ probs); loss = cost + anneal * ep * (probs * (1 - probs)).sum()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
        hdyn = hnew.detach()  # HIDDEN-STATE FEEDBACK (the QIGNN key idea)
        if ep % eval_every == 0:
            pnp = probs.detach().cpu().numpy()
            for _ in range(n_round):
                xc,_ = local_search_1flip(qubo, (rng.random(n) < pnp).astype(np.int8), ls_passes); e = qubo.energy(xc)
                if e < best_e: best_e, best_x = e, xc
    return {"x": best_x, "energy": best_e}


def main():
    PRIOR = {"G14": -0.13, "G22": -0.03}  # our current prob-fb GNN gaps
    print("=== Gset MaxCut: QIGNN HIDDEN-STATE feedback (3 restarts) ===", flush=True)
    for g in ["G14", "G22"]:
        q = maxcut_qubo(f"Gset/{g}.txt"); bk = BK[g]; n = q.n
        deg = np.asarray(np.abs(q.Q).sum(1)).reshape(-1, 1)
        sf = np.column_stack([(deg - deg.mean()) / (deg.std() + 1e-9), np.ones((n, 1))]).astype(np.float32)
        A = np.asarray(q.Q) - np.diag(np.diag(np.asarray(q.Q))); rr,cc = np.nonzero(A); ei = torch.tensor(np.vstack([rr,cc]), dtype=torch.long, device=DEV)
        best = max(cut_value(q, solve_qignn(q, sf, ei, seed=s)["x"]) for s in range(3))
        gap = (best - bk) / bk * 100
        print(f"  {g} (BK {bk}): QIGNN cut={best:.0f} gap={gap:+.3f}% | our prob-fb prior {PRIOR[g]:+.2f}%", flush=True)


if __name__ == "__main__":
    main()
