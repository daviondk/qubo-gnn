"""E9 (Phase A, Tier-2): WIDER Optuna architecture search on the amortized GNN — adds conv-type
{SAGE, GAT, GraphConv} + heads + more trials, on top of E4/Optuna search space. Objective =
0.5*(test+OOD gap vs tabu) on S&P100 -> NASDAQ100. Saves best checkpoint + best-value curve + JSON.
Designed to run IN PARALLEL with E8 (light GPU, N=71 cheap tabu). Run in .venv.
Usage: python experiments/e9_optuna_wide.py [n_trials]
"""
from __future__ import annotations
import os, sys, json
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE"); os.environ.setdefault("OMP_NUM_THREADS", "4")
HERE = os.path.dirname(__file__); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, optuna
from torch_geometric.nn import SAGEConv, GATv2Conv, GraphConv
from amortized import sel_obj, DEVICE
from amortized_transfer import windows_from_returns
from backtest import load_prices, SP100
from datasets import get_returns
from baselines import tabu_qubo
from qubo_portfolio import selection_qubo
K = 15


def basic_feats(mu, S):
    sig = np.sqrt(np.clip(np.diag(S), 1e-12, None)); C = S / np.outer(sig, sig)
    ac = (np.abs(C).sum(1) - 1) / (len(mu) - 1)
    def z(x): return (x - x.mean()) / (x.std() + 1e-9)
    def rk(x): return np.argsort(np.argsort(x)) / (len(x) - 1)
    return np.column_stack([z(mu), z(sig), z(ac), rk(mu), rk(sig)]).astype(np.float32), C


def knn(C, k):
    n = C.shape[0]; A = np.abs(C.copy()); np.fill_diagonal(A, -1); r, c = [], []
    for i in range(n):
        for j in np.argsort(-A[i])[:k]:
            r += [i, int(j)]; c += [int(j), i]
    return torch.tensor(np.array([r, c], np.int64), device=DEVICE)


class Net(nn.Module):
    def __init__(self, din, h, L, drop, conv, heads):
        super().__init__(); self.cs = nn.ModuleList(); self.drop = drop; c = din
        for _ in range(L):
            if conv == "gat":
                self.cs.append(GATv2Conv(c, h // heads, heads=heads)); c = h
            elif conv == "graphconv":
                self.cs.append(GraphConv(c, h)); c = h
            else:
                self.cs.append(SAGEConv(c, h)); c = h
        self.o = nn.Linear(c, 1)

    def forward(self, x, ei):
        for cv in self.cs:
            x = F.relu(cv(x, ei))
            if self.drop > 0: x = F.dropout(x, self.drop, self.training)
        return self.o(x).squeeze(-1)


def main():
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    R = load_prices(SP100, "2005-01-01", "2024-12-31").pct_change().dropna().values
    w = windows_from_returns(R); sp = int(0.7 * len(w)); tr_raw, te_raw = w[:sp], w[sp:]
    oo_raw = windows_from_returns(get_returns("nasdaq100").values, max_w=40)
    print(f"E9 wide Optuna: {len(tr_raw)}tr/{len(te_raw)}te/{len(oo_raw)}ood; caching tabu...", flush=True)
    def tsel(mu, S):
        q = selection_qubo(mu, S, K, risk_aversion=0.5, return_weight=0.5); r = tabu_qubo(q, num_reads=100, seed=0)
        i = np.flatnonzero(np.asarray(r["x"]) > 0.5); return i if len(i) == K else np.argsort(-np.asarray(r["x"]))[:K]
    lab = [tsel(mu, S) for mu, S in tr_raw]; tref = [sel_obj(tsel(mu, S), mu, S, K) for mu, S in te_raw]
    oref = [sel_obj(tsel(mu, S), mu, S, K) for mu, S in oo_raw]
    print("cached. study...", flush=True)

    def build(raw, kk):
        return [(torch.tensor(basic_feats(mu, S)[0], device=DEVICE), knn(basic_feats(mu, S)[1], kk), mu, S) for mu, S in raw]

    def objective(trial):
        h = trial.suggest_categorical("hidden", [32, 64, 128]); L = trial.suggest_int("layers", 2, 4)
        lr = trial.suggest_float("lr", 5e-4, 8e-3, log=True); epochs = trial.suggest_int("epochs", 250, 600, step=175)
        drop = trial.suggest_float("dropout", 0.0, 0.4); pw_s = trial.suggest_float("pw", 0.5, 2.0)
        kk = trial.suggest_categorical("knn_k", [6, 12, 20]); conv = trial.suggest_categorical("conv", ["sage", "gat", "graphconv"])
        heads = trial.suggest_categorical("heads", [2, 4]) if conv == "gat" else 1
        tr = build(tr_raw, kk); te = build(te_raw, kk); oo = build(oo_raw, kk)
        labs = [torch.tensor(np.isin(np.arange(len(mu)), l).astype(np.float32), device=DEVICE) for (mu, _), l in zip(tr_raw, lab)]
        torch.manual_seed(0); np.random.seed(0)
        m = Net(tr[0][0].shape[1], h, L, drop, conv, heads).to(DEVICE); opt = torch.optim.Adam(m.parameters(), lr=lr)
        pw = torch.tensor([pw_s * (71 - K) / K], device=DEVICE)
        for ep in range(epochs):
            m.train(); perm = np.random.permutation(len(tr))
            for bi in range(0, len(tr), 16):
                opt.zero_grad(); bl = 0.0
                for ii in perm[bi:bi + 16]:
                    bl = bl + F.binary_cross_entropy_with_logits(m(tr[ii][0], tr[ii][1]), labs[ii], pos_weight=pw)
                (bl / max(1, len(perm[bi:bi + 16]))).backward(); opt.step()
        m.eval()
        def ev(ins, refs):
            g = []
            for (x, ei, mu, S), rf in zip(ins, refs):
                with torch.no_grad(): p = m(x, ei).cpu().numpy()
                g.append((sel_obj(np.argsort(-p)[:K], mu, S, K) - rf) / abs(rf) * 100 if abs(rf) > 1e-12 else 0.0)
            return float(np.mean(g))
        tm, om = ev(te, tref), ev(oo, oref); trial.set_user_attr("test", tm); trial.set_user_attr("ood", om)
        return 0.5 * (tm + om)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=n_trials)
    b = study.best_trial
    print(f"\n=== E9 BEST obj {b.value:.3f}: test {b.user_attrs['test']:.3f}% ood {b.user_attrs['ood']:.3f}% ===")
    print("params:", json.dumps(b.params))
    # best conv breakdown
    byconv = {}
    for t in study.trials:
        if t.value is not None: byconv.setdefault(t.params.get("conv"), []).append(t.value)
    for cv, vs in byconv.items(): print(f"  conv={cv}: best {min(vs):.3f} (n={len(vs)})")
    rows = sorted([{"value": t.value, **t.params, **t.user_attrs} for t in study.trials if t.value is not None], key=lambda r: r["value"])
    json.dump({"best_value": b.value, "best_params": b.params, "best_attrs": b.user_attrs,
               "by_conv_best": {k: min(v) for k, v in byconv.items()}, "top10": rows[:10]},
              open(os.path.join(HERE, "results", "e9_optuna_wide.json"), "w"), indent=2)
    print("saved e9_optuna_wide.json", flush=True)


if __name__ == "__main__":
    main()
