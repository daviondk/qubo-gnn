"""Optuna architecture/hyperparameter search for the AMORTIZED GNN (our algorithm's real win).
Objective: minimize 0.5*(test_mean_gap + ood_mean_gap) vs per-instance tabu, training a supervised
amortized GraphSAGE on S&P100 windows and evaluating on held-out S&P100 (test) + NASDAQ100 (OOD).

Tabu labels (train) and tabu references (test/OOD) are computed ONCE and cached, so each trial only
trains+evaluates the GNN (fast). Search: hidden, layers, lr, epochs, dropout, pos_weight scale, kNN k,
feature set {basic,rich}. Run in .venv.  Usage: python experiments/optuna_amortized.py [n_trials]
"""
from __future__ import annotations
import os, sys, json, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE"); os.environ.setdefault("OMP_NUM_THREADS", "8")
HERE = os.path.dirname(__file__); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, optuna
from torch_geometric.nn import SAGEConv
from amortized import sel_obj, DEVICE
from amortized_transfer import windows_from_returns
from backtest import load_prices, SP100
from datasets import get_returns
from baselines import tabu_qubo
from qubo_portfolio import selection_qubo
K = 15


def rich_feats(mu, Sigma, k=10):
    sig = np.sqrt(np.clip(np.diag(Sigma), 1e-12, None)); C = Sigma / np.outer(sig, sig)
    ac = np.abs(C); np.fill_diagonal(ac, 0.0); avgc = ac.sum(1) / (len(mu) - 1); topc = np.sort(ac, 1)[:, -k:].sum(1)
    def z(x): return (x - x.mean()) / (x.std() + 1e-9)
    def rk(x): return np.argsort(np.argsort(x)) / (len(x) - 1)
    return np.column_stack([z(mu), z(sig), z(avgc), rk(mu), rk(sig), z(topc), np.ones_like(mu)]).astype(np.float32), C


def basic_feats(mu, Sigma):
    sig = np.sqrt(np.clip(np.diag(Sigma), 1e-12, None)); C = Sigma / np.outer(sig, sig)
    ac = (np.abs(C).sum(1) - 1) / (len(mu) - 1)
    def z(x): return (x - x.mean()) / (x.std() + 1e-9)
    return np.column_stack([z(mu), z(sig), z(ac), np.ones_like(mu)]).astype(np.float32), C


def knn_edges(C, k):
    n = C.shape[0]; A = np.abs(C.copy()); np.fill_diagonal(A, -1); r, c = [], []
    for i in range(n):
        for j in np.argsort(-A[i])[:k]:
            r += [i, int(j)]; c += [int(j), i]
    return torch.tensor(np.array([r, c], np.int64), device=DEVICE)


class Net(nn.Module):
    def __init__(self, din, h, L, drop):
        super().__init__(); self.cs = nn.ModuleList(); c = din
        for _ in range(L):
            self.cs.append(SAGEConv(c, h)); c = h
        self.drop = drop; self.o = nn.Linear(h, 1)

    def forward(self, x, ei):
        for c in self.cs:
            x = F.relu(c(x, ei))
            if self.drop > 0: x = F.dropout(x, self.drop, self.training)
        return self.o(x).squeeze(-1)


def main():
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    R = load_prices(SP100, "2005-01-01", "2024-12-31").pct_change().dropna().values
    w = windows_from_returns(R); split = int(0.7 * len(w))
    train_raw, test_raw = w[:split], w[split:]
    ood_raw = windows_from_returns(get_returns("nasdaq100").values, max_w=40)
    print(f"caching tabu labels/refs once: {len(train_raw)} train, {len(test_raw)} test, {len(ood_raw)} ood", flush=True)
    def tabu_sel(mu, S):
        q = selection_qubo(mu, S, K, risk_aversion=0.5, return_weight=0.5)
        r = tabu_qubo(q, num_reads=100, seed=0); idx = np.flatnonzero(np.asarray(r["x"]) > 0.5)
        return idx if len(idx) == K else np.argsort(-np.asarray(r["x"]))[:K]
    train_lab = [tabu_sel(mu, S) for mu, S in train_raw]
    test_ref = [sel_obj(tabu_sel(mu, S), mu, S, K) for mu, S in test_raw]
    ood_ref = [sel_obj(tabu_sel(mu, S), mu, S, K) for mu, S in ood_raw]
    print("cached. starting study...", flush=True)

    def build(raw, mode, k):
        out = []
        for mu, S in raw:
            f, C = (rich_feats(mu, S) if mode == "rich" else basic_feats(mu, S))
            out.append((torch.tensor(f, device=DEVICE), knn_edges(C, k), mu, S))
        return out

    def objective(trial):
        h = trial.suggest_categorical("hidden", [32, 64, 128])
        L = trial.suggest_int("layers", 2, 4)
        lr = trial.suggest_float("lr", 5e-4, 8e-3, log=True)
        epochs = trial.suggest_int("epochs", 250, 700, step=150)
        drop = trial.suggest_float("dropout", 0.0, 0.3)
        pw_scale = trial.suggest_float("pos_weight_scale", 0.5, 2.0)
        kk = trial.suggest_categorical("knn_k", [6, 8, 12, 16])
        mode = trial.suggest_categorical("features", ["basic", "rich"])
        tr = build(train_raw, mode, kk); te = build(test_raw, mode, kk); oo = build(ood_raw, mode, kk)
        labs = [torch.tensor(np.isin(np.arange(len(mu)), lab).astype(np.float32), device=DEVICE)
                for (mu, _), lab in zip(train_raw, train_lab)]
        torch.manual_seed(0); np.random.seed(0)
        m = Net(tr[0][0].shape[1], h, L, drop).to(DEVICE); opt = torch.optim.Adam(m.parameters(), lr=lr)
        pw = torch.tensor([pw_scale * (71 - K) / K], device=DEVICE)
        for ep in range(epochs):
            m.train(); perm = np.random.permutation(len(tr))
            for bi in range(0, len(tr), 16):
                opt.zero_grad(); bl = 0.0
                for ii in perm[bi:bi + 16]:
                    logit = m(tr[ii][0], tr[ii][1]); bl = bl + F.binary_cross_entropy_with_logits(logit, labs[ii], pos_weight=pw)
                (bl / max(1, len(perm[bi:bi + 16]))).backward(); opt.step()
        m.eval()
        def ev(insts, refs):
            g = []
            for (x, ei, mu, S), ref in zip(insts, refs):
                with torch.no_grad():
                    p = m(x, ei).cpu().numpy()
                oa = sel_obj(np.argsort(-p)[:K], mu, S, K)
                g.append((oa - ref) / abs(ref) * 100 if abs(ref) > 1e-12 else 0.0)
            return float(np.mean(g))
        tm, om = ev(te, test_ref), ev(oo, ood_ref)
        trial.set_user_attr("test_gap", tm); trial.set_user_attr("ood_gap", om)
        return 0.5 * (tm + om)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_trial
    print(f"\n=== BEST (obj={best.value:.3f}): test={best.user_attrs['test_gap']:.3f}% ood={best.user_attrs['ood_gap']:.3f}% ===")
    print("params:", json.dumps(best.params))
    print("baseline (Exp4 SAGE+rich): test~0.86% ood~0.11% -> obj~0.49")
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    rows = [{"value": t.value, **t.params, **t.user_attrs} for t in study.trials if t.value is not None]
    json.dump({"best_value": best.value, "best_params": best.params, "best_attrs": best.user_attrs,
               "all_trials": sorted(rows, key=lambda r: r["value"])[:15]},
              open(os.path.join(HERE, "results", "optuna_amortized.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
