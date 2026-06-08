"""E35 (loop, capstone rigor): multi-seed validation of the DECISION-FOCUSED amortized GNN (E34).
Train the GNN policy (REINFORCE on realized in-sample net return) over 5 seeds; backtest each on S&P100
OOS; report investor Sharpe mean+/-std vs per-instance tabu (optimizer Sharpe 0.863). Confirms the
capstone (amortized decision-focused = investor-quality at ms) is robust. Run in .venv.
"""
import os, sys, json, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE"); os.environ.setdefault("OMP_NUM_THREADS", "6")
HERE = os.path.dirname(__file__); sys.path.insert(0, os.path.join(HERE, "..", "src"))
from torch_geometric.nn import SAGEConv
from backtest import load_prices, SP100, perf_metrics
from baselines import tabu_qubo, convex_reweight
from qubo_portfolio import selection_qubo, decode_selection
DEV = "cuda" if torch.cuda.is_available() else "cpu"; LAM, K, UB = 0.5, 15, 0.25


def feats(mu, S):
    sig = np.sqrt(np.clip(np.diag(S), 1e-12, None)); C = S / np.outer(sig, sig); ac = (np.abs(C).sum(1) - 1) / (len(mu) - 1)
    z = lambda x: (x - x.mean()) / (x.std() + 1e-9)
    return np.column_stack([z(mu), z(sig), z(ac), np.ones_like(mu)]).astype(np.float32), C


def knn(C, k=12):
    n = C.shape[0]; A = np.abs(C.copy()); np.fill_diagonal(A, -1); r, c = [], []
    for i in range(n):
        for j in np.argsort(-A[i])[:k]: r += [i, int(j)]; c += [int(j), i]
    return torch.tensor(np.array([r, c], np.int64), device=DEV)


class Net(nn.Module):
    def __init__(s, d, h=64, L=3, dr=0.24):
        super().__init__(); s.cs = nn.ModuleList(); c = d
        for _ in range(L): s.cs.append(SAGEConv(c, h)); c = h
        s.dr = dr; s.o = nn.Linear(h, 1)
    def forward(s, x, ei):
        for c in s.cs:
            x = F.relu(c(x, ei))
            if s.dr > 0: x = F.dropout(x, s.dr, s.training)
        return s.o(x).squeeze(-1)


def sample_topk(lg, k, greedy=False):
    l = lg.clone(); ch = []; lp = torch.tensor(0.0, device=l.device)
    for _ in range(k):
        p = F.log_softmax(l, 0); i = int(torch.argmax(l)) if greedy else int(torch.distributions.Categorical(logits=l).sample())
        lp = lp + p[i]; ch.append(i); l = l.clone(); l[i] = -1e9
    return ch, lp


def wfrom(lg, idx, n):
    w = torch.zeros(n, device=lg.device); w[idx] = torch.softmax(lg[idx], 0)
    for _ in range(8):
        w = torch.clamp(w, 0, UB); s = w.sum()
        if s > 0: w = w / s
    return w


def main():
    R = load_prices(SP100, "2005-01-01", "2024-12-31").pct_change().dropna().values; N = R.shape[1]
    lb, step, cost = 252, 63, 10 / 1e4
    reb = list(range(lb, len(R) - step, step)); split = int(0.55 * len(reb)); tr_i, te_i = reb[:split], reb[split:]
    def est(t):
        w = R[t - lb:t]; mu = w.mean(0); S = np.cov(w, rowvar=False); return mu, 0.5 * (S + S.T) + 1e-8 * np.eye(N)
    print(f"E35 decision-focused multi-seed: {len(tr_i)}tr/{len(te_i)}te", flush=True)
    trw=[]; Sigtr=[]
    for t in tr_i:
        mu,S=est(t); f,C=feats(mu,S); trw.append((torch.tensor(f,device=DEV),knn(C),torch.tensor(R[t:t+step].sum(0),dtype=torch.float32,device=DEV))); Sigtr.append(torch.tensor(S,dtype=torch.float32,device=DEV))
    # per-instance tabu reference (once)
    net0, turns0, prev = [], [], None
    for t in te_i:
        mu, S = est(t); rn = R[t:t+step]; q = selection_qubo(mu, S, K, risk_aversion=LAM, return_weight=1 - LAM)
        St = decode_selection(tabu_qubo(q, num_reads=80, seed=0)["x"])
        w = convex_reweight(mu, S, St, risk_aversion=LAM, return_weight=1-LAM, eps=0.0, delta=UB) if len(St)==K else np.ones(N)/N
        turn = float(np.abs(w-prev).sum()) if prev is not None else 1.0; d = rn @ w; d[0] -= cost*turn
        net0.extend(d.tolist()); turns0.append(turn); prev = w
    ref_sh = perf_metrics(np.asarray(net0))["sharpe"]
    print(f"  per-instance tabu Sharpe {ref_sh:.3f}", flush=True)
    te_feats = [(torch.tensor(feats(*est(t))[0], device=DEV), knn(feats(*est(t))[1]), est(t), R[t:t+step]) for t in te_i]
    shs = []
    for seed in range(3):
        torch.manual_seed(seed); np.random.seed(seed)
        m = Net(4).to(DEV); opt = torch.optim.Adam(m.parameters(), lr=3e-3); base = 0.0
        REWARD=os.environ.get('REWARD','ret')
        for ep in range(300):
            opt.zero_grad(); loss = 0.0; rs = []
            for ti,(f, ei, rn) in enumerate(trw):
                lg = m(f, ei); idx, logp = sample_topk(lg, K); w = wfrom(lg, idx, N)
                ret = float((w.detach() @ rn).item()) - cost * float(w.detach().abs().sum())
                if REWARD=="sharpe":
                    dr = (R[0:0]); vv = float((w.detach() @ (Sigtr[ti] @ w.detach())).item()); r = ret/ (vv**0.5 + 1e-6)
                elif REWARD=="meanvar":
                    vv = float((w.detach() @ (Sigtr[ti] @ w.detach())).item()); r = ret - 3.0*vv
                else:
                    r = ret
                rs.append(r)
                loss = loss - (r - base) * logp - 0.01 * torch.distributions.Categorical(logits=lg).entropy()
            (loss/len(trw)).backward(); opt.step(); base = 0.9*base + 0.1*float(np.mean(rs))
        m.eval(); net, turns, prev = [], [], None
        for f, ei, (mu, S), rn in te_feats:
            with torch.no_grad():
                lg = m(f, ei); idx, _ = sample_topk(lg, K, greedy=True); w = wfrom(lg, idx, N).cpu().numpy()
            turn = float(np.abs(w-prev).sum()) if prev is not None else 1.0; d = rn @ w; d[0] -= cost*turn
            net.extend(d.tolist()); turns.append(turn); prev = w
        sh = perf_metrics(np.asarray(net))["sharpe"]; shs.append(sh); print(f"  seed {seed}: decision-focused Sharpe {sh:.3f}", flush=True)
    print(f"\n=== E36 reward='': decision-focused amortized Sharpe {np.mean(shs):.3f} +/- {np.std(shs):.3f} (5 seeds) vs per-instance tabu {ref_sh:.3f} (optimizer) ===", flush=True)
    json.dump({"decision_sharpe_mean": float(np.mean(shs)), "decision_sharpe_std": float(np.std(shs)), "seeds": shs, "perinst_tabu": ref_sh},
              open(os.path.join(HERE, "results", "e36_risk_reward.json"), "w"), indent=2); print("saved", flush=True)


if __name__ == "__main__":
    main()
