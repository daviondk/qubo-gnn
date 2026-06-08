"""E34 (loop, synthesis): the amortized GNN can be EITHER the best OPTIMIZER or the best INVESTOR, both at
ms inference. Live S&P100 backtest (K=15, 10bps) comparing:
  (a) per-instance Tabu+reweight                      (optimizer-quality, ~1.7s/reb)
  (b) amortized-IMITATION (imitate tabu)+reweight     (optimizer-quality at ms)
  (c) amortized-DECISION-FOCUSED GNN policy (REINFORCE on realized in-sample net return), top-K + softmax
      weights                                          (investor-quality at ms)
Report realized OOS Sharpe/Sortino/MaxDD/turnover + per-reb time. Shows (c) reaches the modern-ML
investor Sharpe (~1.0) at ms, while (b) reaches the per-instance optimizer Sharpe (~0.86). Run in .venv.
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


def metrics(net, turns):
    d = np.asarray(net); m = perf_metrics(d); m["turn"] = float(np.mean(turns)); return m


def main():
    R = load_prices(SP100, "2005-01-01", "2024-12-31").pct_change().dropna().values; N = R.shape[1]
    lb, step, cost = 252, 63, 10 / 1e4
    reb = list(range(lb, len(R) - step, step)); split = int(0.55 * len(reb)); tr_i, te_i = reb[:split], reb[split:]
    def est(t):
        w = R[t - lb:t]; mu = w.mean(0); S = np.cov(w, rowvar=False); return mu, 0.5 * (S + S.T) + 1e-8 * np.eye(N)
    print(f"E34 amortized modes: {len(tr_i)}tr/{len(te_i)}te", flush=True)
    # (b) imitation training data
    imi = []
    for t in tr_i:
        mu, S = est(t); q = selection_qubo(mu, S, K, risk_aversion=LAM, return_weight=1 - LAM)
        lab = decode_selection(tabu_qubo(q, num_reads=80, seed=0)["x"]); f, C = feats(mu, S)
        imi.append((torch.tensor(f, device=DEV), knn(C), torch.tensor(np.isin(np.arange(N), lab).astype(np.float32), device=DEV), torch.tensor(R[t:t+step].sum(0), dtype=torch.float32, device=DEV)))
    # train (b) imitation
    torch.manual_seed(0); np.random.seed(0)
    mb = Net(4).to(DEV); ob = torch.optim.Adam(mb.parameters(), lr=1.3e-3); pw = torch.tensor([(N - K) / K], device=DEV)
    for ep in range(400):
        mb.train(); perm = np.random.permutation(len(imi))
        for bi in range(0, len(imi), 16):
            ob.zero_grad(); bl = 0.0
            for ii in perm[bi:bi+16]: bl = bl + F.binary_cross_entropy_with_logits(mb(imi[ii][0], imi[ii][1]), imi[ii][2], pos_weight=pw)
            (bl/max(1,len(perm[bi:bi+16]))).backward(); ob.step()
    mb.eval()
    # train (c) decision-focused REINFORCE on in-sample net return
    torch.manual_seed(0); np.random.seed(0)
    mc = Net(4).to(DEV); oc = torch.optim.Adam(mc.parameters(), lr=3e-3); base = 0.0
    for ep in range(300):
        oc.zero_grad(); loss = 0.0; rs = []
        for f, ei, lab, rn in imi:
            lg = mc(f, ei); idx, logp = sample_topk(lg, K); w = wfrom(lg, idx, N)
            r = float((w.detach() @ rn).item()) - cost * float(w.detach().abs().sum()); rs.append(r)
            loss = loss - (r - base) * logp - 0.01 * torch.distributions.Categorical(logits=lg).entropy()
        (loss/len(imi)).backward(); oc.step(); base = 0.9*base + 0.1*float(np.mean(rs))
    mc.eval(); print("  trained (b) imitation + (c) decision-focused", flush=True)
    acc = {x: {"net": [], "turn": [], "prev": None, "t": 0.0} for x in ["PerInst-Tabu", "Amort-Imitation", "Amort-Decision"]}
    for t in te_i:
        mu, S = est(t); rn = R[t:t+step]; q = selection_qubo(mu, S, K, risk_aversion=LAM, return_weight=1 - LAM)
        f, C = feats(mu, S); ft = torch.tensor(f, device=DEV); ei = knn(C)
        t1 = time.time(); St = decode_selection(tabu_qubo(q, num_reads=80, seed=0)["x"]); acc["PerInst-Tabu"]["t"] += time.time()-t1
        wt = convex_reweight(mu, S, St, risk_aversion=LAM, return_weight=1-LAM, eps=0.0, delta=UB) if len(St)==K else np.ones(N)/N
        t1 = time.time()
        with torch.no_grad(): pb = mb(ft, ei).cpu().numpy()
        Sb = np.argsort(-pb)[:K]; tb = time.time()-t1
        wb = convex_reweight(mu, S, Sb, risk_aversion=LAM, return_weight=1-LAM, eps=0.0, delta=UB)
        t1 = time.time()
        with torch.no_grad():
            lg = mc(ft, ei); idx, _ = sample_topk(lg, K, greedy=True); wc = wfrom(lg, idx, N).cpu().numpy()
        tc = time.time()-t1
        acc["Amort-Imitation"]["t"] += tb; acc["Amort-Decision"]["t"] += tc
        for nm, w in [("PerInst-Tabu", wt), ("Amort-Imitation", wb), ("Amort-Decision", wc)]:
            prev = acc[nm]["prev"]; turn = float(np.abs(w - prev).sum()) if prev is not None else 1.0
            daily = rn @ w; daily[0] -= cost*turn; acc[nm]["net"].extend(daily.tolist()); acc[nm]["turn"].append(turn); acc[nm]["prev"] = w
    print(f"\n{'method':<18}{'Sharpe':>8}{'Sortino':>8}{'MaxDD':>8}{'Turn':>7}{'solve/reb':>11}", flush=True)
    out = {}
    for nm in acc:
        mm = metrics(acc[nm]["net"], acc[nm]["turn"]); spr = acc[nm]["t"]/len(te_i)
        out[nm] = {**mm, "solve_per_reb_s": spr}
        print(f"{nm:<18}{mm['sharpe']:>8.3f}{mm['sortino']:>8.3f}{mm['maxdd']:>8.3f}{mm['turn']:>7.2f}{spr:>11.4f}", flush=True)
    json.dump(out, open(os.path.join(HERE, "results", "e34_amortized_modes.json"), "w"), indent=2); print("saved", flush=True)


if __name__ == "__main__":
    main()
