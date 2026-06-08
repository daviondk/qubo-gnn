"""E53 (loop): real-time efficient-frontier tracing via amortization. To trace the risk-return frontier you
solve the cardinality QUBO at many lambda. Compare (a) per-instance tabu at each lambda (slow) vs (b) the
lambda-conditioned amortized model (E52, ms) tracing the whole frontier. Do the frontiers match, and at what
speedup? Saves a figure. Run in .venv."""
import os, sys, json, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
os.environ.setdefault("KMP_DUPLICATE_LIB_OK","TRUE"); os.environ.setdefault("OMP_NUM_THREADS","6")
HERE=os.path.dirname(__file__); sys.path.insert(0, os.path.join(HERE,"..","src"))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from torch_geometric.nn import SAGEConv
from backtest import load_prices, SP100
from baselines import tabu_qubo
from qubo_portfolio import selection_qubo, decode_selection
DEV="cuda" if torch.cuda.is_available() else "cpu"; K=15

def feats(mu,S,lam):
    sig=np.sqrt(np.clip(np.diag(S),1e-12,None)); C=S/np.outer(sig,sig); ac=(np.abs(C).sum(1)-1)/(len(mu)-1)
    z=lambda x:(x-x.mean())/(x.std()+1e-9)
    return np.column_stack([z(mu),z(sig),z(ac),np.full_like(mu,lam),np.ones_like(mu)]).astype(np.float32), C
def knn(C,k=12):
    n=C.shape[0];A=np.abs(C.copy());np.fill_diagonal(A,-1);r,c=[],[]
    for i in range(n):
        for j in np.argsort(-A[i])[:k]: r+=[i,int(j)];c+=[int(j),i]
    return torch.tensor(np.array([r,c],np.int64),device=DEV)
class Net(nn.Module):
    def __init__(s,d,h=64,L=3,dr=0.24):
        super().__init__(); s.cs=nn.ModuleList(); c=d
        for _ in range(L): s.cs.append(SAGEConv(c,h)); c=h
        s.dr=dr; s.o=nn.Linear(h,1)
    def forward(s,x,ei):
        for c in s.cs:
            x=F.relu(c(x,ei))
            if s.dr>0: x=F.dropout(x,s.dr,s.training)
        return s.o(x).squeeze(-1)

def main():
    R=load_prices(SP100,"2005-01-01","2024-12-31").pct_change().dropna().values; N=R.shape[1]
    lb,step=252,21; idx=list(range(lb,len(R),step)); sp=int(0.7*len(idx)); tr_i=idx[:sp]
    def est(t):
        w=R[t-lb:t];mu=w.mean(0);S=np.cov(w,rowvar=False);return mu,0.5*(S+S.T)+1e-8*np.eye(N)
    rng=np.random.default_rng(0); TR=[]
    for t in tr_i:
        for _ in range(2):
            lam=float(rng.choice([0.1,0.3,0.5,0.7,0.9])); mu,S=est(t); q=selection_qubo(mu,S,K,risk_aversion=lam,return_weight=1-lam)
            r=tabu_qubo(q,num_reads=120,seed=0); xi=np.flatnonzero(np.asarray(r["x"])>0.5)
            sel=xi if len(xi)==K else np.argsort(-np.asarray(r["x"]))[:K]; f,C=feats(mu,S,lam)
            TR.append((torch.tensor(f,device=DEV),knn(C),torch.tensor(np.isin(np.arange(N),sel).astype(np.float32),device=DEV)))
    torch.manual_seed(0); np.random.seed(0)
    m=Net(5).to(DEV); opt=torch.optim.Adam(m.parameters(),lr=1.3e-3); pw=torch.tensor([(N-K)/K],device=DEV)
    for ep in range(400):
        m.train(); perm=np.random.permutation(len(TR))
        for bi in range(0,len(TR),16):
            opt.zero_grad(); bl=0.0
            for ii in perm[bi:bi+16]: bl=bl+F.binary_cross_entropy_with_logits(m(TR[ii][0],TR[ii][1]),TR[ii][2],pos_weight=pw)
            (bl/max(1,len(perm[bi:bi+16]))).backward(); opt.step()
    m.eval()
    # trace frontier on a held-out window
    t=idx[-1]; mu,S=est(t); lams=np.linspace(0.05,0.95,20)
    def pt(sel):
        if len(sel)!=K: return None
        w=np.zeros(N); w[sel]=1/K; ret=float(mu@w)*252; vol=float(np.sqrt(w@S@w))*np.sqrt(252); return vol,ret
    f_tabu=[]; t0=time.time()
    for lam in lams:
        q=selection_qubo(mu,S,K,risk_aversion=lam,return_weight=1-lam); sel=decode_selection(tabu_qubo(q,num_reads=100,seed=0)["x"]); p=pt(sel)
        if p: f_tabu.append(p)
    t_tabu=time.time()-t0
    f_am=[]; t0=time.time()
    for lam in lams:
        ff,C=feats(mu,S,float(lam))
        with torch.no_grad(): pr=m(torch.tensor(ff,device=DEV),knn(C)).cpu().numpy()
        p=pt(np.argsort(-pr)[:K])
        if p: f_am.append(p)
    t_am=time.time()-t0
    f_tabu=np.array(f_tabu); f_am=np.array(f_am)
    print(f"E53 frontier trace: tabu {t_tabu:.2f}s | amortized {t_am:.3f}s ({t_tabu/max(t_am,1e-6):.0f}x faster), {len(lams)} points",flush=True)
    plt.figure(figsize=(7,5))
    plt.plot(f_tabu[:,0],f_tabu[:,1],"o-",label=f"per-instance tabu ({t_tabu:.1f}s)",color="black")
    plt.plot(f_am[:,0],f_am[:,1],"s--",label=f"amortized ({t_am:.2f}s, {t_tabu/max(t_am,1e-6):.0f}x)",color="tab:red")
    plt.xlabel("annualized volatility"); plt.ylabel("annualized return"); plt.title("Cardinality efficient frontier: amortized vs per-instance tabu")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(os.path.join(HERE,"..","results","figures","fig_e53_frontier_trace.png"),dpi=130)
    json.dump({"t_tabu":t_tabu,"t_amortized":t_am,"speedup":t_tabu/max(t_am,1e-6),"n_points":len(lams)}, open(os.path.join(HERE,"results","e53_frontier_trace.json"),"w"), indent=2)
    print("saved fig_e53_frontier_trace.png",flush=True)

if __name__=="__main__": main()
