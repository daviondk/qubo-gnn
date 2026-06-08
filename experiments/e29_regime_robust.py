"""E29 (loop): REGIME ROBUSTNESS of the amortization win. Train amortized GNN on S&P100 rebalances
BEFORE 2016; test on 2016-2024 incl the 2020 COVID crash. Report gap vs per-instance tabu overall AND
for the 2020 crash sub-window. Tests temporal/regime-shift robustness. Run in .venv."""
import os, sys, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
os.environ.setdefault("KMP_DUPLICATE_LIB_OK","TRUE"); os.environ.setdefault("OMP_NUM_THREADS","6")
HERE=os.path.dirname(__file__); sys.path.insert(0, os.path.join(HERE,"..","src"))
from torch_geometric.nn import SAGEConv
from backtest import load_prices, SP100
from baselines import tabu_qubo
from qubo_portfolio import selection_qubo, decode_selection
from amortized import sel_obj
DEV="cuda" if torch.cuda.is_available() else "cpu"; LAM,K=0.5,15
def feats(mu,S):
    sig=np.sqrt(np.clip(np.diag(S),1e-12,None));C=S/np.outer(sig,sig);ac=(np.abs(C).sum(1)-1)/(len(mu)-1)
    z=lambda x:(x-x.mean())/(x.std()+1e-9);return np.column_stack([z(mu),z(sig),z(ac),np.ones_like(mu)]).astype(np.float32),C
def knn(C,k=12):
    n=C.shape[0];A=np.abs(C.copy());np.fill_diagonal(A,-1);r,c=[],[]
    for i in range(n):
        for j in np.argsort(-A[i])[:k]:r+=[i,int(j)];c+=[int(j),i]
    return torch.tensor(np.array([r,c],np.int64),device=DEV)
class Net(nn.Module):
    def __init__(s,d,h=64,L=3,dr=0.24):
        super().__init__();s.cs=nn.ModuleList();c=d
        for _ in range(L):s.cs.append(SAGEConv(c,h));c=h
        s.dr=dr;s.o=nn.Linear(h,1)
    def forward(s,x,ei):
        for c in s.cs:
            x=F.relu(c(x,ei))
            if s.dr>0:x=F.dropout(x,s.dr,s.training)
        return s.o(x).squeeze(-1)
px=load_prices(SP100,"2005-01-01","2024-12-31"); dates=px.index[1:]; R=px.pct_change().dropna().values; N=R.shape[1]
lb,step=252,21; idx=list(range(lb,len(R),step))
def yr(t): return dates[t].year
tr_i=[t for t in idx if yr(t)<2016]; te_i=[t for t in idx if yr(t)>=2016]
def est(t):
    w=R[t-lb:t];mu=w.mean(0);S=np.cov(w,rowvar=False);return mu,0.5*(S+S.T)+1e-8*np.eye(N)
print(f"E29 regime: train<2016 ({len(tr_i)}) test>=2016 ({len(te_i)})",flush=True)
tr=[]
for t in tr_i:
    mu,S=est(t);q=selection_qubo(mu,S,K,risk_aversion=LAM,return_weight=1-LAM)
    lab=decode_selection(tabu_qubo(q,num_reads=80,seed=0)["x"]);f,C=feats(mu,S)
    tr.append((torch.tensor(f,device=DEV),knn(C),torch.tensor(np.isin(np.arange(N),lab).astype(np.float32),device=DEV)))
torch.manual_seed(0);np.random.seed(0)
m=Net(4).to(DEV);opt=torch.optim.Adam(m.parameters(),lr=1.3e-3);pw=torch.tensor([(N-K)/K],device=DEV)
for ep in range(400):
    m.train();perm=np.random.permutation(len(tr))
    for bi in range(0,len(tr),16):
        opt.zero_grad();bl=0.0
        for ii in perm[bi:bi+16]:bl=bl+F.binary_cross_entropy_with_logits(m(tr[ii][0],tr[ii][1]),tr[ii][2],pos_weight=pw)
        (bl/max(1,len(perm[bi:bi+16]))).backward();opt.step()
m.eval()
gaps={"all":[], "2020":[], "other":[]}
for t in te_i:
    mu,S=est(t);q=selection_qubo(mu,S,K,risk_aversion=LAM,return_weight=1-LAM)
    r=tabu_qubo(q,num_reads=80,seed=0);xi=np.flatnonzero(np.asarray(r["x"])>0.5)
    ref=sel_obj(xi if len(xi)==K else np.argsort(-np.asarray(r["x"]))[:K],mu,S,K)
    f,C=feats(mu,S)
    with torch.no_grad():p=m(torch.tensor(f,device=DEV),knn(C)).cpu().numpy()
    g=(sel_obj(np.argsort(-p)[:K],mu,S,K)-ref)/abs(ref)*100 if abs(ref)>1e-12 else 0.0
    gaps["all"].append(g); (gaps["2020"] if yr(t)==2020 else gaps["other"]).append(g)
for k,v in gaps.items():
    if v: print(f"  {k:<6}: gap {np.mean(v):.3f}% (median {np.median(v):.3f}%, n={len(v)})",flush=True)
json.dump({k:[float(np.mean(v)),float(np.median(v)),len(v)] for k,v in gaps.items() if v}, open(os.path.join(HERE,"results","e29_regime_robust.json"),"w"), indent=2)
print("saved",flush=True)
