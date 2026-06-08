"""E52 (loop): lambda-CONDITIONED amortization -- one model for the whole efficient frontier. Add the
risk-aversion lambda as a node feature; train on instances with RANDOM lambda; evaluate gap vs tabu at
lambda in {0.1,0.5,0.9}. Can a single amortized model serve any risk preference at inference (vs needing
a per-lambda model)? Run in .venv."""
import os, sys, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
os.environ.setdefault("KMP_DUPLICATE_LIB_OK","TRUE"); os.environ.setdefault("OMP_NUM_THREADS","6")
HERE=os.path.dirname(__file__); sys.path.insert(0, os.path.join(HERE,"..","src"))
from torch_geometric.nn import SAGEConv
from backtest import load_prices, SP100
from baselines import tabu_qubo
from qubo_portfolio import selection_qubo
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
    lb,step=252,21; idx=list(range(lb,len(R),step)); sp=int(0.7*len(idx)); tr_i,te_i=idx[:sp],idx[sp:][:40]
    def est(t):
        w=R[t-lb:t];mu=w.mean(0);S=np.cov(w,rowvar=False);return mu,0.5*(S+S.T)+1e-8*np.eye(N)
    rng=np.random.default_rng(0)
    # train data: each window with a RANDOM lambda
    TR=[]
    for t in tr_i:
        for _ in range(2):  # 2 random lambdas per window
            lam=float(rng.choice([0.1,0.3,0.5,0.7,0.9]))
            mu,S=est(t); q=selection_qubo(mu,S,K,risk_aversion=lam,return_weight=1-lam)
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
    print("E52 lambda-conditioned amortization (one model, eval across lambda)",flush=True)
    out={}
    for lam in [0.1,0.5,0.9]:
        g=[]
        for t in te_i:
            mu,S=est(t); q=selection_qubo(mu,S,K,risk_aversion=lam,return_weight=1-lam)
            ref=tabu_qubo(q,num_reads=120,seed=0)["energy"]; f,C=feats(mu,S,lam)
            with torch.no_grad(): p=m(torch.tensor(f,device=DEV),knn(C)).cpu().numpy()
            xk=np.zeros(N,np.int8); xk[np.argsort(-p)[:K]]=1
            g.append((q.energy(xk)-ref)/abs(ref)*100 if abs(ref)>1e-12 else 0.0)
        out[lam]=float(np.mean(g)); print(f"  lambda={lam}: lambda-conditioned model gap vs tabu {np.mean(g):.3f}%",flush=True)
    json.dump(out, open(os.path.join(HERE,"results","e52_lambda_conditioned.json"),"w"), indent=2); print("saved",flush=True)

if __name__=="__main__": main()
