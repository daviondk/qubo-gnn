"""E50 (loop): does adding GRAPH FEATURES let a linear model match the GNN in the risk-dominated regime?
E46 showed GNN>linear at lambda=0.9 because the pairwise risk term is graph-structured and a per-asset
linear model can't see it. Add eigenvector-centrality + degree of the |corr| graph to the linear model's
features. If linear+graphfeat ~= GNN, it confirms the mechanism (pairwise info) and gives a simpler solver.
Run in .venv."""
import os, sys, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
os.environ.setdefault("KMP_DUPLICATE_LIB_OK","TRUE"); os.environ.setdefault("OMP_NUM_THREADS","6")
HERE=os.path.dirname(__file__); sys.path.insert(0, os.path.join(HERE,"..","src"))
from torch_geometric.nn import SAGEConv
from backtest import load_prices, SP100
from baselines import tabu_qubo
from qubo_portfolio import selection_qubo
LAM_DUMMY=0.0; DEV="cuda" if torch.cuda.is_available() else "cpu"

def base_feats(mu,S):
    sig=np.sqrt(np.clip(np.diag(S),1e-12,None)); C=S/np.outer(sig,sig); ac=(np.abs(C).sum(1)-1)/(len(mu)-1)
    z=lambda x:(x-x.mean())/(x.std()+1e-9)
    return np.column_stack([z(mu),z(sig),z(ac),np.ones_like(mu)]).astype(np.float32), C

def graph_feats(mu,S):
    f,C=base_feats(mu,S); A=np.abs(C.copy()); np.fill_diagonal(A,0.0)
    deg=A.sum(1)
    # eigenvector centrality (power iteration on |C|)
    v=np.ones(len(mu))/np.sqrt(len(mu))
    for _ in range(50): v=A@v; n=np.linalg.norm(v); v=v/(n+1e-12)
    z=lambda x:(x-x.mean())/(x.std()+1e-9)
    extra=np.column_stack([z(deg),z(v)]).astype(np.float32)
    return np.concatenate([f,extra],1), C

def knn(C,k=12):
    n=C.shape[0];A=np.abs(C.copy());np.fill_diagonal(A,-1);r,c=[],[]
    for i in range(n):
        for j in np.argsort(-A[i])[:k]: r+=[i,int(j)];c+=[int(j),i]
    return torch.tensor(np.array([r,c],np.int64),device=DEV)

class GNNNet(nn.Module):
    def __init__(s,d,h=64,L=3,dr=0.24):
        super().__init__(); s.cs=nn.ModuleList(); c=d
        for _ in range(L): s.cs.append(SAGEConv(c,h)); c=h
        s.dr=dr; s.o=nn.Linear(h,1)
    def forward(s,x,ei):
        for c in s.cs:
            x=F.relu(c(x,ei))
            if s.dr>0: x=F.dropout(x,s.dr,s.training)
        return s.o(x).squeeze(-1)

class LinNet(nn.Module):
    def __init__(s,d): super().__init__(); s.o=nn.Linear(d,1)
    def forward(s,x,ei): return s.o(x).squeeze(-1)

def main():
    R=load_prices(SP100,"2005-01-01","2024-12-31").pct_change().dropna().values; N=R.shape[1]
    lb,step=252,21; idx=list(range(lb,len(R),step)); sp=int(0.7*len(idx)); tr_i,te_i=idx[:sp],idx[sp:][:40]
    def est(t):
        w=R[t-lb:t];mu=w.mean(0);S=np.cov(w,rowvar=False);return mu,0.5*(S+S.T)+1e-8*np.eye(N)
    K=15; lam=0.9
    # precompute labels (tabu) + refs
    def prep(ts):
        o=[]
        for t in ts:
            mu,S=est(t); q=selection_qubo(mu,S,K,risk_aversion=lam,return_weight=1-lam)
            r=tabu_qubo(q,num_reads=200,seed=0); xi=np.flatnonzero(np.asarray(r["x"])>0.5)
            sel=xi if len(xi)==K else np.argsort(-np.asarray(r["x"]))[:K]
            o.append({"mu":mu,"S":S,"q":q,"ref":r["energy"],"sel":sel})
        return o
    TR,TE=prep(tr_i),prep(te_i)
    print(f"E50 graph-features in risk-dominated regime (lambda={lam}, K={K})",flush=True)
    def run(name, featfn, Cls, use_graph):
        torch.manual_seed(0); np.random.seed(0)
        TRf=[(featfn(it["mu"],it["S"]),it["sel"]) for it in TR]; din=TRf[0][0][0].shape[1]
        m=Cls(din).to(DEV) if Cls is LinNet else Cls(din,64,3).to(DEV)
        opt=torch.optim.Adam(m.parameters(),lr=1.3e-3); pw=torch.tensor([(N-K)/K],device=DEV)
        cache=[(torch.tensor(f,device=DEV), knn(C), torch.tensor(np.isin(np.arange(N),sel).astype(np.float32),device=DEV)) for (f,C),sel in TRf]
        for ep in range(400):
            m.train(); perm=np.random.permutation(len(cache))
            for bi in range(0,len(cache),16):
                opt.zero_grad(); bl=0.0
                for ii in perm[bi:bi+16]:
                    ft,ei,lab=cache[ii]; bl=bl+F.binary_cross_entropy_with_logits(m(ft,ei),lab,pos_weight=pw)
                (bl/max(1,len(perm[bi:bi+16]))).backward(); opt.step()
        m.eval(); g=[]
        for it in TE:
            f,C=featfn(it["mu"],it["S"])
            with torch.no_grad(): p=m(torch.tensor(f,device=DEV),knn(C)).cpu().numpy()
            xk=np.zeros(N,np.int8); xk[np.argsort(-p)[:K]]=1
            g.append((it["q"].energy(xk)-it["ref"])/abs(it["ref"])*100 if abs(it["ref"])>1e-12 else 0.0)
        print(f"  {name:<22}: gap vs tabu {np.mean(g):.3f}%",flush=True); return float(np.mean(g))
    out={}
    out["Linear (base feats)"]=run("Linear (base 4 feats)", base_feats, LinNet, False)
    out["Linear + graph feats"]=run("Linear + graph feats", graph_feats, LinNet, True)
    out["GNN (base feats)"]=run("GNN (base feats)", base_feats, GNNNet, True)
    json.dump(out, open(os.path.join(HERE,"results","e50_graphfeat.json"),"w"), indent=2); print("saved",flush=True)

if __name__=="__main__": main()
