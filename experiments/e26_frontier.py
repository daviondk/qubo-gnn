"""E26 (loop): quality/speed PARETO FRONTIER of the warm-start system. Train amortized GNN once; on test
windows measure (gap-to-best, time) for amortized-init + k tabu reads (k=0,1,2,4,8,16,40) and cold tabu
at k=4,8,20,80. Plot the frontier -> characterizes the deployable amortized-warm-start contribution.
Run in .venv."""
import os, sys, json, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
os.environ.setdefault("KMP_DUPLICATE_LIB_OK","TRUE"); os.environ.setdefault("OMP_NUM_THREADS","6")
HERE=os.path.dirname(__file__); sys.path.insert(0, os.path.join(HERE,"..","src"))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from torch_geometric.nn import SAGEConv
from backtest import load_prices, SP100
from baselines import tabu_qubo
from qubo_portfolio import selection_qubo, decode_selection
from tabu import TabuSampler
DEV="cuda" if torch.cuda.is_available() else "cpu"; LAM,K=0.5,15
def feats(mu,S):
    sig=np.sqrt(np.clip(np.diag(S),1e-12,None)); C=S/np.outer(sig,sig); ac=(np.abs(C).sum(1)-1)/(len(mu)-1)
    z=lambda x:(x-x.mean())/(x.std()+1e-9); return np.column_stack([z(mu),z(sig),z(ac),np.ones_like(mu)]).astype(np.float32),C
def knn(C,k=12):
    n=C.shape[0];A=np.abs(C.copy());np.fill_diagonal(A,-1);r,c=[],[]
    for i in range(n):
        for j in np.argsort(-A[i])[:k]: r+=[i,int(j)];c+=[int(j),i]
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
R=load_prices(SP100,"2005-01-01","2024-12-31").pct_change().dropna().values; N=R.shape[1]; lb,step=252,21
idx=list(range(lb,len(R),step)); sp=int(0.6*len(idx)); tr_i,te_i=idx[:sp],idx[sp:][:30]
def est(t):
    w=R[t-lb:t];mu=w.mean(0);S=np.cov(w,rowvar=False);return mu,0.5*(S+S.T)+1e-8*np.eye(N)
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
        for ii in perm[bi:bi+16]: bl=bl+F.binary_cross_entropy_with_logits(m(tr[ii][0],tr[ii][1]),tr[ii][2],pos_weight=pw)
        (bl/max(1,len(perm[bi:bi+16]))).backward();opt.step()
m.eval(); print("trained",flush=True)
# precompute per-window q, best (tabu80), amortized init
W=[]
for t in te_i:
    mu,S=est(t);q=selection_qubo(mu,S,K,risk_aversion=LAM,return_weight=1-LAM)
    best=tabu_qubo(q,num_reads=120,seed=0)["energy"]
    f,C=feats(mu,S)
    with torch.no_grad(): p=m(torch.tensor(f,device=DEV),knn(C)).cpu().numpy()
    xa=np.zeros(N,np.int8);xa[np.argsort(-p)[:K]]=1
    W.append((q,best,xa,(time.time())))
def warm(q,x0,k):
    t0=time.time();init=[{i:int(x0[i]) for i in range(q.n)} for _ in range(k)]
    res=TabuSampler().sample(q.to_dimod(),num_reads=k,seed=0,initial_states=init);b=res.first
    x=np.array([b.sample[i] for i in range(q.n)],np.int8);return q.energy(x),time.time()-t0
def cold(q,k):
    r=tabu_qubo(q,num_reads=k,seed=0);return r["energy"],r["time"]
pts={"warm":[],"cold":[],"amort":None}
# amortized-alone
ga=[];ta=[]
for q,best,xa,_ in W:
    ga.append((q.energy(xa)-best)/abs(best)*100 if abs(best)>1e-12 else 0); 
pts["amort"]=(float(np.mean(ga)),0.004)
for k in [1,2,4,8,16,40]:
    g=[];tt=[]
    for q,best,xa,_ in W:
        e,t=warm(q,xa,k);g.append((e-best)/abs(best)*100 if abs(best)>1e-12 else 0);tt.append(t)
    pts["warm"].append((k,float(np.mean(g)),float(np.mean(tt))))
for k in [4,8,20,80]:
    g=[];tt=[]
    for q,best,xa,_ in W:
        e,t=cold(q,k);g.append((e-best)/abs(best)*100 if abs(best)>1e-12 else 0);tt.append(t)
    pts["cold"].append((k,float(np.mean(g)),float(np.mean(tt))))
print("amort-alone gap%.3f t%.4f"%pts["amort"])
for k,g,t in pts["warm"]: print(f"warm k={k}: gap {g:.3f}% t {t:.3f}s")
for k,g,t in pts["cold"]: print(f"cold k={k}: gap {g:.3f}% t {t:.3f}s")
plt.figure(figsize=(7,4.5))
wk=pts["warm"];ck=pts["cold"]
plt.plot([t for _,_,t in wk],[g for _,g,t in wk],"o-",label="amortized warm-start + k tabu",color="red")
plt.plot([t for _,_,t in ck],[g for _,g,t in ck],"s-",label="cold tabu (k reads)",color="black")
plt.scatter([pts["amort"][1]],[pts["amort"][0]],c="green",s=80,label="amortized-alone",zorder=5)
plt.xscale("log");plt.xlabel("time per instance (s, log)");plt.ylabel("gap to best (%)")
plt.title("Quality/speed frontier: amortized warm-start vs cold tabu (S&P100)");plt.legend();plt.grid(alpha=0.3)
plt.tight_layout();plt.savefig(os.path.join(HERE,"..","results","figures","fig_e26_frontier.png"),dpi=130)
json.dump(pts,open(os.path.join(HERE,"results","e26_frontier.json"),"w"),indent=2)
print("saved fig_e26_frontier.png",flush=True)
