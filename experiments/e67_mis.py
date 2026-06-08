"""E67: MIS (Maximum Independent Set) on Erdos-Renyi graphs -- canonical hard QUBO (PI-GNN benchmark class).
QUBO: min -sum x_i + 2*sum_{(i,j) in E} x_i x_j. IS size = sum x_i for a valid IS. Our GNN vs SB vs tabu.
Run in .venv."""
import sys,time,os,json,numpy as np,torch
sys.path.insert(0,'src')
from qubo import QUBO
from baselines import tabu_qubo
from gnn_solver import solve_qubo_gnn, GNNHypers
import simulated_bifurcation as sb
h=GNNHypers(model='qrf',epochs=3000,hidden=256,dim_embedding=30,n_layers=4,lr=1e-3,anneal_rate=0.0,eval_every=100,patience=800,ls_passes=300,n_round_samples=24,refine_sa=True,refine_reads=40)
def mis_qubo(n,p,seed):
    rng=np.random.default_rng(seed); A=(rng.random((n,n))<p).astype(float); A=np.triu(A,1); A=A+A.T
    Q=2.0*A.copy(); np.fill_diagonal(Q,-1.0); return QUBO(0.5*(Q+Q.T)), A
def is_size(x,A):
    x=(np.asarray(x)>0.5).astype(int); viol=int(x@A@x/2)
    return int(x.sum()), viol  # size, #edge-violations (0 = valid IS)
res={}
for (n,p) in [(1000,0.01),(3000,0.005)]:
    q,A=mis_qubo(n,p,0)
    t0=time.time(); r=solve_qubo_gnn(q,h,device='cuda',seed=0); tg=time.time()-t0; sg,vg=is_size(r['x'],A)
    t0=time.time(); x,e=sb.minimize(torch.tensor(q.Q.astype(np.float64)),domain='binary',agents=128,max_steps=30000,best_only=True); ts=time.time()-t0; ss,vs=is_size(np.asarray(x.cpu()).reshape(-1),A)
    t0=time.time(); rt=tabu_qubo(q,num_reads=20,seed=0); tt=time.time()-t0; st,vt=is_size(rt['x'],A)
    best=max(s for s,v in [(sg,vg),(ss,vs),(st,vt)] if v==0) if any(v==0 for v in [vg,vs,vt]) else max(sg,ss,st)
    res[f'MIS_n{n}']={'GNN':[sg,vg,tg],'SB':[ss,vs,ts],'tabu':[st,vt,tt],'best':best}
    f=lambda s,v:f'{s}{"" if v==0 else f"(viol{v})"}({(s-best)/best*100:+.1f}%)'
    print(f'MIS n={n} p={p}: GNN {f(sg,vg)} {tg:.0f}s | SB {f(ss,vs)} {ts:.0f}s | tabu {f(st,vt)} {tt:.0f}s (best IS={best})',flush=True)
json.dump(res,open('experiments/results/e67_mis.json','w'),indent=2); print('saved')
