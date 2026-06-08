"""E66 (USER DIRECTIVE): confirm our GNN-QUBO advantage across HARD QUBO classes vs modern open solvers.
Gset MaxCut G49/G50 (gap to best-known) + Sherrington-Kirkpatrick dense spin-glass (gap to best-found).
Our GNN-QUBO vs simulated-bifurcation (2025) vs tabu. Run in .venv."""
import sys,time,os,json,numpy as np,torch
sys.path.insert(0,'src')
from maxcut import maxcut_qubo, cut_value, GSET_BEST_KNOWN as BK
from qubo import QUBO
from baselines import tabu_qubo
from gnn_solver import solve_qubo_gnn, GNNHypers
import simulated_bifurcation as sb
h=GNNHypers(model='qrf',epochs=3000,hidden=256,dim_embedding=30,n_layers=4,lr=1e-3,anneal_rate=0.0,eval_every=100,patience=800,ls_passes=300,n_round_samples=24,refine_sa=True,refine_reads=40)
def run_qubo(q, ref, label, maxcut=False):
    val=lambda x: cut_value(q,np.asarray(x)) if maxcut else q.energy(np.asarray(x))
    t0=time.time(); r=solve_qubo_gnn(q,h,device='cuda',seed=0); tg=time.time()-t0; vg=val(r['x'])
    t0=time.time(); x,e=sb.minimize(torch.tensor(q.Q.astype(np.float64)),domain='binary',agents=128,max_steps=30000,best_only=True); ts=time.time()-t0; vs=val(np.asarray(x.cpu()).reshape(-1))
    t0=time.time(); rt=tabu_qubo(q,num_reads=20,seed=0); tt=time.time()-t0; vt=val(rt['x'])
    return {'GNN':[vg,tg],'SB':[vs,ts],'tabu':[vt,tt]}
res={}
for g in ['G49','G50']:
    q=maxcut_qubo(f'Gset/{g}.txt'); bk=BK[g]; o=run_qubo(q,bk,g,maxcut=True); res[g]={'BK':bk,**o}
    f=lambda c:f'{c:.0f}({(c-bk)/bk*100:+.2f}%)'
    print(f"{g} (n={q.n} BK={bk}): GNN {f(o['GNN'][0])} {o['GNN'][1]:.0f}s | SB {f(o['SB'][0])} {o['SB'][1]:.0f}s | tabu {f(o['tabu'][0])} {o['tabu'][1]:.0f}s",flush=True)
# SK spin-glass n=800: dense +/-1 couplings, minimize x'Jx (Ising-> use {0,1} QUBO via spin map)
rng=np.random.default_rng(0); N=800
J=rng.choice([-1.0,1.0],size=(N,N)); J=np.triu(J,1); J=J+J.T  # symmetric, zero diag
Q=4*J.copy(); np.fill_diagonal(Q,-4*J.sum(1))  # s=2x-1 mapping: x'Qx + const ~ s'Js
qsk=QUBO(Q)
o=run_qubo(qsk,None,'SK',maxcut=False); best=min(o[k][0] for k in o); res['SK_n800']={'best':best,**o}
print(f"SK spin-glass (n=800, dense): GNN {o['GNN'][0]:.0f}({(o['GNN'][0]-best)/abs(best)*100:+.2f}%) {o['GNN'][1]:.0f}s | SB {o['SB'][0]:.0f}({(o['SB'][0]-best)/abs(best)*100:+.2f}%) {o['SB'][1]:.0f}s | tabu {o['tabu'][0]:.0f}({(o['tabu'][0]-best)/abs(best)*100:+.2f}%) {o['tabu'][1]:.0f}s",flush=True)
json.dump(res,open('experiments/results/e66_hard_qubo.json','w'),indent=2); print('saved')
