"""E68 (USER HYPOTHESIS): does ADDING CONSTRAINTS make portfolio genuinely hard for EXACT solvers?
Build cardinality + MIN-BUY-IN semi-continuous MIQP (Frangioni-Gentile/Bertsimas hard regime) on real data,
solve EXACTLY with SCIP (60s limit) at increasing N, report objective + optimality gap + status.
Compare: (A) plain cardinality MIQP (easy) vs (B) +min-buy-in (hard). Shows WHEN exact breaks.
Run in .venv (pyscipopt)."""
import sys,time,os,json,numpy as np
sys.path.insert(0,'src')
from pyscipopt import Model, quicksum
from backtest import load_prices, SP100
def make_data(N,seed=0):
    R=load_prices(SP100,'2018-01-01','2024-12-31').pct_change().dropna().values
    rng=np.random.default_rng(seed)
    if N<=R.shape[1]: idx=rng.choice(R.shape[1],N,replace=False); w=R[-756:,idx]
    else:  # synthetic factor model for larger N
        F=rng.standard_normal((756,8))*0.01; L=rng.standard_normal((N,8)); w=F@L.T+rng.standard_normal((756,N))*0.01
    mu=w.mean(0)*252; S=np.cov(w,rowvar=False)*252; return mu,0.5*(S+S.T)+1e-8*np.eye(N)
def solve_miqp(mu,S,K,lam,minbuyin,tl=60):
    N=len(mu); m=Model(); m.hideOutput(); m.setParam('limits/time',tl)
    w=[m.addVar(f'w{i}',lb=0,ub=1) for i in range(N)]; z=[m.addVar(f'z{i}',vtype='B') for i in range(N)]
    m.addCons(quicksum(w)==1); m.addCons(quicksum(z)<=K)
    for i in range(N):
        m.addCons(w[i]<=z[i])  # link
        if minbuyin>0: m.addCons(w[i]>=minbuyin*z[i])  # semi-continuous min buy-in (the HARD part)
    aux=m.addVar('obj',lb=-1e9,ub=1e9)
    risk=quicksum(lam*S[i,j]*w[i]*w[j] for i in range(N) for j in range(N))
    ret=quicksum((1-lam)*mu[i]*w[i] for i in range(N))
    m.addCons(risk-ret<=aux)  # epigraph (quadratic constraint, SCIP-supported)
    m.setObjective(aux,'minimize')
    t0=time.time(); m.optimize(); dt=time.time()-t0
    st=m.getStatus(); gap=m.getGap(); obj=m.getObjVal() if m.getNSols()>0 else None
    return {'status':st,'gap':gap,'obj':obj,'t':dt}
out={}
for N in [300,400,500]:
    mu,S=make_data(N); K=max(10,N//10)
    a=solve_miqp(mu,S,K,0.5,0.0)       # plain cardinality
    b=solve_miqp(mu,S,K,0.5,0.02)      # + min-buy-in (hard)
    out[N]={'plain':a,'minbuyin':b}
    f=lambda r:f"{r['status']}/gap{r['gap']*100:.1f}%/{r['t']:.0f}s" if r['gap'] is not None else f"{r['status']}/{r['t']:.0f}s"
    print(f"N={N} K={K}: PLAIN-cardinality {f(a)} | +MIN-BUY-IN {f(b)}",flush=True)
json.dump(out,open('experiments/results/e68_constraint_hardness.json','w'),indent=2,default=str); print('saved')
