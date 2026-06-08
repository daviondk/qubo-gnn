"""E61 (USER DIRECTIVE, revised): benchmark OPEN solvers (ML/DL preferred) on the portfolio QUBO math
problem -- gap-to-best + time + scaling. Gurobi is PROPRIETARY -> cited from Stopfer (optimal in seconds),
NOT run here. Open methods: our GNN-QUBO (DL), SimulatedAnnealing, TabuSearch, SCIP (open exact, time-limited).
Weight-encoded Markowitz QUBO on Stopfer nasdaq data, N=40,80,150. Reference = best-found across methods.
Run in .venv. (ML/DL competitors PI-GNN/PQQA/CRA added separately on identical Q.)"""
import os, sys, json, time, numpy as np, pandas as pd
os.environ.setdefault("KMP_DUPLICATE_LIB_OK","TRUE"); os.environ.setdefault("OMP_NUM_THREADS","6")
HERE=os.path.dirname(__file__); sys.path.insert(0, os.path.join(HERE,"..","src"))
from qubo_portfolio import weight_qubo
from baselines import tabu_qubo, sa_qubo, scip_qubo
from gnn_solver import solve_qubo_gnn, GNNHypers
DATA=os.path.join(HERE,"..","competitors","portfolio_opt_benchmark","src","problems","MarkowitzPortfolio")

def main():
    mu_all=pd.read_csv(os.path.join(DATA,"nasdaq_annual_returns.csv"),sep="\t").iloc[0]
    cov_all=pd.read_csv(os.path.join(DATA,"nasdaq_annualized_covariance_matrix.csv"),sep="\t",index_col=0)
    tk=[t for t in mu_all.index if t in cov_all.columns]; rng=np.random.default_rng(1)
    h=GNNHypers(model="qrf",epochs=1500,hidden=128,dim_embedding=20,n_layers=3,lr=1e-3,anneal_rate=0.0,
                eval_every=50,patience=400,ls_passes=150,n_round_samples=20,refine_sa=True,refine_reads=40)
    print("E61 OPEN-solver scaling (weight-encoded Markowitz QUBO, Stopfer nasdaq). Gurobi cited from paper (optimal/sec).",flush=True)
    out={}
    for N in [40,80,150]:
        a=sorted(rng.choice(tk,size=N,replace=False).tolist())
        mu=mu_all[a].values.astype(float); S=cov_all.loc[a,a].values.astype(float); S=0.5*(S+S.T)+1e-10*np.eye(N)
        q,_=weight_qubo(mu,S,n_bits=3,risk_aversion=0.5,return_weight=0.5,w_max=0.2); nv=q.n
        res={}
        t0=time.time()
        try: res["SCIP(open exact)"]=(scip_qubo(q,time_limit=45)["energy"],time.time()-t0)
        except Exception: res["SCIP(open exact)"]=(float("nan"),time.time()-t0)
        r=tabu_qubo(q,num_reads=150,seed=0); res["tabu"]=(r["energy"],r["time"])
        r=sa_qubo(q,num_reads=150,seed=0); res["SA"]=(r["energy"],r["time"])
        r=solve_qubo_gnn(q,h,device="cuda",seed=0); res["GNN-QUBO(ours,DL)"]=(r["energy"],r["time"])
        best=min(v[0] for v in res.values() if np.isfinite(v[0]))
        out[N]={m:{"gap%":(e-best)/abs(best)*100 if np.isfinite(e) and abs(best)>1e-12 else None,"t":t} for m,(e,t) in res.items()}
        print(f"  N={N:>3} ({nv} vars): "+" | ".join(f"{m} {out[N][m]['gap%'] if out[N][m]['gap%'] is None else round(out[N][m]['gap%'],2)}%/{out[N][m]['t']:.1f}s" for m in res),flush=True)
    json.dump(out,open(os.path.join(HERE,"results","e61_open_scaling.json"),"w"),indent=2); print("saved",flush=True)

if __name__=="__main__": main()
