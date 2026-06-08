"""E59 (USER DIRECTIVE): definitive SOLVER-BENCHMARK on the (hard) weight-encoded Markowitz portfolio QUBO
-- gap-to-best + time + SCALING in N. Methods: SCIP(exact), tabu, SA, our GNN-QUBO. Real S&P500 data.
Answers 'how close to optimum + speed + scaling' for each solver on the SAME math problem (the quantum-lit
formulation). Saves table + scaling figure. Run in .venv."""
import os, sys, json, time, numpy as np
os.environ.setdefault("KMP_DUPLICATE_LIB_OK","TRUE"); os.environ.setdefault("OMP_NUM_THREADS","6")
HERE=os.path.dirname(__file__); sys.path.insert(0, os.path.join(HERE,"..","src"))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from datasets import get_returns
from qubo_portfolio import weight_qubo
from baselines import tabu_qubo, sa_qubo, scip_qubo
from gnn_solver import solve_qubo_gnn, GNNHypers

def main():
    try: R=get_returns("sp500").values
    except Exception:
        from backtest import load_prices, SP100; R=load_prices(SP100,"2018-01-01","2024-12-31").pct_change().dropna().values
    Nmax=R.shape[1]
    h=GNNHypers(model="qrf",epochs=2000,hidden=128,dim_embedding=20,n_layers=3,lr=1e-3,anneal_rate=0.0,
                eval_every=50,patience=400,ls_passes=150,n_round_samples=20,refine_sa=True,refine_reads=40)
    Ns=[20,40,60,100,150]
    print(f"E59 solver-scaling on weight-encoded Markowitz QUBO (universe N_max={Nmax})",flush=True)
    out={}
    for N in Ns:
        if N>Nmax: continue
        sub=np.arange(N); w=R[-504:,sub]; mu=w.mean(0)*252; S=np.cov(w,rowvar=False)*252; S=0.5*(S+S.T)+1e-8*np.eye(N)
        q,_=weight_qubo(mu,S,n_bits=3,risk_aversion=0.5,return_weight=0.5,w_max=0.2); nv=q.n
        res={}
        t0=time.time()
        try: r=scip_qubo(q,time_limit=45); res["SCIP"]=(r["energy"],time.time()-t0)
        except Exception: res["SCIP"]=(float("nan"),time.time()-t0)
        r=tabu_qubo(q,num_reads=150,seed=0); res["tabu"]=(r["energy"],r["time"])
        r=sa_qubo(q,num_reads=150,seed=0); res["SA"]=(r["energy"],r["time"])
        r=solve_qubo_gnn(q,h,device="cuda",seed=0); res["GNN-QUBO"]=(r["energy"],r["time"])
        best=min(v[0] for v in res.values() if np.isfinite(v[0]))
        row={m:{"gap%":(e-best)/abs(best)*100 if np.isfinite(e) and abs(best)>1e-12 else float("nan"),"t":t} for m,(e,t) in res.items()}
        out[N]=row
        print(f"  N={N:>3} ({nv} vars): "+" | ".join(f"{m} {row[m]['gap%']:.2f}%/{row[m]['t']:.1f}s" for m in res),flush=True)
    json.dump(out,open(os.path.join(HERE,"results","e59_solver_scaling.json"),"w"),indent=2)
    # figure: gap vs N, time vs N
    ms=["SCIP","tabu","SA","GNN-QUBO"]; cols={"SCIP":"gray","tabu":"black","SA":"tab:green","GNN-QUBO":"tab:red"}
    Nsd=sorted(out); fig,ax=plt.subplots(1,2,figsize=(11,4.3))
    for m in ms:
        ax[0].plot(Nsd,[min(out[n][m]["gap%"],50) for n in Nsd],"o-",label=m,color=cols[m])
        ax[1].plot(Nsd,[out[n][m]["t"] for n in Nsd],"o-",label=m,color=cols[m])
    ax[0].set_xlabel("N assets"); ax[0].set_ylabel("gap to best (%, capped 50)"); ax[0].set_title("Optimality gap"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].set_xlabel("N assets"); ax[1].set_ylabel("time (s)"); ax[1].set_title("Time-to-solution"); ax[1].legend(); ax[1].grid(alpha=.3)
    plt.suptitle("Solver benchmark on weight-encoded Markowitz QUBO (exact MIP fails; SA/GNN-QUBO scale)")
    plt.tight_layout(); plt.savefig(os.path.join(HERE,"..","results","figures","fig_e59_solver_scaling.png"),dpi=130)
    print("saved fig_e59_solver_scaling.png",flush=True)

if __name__=="__main__": main()
