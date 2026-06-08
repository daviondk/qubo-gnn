"""E25 (loop): solver comparison on the WEIGHT-ENCODED (integer-lot-style) QUBO — structurally different
(N x n_bits binary vars, budget-constrained, dense) from binary cardinality selection. Compares GNN vs
tabu vs SA vs SCIP-global on the QUBO energy. Does the GNN help on this harder/denser QUBO? Run in .venv."""
import os, sys, json, time, numpy as np
os.environ.setdefault("KMP_DUPLICATE_LIB_OK","TRUE"); os.environ.setdefault("OMP_NUM_THREADS","6")
HERE=os.path.dirname(__file__); sys.path.insert(0, os.path.join(HERE,"..","src"))
from qubo_portfolio import weight_qubo
from baselines import sa_qubo, tabu_qubo
from gnn_solver import solve_qubo_gnn, GNNHypers
rng=np.random.default_rng(3); N=40; B=rng.standard_normal((N,6))*0.02
Sig=B@B.T+np.diag((np.abs(rng.standard_normal(N))*0.01+0.005)**2); Sig=0.5*(Sig+Sig.T)
mu=rng.standard_normal(N)*0.004+0.003
for nb in [3,4]:
    q,spec=weight_qubo(mu,Sig,n_bits=nb,risk_aversion=0.5,return_weight=0.5)
    nv=q.n; res={}
    r=tabu_qubo(q,num_reads=100,seed=0); res["Tabu"]=(r["energy"],r["time"])
    r=sa_qubo(q,num_reads=100,seed=0); res["SA"]=(r["energy"],r["time"])
    try:
        from baselines import scip_qubo; r=scip_qubo(q,time_limit=60); res["SCIP"]=(r["energy"],r["time"])
    except Exception as e: res["SCIP"]=(float("nan"),0)
    h=GNNHypers(model="qrf",epochs=2000,hidden=128,dim_embedding=20,n_layers=3,lr=1e-3,anneal_rate=0.0,
                eval_every=50,patience=500,ls_passes=100,n_round_samples=16,refine_sa=True,refine_reads=30)
    r=solve_qubo_gnn(q,h,device="cuda",seed=0); res["GNN"]=(r["energy"],r["time"])
    best=min(v[0] for v in res.values() if np.isfinite(v[0]))
    print(f"\n=== weight_qubo N={N} n_bits={nb} (nv={nv} binary vars) ===")
    print(f"{'method':<8}{'energy':>12}{'gap%':>9}{'t(s)':>8}")
    for m,(e,t) in res.items():
        g=(e-best)/abs(best)*100 if np.isfinite(e) and abs(best)>1e-12 else float("nan")
        print(f"{m:<8}{e:>12.5f}{g:>9.3f}{t:>8.1f}")
    json.dump({m:{"energy":e,"t":t} for m,(e,t) in res.items()}, open(os.path.join(HERE,"results",f"e25_weightqubo_nb{nb}.json"),"w"), indent=2)
