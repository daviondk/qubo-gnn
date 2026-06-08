import os,sys,json,numpy as np,torch,torch.nn.functional as F
os.environ.setdefault("KMP_DUPLICATE_LIB_OK","TRUE")
HERE=os.path.dirname(__file__); sys.path.insert(0,os.path.join(HERE,"..","src"))
from optuna_amortized import (rich_feats,basic_feats,knn_edges,Net,K,sel_obj,windows_from_returns,
                              load_prices,SP100,get_returns,tabu_qubo,selection_qubo,DEVICE)
bp=json.load(open(os.path.join(HERE,"results","optuna_amortized.json")))["best_params"]
print("validating best:",bp,flush=True)
R=load_prices(SP100,"2005-01-01","2024-12-31").pct_change().dropna().values
w=windows_from_returns(R); sp=int(0.7*len(w)); tr_raw,te_raw=w[:sp],w[sp:]
oo_raw=windows_from_returns(get_returns("nasdaq100").values,max_w=40)
def tsel(mu,S):
    q=selection_qubo(mu,S,K,risk_aversion=0.5,return_weight=0.5); r=tabu_qubo(q,num_reads=100,seed=0)
    i=np.flatnonzero(np.asarray(r["x"])>0.5); return i if len(i)==K else np.argsort(-np.asarray(r["x"]))[:K]
lab=[tsel(mu,S) for mu,S in tr_raw]; tref=[sel_obj(tsel(mu,S),mu,S,K) for mu,S in te_raw]; oref=[sel_obj(tsel(mu,S),mu,S,K) for mu,S in oo_raw]
def build(raw,mode,k):
    out=[]
    for mu,S in raw:
        f,C=(rich_feats(mu,S) if mode=="rich" else basic_feats(mu,S)); out.append((torch.tensor(f,device=DEVICE),knn_edges(C,k),mu,S))
    return out
mode=bp["features"]; kk=bp["knn_k"]
tr=build(tr_raw,mode,kk); te=build(te_raw,mode,kk); oo=build(oo_raw,mode,kk)
labs=[torch.tensor(np.isin(np.arange(len(mu)),l).astype(np.float32),device=DEVICE) for (mu,_),l in zip(tr_raw,lab)]
def run(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    m=Net(tr[0][0].shape[1],bp["hidden"],bp["layers"],bp["dropout"]).to(DEVICE); opt=torch.optim.Adam(m.parameters(),lr=bp["lr"])
    pw=torch.tensor([bp["pos_weight_scale"]*(71-K)/K],device=DEVICE)
    for ep in range(bp["epochs"]):
        m.train(); perm=np.random.permutation(len(tr))
        for bi in range(0,len(tr),16):
            opt.zero_grad(); bl=0.0
            for ii in perm[bi:bi+16]:
                bl=bl+F.binary_cross_entropy_with_logits(m(tr[ii][0],tr[ii][1]),labs[ii],pos_weight=pw)
            (bl/max(1,len(perm[bi:bi+16]))).backward(); opt.step()
    m.eval()
    def ev(ins,ref):
        g=[]
        for (x,ei,mu,S),rf in zip(ins,ref):
            with torch.no_grad(): p=m(x,ei).cpu().numpy()
            g.append((sel_obj(np.argsort(-p)[:K],mu,S,K)-rf)/abs(rf)*100 if abs(rf)>1e-12 else 0.0)
        return float(np.mean(g))
    return ev(te,tref),ev(oo,oref)
res=[run(s) for s in range(5)]
te_g=[r[0] for r in res]; oo_g=[r[1] for r in res]
print(f"\n5-seed VALIDATION best config: test {np.mean(te_g):.3f}+/-{np.std(te_g):.3f}%  OOD {np.mean(oo_g):.3f}+/-{np.std(oo_g):.3f}%")
print("per-seed test",[round(x,2) for x in te_g],"ood",[round(x,2) for x in oo_g])
json.dump({"test_mean":float(np.mean(te_g)),"test_std":float(np.std(te_g)),"ood_mean":float(np.mean(oo_g)),"ood_std":float(np.std(oo_g)),"seeds_test":te_g,"seeds_ood":oo_g},open(os.path.join(HERE,"results","optuna_validate.json"),"w"),indent=2)
