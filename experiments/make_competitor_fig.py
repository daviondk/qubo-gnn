import os, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT="results/figures"; os.makedirs(OUT, exist_ok=True)
# (method, optGap%, Sharpe, marker color)
d=[("EqualWeight",29.9,0.764,"gray"),("Exact-MIQP",0.0,0.853,"black"),
   ("GNN-QUBO (ours)",0.0,0.863,"red"),("DiffOpt (ML)",26.6,1.086,"green"),
   ("DRL (ML)",32.2,1.050,"blue"),("E2E-DRO-style",26.6,1.086,"purple")]
plt.figure(figsize=(8,5.5))
for n,g,s,c in d:
    plt.scatter(g,s,s=130,c=c,zorder=3,edgecolor='k')
    plt.annotate(n,(g,s),textcoords="offset points",xytext=(8,5),fontsize=9)
plt.axhspan(0.95,1.15,alpha=0.07,color='green'); plt.axvspan(-2,5,alpha=0.07,color='red')
plt.text(1,0.79,"better OPTIMIZER\n(gap~0)",color='red',fontsize=8)
plt.text(20,1.12,"better INVESTOR (OOS Sharpe)",color='green',fontsize=8)
plt.xlabel("optimality gap vs exact MIP  (%, lower=better optimizer)")
plt.ylabel("out-of-sample Sharpe  (higher=better investor)")
plt.title("Optimizer vs Investor: QUBO-GNN solves the problem; modern ML wins OOS\nS&P100, cardinality+tx-cost+turnover")
plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(f"{OUT}/fig_competitors.png",dpi=130); print("saved",f"{OUT}/fig_competitors.png")
