import os, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT="results/figures"
# (method, OOS Sharpe, inference s/instance, color, role)
d=[("EqualWeight",0.764,0.0001,"gray"),
   ("Exact/Tabu (per-inst)",0.863,1.68,"black"),
   ("Amort-Imitation",0.862,0.0019,"tab:blue"),
   ("Amort-Decision",1.009,0.0077,"tab:red"),
   ("DiffOpt (ML)",1.086,0.0,"tab:green"),
   ("DRL (ML)",1.050,0.0,"tab:green")]
plt.figure(figsize=(8,5))
for n,sh,t,c in d:
    x=max(t,0.0005)
    plt.scatter(x,sh,s=140,c=c,edgecolor='k',zorder=3)
    plt.annotate(n,(x,sh),textcoords="offset points",xytext=(7,4),fontsize=9)
plt.axhspan(1.0,1.12,alpha=0.06,color='green'); plt.axhspan(0.84,0.88,alpha=0.06,color='blue')
plt.text(0.0006,1.02,"investor band (decision-focused)",color='green',fontsize=8)
plt.text(0.0006,0.845,"optimizer band (imitation=tabu)",color='blue',fontsize=8)
plt.xscale("log"); plt.xlabel("inference time per instance (s, log)"); plt.ylabel("out-of-sample Sharpe")
plt.title("One amortized GNN spans optimizer<->investor at ms inference\n(imitation->0.86; decision-focused->1.01; both ~ms)")
plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(f"{OUT}/fig_e34_synthesis.png",dpi=130); print("saved",f"{OUT}/fig_e34_synthesis.png")
