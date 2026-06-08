import os, json, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
d=json.load(open("experiments/results/e46_riskaversion.json"))
lams=sorted(float(k) for k in d); 
gnn=[d[str(l) if str(l) in d else l]["GNN"] for l in lams]
lin=[d[str(l) if str(l) in d else l]["Linear"] for l in lams]
# handle key types
def get(l,k):
    return d[str(l)][k] if str(l) in d else d[l][k]
gnn=[get(l,"GNN") for l in lams]; lin=[get(l,"Linear") for l in lams]; grd=[abs(get(l,"greedy_gap")) for l in lams]
plt.figure(figsize=(7.5,4.6))
plt.plot(lams,gnn,"o-",label="GNN amortization",color="tab:red",lw=2)
plt.plot(lams,lin,"s--",label="Linear amortization",color="tab:blue",lw=2)
plt.plot(lams,grd,"^:",label="problem hardness (|greedy gap|)",color="gray")
plt.axvspan(0.85,1.0,alpha=0.08,color="red")
plt.text(0.86,max(gnn)*0.6,"risk-dominated:\nGNN > Linear\n(exact SCIP times out)",fontsize=8,color="darkred")
plt.text(0.12,max(gnn)*0.25,"return-dominated:\nGNN = Linear (easy)",fontsize=8,color="navy")
plt.xlabel("risk-aversion $\lambda$"); plt.ylabel("gap vs tabu (%)")
plt.title("Where the GNN earns its keep: GNN vs Linear amortization across risk-aversion")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("results/figures/fig_e46_regime.png",dpi=130); print("saved fig_e46_regime.png")
