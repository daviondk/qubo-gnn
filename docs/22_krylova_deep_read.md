# Deep read: Krylova (2024) — the closest prior work, and what it means for our comparison & architecture

Full thesis read (`papers/Krylova2024_GNN-QUBO-portfolio_MScThesis.pdf`, 78 pp; text in
`papers/_krylova_text.txt`). She is essentially doing *our* topic: an unsupervised PI-GNN/Schuetz-style
QUBO solver applied to cardinality portfolio. Studying her experience to (a) confirm we compare
correctly and (b) extract architecture-improvement ideas.

## Her setup
- **Solver:** PI-GNN (Schuetz lineage), **GraphSAGE** layers, trainable node embedding of size
  $\lfloor\sqrt{|V|}\rfloor$, loss $x^\top Q x$, **Rprop** optimizer, PyTorch-Geometric, multi-restart.
- **Problems:** MaxCut + MIS (to probe limits), then portfolio on **Nikkei225** (N∈{100,150,225}).
- **Portfolio Model 1** = Phillipson & Bhatia (2021) D-Wave formulation: budget ($\sum x=n$) + return
  floor ($\mu^\top x\ge R$) + slack vars. **Model 2** = classical Markowitz $\min x^\top\Sigma x-\lambda\mu^\top x$
  (MIS-like, risk matrix as adjacency).
- **Baselines/metric:** **Gurobi** (exact Pareto front), **random portfolios** (500 of same size),
  D-Wave hybrid (from Phillipson & Bhatia); evaluated as **proximity to the Gurobi Pareto front in
  (risk, return)** + selection frequency + portfolio sizes + solution time per $\lambda$ and $N$.

## Her findings (the lessons)
1. **Density is the core obstacle** for the basic GCN solver — works only on sparse graphs; for
   regular degree $d>7$ it fails (MaxCut & MIS).
2. **Architecture fixes that broadened the solvable density range:** (i) **Rprop** (denser + far faster
   → more restarts → better best-of); (ii) **GraphSAGE > GCN**; (iii) **embedding transfer**
   (pretrain on MaxCut, transfer to MIS). Even so, very high degree still impedes it → "investigate
   higher-expressivity GNNs" (future work).
3. **Hard constraints mislead the GNN** — Model 1 (budget/cardinality) gave "distinctly suboptimal"
   results, "budget almost never met". Matches our official-PI-GNN collapse (docs/19) and the general
   PI-GNN-can't-do-constraints result.
4. **Model 2 (no budget) reaches the Gurobi Pareto front.** Insight: although the portfolio "MIS graph"
   is fully connected, the **continuous** risk-matrix adjacency does NOT impede the GNN like a
   **discrete** adjacency would — relaxing only the variables (not the adjacency) may be insufficient
   for discrete-dense problems.
5. **Data scaling matters** (NN best practice); she re-optimized penalty $\lambda$s.

## What this means for OUR comparison (sanity check — are we doing it right?)
- **We are on her exact dataset:** Nikkei225 $=$ OR-Library **port5** (N=225). So our port5 numbers are
  directly comparable to her.
- **We compare more rigorously than she does:** standard **MED** metric vs published GA/TS/SA/PSO/
  IPSO-SA/Firefly, PLUS exact (Gurobi+SCIP), tabu, SA, greedy. She used Pareto-vs-Gurobi-vs-random.
  ⇒ we are not doing "дичь"; we are aligned and broader.
- **We already improve on her:** she fails the constrained Model 1 and only gets *close* to the Pareto
  front on Model 2; our selection-QUBO + auto-penalty + explore→exploit local search + hybrid
  re-weight **matches the exact optimum** on port5 (MED ≈ 0). The official PI-GNN we ran collapses just
  like her Model 1.
- **To be directly, visually comparable to her, add:** a **Pareto front (risk, return) plot: GNN vs
  Gurobi-exact vs random portfolios on port5 (Nikkei225)** — her headline figure. (We have frontier
  plots but not the random overlay / Nikkei framing.) → `experiments/` TODO.

## Architecture-improvement ideas to test (from her experience + ours), tracked in `experiments/LOG.md`
1. **Optimizer: Rprop vs Adam** — her biggest practical lever (denser-capable + faster → more restarts).
2. **Isolate the GNN's own contribution** — our ablation showed local search carries the result; like
   her, the bare GNN may be weak on dense Q. Measure GNN-alone (no LS) vs LS-from-random.
3. **Graph sparsification** of the dense covariance graph (kNN on |corr|) — directly attacks her
   "density obstacle".
4. **Edge features = covariance / higher-expressivity GNN** (GAT/GINE) — her "future work".
5. **Embedding transfer / pretraining** — relates to our amortization (train once, reuse) — promising.
6. **Soft vs hard constraints** — confirm soft selection-surrogate + reweight beats hard-penalty QUBO
   (she and we both see hard penalties mislead the GNN).
