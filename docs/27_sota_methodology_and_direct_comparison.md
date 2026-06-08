# How the SOTA papers evaluate, and what we can compare DIRECTLY (their code on our data)

User goal: compare directly with the newest SOTA (2024–2026), study how they evaluate (metrics, task,
protocol), and avoid "tricks" (prefer their actual code over reimplementations). This documents both the
**methodology study** and the **runnability reality** (verified by inspecting the actual repos).

---
## A. How the newest SOTA evaluates (metrics / task / protocol) — what to look at
| Paper (year) | Task solved | Objective | Metrics reported (how) | Dataset / protocol | Baselines |
|---|---|---|---|---|---|
| **THRML** (Mancilla 2026, 2601.07792) | cardinality **index tracking** (pick K=30 of 100) | Ising H: tracking + momentum + liquidity bias, correlation-diversification couplings, VIX-scaled β; sector caps post-hoc | **annualized tracking error**, correlation, total return, Sharpe(rf=2%), Sortino, MaxDD, Information Ratio; **Diebold–Mariano** significance on squared TE | 100-stock S&P500 subset, 2023–2025, **quarterly** rebal, **10 bps/trade**, K=30 | Greedy, MIP, Robust MVO, NSGA-II, SA |
| **VNA** (Ranabhat 2025, 2507.07159) | cardinality + **turnover + transaction-cost** MINLP | risk-adjusted return → classical Ising H, solved by autoregressive-RNN variational annealing | objective value & feasibility **vs Mosek**; convergence time; **finite-size scaling** | S&P500, Russell 1000/3000 | Mosek (conic) |
| **DSL** (Kim 2025, 2503.13544) | supervised allocation (predict optimal weights) | cross-entropy to Sharpe/Sortino-max target portfolios + deep ensemble | median return, **Sharpe, Sortino, MaxDD, turnover**, ensemble-size ablation | 8 universes incl S&P-Top-30, S&P500-rolling, NASDAQ100; OHLCV | PFL (predict-then-opt), E2E/DFL, MVO |
| **QAOA-XY** (Mancilla 2026, 2602.14827) | cardinality direct indexing (Dicke/XY-mixer = hard K) | select K then Sharpe-max weights | Sharpe, turnover, vs SA/HRP; tx-cost backtest | 10 equities, 2025 walk-forward | SA, HRP |
| **Lozano penalty-free** (2026, 2605.17628) | cardinality selection on annealer | objective-only QUBO + classical feasibility projector | **regret vs greedy** (≤0.03%), chain-break fraction | Fama-French 49 | greedy, penalty-QUBO |
| **RTS-PnO** (Lyu 2025, 2505.24835) | fund **timing** (buy-cost over window) — NOT selection | predict-and-optimize + conformal | **regret** vs optimal cost | Currency/Stock/Crypto | PtO, heuristics |

**Takeaways for what matters in our task:**
- The risk/index-tracking SOTA report **annualized tracking error + Sharpe/Sortino/MaxDD/Information-Ratio + DM significance**, quarterly rebalance, **10 bps** cost — we should emit exactly this set (our `exp_index_tracking.py` already gives TE% + gap; extend with Sharpe/Sortino/MaxDD/IR).
- The solver SOTA (VNA) reports **objective vs Mosek + convergence time + finite-size scaling** — i.e. quality-at-equal-time and scaling, which aligns with our optimality-gap + speedup framing.
- **Almost none report an optimality gap vs exact MIP** (only Lozano via regret, VNA via Mosek) — our optimality-gap column remains a distinctive contribution.

---
## B. Runnability reality — can we run THEIR code on OUR data? (verified by repo inspection)
| Method (year) | Repo | Portfolio code public? | Blockers | Verdict |
|---|---|---|---|---|
| **THRML** (2026) | extropic-ai/thrml | **NO** — only the generic JAX Gibbs-sampling library; no portfolio/VIX/backtest code | must reimplement the Ising+VIX+sector recipe | **reimplement only** (we have `exp_index_tracking.py`, `.venv-jax`) |
| **VNA** (2025) | NishanRanabhat/VNA_spin_models | **NO** — generic **field-free** spin-glass RNN annealer, **DDP-hardwired** (`mp.spawn`+DDP); no portfolio code | energy has no linear field h (portfolio needs it); single-GPU driver needed; portfolio Ising encoding needed | **adapt core** (their RNN ansatz) — substantial |
| **DSL** (2025) | DSLwDE/DSLwDE | **YES** (full: target-portfolio solver + LSTM/Transformer/Mamba train + backtest) | imports **mamba_ssm** (CUDA-compiled, Windows-hostile) at top; heavy frozen deps (bt, boto3…); needs OHLCV input | **runnable with effort** (stub mamba → LSTM/Transformer; isolated venv; feed our tickers' OHLCV) |
| QAOA-XY, Lozano, RTS-PnO | — | no / different task | — | cite numbers / reimplement |
| E2E-DRO (2023) | Iyengar-Lab/E2E-DRO | yes | cvxpylayers+diffcp+torch1+numpy<2 (breaks our stack) | HARD, older (deprioritize) |
| X2GNN / GCON / CRA4CO | public | graph-CO only / we already wrapped CRA | not portfolio | done (CRA=E5) / skip |

**KEY HONEST FINDING:** the **closest learned-solver SOTA for portfolios (THRML, VNA) do NOT publish
portfolio code** — only generic, field-free, DDP-coupled solver libraries. So a faithful "their exact
code on our portfolio" run is **not available** for the nearest competitors; it requires reimplementing
their portfolio encoding (THRML) or adding field support + a single-GPU driver (VNA). This
**reproducibility gap is itself a finding** and strengthens the value of (a) our honest reimplementations
with clearly-labeled provenance and (b) our optimality-gap-vs-exact-MIP reporting that these papers omit.

---
## C. Direct-comparison plan (effort-rated)
1. **DSL (2025) — most legitimate runnable SOTA** (EFFORT: medium): isolated `.venv-dsl`; stub `mamba_ssm`
   → use LSTM/Transformer; pull OHLCV for our S&P100 tickers (yfinance); run their target-portfolio +
   train + backtest → get **Sharpe/Sortino/MaxDD/turnover**; drop into our common table. = their actual
   code, comparable universe/metrics.
2. **VNA (2025) — closest learned-solver** (EFFORT: high): write a single-GPU driver around their RNN
   `model_ansatz` + add a field term to `energy()`; encode our selection QUBO → Ising (J,h); run their
   variational-annealing loss; decode best sample → gap vs tabu/exact + vs our GNN on the IDENTICAL
   instance. = their solver architecture, our instance (modification: +field, single-GPU — labeled).
3. **THRML (2026)** (EFFORT: medium): reimplement the Ising+VIX+sector index-tracking recipe in
   `exp_index_tracking.py` (+`.venv-jax` block-Gibbs) → TE/Sharpe/Sortino/MaxDD/IR vs our GNN.
   = their formulation, our implementation (labeled "reimplementation").
4. Keep the DIRECT published-number comparisons we already have: classical cardinality MED (Cura/
   Mozafari/Bacanin), French49 regret (Lozano), Gset best-known.

**Provenance labels (always state in the paper):** "published numbers, same dataset+metric" (MED,
Lozano, Gset) vs "their public code, comparable data" (DSL) vs "our reimplementation of their method"
(DiffOpt/DRL/E2E-DRO-style, VNA-core, THRML) — never blur these.

Repos cloned: `competitors/DSLwDE`, `competitors/VNA_spin_models`. PDFs in `papers/lit_2025_2026/`.
