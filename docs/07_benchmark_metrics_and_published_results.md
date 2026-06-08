# OR-Library cardinality portfolio: exact metric definitions + published results (for comparison)

Verified literature extraction (2026-06-04) of the standard cardinality-constrained portfolio
benchmark, its EXACT metric definitions, and published per-instance numbers — so our method can be
compared on the same footing. **This is the basis for the comparison table in `08_comparison.md`.**

## Standard experimental setup (Chang et al. 2000; reused by all later papers)
- Instances = OR-Library port1–port5: Hang Seng (N=31), DAX 100 (85), FTSE 100 (89), S&P 100 (98),
  Nikkei 225 (225). Data: weekly returns, Mar 1992–Sep 1997 (291 obs).
- Cardinality **K = 10**; per-asset bounds **εᵢ = 0.01** (floor if selected), **δᵢ = 1** (ceiling).
- Objective traced over the risk–return tradeoff: `min λ·wᵀΣw − (1−λ)·μᵀw`, `Σw=1`,
  `ε·zᵢ ≤ wᵢ ≤ δ·zᵢ`, `Σzᵢ=K`, `z∈{0,1}`.
- **Frontier points:** Chang uses E=50 (λ=(e−1)/49); Cura/Mozafari/Bacanin use **ξ=51** (Δλ=0.02).
- **Reference (unconstrained) frontier:** 2000 points (standard Markowitz, budget + no-short only),
  linearly interpolated.

## ⚠ Two non-interchangeable metric families
1. **Chang (2000) "percentage error"** = min(horizontal same-return %, vertical same-risk %) deviation
   from the unconstrained frontier, using **standard deviation**; reports median & mean %.
2. **Cura (2009) MED/VRE/MRE** (the one all modern metaheuristic papers use). For each of the ξ
   heuristic frontier points (vʰ, rʰ) = (variance, return), find the nearest of the 2000 standard
   points (vˢ, rˢ) by Euclidean distance in (variance, return):
   - **MED** = mean over points of `sqrt((vˢ−vʰ)² + (rˢ−rʰ)²)`
   - **VRE %** = mean of `100·|vˢ−vʰ|/vʰ`
   - **MRE %** = mean of `100·|rˢ−rʰ|/rʰ`
   The famous "GA/TS/SA baseline MED" numbers are **Cura's recomputation**, NOT from Chang's tables.
   → We implement the **Cura MED/VRE/MRE** (so we are comparable to the modern literature).

## Published MED (Cura 2009, Table 1) — the standard baseline everyone cites
| Instance (N) | GA | TS | SA | PSO |
|---|---|---|---|---|
| Hang Seng (31) | 0.0040 | 0.0040 | 0.0040 | 0.0049 |
| DAX 100 (85) | 0.0076 | 0.0082 | 0.0078 | 0.0090 |
| FTSE 100 (89) | 0.0020 | 0.0021 | 0.0021 | 0.0022 |
| S&P 100 (98) | 0.0041 | 0.0041 | 0.0041 | 0.0052 |
| Nikkei (225) | 0.0093⚠ | 0.0010 | 0.0010 | 0.0019 |
⚠ Nikkei GA=0.0093 is a known outlier/likely typo propagated through later papers.

## Better published methods (MED)
- **IPSO-SA** (Mozafari, Tafazzoli, Jolai 2011, *IJIEC* 2(2):249): MED ≈ **0.0001 / 0.0001 / 0.0000 /
  0.0001 / 0.0000** for port1–5 (near-zero — essentially recovers the exact frontier).
- **Firefly mFA** (Bacanin & Tuba 2014, *Sci. World J.* 721521): MED = **0.0003 / 0.0009 / 0.0004 /
  0.0003 / 0.0000** for port1–5.
- Other same-benchmark papers (numbers paywalled, not extracted): Deng, Lin, Lo 2012 (improved PSO,
  *ESWA* 39(4)); Zheng, Zhang, Zhang 2023 (Mayfly, *ESWA* 230:120656).

## Chang (2000) CCEF mean % error (Table 4, the original metric — different family)
| Instance | GA-H | TS-H | SA-H | pooled |
|---|---|---|---|---|
| Hang Seng | 0.9457 | 0.9908 | 0.9892 | 0.9332 |
| DAX | 1.9515 | 3.0635 | 2.4299 | 2.1927 |
| FTSE | 0.8784 | 1.3908 | 1.1341 | 0.7790 |
| S&P | 1.7157 | 3.1678 | 2.6970 | 1.3106 |
| Nikkei | 0.6431 | 0.8981 | 0.6370 | 0.5690 |

## Cross-paper comparability flags (must respect for a fair paper)
1. Use Cura MED (not Chang %) to compare to modern methods. State the metric explicitly.
2. ξ=51 λ-points (Δλ=0.02), unconstrained frontier = 2000 points, K=10, ε=0.01, δ=1.
3. MED is unit-dependent (variance ~1e-3, return ~1e-2) — only comparable on identical instances +
   identical standard-frontier construction. We use the OR-Library files directly.
4. CCEF errors are measured vs the *unconstrained* frontier (which dominates), so they overestimate
   true deviation — same for everyone.

## Sources
- Chang, Meade, Beasley, Sharaiha (2000), *Comput. & Oper. Res.* 27(13):1271–1302. DOI
  10.1016/S0305-0548(99)00074-X. Data: people.brunel.ac.uk/~mastjjb/jeb/orlib/portinfo.html
- Cura (2009), *Nonlinear Analysis: RWA* 10(5):2396–2406. DOI 10.1016/j.nonrwa.2008.04.023
- Mozafari, Tafazzoli, Jolai (2011), *IJIEC* 2(2):249–262. DOI 10.5267/j.ijiec.2011.01.004 (open)
- Bacanin & Tuba (2014), *The Scientific World Journal* 2014:721521 (open, PMC4060745)
- Deng, Lin, Lo (2012), *ESWA* 39(4):4558–4566; Zheng, Zhang, Zhang (2023), *ESWA* 230:120656
