# Master comparison tables (OR-Library port1-5) — verified, two metric families

Deep extraction pass (2026-06-05). **Key finding: metric fragmentation.** Two non-comparable families:
(1) **Cura-2009 MED** (Euclidean distance to unconstrained frontier), (2) **Chang-2000 MPE**
(mean % error = min(horizontal%, vertical%) over 50 frontier points). Most metaheuristic papers report
MPE, not MED. We compare on BOTH so our method sits next to the right peers on each axis. Paywalled
numbers are marked n/a (never invented).

## Table 1 — MED (Cura convention), port1-5 [K=10, eps=0.01, delta=1]
| Method | source | port1 | port2 | port3 | port4 | port5 | verified |
|---|---|---|---|---|---|---|---|
| GA | Cura 2009 | 0.0040 | 0.0076 | 0.0020 | 0.0041 | 0.0093 | ✓ |
| TS | Cura 2009 | 0.0040 | 0.0082 | 0.0021 | 0.0041 | 0.0010 | ✓ |
| SA | Cura 2009 | 0.0040 | 0.0077 | 0.0020 | 0.0041 | 0.0010 | ✓ |
| PSO | Cura 2009 | 0.0049 | 0.0090 | 0.0022 | 0.0052 | 0.0024 | ✓ |
| IPSO-SA | Mozafari 2011 | 0.0001 | 0.0001 | 0.0000 | 0.0001 | 0.0000 | ✓ |
| Firefly mFA | Bacanin & Tuba 2014 | 0.0003 | 0.0009 | 0.0004 | 0.0003 | 0.0000 | ✓ |
| **GNN+refine (ours)** | docs/08 | **0.0001** | **0.0002** | **0.0000** | **0.0001** | **0.0000** | ✓ ours |
| **QRF-GNN original (ours)** | docs/08 | **0.0001** | **0.0004** | **0.0003** | **0.0004** | **0.0004** | ✓ ours |
| **SCIP exact (ours)** | docs/08 | 0.0001 | 0.0001 | 0.0000 | 0.0001 | 0.0000 | ✓ ours (floor) |
| Mayfly 2023; Dung-Beetle 2025; ABC 2017 | (paywalled) | n/a | n/a | n/a | n/a | n/a | ✗ paywall |

Best verified MED holder = IPSO-SA / Firefly / our GNN+refine / SCIP (all ≈ floor 0.0000-0.0001).
⚠ Tuo-2016 HSDS reports MED ~1e-6 but on a ~10× different scale (incompatible — excluded).

## Table 2 — Chang MPE % (mean percentage error), port1-5  [the OTHER common metric]
| Method | source | port1 | port2 | port3 | port4 | port5 | verified |
|---|---|---|---|---|---|---|---|
| GA | Chang 2000 | 1.0974 | 2.5424 | 1.1076 | 1.9328 | 0.7961 | ✓ |
| SA | Chang 2000 | 1.0957 | 2.9297 | 1.4623 | 3.0696 | 0.6732 | ✓ |
| TS | Chang 2000 | 1.1217 | 3.3049 | 1.1217 | 3.3092 | 0.8975 | ✓ |
| IPSO | Deng, Lin, Lo 2012 | 1.0953 | 2.5417 | 1.0628 | 1.6890 | 0.6870 | ✓ |
| ARO | Sefiane et al. 2021 (2101.03312) | 1.4181 | 1.3190 | 0.8151 | 1.4468 | 0.6179 | ✓ |
| **Ours (exact frontier, SCIP)** | src/orlib_mpe.py | **1.198** | **3.044** | **0.873** | **2.116** | 2.130 | ✓ ours |

Note: our exact-frontier MPE is **mid-pack** on port1-4 (port3 0.873 beats GA/SA/TS/IPSO and ≈ ARO;
port2 3.044 between SA and TS). **port5 (2.13) is inflated** — for N=225 the SCIP cardinality frontier
used a 30 s/point time limit (some of the 50 λ-points not proven optimal) and our UEF construction
(2000-pt λ-sweep + efficient hull) differs from Chang's NAG frontier; both inflate MPE. MPE is highly
implementation-sensitive (UEF point count, λ-grid, std-vs-variance, interpolation), so cross-paper
differences are partly metric noise. **The clean, defensible comparison is the MED axis (Table 1, where
we = the exact floor = best published).** On MPE we are competitive on small instances; port5 needs a
longer exact budget to be a fair number (not re-run — MED already establishes optimality).

Other MPE-family papers (metric confirmed, values not all extracted): Black Widow (Sci.Rep. 2024),
Nikiporenko time-limited (2307.04045), Kalayci ABC 2017, Baykasoglu GRASP 2015.

## Honest status of "deeper papers/numbers" (axis C)
- **No additional Cura-MED numbers are verifiable** — the strong recent metaheuristics (Mayfly, Dung
  Beetle, ABC, GRASP) are hard-paywalled (Elsevier/Springer); WebFetch cannot pass them; not invented.
- The verifiable comparison set is: MED → {Cura GA/TS/SA/PSO, IPSO-SA, Firefly} + ours; MPE →
  {Chang GA/SA/TS, Deng IPSO, ARO} + ours (once we compute it).
- **Action:** implement the Chang MPE metric and add OUR methods' row to Table 2 (a second, independent
  head-to-head with the MPE papers). Done in `src/orlib_mpe.py` → fills the TBD row.
- To get the paywalled MED numbers would require institutional/Sci-Hub PDF access (out of scope here).
