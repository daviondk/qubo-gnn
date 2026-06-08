# Extended published results on OR-Library port1–5 (more methods, verified)

Second, deeper sweep (2026-06-04) to enrich the comparison. **Critical caveats first.**

## ⚠ Two incompatible "MED" scales — never merge them
1. **Cura-scale MED** (the dominant convention: Cura 2009, Mozafari 2011, Bacanin & Tuba 2014):
   canonical GA Hang Seng MED = **0.0040**. This is the scale our results use.
2. **~10× smaller MED scale** (Tuo et al. 2016 HSDS): GA Hang Seng = 5.90e-4. Internally consistent
   but **not comparable** to the Cura scale — keep separate.
Also: many "new method" papers report the **Chang (2000) percentage-error (MPE)**, NOT MED — a
different metric family (separate table below).

## Correction to our earlier numbers
The Cura Nikkei (port5) value **0.0093 belongs to GA**, with TS/SA = 0.0010, PSO = 0.0019 (verified in
the original Cura PDF). Our tables already use these correct values.

## Consolidated MED table (Cura scale — directly comparable, verified vs primary PDFs)
| Method (paper) | port1 | port2 | port3 | port4 | port5 |
|---|---|---|---|---|---|
| GA (Chang 2000 → Cura 2009) | 0.0040 | 0.0076 | 0.0020 | 0.0041 | 0.0093 |
| TS (Chang 2000 → Cura 2009) | 0.0040 | 0.0082 | 0.0021 | 0.0041 | 0.0010 |
| SA (Chang 2000 → Cura 2009) | 0.0040 | 0.0078 | 0.0021 | 0.0041 | 0.0010 |
| PSO (Cura 2009) | 0.0049 | 0.0090 | 0.0022 | 0.0052 | 0.0019 |
| IPSO-SA (Mozafari 2011) | 0.0001 | 0.0001 | 0.0000 | 0.0001 | 0.0000 |
| Firefly mFA (Bacanin & Tuba 2014) | 0.0003 | 0.0009 | 0.0004 | 0.0003 | 0.0000 |
| **QRF-GNN original (ours)** | **0.0001** | **0.0004** | **0.0003** | **0.0004** | **0.0004** |
| **GNN+refine PyG (ours)** | **0.0001** | **0.0002** | **0.0000** | **0.0001** | **0.0000** |

**Best published (Cura scale): IPSO-SA** ties/holds best MED on all five (0.0000–0.0001); Firefly ties
on port3/port5. Our PyG GNN+refine matches this; our verbatim-original GNN beats all classic GA/TS/SA/PSO.

## Separate scale — HSDS (Tuo et al. 2016), NOT comparable to above
GA 5.90e-4 / 1.15e-3 / 3.03e-4 / 6.20e-4 / 1.50e-3; **HSDS 9.71e-7 / 3.39e-6 / 3.64e-6 / 3.86e-6 /
1.01e-5**. Different MED normalization → cannot be ranked against the Cura-scale table.

## Percentage-error (MPE %) family — same instances, different metric (flagged)
| Method (paper) | port1 | port2 | port3 | port4 | port5 |
|---|---|---|---|---|---|
| GA (Chang 2000) | 1.0974 | 2.5424 | 1.1076 | 1.9328 | 0.7961 |
| SA (Chang 2000) | 1.0957 | 2.9297 | 1.4623 | 3.0696 | 0.6732 |
| TS (Chang 2000) | 1.1217 | 3.3049 | 1.6080 | 3.3092 | 0.8975 |
| PSO (Deng, Lin, Lo 2012) | 1.0953 | 2.5417 | 1.0628 | 1.6890 | 0.6870 |
| ARO (Mansouri & Sadeghi-Moghadam 2021, arXiv:2101.03312) | 1.4181 | 1.3190 | 0.8151 | 1.4468 | 0.6179 |
| SA 25s (Nikiporenko 2023, arXiv:2307.04045) | 1.0950 | 2.3280 | 0.8456 | 1.3869 | 0.5980 |

## Exact / optimal floor
- **Xu, Tang, Yiu, Peng (2024)**, *INFORMS J. Computing* 36(2):690–704, DOI 10.1287/ijoc.2022.0344 —
  Lagrangian + branch-and-bound finds the **provable global optimum** on port1–4 (reports nodes/CPU,
  K=4/6/8, not MED). Implication: **the true MED floor is ≈ 0** (the cardinality frontier can be
  computed exactly). We confirm this directly with our own SCIP exact solver (see `11_*` / results).

## Exist but per-instance numbers paywalled / unverified (do NOT cite numbers)
Mayfly (Zheng 2023, ESWA 230:120656); ABC (Kalayci 2017, ESWA 85:61; Chen 2012, IEEE CEC); GRASP
(Baykasoğlu 2015, C&IE 90:339); hybrid metaheuristic (Swarm&Evol.Comp. 2020, 100662); clustering
(Ann. Oper. Res. 2026); PSO+Hopfield; cuckoo search. All use port1-5 but numbers not retrievable.

## Takeaway for the paper
- Use **Cura-scale MED**, K=10/ε=0.01/δ=1, 51 λ-points, 2000-point reference frontier. State the scale
  explicitly (HSDS shows how easily scales are conflated).
- The honest framing: our GNN-QUBO **matches the best published (IPSO-SA) and the exact optimum**, and
  **beats all classic baselines** — but the true floor is 0 (exact solvers reach it), so the
  contribution is "**first learned GNN-QUBO at SOTA accuracy on this benchmark + scaling**," not a new
  optimum. (See also Xu 2024 for the exact-method angle.)

### New verified citations to add to `02_references.md`
- Cura 2009 (already); Mozafari 2011 (already); Bacanin & Tuba 2014 (already);
- Tuo, Geng, Zhou 2016, *ECECSR* 50(1):311 (HSDS) — separate MED scale;
- Mansouri & Sadeghi-Moghadam 2021, arXiv:2101.03312 (ARO, MPE);
- Deng, Lin, Lo 2012, *ESWA* 39(4):4558 (improved PSO, MPE);
- Nikiporenko 2023, arXiv:2307.04045 (time-limited metaheuristics, MPE);
- Xu, Tang, Yiu, Peng 2024, *INFORMS J. Comp.* 36(2):690 (exact global optimum).
