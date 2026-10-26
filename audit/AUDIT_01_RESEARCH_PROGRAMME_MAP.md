# AUDIT_01 — Research programme map (contribution → study → experiment → evidence)

The repository contains **three distinct bodies of work**. The audit must not conflate them.

| Body | Where | Status | GPU(s) | Used by the new paper as |
|---|---|---|---|---|
| **A. Prior preprint** | external, `10.5281/zenodo.23395414` | under review (TCSVT-32491-2026, per author) | T4 (+ an A100 arm) | **prior work** (cited, §1, §3) |
| **B. JevGAG report** | `jevgag/` incl. `jevgag/report/main.tex` | **unpublished manuscript in-repo** | T4 + A100 | **background** (§3) |
| **C. DDAG method** | `ddag/` | experiments + analyses, no manuscript | T4 + A100 | **the contribution** (§4–§6) |

---

## 1. Contribution → evidence map for `paper/main.tex`

| # | Manuscript claim (location) | Originating body | Experiment | Source code | Source result file · field |
|---|---|---|---|---|---|
| C1 | two-GPU regime map; adaptive region empty (§1, §4.1, §6.5, Tab. 3, Fig. 2) | **C** | 07 online policy, 08 band | `ddag/analysis/online_policy.py` | `ddag/results/{online_t4,online_a100,band_t4}/e5_online.json` · `cost_ratio_coarse_over_fine`, `mean_q_*`, `fixed_*` |
| C2 | intrinsic probe 0.798 (T4) (§1, §4.2, §6.2, Fig. 3) | **C** | 01 intrinsic probe, 05 convergence | `ddag/analysis/crv_analysis.py` | `ddag/results/e2_crv.json` · `probe_block11_direct_auroc` |
| C3 | probe inverts on A100 (0.143) (§1, §6.2, Fig. 3, Tab. 4) | **C** | 05 convergence (A100) | `crv_analysis.py` | `ddag/results/a100/e2_crv.json` · `probe_block11_direct_auroc` |
| C4 | convergence residual 0.752; probe+residual 0.842 (§1, §4.3, §6.2) | **C** | 05 convergence | `crv_variants.py` | `ddag/results/e2b_residual_variants.json` · `step_last_over_first` |
| C5 | seam repair −25 %, fidelity improves (§1, §4.5, §6.3, Tab. 2) | **C** | 02 seam repair, 11 flicker | `flicker_analysis.py` | `ddag/results/e8_flicker.json` · `seams`, `fids`, `flicker_*` |
| C6 | flicker rises under repair (§6.3, Tab. 2) | **C** | 11 flicker | `flicker_analysis.py` | `e8_flicker.json` · `flicker_delta_blend_minus_none` |
| C7 | drift propagates on T4, 4/5 tail segments (§1, §6.4, Fig. 4) | **C** | 06 drift (n = 32) | `drift_analysis.py` | `ddag/results/t4_n32/e3_drift.json` · `propagation[*]` |
| C8 | drift reverses on A100 (§6.4, Fig. 4) | **C** | 06 drift (n = 8) | `drift_analysis.py` | `ddag/results/a100/e3_drift.json` · `mean_drift` |
| C9 | drift linear, not policy-predictable, R²≈0.94 / 0.01 (§6.4) | **C** | 06 drift, E4 model | `propagation_model.py` | `ddag/results/t4_n32/e4_propagation.json` · `M1_current_drift_r2`, `M4_increment_from_mode_r2` |
| C10 | cost is hardware-conditional; ratios 3.01 / 1.60, break-even 2.33 (§6.1) | **A (prior study)** | — | — | **not in this repo** — from the prior preprint |
| C11 | one-step coarse on T4 → ratio 0.73 (§6.1) | **C** | 08 band | `online_policy.py` | `ddag/results/band_t4/e5_online.json` · `cost_ratio_coarse_over_fine` = 0.728 |
| C12 | Jev critic at chance (0.497) (§1, Tab. 4) | **C** | 10 Jev critic | `jcd_analysis.py` | `ddag/results/e7_jcd.json` · `tests.jev_discontinuity.auroc` |
| C13 | preconditions P1/P2/P3; hysteresis + recovery; seam exists (§3) | **B (report)** | report E1–E9 | `jevgag/code/e1…e9` | `jevgag/results/**`; `claim_audit.json` (18/18) |
| C14 | 0.491 hand-crafted / 0.788 Jev text (§1, §6.2, Fig. 3) | **A + B** | report E2 | `jevgag/code/e2_*.py` | `jevgag/results/e2_signal/e2_audit.json` |
| C15 | qualitative: modes differ, neither worse (§6.6, Fig. 6) | **C** | 12 qualitative | kernel `qual.py` | `ajjisan/ddag-matched-frames` (frames) |

---

## 2. Conflation risks found

| Risk | Detail | Severity |
|---|---|---|
| **R-1** | C10 attributes the prior study's cost ratios (3.01/1.60/2.33, single-step regime) to a paper whose own cost table (Tab. 3) uses the two-step regime (ratios 1.030 / 0.536). The prose and the table describe **different regimes** | **P1** |
| **R-2** | C13 ("the report") is summarised in §3 but the report is a **standalone unpublished manuscript** in the same repository; the boundary between "prior work" and "this work" is blurred | **P1** |
| **R-3** | §6.4 mixes T4 (**n = 32**) and A100 (**n = 8**) drift results without stating the asymmetry in the prose (it is only in Fig. 4's panel titles) | P2 |
| **R-4** | §6.1 calls 0.728 "the measured ratio" while the surrounding sentence uses "per-forward ratio" for a different quantity (3.01). Two different ratios share one name | **P1** |
| **R-5** | The claim-to-artifact audit (18/18) covers **body B only**. Every C-series claim (C1–C9, C11, C12, C15) is **outside its scope**, yet the README badge "claim-to-artifact audit 18/18 PASS" is displayed next to the new paper | **P1** |
| **R-6** | The A100 signal (0.143) is reported as an "inversion"; the same A100 run also yields `probe+r_conv = 0.743`, which is **above chance** and is not reported anywhere | P2 |
| **R-7** | §6.5 "the optimal policy is a fixed mode" is stated without defining the objective (cost-optimal? Pareto? quality-constrained?) | **P1** |

---

## 3. Experiment inventory (body C) — status and completeness

| # | Folder | Kernel present | Results present | Analysis script | Reported in paper |
|---|---|---|---|---|---|
| 01 | `01_intrinsic_probe` | ✅ | ✅ npz + json | `crv_analysis.py` (shared) | yes (C2) |
| 02 | `02_seam_repair` | ✅ | ✅ npz + json | `analyze.py` | yes (C5) |
| 03 | `03_closed_loop_evidence` | ✅ | ✅ | `closedloop.py` | **no** (only in README/site) |
| 04 | `04_escalation_gate` | ❌ (0-GPU only) | ✅ `gate_check.json` | `gate_check.py` | **no** (only in README/site) |
| 05 | `05_convergence_residual` | ✅ | ✅ (shared `ddag/results/`) | `crv_analysis.py`, `crv_variants.py` | yes (C3, C4) |
| 06 | `06_drift_propagation` | ✅ | ✅ | `drift_analysis.py` | yes (C7–C9) |
| 07 | `07_online_policy` | ✅ | ✅ | `online_policy.py` | yes (C1, Tab. 3) |
| 08 | `08_intermediate_band` | ✅ | ✅ | `online_policy.py` | yes (C11) |
| 09 | `09_pixel_endpoint` | ✅ | ✅ npz | *(analysis not shipped)* | **mentioned in README, not in the paper** |
| 10 | `10_jev_critic` | ✅ | ✅ | `jcd_analysis.py` | yes (C12) |
| 11 | `11_flicker_endpoint` | ✅ | ✅ | `flicker_analysis.py` | yes (C6) |
| 12 | `12_qualitative` | ✅ | frames (Kaggle) | — | yes (C15) |

**Gaps:** experiments **03** (closed-loop, 0.725/0.737), **04** (escalation gate, NO-GO) and **09**
(pixel endpoint: coarse cheaper *and* closer) carry **substantive results that are absent from the
manuscript** although they appear in the README/site. Either they belong in the paper (they strengthen
the "cheap mode is not worse" argument) or they should be explicitly deferred to supplementary material.
**P1.**
