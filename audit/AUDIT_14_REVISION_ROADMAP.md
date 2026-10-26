# AUDIT_14 — Prioritized revision roadmap

Priorities: **P0** scientific validity · **P1** essential methodology · **P2** manuscript/presentation ·
**P3** optional. "New compute?" answers whether a GPU run or API call is required. Nothing here alters
raw results.

---

## P0 — Scientific validity (must be fixed before submission)

### F-1 · The convergence residual is sign-inverted; the 0.752 and 0.842 numbers are not what the paper says
- **Location:** Eq. (2); §4.3; §6.2; Fig. 3; abstract; §1 contribution 2.
- **Evidence:** `ddag/results/features_t4.npz`, re-derived without a logistic layer: direct AUROC of
  `v_S/v_1` = **0.2403** (inverted); of `v_1/v_S` = **0.7597**; of the first-step magnitude `v_1` =
  **0.7101**. `ddag/analysis/crv_variants.py` passes the 1-D ratio through a single-feature logistic
  (`oof(...)`), which re-learns the sign and reports ≈0.752. The stored `e2_crv.json` reports
  `probe+r_conv = 0.7621`; the manuscript's **0.842** appears in **no** result file (it re-derives to
  0.8418 from a temporary script).
- **Action:** (i) redefine the residual — recommended `κ = v_1/v_S` or, simpler and stronger, the
  **first-step magnitude** `v_1`; (ii) re-derive and **store** the combined model in a result file;
  (iii) update Eq. (2), §4.3, §6.2, §1, the abstract and Fig. 3; (iv) add the methodological note that
  scalar signals must be scored by *direct* AUROC, never through a single-feature logistic; (v) fix the
  same defect in `ddag/analysis/crv_variants.py` and in `README_E2_T4.md` / `README_E2_crosshw.md`.
- **New compute?** No. **Effect:** removes a false headline; the corrected first-step signal (0.7101) is
  still a genuine positive.
- **Verification:** re-run `crv_variants.py` after the fix and confirm the stored AUROC equals the direct
  value.

### F-2 · Table 1 (seam repair) reports two wrong flicker means
- **Location:** Table 1, rows `none` and `fine (no switch)`.
- **Evidence:** `ddag/results/e8_flicker.json`: `none` = **0.01477** (not 0.0213); `fine` = **0.01890**
  (not 0.0254). The deltas (−0.0041, +0.0071) are correct.
- **Action:** correct the two means in the manuscript, in `README_E8_flicker.md`, in the site, and in the
  Kaggle dataset `ajjisan/ddag-experiments`; version-bump the dataset.
- **New compute?** No. **Effect:** makes the flicker trade-off *stronger* (the repaired value is the
  highest of the three, which the wrong numbers concealed).

### F-3 · The regime map asserts "optimal" without an objective
- **Location:** Eq. (1); §4.1; §6.5; conclusion.
- **Action:** define the objective explicitly (recommended: minimise DiT seconds subject to expected
  quality not worse than fixed-fine), state that the fixed policy is optimal **for that objective**, and
  note that at every measured point the cost-, quality- and Pareto-objectives coincide (the reason the
  adaptive region is empty).
- **New compute?** No.

### F-4 · Cost numbers are the prior study's, and two different ratios share one name
- **Location:** §6.1.
- **Evidence:** 3.01 / 1.60 / 2.33 come from the **prior preprint** (single-step regime); the repository's
  matched matrix gives T4 s2/s2 = 2.969 (break-even 2.684) and A100 s1/s1 = 1.561 (break-even 2.28); the
  DDAG band run gives a *per-segment cost ratio* `r = 0.728`.
- **Action:** attribute 3.01/1.60/2.33 to the prior study; use `r` (per-segment cost ratio) for the
  paper's own claims; introduce distinct symbols for the per-forward ratio and the cost ratio.
- **New compute?** No.

---

## P1 — Essential methodology

### F-5 · The "18/18 PASS" audit does not cover this paper
- **Evidence:** `jevgag/results/e9_audit/claim_audit.json` checks report claims E1–E5, E7 only.
- **Action:** extend the audit to the new paper's claims (CL-01…CL-25 in `AUDIT_06`), or re-word the
  README badge to state its scope.
- **New compute?** No.

### F-6 · Two manuscripts in one repository (self-overlap)
- **Evidence:** `paper/main.tex` (CVIU) and `jevgag/report/main.tex` (IEEEtran) both describe the audit.
- **Action:** decide the relationship before submission: (a) the report stays an unpublished repository
  artifact cited as such, or (b) the report is folded into this paper. State the decision in §3 and in
  the cover letter. **This is the largest non-numerical risk to the submission.**

### F-7 · Missing limitations / threats-to-validity section
- **Action:** add an explicit subsection: one backbone, one schedule family, two GPUs, n = 8 on A100,
  single seed, latent endpoint on A100 (no pixel endpoint there). This is load-bearing given the paper's
  own thesis.

### F-8 · Missing ablations
- **Evidence:** the data already exist (`probe_results.json` 30-block sweep; `e5_online.json` τ frontier).
- **Action:** add REC-5 and REC-7 (`AUDIT_12`). **New compute?** No.

### F-9 · No reproducibility detail in the paper
- **Action:** add a protocol table: backbone checkpoint IDs, upstream commit `ed3cb14`, seed 17, the 16
  prompts, horizon, mode/schedule pairs, and per-experiment n. All values exist in the repository.

### F-10 · Results absent from the paper but present in the repo
- **Evidence:** exp 03 (closed-loop 0.725 / 0.737), exp 04 (escalation gate NO-GO), exp 09 (pixel
  endpoint: coarse cheaper *and* closer).
- **Action:** exp 09 is direct support for "the cheap mode is not worse" — **add it**. exp 03/04 belong
  in supplementary or a one-sentence mention.

### F-11 · Endpoint asymmetry is not disclosed
- **Action:** state that the pixel endpoint was run on T4 only; that the A100 quality axis is latent-only.

---

## P2 — Manuscript and presentation

| ID | Location | Issue | Action |
|---|---|---|---|
| F-12 | Table 1 caption | "lower seam and lower fidelity are better" is ambiguous | re-word: "lower is better (fidelity = distance to the reference)" |
| F-13 | Table 2 | no CIs, no n, unqualified "dominant" | add CIs, n, and the objective |
| F-14 | Table 3 | row 1 mixes the report's 0.491 with this work's 0.798 | attribute each number |
| F-15 | Fig. 2 | axis sign easy to misread | add "(negative = coarse closer to the reference)" |
| F-16 | Fig. 4 / §6.4 | n = 32 vs n = 8 only in panel titles | state in prose and caption; add CI band |
| F-17 | Fig. 6 caption | device and segment definition missing | add both |
| F-18 | Fig. 5 | drawn but unused | use it or remove it |
| F-19 | §2 | novelty delta never stated explicitly | add a positioning paragraph |
| F-20 | §1 | no explicit research question; no falsification condition | add one sentence each |
| F-21 | §8 conclusion | no scope-of-generalisation statement | add |
| F-22 | Keywords | 7 (typically ≤ 6) | drop one |
| F-23 | Declarations | ethics statement missing | add "not applicable" |
| F-24 | §3 | label notation `ℓ(q)` ill-typed | use `d(m_c) > d(m_f)` |
| F-25 | §4.4 | gating algebra omits the probe cost | add `c_p` |

---

## P3 — Optional

| ID | Item | Note |
|---|---|---|
| F-26 | Random-router and oracle-router baselines (REC-2) | zero GPU; answers the obvious reviewer question |
| F-27 | Probe calibration diagram (REC-3) | zero GPU |
| F-28 | Compute/energy accounting (REC-10) | zero GPU |
| F-29 | Cite the report's endpoint result (ρ = +1.000) as evidence that an independent endpoint is achievable | zero GPU |
| F-30 | Make `ajjisan/wan21-t5` / `wan21-dit-vae` public, or document the HuggingFace fallback | affects reproducibility |

---

## Findings that do NOT require new experiments (all of F-1 … F-30 except F-6's decision)

Every correction above is derivable from stored result files. **No GPU run and no API call is required
to fix any P0 or P1 finding.** The only items that *would* need compute are the optional additional
baselines in `AUDIT_11` §3 (a step-allocation and a chunk-shape baseline), which are **not** required for
the paper's central claim.

---

## Does the central conclusion survive?

**Yes, and it is strengthened.** The corrected picture is:

1. The **cheap mode was never worse** at any measured operating point — verified on the latent endpoint
   (both GPUs), the decoded-pixel endpoint (T4) and the temporal-flicker endpoint (T4). The regime-map
   claim therefore survives, and the pixel-endpoint result (exp 09) makes it stronger.
2. The **regime map** survives; it only needs an objective to be well-posed (F-3).
3. The **intrinsic probe** survives on T4 (0.798 direct, verified) and the **inversion on A100** (0.143
   direct, verified) survives — indeed it is the paper's most citable finding.
4. The **convergence residual does NOT survive as written** (F-1) and must be redefined; the corrected
   first-step signal (0.7101 direct) is still a legitimate, smaller positive.
5. The **seam repair** survives (−25 %, verified); its flicker trade-off becomes *more* visible after the
   F-2 correction.
6. The **drift propagation** survives (verified, with CIs), with the n asymmetry to be disclosed.

**Verdict:** the paper is scientifically defensible after F-1…F-11, none of which needs new computation.
The single largest risk is not numerical but editorial: **F-6, two manuscripts in one repository.**
