# AUDIT_12 — Recommended figures (evidence-based, from existing results)

Inventory of what already exists: 6 new-paper figures + 5 report figures (see `AUDIT_08`); result files
with per-prompt data (`features_*.npz`, `e5_online.json`, `e3_drift.json`, `e2_crv.json`,
`e2b_residual_variants.json`, `e7_jcd.json`, `e8_flicker.json`, `gate_check.json`, `e1_closedloop.json`).

Each recommendation below states the scientific question, the data, the panels, placement, priority,
effort, and whether new computation is needed. **No figure is recommended merely to raise the count.**

---

## REC-1 — Cost–quality Pareto frontier for all verified policies ★★★ (P1, zero GPU)

- **Question.** Where do the measured policies lie in the (DiT seconds, quality) plane, and is any
  adaptive policy on the frontier?
- **Data.** `ddag/results/{online_t4,online_a100,band_t4}/e5_online.json` → `fixed_fine`,
  `fixed_coarse`, `oracle_route`, `oracle_escalate`, and the 41-point `frontier` array.
- **Panels.** One panel per operating point; x = DiT total seconds, y = mean endpoint distance;
  markers for fixed fine / fixed coarse / oracle route / the probe-routed frontier; the Pareto front
  drawn as a line.
- **Why it matters.** It is the single clearest statement of the paper's thesis and it makes the
  "adaptive region is empty" claim visual and falsifiable.
- **Placement.** Main paper (replaces or supplements Table 2). **Effort:** low (values already stored).

## REC-2 — Random-router and oracle-router baselines ★★★ (P1, zero GPU)

- **Question.** How much does the *signal* buy over chance, and how far is it from a perfect signal?
- **Data.** `oracle_route` is already stored in `e5_online.json`; a random router can be computed from
  `features_*.npz` (labels + costs) by sampling escalation rates.
- **Panels.** Bar/interval chart of DiT seconds and quality for: fixed fine, fixed coarse, random
  router (matched escalation rate), probe router, oracle router.
- **Why it matters.** Reviewers will ask for a chance baseline; the oracle bound shows the headroom is
  small, which *supports* the negative result.
- **Placement.** Main paper §6.5. **Effort:** low.

## REC-3 — Probe calibration / reliability diagram ★★ (P2, zero GPU)

- **Question.** Is the intrinsic probe calibrated, or only rank-accurate?
- **Data.** `features_t4.npz` + `ddag/analysis/crv_analysis.py` (out-of-fold scores → probabilities).
- **Panels.** Reliability diagram (predicted vs empirical frequency) + a histogram of predicted scores,
  for T4 and A100 (the A100 curve will show the inversion dramatically).
- **Placement.** Supplementary, or a compact inset in Fig. 3. **Effort:** low.

## REC-4 — Endpoint agreement matrix ★★ (P1, zero GPU)

- **Question.** Do the three endpoints agree on the *ranking* of the modes, per prompt?
- **Data.** `features_pxq_t4.npz` (latent + pixel), `e8_flicker.json` (flicker), `e5_online.json`.
- **Panels.** Per-prompt paired differences (latent Δ, pixel Δ, flicker Δ) with prompt-grouped CIs;
  plus a Spearman agreement matrix between endpoints.
- **Why it matters.** The paper claims "no endpoint ranks coarse worse". This figure *proves* it per
  prompt rather than in aggregate, and it is the paper's most defensible methodological contribution.
- **Placement.** Main paper §6.3/§6.6. **Effort:** low.

## REC-5 — Corrected signal figure with CI whiskers ★★★ (P0-adjacent, zero GPU)

- **Question.** (replaces Fig. 3) Which signal predicts coarse-unsafe segments, per GPU?
- **Data.** `e2_crv.json`, `a100/e2_crv.json`, `e2b_residual_variants.json`, plus the **corrected**
  residual definition from `AUDIT_07` M-1.
- **Panels.** Grouped bars with 95 % CI whiskers; direct AUROCs only (never single-feature logistic);
  a marker for the *first-step magnitude* (0.7101) and the *reciprocal ratio* (0.7597).
- **Placement.** Main paper. **Effort:** low. **Must be regenerated after M-1.**

## REC-6 — Drift trajectories with uncertainty bands ★★ (P1, zero GPU)

- **Question.** Is the propagation real, and how large is it relative to between-prompt variance?
- **Data.** `t4_n32/e3_drift.json` (mean + the `propagation` CIs), `a100/e3_drift.json`.
- **Panels.** Existing Fig. 4 plus a shaded bootstrap CI band on the one-coarse chain and an explicit
  `n = 32` / `n = 8` annotation.
- **Placement.** Main paper (upgrade Fig. 4). **Effort:** low.

## REC-7 — Ablation panel: probe block depth and gating threshold ★★ (P1, zero GPU)

- **Question.** Is the probe result an artefact of the chosen block? Is the gate sensitive to τ?
- **Data.** `01_intrinsic_probe/results/probe_results.json` (30-block sweep), `e5_online.json`
  (`frontier` over 41 τ values).
- **Panels.** (a) AUROC vs block depth, T4 and A100; (b) escalation rate vs τ, with the cost-optimal
  point marked.
- **Placement.** Main paper §6.2 (compact) or supplementary. **Effort:** low.

## REC-8 — Architecture of the deployable controller ★★ (P1, low effort, TikZ)

- **Question.** What exactly would a practitioner build?
- **Content.** Pre-generation probe → gate (with τ) → mode choice; escalation branch; post-generation
  seam repair; the regime-map decision upstream. Distinct from Fig. 1 (which is the *research
  programme*, not the *system*).
- **Placement.** Main paper §4 (system figure). **Effort:** low (TikZ, same style as Fig. 1).

## REC-9 — Qualitative temporal sequences (video strips) ★ (P2, needs media)

- **Question.** How do the modes differ over *time*, not just at one frame?
- **Data.** `ajjisan/ddag-matched-frames` (54 frames, 3 frames per variant per segment).
- **Panels.** 3 × 4 strips (fine / coarse / repaired × 3 timesteps) for 2 prompts, with the difference
  strip.
- **Placement.** Supplementary (the current Fig. 6 already covers the main point). **Effort:** low.

## REC-10 — Compute / energy accounting ★ (P3, zero GPU)

- **Question.** What did the study cost to run?
- **Data.** Kernel logs (elapsed seconds), Kaggle/Colab session logs.
- **Placement.** Supplementary table. **Effort:** low.

---

## NOT recommended

| Candidate | Why not |
|---|---|
| ROC/PR curves for the probe (candidate C) | AUROC + calibration (REC-3) carries the same information with less space; PR is misleading at 28 % prevalence. |
| A separate "cost breakdown by NFE" figure (candidate E) | Already covered by REC-1's axes and Table 2. |
| A seam-vs-flicker figure (candidate G) | **Already drawn** (`fig_seam.pdf`) — the action is to *use* it (AUDIT_10 MM-6), not to add another. |
| Hysteresis/recovery ablation figure (candidate J) | The relevant data live in the **report** (E4), not in the new experiments; plotting them would import report results into the method paper (overlap risk). |

## Priority order (all zero-GPU except REC-9)

1. **REC-5** (corrected signal figure) — required by M-1.
2. **REC-1** (Pareto frontier) — the thesis, visually.
3. **REC-2** (random + oracle baselines) — answers the obvious reviewer question.
4. **REC-4** (endpoint agreement) — makes "no endpoint ranks coarse worse" rigorous.
5. **REC-8** (controller architecture) — makes the method deployable.
6. REC-6, REC-7, REC-3, REC-9, REC-10.
