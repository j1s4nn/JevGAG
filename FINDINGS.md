# JevGAG — Findings so far (E1, E2, E5, E7 complete; E3/E4/E6/E8 pending GPU)

All experiments reuse prior artifacts read-only (`../idea_05_adaptive_granularity_WM/04_Experiment`).
Jev pinned to `jev-1.14.0`. Image calls use `TypeSafe-Danger-Zone-Images: true`.

## E1 — Break-even predictor (POSITIVE tool)
Matched matrix over {T4, A100} x {fine s1/s2/s4} x {coarse c3-3 s1/s2/s4}, from the prior
mode-cost sweeps. Rule: coarse is cheaper iff `s_per_nfe_coarse/s_per_nfe_fine < NFE_fine/NFE_coarse`.

- **T4**: `coarse_s1` beats fine only at `fine_s2`/`fine_s4`; `coarse_s4` is **never** cheaper.
- **A100**: `coarse_s1` beats **all** fine modes; `coarse_s4` beats only `fine_s2`/`fine_s4`.
- Resolves the prior paper's confound (T4 used expensive coarse, A100 used cheap coarse):
  with matched modes the effect is **2D — hardware x operating point**.

Artifacts: `results/e1_breakeven/`, figure `assets/figures/fig_e1_breakeven.png`.

## E3 — Matched A100 online factorial (POSITIVE, operating-point)
Same A100, same 12 prompts (manifest a100-01), same horizon. Only the coarse step differs:
prior arm A2 v1 = expensive `c3-3 s4`; arm A2 v2 = cheap `c3-3 s1`.

| policy | exp s4 DiT (s) | exp refL1 | cheap s1 DiT (s) | cheap refL1 |
|---|---|---|---|---|
| fixed_fine | 3.81 | 0.2812 | 3.95 | 0.2832 |
| fixed_coarse | 4.43 | 0.2874 | **2.55** | **0.2728** |
| adaptive_warp | 3.87 (92% fine) | 0.2820 | **2.85 (21% fine)** | **0.2733** |

- EXPENSIVE s4: coarse **not** cheaper online (4.43 > 3.81) → adaptive forced to 92% fine.
- CHEAP s1: coarse **is** cheaper (2.55 < 3.95) → adaptive uses 21% fine and beats fixed_fine.
**Finding: the adaptive result flips with the coarse operating point alone** — this is the confound
in the prior paper (T4 used expensive coarse, A100 used cheap coarse) now isolated.

Artifacts: `results/e3_online/`.

## E4 — Recovery+hysteresis and the switch seam (POSITIVE, both GPUs)

| GPU | controller | switches | AUH refL1 | seam (switch/interior) |
|---|---|---|---|---|
| T4 | threshold_only | 0.50 | 0.3133 | 1.30 |
| T4 | full | 0.38 | 0.3116 | 1.18 |
| T4 | threshold_only_noisy | 1.12 | 0.3159 | 2.17 |
| T4 | full_noisy | 0.62 | 0.3150 | 1.53 |
| A100 | threshold_only | 0.50 | 0.2767 | 2.20 |
| A100 | full | 0.42 | 0.2772 | 1.98 |
| A100 | threshold_only_noisy | 0.92 | 0.2741 | 2.43 |
| A100 | full_noisy | 0.50 | 0.2718 | 1.75 |

**Finding:** recovery+hysteresis cuts switches on both GPUs without worsening regret; a switch
**seam** exists (switch-adjacent L1 > interior L1, ratio > 1) on both GPUs.

Artifacts: `results/e4_controller/`.

## E2 — Image-Jev vs text-Jev signal (NEGATIVE)
Paired, same 80 segments, same questions. Arms: text-1.13 (prior), text-1.14, image-1.14, hybrid-1.14.

| signal | text-1.13 | text-1.14 | image-1.14 | hybrid-1.14 |
|---|---|---|---|---|
| jev_difficulty | 0.788 | 0.777 | 0.667 | 0.687 |
| jev_drift | 0.738 | 0.755 | 0.647 | 0.656 |
| jev_motion | 0.703 | 0.706 | 0.510 | 0.568 |

Paired Δ(image−text14): difficulty −0.110, drift −0.108, **motion −0.195 (95% CI [−0.356,−0.030])**.
**Finding: image-grounded Jev does NOT beat text Jev as a pre-generation signal; it is worse.**
Text-1.13 ≈ text-1.14, so the deficit is the modality, not the model version.

Artifacts: `results/e2_signal/`, figure `assets/figures/fig_e2_signal.png`.

## E5 — Image-Jev as an independent endpoint (STRONG POSITIVE)
48 f4lite videos, 6 policies. Jev 1.14 asked whether the subject is preserved first→last and
how consistent the subject is. Compared to CLIP subject consistency (prior result).

| policy | Jev consistency | Jev same-subject | CLIP subj.cons. |
|---|---|---|---|
| reference | 3.461 | 0.894 | 0.990 |
| fixed_fine | 2.151 | 0.416 | 0.984 |
| adaptive_warp | 2.081 | 0.410 | 0.980 |
| random50 | 1.978 | 0.286 | 0.965 |
| alternating50 | 1.626 | 0.268 | 0.959 |
| fixed_coarse | 1.775 | 0.230 | 0.949 |

**Spearman(Jev same-subject, CLIP) = +1.000; Spearman(Jev consistency, CLIP) = +0.943.**
**Finding: the image-grounded oracle is a poor *signal* but an excellent, off-device *endpoint*.**

Artifacts: `results/e5_endpoint/`, figure `assets/figures/fig_e5_endpoint.png`.

## E7 — Confidence-gated routing (NEUTRAL)
Offline policy simulation on the 80 segments (0 GPU). Confidence-gating vs difficulty-only:

- 25% budget: +0.0004 · 50%: −0.0000 · 75%: −0.0013 (all ~0; oracle is far better).
**Finding: gating on Jev's own confidence does not improve the frontier on this set.**

Artifacts: `results/e7_confidence/`, figure `assets/figures/fig_e7_frontier.png`.

## Status
7 of 9 experiments complete (E1-E7, E9) with audited results and figures; E6 derived; **E8 pending Colab**.

## Done (0 GPU, all committed locally)
E1 break-even · E2 image-vs-text signal · E3 matched A100 online · E4 controller+seam ·
E5 image endpoint (rho=+1.0) · E6 T4 boundary (derived) · E7 confidence-gating · E9 claim audit (18/18) ·
figures · FINDINGS · paper skeleton.

## E6 — matched T4 boundary (DERIVED, negative)
On T4 with fine `c1-1 s1` (the online fine mode), coarse `c3-3 s1` saves **-4.57 s** (slower), so
adaptive routing cannot save wall-clock regardless of the signal → the T4 negative is an
**operating-point boundary**, not a signal failure. Coarse only wins on T4 once fine uses ≥2 steps
(vs fine_s2: +7.01 s; vs fine_s4: +29.70 s). Artifacts: `results/e6_t4_boundary/`.

## E9 — claim-to-artifact audit (18/18 PASS)
Every headline number re-derived from the stored result files. Artifacts: `results/e9_audit/`.

## Still pending
- **E8** stability-gated recalibration — the only experiment needing new A100 generation (Colab, ~2 h).
- **GitHub push** — blocked on the token (read-only for JevGAG).

## Honest headline so far
The image capability is **not** a better routing signal, but it **is** a validated independent
endpoint (ρ=1.0 vs CLIP). The break-even law is the reusable tool. This is a characterization
paper with one strong positive (endpoint) and two honest negatives (signal, confidence).
