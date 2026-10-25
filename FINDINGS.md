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

## Pending (need GPU / external auth)
- **E3** matched online factorial on A100 (both coarse modes, same prompts) — Colab, ~3 A100-h.
- **E4** recovery+hysteresis + semantic seam — A100, ~1 h.
- **E8** stability-gated recalibration — A100, ~2 h.
- **E6** matched T4 boundary — Kaggle, free.
- **E9** claim-to-artifact audit + manuscript.

## Honest headline so far
The image capability is **not** a better routing signal, but it **is** a validated independent
endpoint (ρ=1.0 vs CLIP). The break-even law is the reusable tool. This is a characterization
paper with one strong positive (endpoint) and two honest negatives (signal, confidence).
