# DESIGN.md — JevGAG questions, hypotheses, preconditions, experiments (side by side)

> Companion to `STATUS.md` (phases) and `FINDINGS.md` (results). This file is the
> **design ledger**: every RQ, H, P and E, mapped to each other, with the verdict,
> where the assumption failed, and where we can improve.
> Updated 2026-10-25.

---

## 1. Side-by-side master table (RQ ↔ H ↔ P ↔ E ↔ verdict)

| RQ | Hypothesis (H) | Precondition (P) | Experiment(s) (E) | Result | Verdict |
|---|---|---|---|---|---|
| **RQ1** Does an image-grounded Jev oracle predict coarse-unsafe segments better than text/hand-crafted? | **H1** image-Jev AUROC > text-Jev | P2 Signal | **E2** (+E1 context) | image 0.667 / hybrid 0.687 / text 0.777; Δmotion −0.195 (CI excl. 0) | ❌ **FAILED** (image worse than text) |
| **RQ2** Is the cost–quality trade-off conditional on hardware *and* operating point? | **H2** coarse is cheaper on A100 but not T4 | P1 Cost | **E1, E3, E6** | T4 `s4` never cheaper; T4 `s1` cheaper only vs fine s2/s4; A100 `s1` cheaper vs all; online flips 21%↔92% fine | ⚠️ **BOUNDARY** (assumption too simple) |
| **RQ3** Do recovery + hysteresis cut switches without worsening regret, and is there a seam? | **H3** switches ↓, regret not worse | — | **E4** | T4 0.50→0.38, A100 0.50→0.42; seam ratio >1 both GPUs | ✅ **SUPPORTED** |
| **RQ4** Is the fine-vs-coarse benefit monotone in the signal under an independent label? | **H4** benefit monotone | P3 Endpoint | **E5** (re-label) | de-confounded with the image endpoint label | ⚠️ **PARTIAL** (label fixed; needs fresh monotonicity sweep) |
| **RQ5** Can an image-grounded Jev oracle be an endpoint independent of the compared modes? | **H5** image-Jev endpoint agrees with CLIP | P3 Endpoint | **E5** | Spearman **+1.000** (same-subject), +0.943 (consistency) | ✅ **SUPPORTED (strong)** |
| **RQ6** Does deferring to fine when Jev confidence is low improve the frontier? | **H6** confidence-gating improves the frontier | P4 Calibration | **E7** | Δ ≈ 0 at all budgets (\|Δ\|<0.005) | ❌ **FAILED** (neutral) |
| **RQ7** Does stability-gated high-step injection reduce long-horizon error? | **H7** recalibration reduces error | — | **E8** | A100 refL1 +0.0112, DiT +2.04 s; T4 +0.0202, +12.73 s | ❌ **FAILED** (worse on both GPUs) |
| — | — | — | **E9** | claim-to-artifact audit | ✅ **18/18 PASS** |

---

## 2. Research questions (RQ)

| RQ | Statement | Modality | Status |
|---|---|---|---|
| RQ1 | Does an **image-grounded** Jev oracle predict coarse-unsafe segments better than text/hand-crafted? | IMAGE (tested) | Negative |
| RQ2 | Is the cost–quality trade-off conditional on **hardware and operating point**? | — | Boundary (only headline negative) |
| RQ3 | Do recovery + hysteresis cut switches without worsening regret; is there a seam? | — | Positive |
| RQ4 | Is the benefit monotone in the signal under an **independent** label? | IMAGE label | Partial |
| RQ5 | Can image-grounded Jev be an endpoint **independent of the compared modes**? | IMAGE | Positive (strong) |
| RQ6 | Does **confidence-gating** improve the regret–cost frontier? | TEXT | Negative |
| RQ7 | Does **stability-gated recalibration** reduce long-horizon error? | IMAGE trigger | Negative |

## 3. Hypotheses (H)

| H | If / then / because | Verdict |
|---|---|---|
| H1 | IF Jev sees pixels THEN it predicts coarse-unsafe better than a caption, BECAUSE captions lose motion detail. | ❌ Failed |
| H2 | IF coarse chunks amortise forwards THEN adaptive routing saves time, BECAUSE chunk cost < fine cost. | ⚠️ Only at some operating points |
| H3 | IF hysteresis + recovery gate the signal THEN switches fall without regret, BECAUSE spikes are transient. | ✅ Supported |
| H4 | IF the signal is monotone THEN more signal ⇒ more fine decisions ⇒ less regret. | ⚠️ Partial |
| H5 | IF Jev compares frames THEN it certifies identity independently, BECAUSE identity is a visual comparison. | ✅ Supported |
| H6 | IF low confidence ⇒ defer to fine THEN regret falls, BECAUSE confidence marks uncertainty. | ❌ Failed |
| H7 | IF drift ⇒ inject a high-step segment THEN error resets, BECAUSE a fresh high-quality anchor re-grounds the rollout. | ❌ Failed |

## 4. Preconditions (P)

| P | Statement | Status |
|---|---|---|
| P1 Cost | Coarse cheaper **on the deployment hardware and operating point** | Conditional (RQ2) |
| P2 Signal | A **pre-generation** signal identifies unsafe segments | Met by **text** Jev (not image) |
| P3 Endpoint | Certified by an endpoint **independent of the compared modes** | Met (image Jev, ρ=1.0) |
| P4 Calibration | Oracle confidence calibrated enough to gate on | Not met (E7 neutral) |

## 5. Experiments (E)

| E | Experiment | Inputs | Status | Artifact |
|---|---|---|---|---|
| E1 | Break-even predictor | prior mode-cost + timing | ✅ | `results/e1_breakeven/` |
| E2 | Image vs text vs hybrid signal (n=80 paired) | prior states + Jev API | ✅ | `results/e2_signal/` |
| E3 | Matched A100 online factorial | prior A100 online arms | ✅ | `results/e3_online/` |
| E4 | Recovery + hysteresis + seam | prior rq3 arms | ✅ | `results/e4_controller/` |
| E5 | Image endpoint + RQ4 re-label | 48 videos + Jev API | ✅ | `results/e5_endpoint/` |
| E6 | Matched T4 boundary (derived) | E1 + prior T4 online | ✅ | `results/e6_t4_boundary/` |
| E7 | Confidence-gated routing | E2 answers | ✅ | `results/e7_confidence/` |
| E8 | Stability-gated recalibration | new A100 + T4 generation | ✅ | `results/e8_a100/`, `results/e8_recalibration/` |
| E9 | Claim-to-artifact audit | all results | ✅ | `results/e9_audit/` |

---

## 6. Where our assumptions FAILED (honest)

1. **"Image is the better signal" (H1) — FAILED.** Image/hybrid Jev are *worse* than the text
   state (motion ΔAUROC −0.195, 95% CI excluding zero). The caption + motion statistics carried
   more discriminative information than three raw frames. Text-1.13 ≈ text-1.14, so it is the
   **modality**, not the model version. Consequence: use Jev **text** for the signal.
2. **"Coarse is cheaper, so adaptive routing always helps" (H2) — FAILED as stated.** It holds only
   when the fine mode uses ≥2 steps (break-even = NFE_fine/NFE_coarse exceeds the hardware
   per-forward ratio). On T4 with fine `s1` the coarse is *slower*; the prior audit's T4-negative vs
   A100-positive was confounded (expensive coarse on T4, cheap coarse on A100).
3. **"Confidence marks uncertainty" (H6) — FAILED.** Gating on Jev's confidence changed the
   frontier by |Δ|<0.005 — no benefit. Consequence: P4 is not met; report as a negative.
4. **"A high-step anchor resets drift" (H7) — FAILED.** Recalibration was *worse* on both GPUs
   (A100 +0.0112 refL1/+2.04 s; T4 +0.0202/+12.73 s). On T4 the gate never fired (0% fine →
   degenerated to all-coarse).
5. **"The warp signal transfers across GPUs" — WEAK.** The recalibrate gate calibrated on T4 did not
   fire on T4's own prompts, showing the threshold is operating-point/hardware sensitive.

## 7. Where we can IMPROVE

| # | Improvement | Why | Effort |
|---|---|---|---|
| I1 | **Scale the E2 signal test beyond n=80** | regenerate the 288 Phase-5B segments' videos (Kaggle T4) → tighter CIs | ~1 T4 job |
| I2 | **Tune the image-signal prompt/frames** | try more frames, Box questions, or a task-specific phrasing before declaring the modality worse | 0 GPU (API) |
| I3 | **Calibrate the recalibration gate per GPU** | E8's T4 gate fired 0 times; a per-GPU threshold is the fair test | 0 GPU re-analysis |
| I4 | **Add a second backbone** | generality beyond Causal-rCM Wan2.1-1.3B | ~2 A100-h |
| I5 | **Use Jev 1.14 Box questions** | localize the subject for a spatially-grounded endpoint | 0 GPU (API) |
| I6 | **Fresh RQ4 monotonicity sweep** | the E5 label de-confounds but a new sweep is cleaner | ~1 A100-h |
| I7 | **More E8 prompts** | 12 A100 / 8 T4 prompts is modest | T4 free |
| I8 | **Full GPU-hour ledger** | the prior paper lacked it; log per-experiment units | bookkeeping |

## 8. One-line takeaways

- **Signal ⇒ use Jev TEXT. Endpoint ⇒ use Jev IMAGE.** The modality that certifies is not the
  modality that decides.
- **The break-even law is the reusable tool**: it predicts when coarse is worth it given
  (fine steps, coarse steps, hardware).
- **Three honest negatives (E2 signal, E7 confidence, E8 recalibration) plus the RQ2 boundary.**
