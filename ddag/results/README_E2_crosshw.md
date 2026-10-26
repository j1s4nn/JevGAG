# DDAG E2 — CRV cross-hardware: T4 vs A100 (CORRECTED)

> **Correction notice.** An earlier version of this file reported the A100 probe at **0.833** and
> concluded "the probe is robust". That was an **analysis artifact**: `crv_analysis.py` trained a second
> logistic on the probe score, which silently re-learns the sign and reports `max(AUROC, 1−AUROC)`.
> The **direct** AUROCs are below. The corrected conclusion is *stronger and opposite*.

**Verdict: BOTH signals are hardware-conditional. The intrinsic probe works on T4 (0.798) and is
INVERTED on A100 (0.143). The convergence ratio works on T4 (0.752) and is at chance on A100 (0.485).**

## Corrected cross-hardware table (16 prompts, 64 matched segments, seed 17)

| signal | T4 | A100 | transfers? |
|---|---|---|---|
| intrinsic prefill probe (block 11, **direct AUROC**) | **0.798** | **0.143** (inverted) | ❌ **no — sign flips** |
| convergence ratio `step_last/step_first` | **0.752** [0.623, 0.863] p=0.000 | **0.485** [0.303, 0.715] p=0.558 | ❌ no |
| `step_std` / `step_range` | 0.507 (chance) | 0.674 p=0.042 | ❌ |
| naive `step_last` residual | 0.621 | 0.493 | ❌ |
| `prefill_norm` | 0.164 (confounded) | 0.154 (confounded) | — |

Label base rate differs by hardware: **18/64 positives on T4**, **10/64 on A100**.

## Why this matters (and is more interesting than the artifact)

1. **RQ6/H7 answered at the signal level, decisively:** *granularity-safety signals do not transfer
   across GPUs — and can invert.* A probe validated at 0.798 on one GPU is **anti-predictive** on
   another. Any deployable controller must **calibrate its signal per GPU**.
2. **Mechanistic reading:** on A100 the *coarse* mode is the better mode (E3: coarse drifts less;
   E5: coarse is 46% cheaper and closer to the s4 reference). So "coarse-unsafe" segments are a
   *different population* on A100, and a probe fitted to T4's notion of unsafe points the wrong way.
3. **This is the base paper's P1 lesson, sharpened:** cost, drift **and signal** are all regime
   properties. The contribution is the **regime map**, not a universal controller.

## Methodology note (worth keeping in the paper)

A single-score logistic evaluation (`oof(score)`) is **sign-agnostic** and will hide an inverted
signal. Report **direct AUROCs** for scalar signals, and only use logistic models for *multi-feature*
combinations (where learning the sign is legitimate).

## Files

| File | What |
|---|---|
| `features_t4.npz`, `features_a100.npz` | captures (identical labels/features across the two A100 copies) |
| `e2_crv.json`, `a100/e2_crv.json` | corrected analyses |
| `e2b_residual_variants.json`, `a100/e2b_residual_variants.json` | residual-variant sweeps (direct AUROCs) |

## Reproduce

```bash
python ideas/ddag/scripts/crv_analysis.py --features ideas/ddag/results/features_a100.npz --out ideas/ddag/results/a100
python ideas/ddag/scripts/crv_variants.py  --features ideas/ddag/results/features_a100.npz --out ideas/ddag/results/a100
```
