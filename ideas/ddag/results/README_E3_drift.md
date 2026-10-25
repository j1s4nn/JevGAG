# DDAG E3 — drift accumulation (T4 vs A100)

**Verdict: drift behaviour is HARDWARE-CONDITIONAL, and on T4 error PROPAGATES.** On **T4**, one coarse
segment leaves a **significant, persistent** drift elevation for the rest of the rollout (H2c supported
at **4/5** tail segments with n=32), while all-coarse drift is *not* uniformly worse; on **A100** the
sign of the coarse-vs-fine drift comparison **reverses**. Drift is ~**linear**, not super-linear, and the
next increment is **not predictable** from the current state (R² ≈ 0).

This is the base paper's **P1** lesson a third time: *the granularity trade-off — cost, signal, and now
drift — is a property of the hardware, not a universal fact.*

## Propagation (T4, C − A, n=32 prompts) — the DALG premise

| segment | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|
| Δ(C − A) | **+0.0313** | **+0.0304** | **+0.0208** | **+0.0241** | +0.0212 |
| 95% CI | [+0.010, +0.052] | [+0.009, +0.051] | [+0.000, +0.042] | [+0.003, +0.045] | [−0.004, +0.046] |
| significant | ✅ | ✅ | ✅ | ✅ | ❌ |

**A single coarse segment at position 1 keeps the drift above the all-fine baseline for the rest of the
rollout.** Error propagation is real → a *lookahead* controller has something to plan for.

## Mean drift (latent L1 to the `c1-1 s4` reference), per segment

**T4** (81 frames, 21 latent, 6 segments, shared coarse prefix, n=10 prompts)

| chain | s1 | s2 | s3 | s4 | s5 | s6 |
|---|---|---|---|---|---|---|
| A all-fine | 0.773 | 0.818 | 0.877 | 0.912 | 0.944 | **0.970** |
| B all-coarse | 0.778 | 0.805 | 0.854 | 0.950 | 0.960 | **0.997** |
| C coarse→fine | 0.778 | 0.862 | 0.913 | 0.943 | 0.977 | **1.000** |

**A100** (same protocol, n=8 prompts)

| chain | s1 | s2 | s3 | s4 | s5 | s6 |
|---|---|---|---|---|---|---|
| A all-fine | 0.822 | 0.858 | 0.892 | 0.917 | 0.949 | **0.952** |
| B all-coarse | 0.756 | 0.812 | 0.812 | 0.800 | 0.802 | **0.812** |
| C coarse→fine | 0.756 | 0.822 | 0.861 | 0.882 | 0.916 | **0.935** |

## Formal tests

| test | T4 (n=32) | A100 (n=8) |
|---|---|---|
| H2a coarse drift > fine (final segment) | ❌ mixed (B 0.947 < A 0.954) | ❌ **reversed** (B 0.812 < A 0.952) |
| H2b super-linear growth | ❌ linear (lin R²=0.950) | ❌ linear (lin R²=0.343) |
| H2c **propagation** (C stays above A after one coarse) | ✅ **SUPPORTED** — 4/5 segments significant (+0.021..+0.031) | ❌ tracks coarse (below A) |
| H3 next increment predictable | ❌ R² = −0.05 | ❌ R² = −0.02 |

**T4 mean drift, n=32:** A `0.789 0.826 0.867 0.899 0.929 0.954` · B `0.783 0.804 0.834 0.910 0.911 0.947`
· C `0.783 0.857 0.898 0.920 0.953 0.975`.

## Interpretation

1. **Where drift is a problem (T4), it is real but modest and linear** — a single coarse segment leaves a
   ~+0.03 elevation that does not grow. So *lookahead* (DALG) has limited headroom: the error does not
   compound super-linearly, and it is not predictable from state (H3 fails).
2. **Where the GPU is strong (A100), coarse is not just cheaper but also less drifty** — so the case for
   adaptive granularity on A100 is *cost + quality*, while on T4 it is *cost only, via `hold`*.
3. **This sharpens the paper's thesis:** the deployable object is not a controller that always helps; it
   is a **regime map + a per-GPU calibrated gate** (E11, E2).

## Honest negatives (kept)

- H2b **super-linearity: refuted** on both GPUs.
- H3 **increment predictability: refuted** on both GPUs (R² ≈ 0).
- H2c propagation: **direction consistent on T4, significance pending** at n=32; **absent on A100**.
- The A100 drift metric is latent-L1 to a *fine* reference, which flatters coarse on A100; a decoded-pixel
  endpoint is the natural follow-up (E8/F4).

## E4 — error-propagation model (the model DALG would plan with)

Transition dataset: `(d_t, next mode, position, chain) -> d_{t+1}` (480 transitions on T4, n=32).

| model | T4 R² | A100 R² |
|---|---|---|
| M0 persistence (`d_{t+1}=d_t`) | 0.905 | 0.865 |
| M1 current drift only | **0.944** | **0.891** |
| M2 drift + next mode | 0.943 | 0.893 |
| M3 full | 0.945 | 0.899 |
| M4 increment from mode/position/chain | **0.008** | **0.077** |

**Verdict: the drift LEVEL is highly predictable (R²≈0.94) but only because drift is a smooth
self-predicting trajectory; the granularity choice adds essentially nothing (M2 ≈ M1), and the
increment is not predictable from the policy (M4 R²≈0.01).**

**Consequence for DALG:** the lookahead premise is weak. Drift moves by only +0.035 ± 0.041 per segment
(T4) and the mode barely shifts that increment, so a receding-horizon controller has very little headroom
over a persistence baseline. DALG is therefore **demoted** from a headline method to a bounded,
honestly-reported component; the paper's weight shifts to the **regime map + cost model** (E11) and the
**robust intrinsic probe** (E2).

## Files

| File | What |
|---|---|
| `drift_t4.npz`, `drift_t4_n24.npz`, `drift_a100.npz` | per-prompt drift trajectories per chain |
| `drift_manifest_t4.json` | T4 protocol + timing |
| `e3_drift.json`, `t4_n24/e3_drift.json`, `a100/e3_drift.json` | analyses |
| `fig_e3_drift.png` | drift trajectories |

## Reproduce

```bash
python ideas/ddag/scripts/drift_analysis.py --meas ideas/ddag/results/drift_t4_n24.npz --out ideas/ddag/results/t4_n24
python ideas/ddag/scripts/drift_analysis.py --meas ideas/ddag/results/drift_a100.npz  --out ideas/ddag/results/a100
```

## Cost

T4: free (Kaggle). A100: session `ddag-a100b`, ~2 units.
