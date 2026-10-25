# DDAG E2 — CRV cross-hardware: T4 vs A100

**Verdict: the convergence residual is HARDWARE-CONDITIONAL; the intrinsic probe is not.** The
`step_last/step_first` ratio is **0.752 on T4** but **0.485 (chance) on A100** — it does **not** transfer.
The intrinsic pre-generation probe **does** transfer and is *stronger* on A100 (**0.833** vs 0.792).

This is the base paper's **P1** lesson reappearing at the signal level: a signal can be a hardware
property, not a universal one.

## Cross-hardware table (16 prompts, 64 matched segments, seed 17)

| signal | T4 | A100 | transfers? |
|---|---|---|---|
| **convergence ratio** `step_last/step_first` | **0.752** [0.623, 0.863] p=0.000 | **0.485** [0.303, 0.715] p=0.558 | ❌ **no** |
| intrinsic prefill probe (block 11) | 0.792 [0.626, 0.913] | **0.833** [0.582, 0.979] | ✅ **yes** |
| `step_std` / `step_range` | 0.507 (chance) | 0.674 [0.592, 0.806] p=0.042 | ❌ (opposite) |
| naive `step_last` residual | 0.621 | 0.493 | ❌ |
| `prefill_norm` | 0.164 (confounded) | 0.154 (confounded) | — |

Label base rate differs by hardware: **18/64 positives on T4**, **10/64 on A100** — the same seed and
schedule produce different coarse-unsafe sets because the numerics differ.

## What this means

1. **RQ6/H7 answered directly:** adaptive-granularity *signals* are hardware-conditional. A verification
   statistic validated on one GPU can be at chance on another. Any claim must be measured per GPU —
   exactly the base paper's cost-precondition argument, now shown for the *signal*.
2. **The intrinsic probe is the robust component.** It improves on A100 (0.833), so the P1/idea-03 result
   generalises; the CRV ratio does not.
3. **Practical consequence:** a deployable controller must **calibrate its verification gate on the
   target GPU** (or use the probe, which transfers). This motivates the **regime map** (E11) and the
   **SDGP compilation** step (E10).

## Honest status of H1

| GPU | H1 (convergence residual verifies coarse-safety) |
|---|---|
| T4 | **supported** (ratio 0.752, p=0.000) |
| A100 | **refuted** (ratio 0.485, p=0.558; `step_std` weakly 0.674) |
| **overall** | **hardware-conditional** — reported as such, not averaged away |

## Files

| File | What |
|---|---|
| `features_t4.npz`, `manifest_t4.json` | T4 capture |
| `features_a100.npz`, `manifest_a100.json` | A100 capture |
| `a100/e2_crv.json`, `a100/e2b_residual_variants.json` | A100 analysis |
| `e2_crv.json`, `e2b_residual_variants.json` | T4 analysis |

## Reproduce

```bash
python ideas/ddag/scripts/crv_analysis.py --features ideas/ddag/results/features_a100.npz --out ideas/ddag/results/a100
python ideas/ddag/scripts/crv_variants.py  --features ideas/ddag/results/features_a100.npz --out ideas/ddag/results/a100
```

## Cost

A100: 256.30 → 261.24 → 256.30 compute units for the session (~5 units). Kaggle T4: free.
