# DDAG E2 — CRV (Convergence-Residual Verification), T4

**Verdict: H1 SUPPORTED in refined form.** The *naive* final-step residual is weak (0.621), but the
**ratio of the last to the first step-norm** of the coarse run is a strong, **free, training-free**
signal (**0.752**, CI [0.623, 0.863], permutation p = 0.000), and **probe + ratio = 0.842 AUROC** — the
best signal in the project, fully intrinsic (no external oracle, no caption).

## Headline (T4, 16 prompts, 64 matched segments)

| signal | AUROC | 95% CI | perm p | cost |
|---|---|---|---|---|
| chance | 0.500 | — | — | — |
| hand-crafted context (base paper) | 0.491 | — | — | free |
| **convergence ratio** (`step_last / step_first`) | **0.752** | [0.623, 0.863] | **0.000** | **free** (coarse run) |
| intrinsic prefill probe (idea 03, block 11) | 0.792 | [0.626, 0.913] | — | trained probe |
| **probe + convergence ratio** | **0.842** | — | — | probe + free |
| Jev text oracle (base paper) | 0.788 | [0.575, 0.946] | — | external API |
| probe + ratio + free evidence | 0.819 | — | — | probe + free |

Evaluation: leave-one-prompt-out CV grouped by prompt, prompt-grouped bootstrap (3,000 resamples),
1,000 label permutations.

## What failed (honest negatives)

| variant | AUROC | note |
|---|---|---|
| `step_last` (naive final-step norm) | 0.621 | weak — the distilled 2-step sampler does not expose convergence this way |
| `step_std`, `step_range` | 0.507 | **chance** |
| `prefill_norm` | 0.164 | **sign-inconsistent / confounded** — with a 1-D feature and LOO-CV the model learns the sign, so a value far below 0.5 is *not* a usable (flipped) signal |
| `r_std` (an earlier, careless variant) | 0.738 | **artifact** — it accidentally included the prefill and append norms in the "steps" list; corrected here |

**Lesson:** the residual must be computed over the **denoising steps only**, and the informative quantity
is the **step-to-step change ratio**, not the absolute final magnitude.

## Interpretation

For a distilled few-step sampler, an unconverged chunk is one whose **update magnitude does not shrink**
from the first to the last step. `step_last / step_first ≈ 1` means the sampler is still "moving" at the
end → coarse-unsafe. This is a causal reading, and it is **free**: it uses only the velocity norms the
coarse run already computes.

## Protocol

Causal-rCM Wan2.1-1.3B (`ed3cb14`), Kaggle T4, 57 frames, seed 17, 16 prompts, matched fine
(`c1-1_step2`) vs coarse (`c3-3_step2`) from a shared boundary; label = coarse-unsafe (latent `refL1` vs
the `c1-1_step4` reference). Per-call velocity norms captured by a forward hook on the DiT.

## Files

| File | What |
|---|---|
| `features_t4.npz` | prefill activations, per-call velocity norms (`vn_coarse`, `vn_fine`), labels, groups, evidence |
| `manifest_t4.json` | protocol, per-segment records, timing |
| `e2_crv.json` | AUROCs, CIs, calibration sweep |
| `e2b_residual_variants.json` | residual-variant sweep with CIs and permutation nulls |
| `fig_e2_crv.png` | AUROC by test |

## Reproduce

```bash
python ideas/ddag/scripts/crv_analysis.py --features ideas/ddag/results/features_t4.npz --out ideas/ddag/results
python ideas/ddag/scripts/crv_variants.py --features ideas/ddag/results/features_t4.npz --out ideas/ddag/results
```

## Next

- Run the same kernel on **A100** (cross-hardware transfer of the convergence ratio).
- E3: drift accumulation study.
