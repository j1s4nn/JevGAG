# SelfAware-DiT — results (P1)

**Verdict: SUPPORTED.** An intrinsic, *pre-generation* probe on the DiT's own activations
predicts coarse-unsafe segments at **AUROC 0.798** (block 11), matching the external Jev text
oracle (0.788) and far above the hand-crafted context signal (0.491).

## Headline

| Predictor | AUROC | 95% CI | perm p |
|---|---|---|---|
| chance | 0.500 | — | — |
| hand-crafted context (base paper) | 0.491 | — | — |
| **SelfAware-DiT prefill probe (block 11)** | **0.798** | [0.640, 0.914] | **0.000** |
| SelfAware-DiT all-block concat | 0.737 | [0.524, 0.889] | 0.003 |
| generation-activation ablation (block 29) | 0.717 | — | — |
| Jev text oracle (base paper) | 0.788 | [0.575, 0.946] | — |

Evaluation: leave-one-prompt-out CV, grouped by prompt (16 prompts, 64 segments, 18 positive),
prompt-grouped bootstrap (2,000 resamples), 1,000 label permutations.

## Layer sweep (prefill, AUROC per block 0..29)

```
0.773 0.768 0.721 0.717 0.745 0.717 0.720 0.721 0.743 0.745
0.763 0.798 0.791 0.758 0.763 0.748 0.729 0.767 0.766 0.768
0.760 0.762 0.743 0.768 0.756 0.710 0.722 0.744 0.727 0.731
```

Every block is informative (0.71–0.80); the peak is mid-depth (blocks 11–13). This is not a
single lucky layer.

## Protocol (see ../README.md)

- Backbone Causal-rCM Wan2.1-1.3B (`ed3cb14`), T4, 57 frames, seed 17.
- fine `c1-1_step2` (2 steps/frame) vs coarse `c3-3_step2` (2 steps/chunk), matched boundary.
- Label: `coarse_unsafe = 1[ refL1_coarse > refL1_fine ]` vs the `c1-1_step4` reference, in
  **latent space** (the faithful analogue of the base paper's decoded-pixel `refL1`; the T4
  VAE decode dominates wall-clock and OOMs when batched, so the pixel path is disabled here).
- Feature: pooled DiT block activations of the **I2V prefill forward on the boundary**
  (strictly pre-generation). The `gen` variant (activations during generation) is an ablation.

## Files

| File | What |
|---|---|
| `features.npz` | `prefill` [64,30,1536], `gen` [64,30,1536], `labels`, `labels_latent`, `labels_smooth`, `prompt_idx`, `segment` |
| `manifest.json` | protocol, dataset paths, per-segment records, timing |
| `probe_results.json` | layer sweep, best block, CIs, permutation nulls |
| `fig_probe_layer_sweep.png` | AUROC vs block depth |

## Reproduce

```bash
python ideas/selfaware-dit/scripts/train_probe.py --features ideas/selfaware-dit/results/features.npz --out ideas/selfaware-dit/results
python ideas/selfaware-dit/scripts/audit.py --features ideas/selfaware-dit/results/features.npz --results ideas/selfaware-dit/results/probe_results.json
```

## Honest caveats

- Label is latent-space (not decoded-pixel) — a compute-driven deviation from the base paper.
- n = 64 segments / 16 prompts; CIs are wide (as in the base paper's own 80-segment Jev result).
- Single backbone, single seed, T4 only. A scale-up (more prompts, A100, pixel labels, multi-seed)
  is the natural next step.
