# Idea 03 — SelfAware-DiT: intrinsic granularity-safety probes

**Question.** The base paper shows that hand-crafted *context* signals cannot predict
coarse-unsafe segments (AUROC **0.491**, chance), while an **external** System One oracle
(Jev 1.14, text) can (**0.788**) — but only off-device, caption-mediated, proprietary and
versioned. Can the generator predict its **own** granularity safety from its **own**
internal state, with no external oracle?

**Hypothesis.** The DiT's internal activations at the decision point (the I2V prefill
forward on the segment boundary) carry a usable coarse-unsafe signal.

## Protocol

- Backbone: Causal-rCM Wan2.1-T2V-1.3B, upstream `ed3cb14`.
- Modes: `fine = c1-1_step2` (2 sampling steps/frame), `coarse = c3-3_step2`
  (2 sampling steps/chunk), `ref = c1-1_step4` (independent reference rollout).
- Horizon 57 frames (15 latent frames), seed 17, shared 3-frame coarse prefix, 4 segments.
- **Matched design:** both modes at segment *g* start from the **same** boundary (the fine
  chain), so the only difference is the generator.
- **Label** (identical to the base paper): `coarse_unsafe = 1[ refL1_coarse > refL1_fine ]`
  where `refL1` is the mean decoded-pixel L1 against the `c1-1 s4` reference over the
  segment. A no-reference proxy (`label_smooth`: within-segment adjacent-frame jitter) is
  also recorded.

## Features

| Feature | How | Legal pre-generation? |
|---|---|---|
| `prefill` | pooled DiT block activations of the **prefill forward on the boundary** (call 0 of the I2V rollout) | **yes** |
| `gen` | pooled activations averaged over the segment's **generation** forwards | no (ablation / upper bound) |

## Evaluation

`scripts/train_probe.py` (0 GPU): logistic probe per DiT block (layer sweep), all-block
concat, **leave-one-prompt-out** CV (grouped by prompt), prompt-grouped bootstrap 95% CIs,
and a label-permutation null. Reference points: chance 0.5, hand-crafted 0.491, Jev text 0.788.

## Run

```bash
# 0-GPU probe training (after the capture run)
python ideas/selfaware-dit/scripts/train_probe.py \
  --features <capture_out>/results/selfaware_dit/features.npz \
  --out ideas/selfaware-dit/results

# GPU capture (Kaggle T4); LIMIT is rewritten in a temp copy
python ideas/selfaware-dit/scripts/kaggle_run.py --job capture --limit 1 \
  --slug ajjisan/jevgag-selfaware-capture-smoke --out <dir>
python ideas/selfaware-dit/scripts/kaggle_run.py --job capture --limit 16 \
  --slug ajjisan/jevgag-selfaware-capture --out <dir>
```

## Outputs

- `results/selfaware_dit/features.npz`, `manifest.json` (from the kernel)
- `results/probe_results.json`, `results/fig_layer_sweep.png` (0 GPU)
- public Kaggle dataset `ajjisan/jevgag-selfaware-dit`

## Verdict rule (pre-registered)

- **Supported** if the pre-generation probe AUROC ≥ 0.70 (approaching Jev text 0.788) with a
  permutation p < 0.05.
- **Partial** if 0.55 ≤ AUROC < 0.70.
- **Negative** if AUROC < 0.55 — then the external oracle is *necessary*, which is itself a
  publishable result.
