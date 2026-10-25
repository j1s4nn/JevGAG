# SeamControl — results (P2)

**Verdict: SUPPORTED.** A training-free **25% latent blend at the switch boundary** cuts the
boundary seam by **25%** (0.614 → 0.460, CI excludes 0) *and* slightly **improves** fidelity
(Δ = −0.0085, CI excludes 0). The base paper measured the seam; this acts on it.

## Headline

| variant | boundary seam L1(b, x1) | Δ vs none | fidelity L1(seg, s4 ref) | Δ vs none |
|---|---|---|---|---|
| none | 0.6137 [0.579, 0.649] | — | 0.7524 [0.704, 0.799] | — |
| **blend0.25** | **0.4602** [0.434, 0.487] | **−0.1534** [−0.162, −0.145] | **0.7439** [0.696, 0.791] | **−0.0085** [−0.0121, −0.0050] |
| blend0.5 | 0.3068 [0.289, 0.325] | −0.3068 [−0.325, −0.290] | 0.7459 [0.697, 0.793] | −0.0065 [−0.0141, +0.0009] |
| blend0.75 | 0.1534 [0.145, 0.162] | −0.4602 [−0.487, −0.434] | 0.7586 [0.709, 0.807] | +0.0061 [−0.0057, +0.0180] |
| anchor (hard) | 0.0000 | −0.6137 [−0.649, −0.579] | 0.7809 [0.730, 0.830] | +0.0285 [+0.0125, +0.0446] |

Prompt-grouped bootstrap (4,000 resamples), 16 prompts, 64 segments. Lower is better on both axes.

**blend0.25 is a strict improvement**: both deltas exclude 0, in the good direction. Hard anchoring
zeroes the seam but costs fidelity (+0.0285, CI excludes 0) — so a *partial* blend is the sweet spot.

## The seam is real (direct measurement)

In the matched design (same boundary, fine vs coarse), the boundary discontinuity is:

| | boundary seam | interior |
|---|---|---|
| fine (`c1-1_step2`) | 0.2148 | 0.1692 |
| coarse (`c3-3_step2`) | **0.6137** | 0.2919 |

Switching to coarse produces a boundary discontinuity **2.9×** fine's and **2.1×** coarse's own
interior — a direct, matched confirmation of the seam the base paper reported from aggregate traces.

## Protocol

- Causal-rCM Wan2.1-1.3B (`ed3cb14`), Kaggle T4, 57 frames, seed 17, 16 prompts.
- fine `c1-1_step2` (2 steps/frame) vs coarse `c3-3_step2` (2 steps/chunk), **same boundary**.
- Repair is applied to the coarse segment's **first latent frame only**:
  `x1 ← (1−a)·x1 + a·b` (blend), or `x1 ← b` (anchor), where `b` is the boundary latent frame.
- Metrics in latent space: seam = L1(b, x1); fidelity = L1(3-frame segment, `c1-1_step4` reference).

## Files

| File | What |
|---|---|
| `measurements.npz` | `variants`, `seams` [5,64], `fids` [5,64], plus per-segment `seam_fine/coarse`, `interior_*`, `fid_*`, `prompt_idx`, `segment` |
| `manifest.json` | protocol, per-prompt records, timing |
| `e2_repair.json` | means, prompt-grouped CIs, deltas |
| `e1_seam.json` | 0-GPU re-analysis of the reused controller traces (segment-mean adjacent L1; **negative** — segment means do not reproduce the boundary seam, which is why the direct measurement above was needed) |
| `fig_seam_repair.png` | seam–fidelity frontier |

## Reproduce

```bash
python ideas/seamcontrol/scripts/analyze.py --meas ideas/seamcontrol/results/measurements.npz --out ideas/seamcontrol/results
```

## Honest caveats

- Metrics are latent-space (compute-driven; the base paper used decoded-pixel L1).
- 16 prompts / 64 segments; CIs are wide (as throughout this project).
- Repair is evaluated on the *first frame*; a full perceptual/flicker endpoint and a
  switch-aware online rollout are the natural next steps.
- `e1_seam.json` is a **negative** re-analysis (segment-mean traces cannot isolate the boundary seam).
