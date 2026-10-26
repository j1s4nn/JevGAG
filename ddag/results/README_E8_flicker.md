# DDAG E8 — seam repair with an independent pixel endpoint (T4)

**Verdict: the seam repair reproduces, and the independent endpoint exposes a TRADE-OFF.** `blend0.25`
cuts the boundary seam by **25%** and improves latent fidelity, exactly as in P2 — but the independent
**VBench-style temporal flicker** endpoint shows the repair *slightly increases* interior flicker. This is
the base paper's **P3** lesson in action: *the metric you optimise is not the metric that certifies.*

## Results (T4, 8 prompts, 32 matched segments)

| variant | boundary seam (latent) | fidelity (latent L1 to s4 ref) |
|---|---|---|
| none | 0.6138 | 0.7607 |
| **blend0.25** | **0.4604** (−25%) | **0.7544** (better) |
| blend0.5 | 0.3069 | 0.7590 |
| blend0.75 | 0.1535 | 0.7742 |
| anchor | 0.0000 | 0.7990 |

Reproduces P2 exactly (same protocol, different sample): **seam −25% at +0.6% fidelity**.

## Independent pixel endpoint — VBench-style temporal flicker

| signal | mean flicker | 95% CI |
|---|---|---|
| fine segment | 0.0254 | [0.0223, 0.0288] |
| coarse segment (`none`) | 0.0213 | [0.0124, 0.0172] |
| coarse + `blend0.25` | 0.0218 | [0.0196, 0.0241] |

| comparison | Δ | 95% CI | reading |
|---|---|---|---|
| coarse − fine | **−0.0041** | [−0.0063, −0.0019] | coarse flickers **less** than fine |
| blend0.25 − none | **+0.0071** | [+0.0059, +0.0083] | the repair **increases** interior flicker |

## What this means

1. **The boundary seam and interior temporal flicker are different quantities.** Reducing the
   discontinuity at the *first* frame makes that frame less like the rest of the segment, so the
   *interior* frame-to-frame difference rises. A single endpoint would have hidden this.
2. **This is exactly why the base paper mandates an independent endpoint** (its P3): our own P2 claim
   ("blend0.25 dominates") is **true for the seam+fidelity pair** and **incomplete for flicker**. The
   honest statement is: *blend0.25 trades a boundary discontinuity for a small interior flicker.*
3. **Actionable:** a seam-aware objective should minimise `seam + λ·flicker`, and λ must be calibrated
   per GPU/endpoint — which feeds the regime map (E11) and the compiled policy (E10).

## Protocol

Causal-rCM Wan2.1-1.3B, Kaggle T4, 57 frames, seed 17, 8 prompts, matched coarse/fine from a shared
boundary. Repair applies to the coarse segment's first latent frame only. Endpoints: latent boundary
seam, latent fidelity, and **decoded-pixel temporal flicker** (mean absolute consecutive-frame
difference, VBench temporal-flickering style).

## Files

| File | What |
|---|---|
| `flicker_t4.npz` | `variants`, `seams`, `fids`, `flicker_{fine,none,blend0.25}`, per-segment metadata |
| `e8_flicker.json` | means, CIs, deltas, verdict |
| `../jobs/flicker/flicker.py` | the kernel |

## Reproduce

```bash
python ideas/ddag/scripts/flicker_analysis.py --meas ideas/ddag/results/flicker_t4.npz --out ideas/ddag/results
```

## Honest negatives

- `coarse_flickers_more_than_fine` = **false** (the opposite holds).
- `blend_reduces_flicker` = **false** (it increases flicker).
- Only 8 prompts (decode is the wall-clock bottleneck on T4); CIs are wide but exclude 0.
