# ClosedLoop-Gran — results (P3)

**Verdict: PARTIAL (positive but not decisive at this n).** Post-generation evidence carries real
signal: free evidence from the previous segment alone predicts the next segment's coarse-unsafety at
**AUROC 0.725**, and closing the loop (prior + previous-segment evidence) improves the open loop on the
causal subset from **0.690 → 0.737**. Adding the costly coarse-vs-fine discrepancy reaches **0.799**.
CIs overlap, so the gain is suggestive rather than significant at 48 causal pairs.

## Headline

| test | n | AUROC | 95% CI |
|---|---|---|---|
| A · prior only (open loop, all segments) | 64 | 0.714 | [0.508, 0.878] |
| A2 · prior only (open loop, causal subset) | 48 | 0.690 | [0.432, 0.908] |
| B · previous-segment evidence only (causal) | 48 | 0.725 | [0.528, 0.877] |
| **C · CLOSED LOOP: prior + previous-segment evidence** | 48 | **0.737** | [0.540, 0.904] |
| D · prior + same-segment evidence (non-causal bound) | 64 | 0.734 | [0.548, 0.882] |
| **E · closed loop + costly discrepancy** | 48 | **0.799** | [0.624, 0.938] |

Reference points: chance 0.500 · hand-crafted 0.491 · Jev text 0.788 · idea 03 intrinsic probe 0.798.

## What this means

- **The open loop is not the whole story.** Free, causal post-generation evidence (`seam_coarse`,
  `interior_coarse` from the segment just generated) is by itself a **0.725** signal for the *next*
  segment — nearly as good as the pre-generation prior on the causal subset (0.690).
- **Closing the loop helps** (C > A2, +0.047) and matches the non-causal same-segment bound (D, 0.734),
  i.e. the previous segment's evidence is almost as informative as the current segment's.
- **A costly discrepancy is the strongest signal** (E, 0.799) but it requires running both modes, which
  defeats the compute saving — it is an upper bound, not a deployable policy.

## Signals

| name | definition | free? |
|---|---|---|
| `prior` | intrinsic pre-generation probe score from prefill activations (idea 03) | yes |
| `evidence` | `seam_coarse` = L1(boundary, coarse first frame); `interior_coarse` = within-coarse jitter | **yes** (from the coarse run) |
| `disc` | realized L1(coarse segment, fine segment) | no (needs both modes) |

Evaluation: leave-one-prompt-out CV grouped by prompt, prompt-grouped bootstrap (3,000 resamples).

## Protocol

Causal-rCM Wan2.1-1.3B (`ed3cb14`), Kaggle T4, 57 frames, seed 17, 16 prompts, 64 matched segments,
fine `c1-1_step2` vs coarse `c3-3_step2` from a shared boundary, label = coarse-unsafe (latent `refL1`
vs the `c1-1_step4` reference).

## Files

| File | What |
|---|---|
| `features.npz` | `prefill` [64,30,1536], `labels`, `prompt_idx`, `segment`, `disc`, `seam_coarse/fine`, `interior_coarse/fine`, `ll1_fine/coarse` |
| `manifest.json` | protocol, per-segment records, timing |
| `e1_closedloop.json` | AUROCs, CIs, reference points |
| `fig_closedloop.png` | AUROC by test |

## Reproduce

```bash
python ideas/closedloop-gran/scripts/closedloop.py --features ideas/closedloop-gran/results/features.npz --out ideas/closedloop-gran/results
```

## Honest caveats

- Causal subset is 48 pairs (segments 2–4); CIs are wide and overlap — the closed-loop gain is
  **suggestive, not decisive**.
- Latent-space metrics (compute-driven); single seed; T4 only.
- A deployable closed loop needs a *free* surprise estimate; `disc` is reported only as an upper bound.
- Natural next step: a switch-aware online rollout that uses `evidence_{g-1}` to gate the switch, with a
  fidelity/perceptual endpoint.
