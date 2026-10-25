# DDAG E7 — JCD: Jev as a typed critic (T4)

**Verdict: H6 REFUTED.** A typed Jev critic looking at two frames of the coarse segment does **not**
predict granularity safety: `discontinuity` **0.497**, `plausible` **0.514**, combined **0.500** — all at
chance. A decision model shown only pixels cannot independently certify granularity safety.

This is consistent with **JEVQA** (arXiv 2609.24395), where a *pixel-only* Jev quality variant **failed**,
and it contrasts with the base paper's **Jev text** result (0.788) — which supplied a BLIP caption **plus
motion statistics**, i.e. far richer context than two raw frames.

## Results (T4, 16 prompts, 64 matched segments, all 64 Jev calls OK)

| Jev signal | AUROC | 95% CI | perm p |
|---|---|---|---|
| `discontinuity` (Noul) | 0.497 | [0.359, 0.644] | 0.520 |
| `plausible` (Noul) | 0.514 | [0.372, 0.648] | 0.417 |
| `plausible` inverted | 0.486 | [0.352, 0.628] | 0.588 |
| `disc − plausible` | 0.500 | [0.367, 0.635] | 0.511 |

Reference points: chance 0.5 · hand-crafted 0.491 · **Jev text (base paper) 0.788** · intrinsic probe
0.792 · CRV ratio (T4) 0.752.

## What we learned

1. **Framing matters more than the model.** The same model reaches 0.788 with caption + motion context
   and chance with two raw frames. "Show Jev pixels and ask if it looks wrong" is not a usable critic.
2. **Negative but informative for the paper:** JCD is **demoted** from a headline method to a documented
   negative, and the pixel-only caveat from JEVQA is reproduced independently.
3. **Design implication:** if a Jev critic is wanted, it must be given (a) more temporal context (a frame
   *history*, per the docs' "What changed" pattern), (b) explicit motion/caption state, or (c) a **Box**
   question to localise, not a global Noul judgement.

## Protocol

Causal-rCM Wan2.1-1.3B, Kaggle T4, 57 frames, seed 17. Per prompt/segment: matched coarse and fine
generation from a shared boundary; the coarse segment is decoded and its first and last frames are
composed into one PNG within Jev's limits (280–1568 px per side) and sent as a `multipart/form-data`
`assets` part with the header `TypeSafe-Danger-Zone-Images: true`. Label = coarse-unsafe (latent refL1).

## Engineering note (kept)

The Jev **image** API requires `multipart/form-data` with the JSON body as a part named `request` and
each image as its own part named in `assets` (a **map** keyed by a reference string that must appear in
`state`). A JSON body with `images=[...]` returns `400 Invalid request`. First E7 run: 0/64 calls OK;
after the fix: **64/64**.

## Files

| File | What |
|---|---|
| `jcd_t4.npz` | labels, Jev `discontinuity`/`plausible` scores, groups |
| `e7_jcd.json` | AUROCs, CIs, permutation nulls, verdict |
| `../jobs/jcd/jcd.py` | the JCD kernel |

## Reproduce

```bash
python ideas/ddag/scripts/jcd_analysis.py --meas ideas/ddag/results/jcd_t4.npz --out ideas/ddag/results
```
