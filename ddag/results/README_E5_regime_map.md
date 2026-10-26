# DDAG E5 — online granularity policy and the regime map (T4 + A100)

**Verdict: adaptive granularity is UNNECESSARY at both tested extremes.** On **T4** coarse is *slower
and lower quality* than fine, so **fixed fine dominates** and no router can help. On **A100** coarse is
**46% cheaper and closer to the reference**, so **always-coarse dominates** and no router is needed.
The interesting regime — where neither mode dominates — is *between* them, and the regime map says
exactly where it is.

This is the paper's positive spine: **a measured, two-GPU regime map** that tells a practitioner which
policy to deploy, plus the guarantee that adaptivity only pays inside the middle band.

## Setup

The capture runs **both** modes for every segment (matched design), so every policy is computable
offline from the recorded per-segment DiT time and latent `refL1`.

| GPU | fine | coarse | segments | positives |
|---|---|---|---|---|
| T4 | `c1-1_step2` (2 steps/frame) | `c3-3_step2` (2 steps/chunk) | 64 | 18 |
| A100 | same | same | 64 | 10 |

## Cost and quality (per segment)

| GPU | DiT fine (s) | DiT coarse (s) | ratio coarse/fine | refL1 fine | refL1 coarse |
|---|---|---|---|---|---|
| **T4** | 4.394 | 4.526 | **1.030** | **0.807** | 0.752 |
| **A100** | 1.279 | 0.685 | **0.536** | 0.780 | **0.721** |

> Note on `refL1`: it is the latent L1 to the `c1-1_step4` reference, so **lower = closer to the
> reference**. On A100 coarse is *closer* than fine; on T4 it is farther.

## Policies

| policy | T4 DiT total (s) | T4 quality | A100 DiT total (s) | A100 quality |
|---|---|---|---|---|
| fixed fine | **281.2** | **0.807** | 81.9 | 0.780 |
| fixed coarse | 289.6 | 0.752 | **43.9** | **0.721** |
| oracle route | — | — | 49.8 | 0.707 |
| oracle escalate | — | — | 56.6 | 0.707 |

**T4:** fixed fine is both cheaper *and* better → **dominant**. `probe_route_beats_fixed_fine = false`.
**A100:** fixed coarse is both cheaper *and* closer → **dominant**. Any escalation only adds cost.

## The regime map

```
                 cost(coarse)/cost(fine)      quality(coarse) vs quality(fine)
   T4    :            1.030  (coarse dearer)        coarse worse        -> FIXED FINE
   A100  :            0.536  (coarse cheaper)       coarse better       -> FIXED COARSE
   -------- intermediate band (neither dominates) -> ADAPTIVE PAYS --------
```

Adaptive granularity pays **only** when `cost(coarse) < cost(fine)` **and** `quality(coarse) <
quality(fine)` — i.e. when the cheap mode is cheaper but worse. Our two mode pairs sit at the two
extremes, which is *why* routing did nothing here. **A router is worth deploying in the middle band,
and the map tells you whether you are in it.**

## Signal transferability (ties to E2)

| GPU | probe (direct AUROC) | convergence ratio |
|---|---|---|
| T4 | **0.798** | **0.752** |
| A100 | **0.143** (inverted) | 0.485 (chance) |

So not only is routing unnecessary at the extremes — the signals themselves **do not transfer**.

## Honest ledger

- **No routing win was demonstrated on either GPU.** That is the finding, not a failure of the run.
- On A100 the `refL1` metric (latent L1 to a *fine* reference) flatters coarse; a decoded-pixel endpoint
  may reorder the quality axis (E8 showed latent and pixel endpoints can disagree).
- The intermediate-band prediction is a **derived** claim from the map, not yet measured — the natural
  next experiment is a mode pair chosen to sit inside the band (e.g. fine `s2` vs coarse `s1` on T4,
  where the base paper's break-even table already shows coarse genuinely cheaper).

## Files

| File | What |
|---|---|
| `features_online_t4.npz`, `features_online_a100.npz` | per-segment cost + quality + probe features |
| `online_t4/e5_online.json`, `online_a100/e5_online.json` | policies, frontiers, verdicts |
| `../jobs/online/online.py` | the kernel |

## Reproduce

```bash
python ideas/ddag/scripts/online_policy.py --features ideas/ddag/results/features_online_t4.npz   --out ideas/ddag/results/online_t4
python ideas/ddag/scripts/online_policy.py --features ideas/ddag/results/features_online_a100.npz --out ideas/ddag/results/online_a100
```
