# DDAG E5b / E5c — the intermediate band, and the endpoint problem

## E5b — a mode pair INSIDE the band (T4, fine `s2` vs coarse **`s1`**)

| quantity | value |
|---|---|
| segments / positives | 64 / 17 |
| probe (direct AUROC) | **0.785** |
| DiT fine (s) | 4.686 |
| DiT coarse (s) | 3.411 |
| **cost ratio coarse/fine** | **0.728 → coarse 27% CHEAPER** |
| refL1 fine / coarse (latent) | 0.806 / **0.755** (coarse closer to the reference) |

**This is the intermediate band**: coarse is genuinely cheaper *and* (by the latent endpoint) not worse.
So even here the best policy is **always-coarse**, saving **27.2%** — and the probe is not needed.

## E5c — the decisive test with a DECODED-PIXEL endpoint: FAILED

The pixel path (`DECODE_PIXELS=True`, never exercised before) crashed:
`ValueError: operands could not be broadcast together with shapes (0,9,480,832,3) (0,57,480,832,3)`.
The decoded tensor comes back with a zero batch dimension and the segment/reference frame counts
disagree (9 vs 57). **Kept as-is (not deleted).** One more debug cycle would settle the quality question.

## The endpoint problem — the real finding

Across every endpoint we have measured, **the coarse mode is never worse than fine**:

| endpoint | T4 result |
|---|---|
| latent `refL1` to the `s4` reference (E2/E5) | coarse **closer** (0.755 vs 0.806) |
| decoded-pixel **temporal flicker** (E8) | coarse flickers **less** (−0.0041, CI excludes 0) |
| decoded-pixel `refL1` | **not yet measured** (E5c failed) |

So our claim "adaptive granularity is unnecessary at both extremes" currently rests on endpoints that
**do not detect coarse degradation**. The base paper's *decoded-pixel* metrics did find coarse worse on
A100 — so the regime map is **endpoint-dependent**, and the pixel endpoint is the missing piece.

## Honest status of the E5 line

| claim | status |
|---|---|
| coarse is 27% cheaper than fine with 1-step coarse on T4 | ✅ measured |
| the intrinsic probe works on T4 (0.785–0.798) | ✅ measured |
| routing beats fixed fine on T4 | ❌ **no** — coarse dominates by the latent endpoint |
| the quality axis is trustworthy | ⚠️ **no** — the latent endpoint does not penalise coarse |
| regime map | ✅ T4 → coarse/fine depending on the step gap; A100 → coarse |

**Bottom line:** with a *cheaper-and-not-worse* coarse mode, the optimal policy is trivially
"always coarse" and adaptivity adds nothing. Whether adaptivity *would* pay depends entirely on the
**pixel endpoint**, which is the one measurement still missing.

## Files

| File | What |
|---|---|
| `features_band_t4.npz` | E5b capture (per-segment cost + latent quality + probe) |
| `band_t4/e5_online.json` | E5b policies and verdict |
| `../jobs/band/band.py`, `../jobs/pxq/pxq.py` | E5b and E5c kernels |
