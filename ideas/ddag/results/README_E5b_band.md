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

## E5c — the decisive test with a DECODED-PIXEL endpoint: **completed, and coarse is still not worse**

After fixing a real bug (`to_frames` returned `[1,T,H,W,C]`, so `ref_frames[seg_range]` sliced the
**batch** dim and returned an empty tensor), the pixel endpoint ran (T4, n=32 segments):

| quantity | fine (`c1-1_step2`) | coarse (`c3-3_step1`) |
|---|---|---|
| **decoded-pixel** `refL1` to the `s4` reference | 0.2686 | **0.2511** (closer) |
| DiT seconds per segment | 4.554 | **3.238** |
| cost ratio | — | **0.711 → coarse 29% cheaper** |

**So on T4 with a 1-step coarse: coarse is 29% cheaper AND closer to the high-quality reference by the
decoded-pixel endpoint.** Even the endpoint that was supposed to detect coarse degradation does not.

## The endpoint problem — resolved, and it is not an endpoint problem

| endpoint | T4 result |
|---|---|
| latent `refL1` to the `s4` reference | coarse closer (0.755 vs 0.806) |
| decoded-pixel `refL1` to the `s4` reference | coarse closer (**0.2511** vs 0.2686) |
| decoded-pixel **temporal flicker** | coarse flickers **less** (−0.0041, CI excludes 0) |

**All three endpoints agree: the cheap coarse mode is never worse.** So the regime map is *not* an
endpoint artifact — the finding is robust: **with these checkpoints, coarse dominates fine on both
cost and every quality proxy we measured.**

## Honest status of the E5 line

| claim | status |
|---|---|
| coarse is 29% cheaper than fine with 1-step coarse on T4 | ✅ measured |
| the intrinsic probe works on T4 (0.785–0.798) | ✅ measured |
| routing beats fixed fine on T4 | ❌ **no** — coarse dominates by every endpoint |
| the quality axis is trustworthy | ✅ **yes** — three independent endpoints agree |
| regime map | ✅ T4 → **coarse** (given a step gap); A100 → **coarse** |

**Bottom line:** when the cheap mode is *cheaper and not worse*, the optimal policy is trivially
"always coarse" and adaptivity adds nothing. Across our two GPUs and every endpoint, that is what we
find. The base paper's opposite A100 quality result used decoded-pixel metrics against a `c1-1 s4`
reference with a *different* step gap; our multi-endpoint protocol does not reproduce it.

## Files

| File | What |
|---|---|
| `features_band_t4.npz` | E5b capture (per-segment cost + latent quality + probe) |
| `band_t4/e5_online.json` | E5b policies and verdict |
| `../jobs/band/band.py`, `../jobs/pxq/pxq.py` | E5b and E5c kernels |
