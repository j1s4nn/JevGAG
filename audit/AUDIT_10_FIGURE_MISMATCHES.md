# AUDIT_10 — Figure and table mismatches (with source evidence)

Method: every figure/table value was compared with its source data file. Values were **not** inferred
from pixels; the `.npz` / `.json` sources were read directly.

---

## MM-1 (P0) — Table 1 (seam repair): two flicker means are wrong

| row | manuscript Table 1 | source `ddag/results/e8_flicker.json` | verdict |
|---|---|---|---|
| `none` | 0.0213 | **`flicker.none.mean = 0.01477`** | **WRONG** |
| `blend 0.25` | 0.0218 | `flicker.blend0.25.mean = 0.02184` | ✅ correct |
| `fine (no switch)` | 0.0254 | **`flicker.fine.mean = 0.01890`** | **WRONG** |

**Evidence (verbatim from the JSON):**
```
flicker.fine.mean      = 0.01889913246850483   ci [0.01680, 0.02111]
flicker.none.mean      = 0.014768498454941437  ci [0.01244, 0.01721]
flicker.blend0.25.mean = 0.021842832793481648  ci [0.01962, 0.02412]
flicker_delta_none_minus_fine  = -0.00413  ci [-0.00628, -0.00188]
flicker_delta_blend_minus_none = +0.00707  ci [+0.00589, +0.00834]
```

**Why the deltas are still right.** The deltas are computed from the arrays, so `−0.0041` and `+0.0071`
are correct. Only the *printed absolute means* for `none` and `fine` are wrong.

**Origin.** The same wrong values are in `ddag/results/README_E8_flicker.md` (and therefore in the
Kaggle dataset `ajjisan/ddag-experiments`). The error was introduced when the README table was written
by hand rather than copied from the analysis output.

**Impact.** Does **not** change the qualitative conclusion (the repair raises flicker above both the
no-repair coarse value **and** the fine value: 0.0218 > 0.0189 > 0.0148). It **does** change the
sentence "coarse flickers *less* than fine": still true (0.0148 < 0.0189), but the margin is smaller than
the printed numbers suggest, and the repair makes the flicker **the highest of the three**, which the
current numbers hide (0.0218 vs 0.0254 printed would suggest the repair is still below fine).

**Action.** Correct Table 1 and the dataset README; version-bump `ajjisan/ddag-experiments`.

---

## MM-2 (P0) — Fig. 3 (signal study): the 0.752 bar is not the direct AUROC of Eq. (2)

**Fig. 3** plots "convergence ratio 0.752" and "probe + ratio 0.842".

**Source.** `ddag/results/e2b_residual_variants.json` contains `step_last_over_first: 0.752`, but that
value was produced by `crv_variants.py` via a **single-feature logistic** (`oof(...)`), which re-learns
the sign. The **direct** AUROC of the ratio `v_S/v_1` is **0.2403**; the direct AUROC of its reciprocal
is 0.7597; the direct AUROC of the **first-step magnitude** is **0.7101** (AUDIT_07, M-1).

**Additionally:** the 0.842 value is **not present in any stored result file**. The stored
`e2_crv.json` reports `probe+r_conv = 0.7621` (a different residual). Re-deriving `probe + ratio`
independently gives **0.8418** — so the number is reproducible, but its provenance is a temporary
script, not an artifact.

**Action.** Correct Eq. (2) / §4.3 / §6.2 / the abstract; regenerate Fig. 3; store the combined-model
result in a result file; add CI whiskers (CIs exist for the ratio but not for the combination).

---

## MM-3 (P1) — Fig. 2 (regime map): axis direction is easy to misread

The y-axis is "quality gap `q_coarse − q_fine`" with `q` defined as *lower-is-better*. A **negative** gap
therefore means the coarse mode is **better**. The shaded regions are labelled correctly
("ADAPTIVE PAYS (cheaper and worse)" above the axis; "FIXED COARSE (cheaper and no worse)" below), and
the point markers sit below the axis — consistent. But a reader who assumes "higher = better" will
invert the map. **Action:** add "(negative = coarse closer to the reference)" to the y-axis label, and
state the sign convention in the caption.

---

## MM-4 (P1) — Fig. 4 (drift): n asymmetry is only in the panel titles

Panel titles read `Tesla T4 (n = 32)` and `A100-80GB (n = 8)`. The prose of §6.4 says only "on T4 … on
A100 …". A reader of the text alone cannot know that the A100 curve rests on **8** prompts. **Action:**
state both n values in the prose and in the caption; add a bootstrap CI band for the propagation chain
(the CIs exist in `t4_n32/e3_drift.json`).

---

## MM-5 (P1) — Table 2 (policies): no uncertainty, no n, and "dominant" is unqualified

The values match the source exactly, but the table gives no CIs (available in `e5_online.json` via the
frontier and in the per-prompt data), no prompt count, and asserts a "verdict" of *dominant* without an
objective. **Action:** add Δ vs fixed-fine with CIs, add "16 prompts / 64 segments", and define
"dominant" as "optimal for Eq. (X)".

---

## MM-6 (P2) — Fig. 5 (seam/flicker) exists but is not used

`assets/figures/fig_seam.pdf` is a two-panel figure (repair frontier + flicker endpoint) that is
**not referenced** by the manuscript, which uses Table 1 instead. The figure conveys the seam↔flicker
trade-off more clearly than the table. **Action:** decide figure vs table; if the table is kept, either
use the figure as a supplementary panel or remove it from `assets/figures/`.

---

## MM-7 (P2) — Fig. 6 (qualitative): device and segment definition missing from the caption

The caption does not state that the segments are T4 outputs, nor that a "segment" is three latent frames
(12 pixel frames) following a shared three-frame coarse prefix. **Action:** add both.

---

## MM-8 (P2) — report figures are unused, including the strongest positive

`jevgag/figures/fig_e5_endpoint.png` documents the report's **image-Jev endpoint agreeing with CLIP at
ρ = +1.000** — the report's strongest positive and direct evidence that an *independent endpoint is
achievable*, which the new paper's P3 precondition asserts. The new paper cites no report figure.
**Action:** cite the report's endpoint result (text, not necessarily the figure) in §3/§5.

---

## Summary

| ID | Priority | Artifact | Mismatch |
|---|---|---|---|
| MM-1 | **P0** | Table 1 flicker means | 0.0213→0.0148, 0.0254→0.0189 |
| MM-2 | **P0** | Fig. 3 + §4.3/§6.2/abstract | 0.752 is sign-corrected (direct 0.2403); 0.842 unstored |
| MM-3 | P1 | Fig. 2 axis sign | ambiguous |
| MM-4 | P1 | Fig. 4 / §6.4 | n = 32 vs n = 8 not in prose |
| MM-5 | P1 | Table 2 | no CI, no n, unqualified "dominant" |
| MM-6 | P2 | Fig. 5 | unused figure |
| MM-7 | P2 | Fig. 6 caption | device/segment missing |
| MM-8 | P2 | report figures | strongest positive uncited |
