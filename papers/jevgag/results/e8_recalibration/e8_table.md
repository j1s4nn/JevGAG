# E8 (Kaggle T4) — recalibration (n=8 prompts)

| policy | mean refL1 | fine frac | switches | DiT s |
|---|---|---|---|---|
| fixed_coarse | 0.3186 | 0.00 | 0.00 | 38.82 |
| fixed_fine | 0.2998 | 1.00 | 0.00 | 19.87 |
| adaptive_warp | 0.2949 | 0.75 | 0.00 | 24.05 |
| recalibrate | 0.3151 | 0.00 | 0.00 | 36.78 |

## Verdict

- recalibrate minus adaptive_warp: refL1 +0.0202, DiT +12.73 s.
- **Recalibration does NOT help** (worse refL1 at higher cost).
