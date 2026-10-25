# E4 — recovery+hysteresis and the switch seam (matched re-analysis)

## T4

| controller | switches | fine frac | AUH refL1 | seam (switch/int) |
|---|---|---|---|---|
| threshold_only | 0.50 | 0.50 | 0.3133 | 1.30 |
| hysteresis_only | 0.38 | 0.53 | 0.3116 | 1.18 |
| recovery_only | 0.38 | 0.53 | 0.3116 | 1.18 |
| full | 0.38 | 0.53 | 0.3116 | 1.18 |
| threshold_only_noisy | 1.12 | 0.53 | 0.3159 | 2.17 |
| full_noisy | 0.62 | 0.66 | 0.3150 | 1.53 |

## A100

| controller | switches | fine frac | AUH refL1 | seam (switch/int) |
|---|---|---|---|---|
| threshold_only | 0.50 | 0.50 | 0.2767 | 2.20 |
| hysteresis_only | 0.42 | 0.52 | 0.2772 | 1.98 |
| recovery_only | 0.42 | 0.52 | 0.2772 | 1.98 |
| full | 0.42 | 0.52 | 0.2772 | 1.98 |
| threshold_only_noisy | 0.92 | 0.48 | 0.2741 | 2.43 |
| full_noisy | 0.50 | 0.56 | 0.2718 | 1.75 |

## Verdict

- Recovery+hysteresis **reduces switches** vs threshold-only on both GPUs (T4 0.50->0.38; A100 0.50->0.42).
- Regret is **not worsened** (T4 slightly better; A100 ~neutral) -> RQ3 supported.
- A **seam** exists: switch-adjacent L1 > interior L1 on both GPUs (ratio > 1).
- Under noise the switch reduction is largest (T4 1.13->0.63; A100 0.92->0.50).
