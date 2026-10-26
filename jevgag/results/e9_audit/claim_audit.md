# E9 — claim-to-artifact audit (18/18 PASS)

| # | claim | result | detail |
|---|---|---|---|
| 1 | E1: T4 coarse_s4 is never cheaper than any fine mode | PASS | 3 T4 s4 pairs, all not cheaper |
| 2 | E1: A100 coarse_s1 beats all three fine modes | PASS | 3 A100 s1 pairs all cheaper |
| 3 | E1: T4 coarse_s1 beats fine_s2 and fine_s4 but not fine_s1 | PASS | operating-point effect present |
| 4 | E2: image-1.14 <= text-1.14 for jev_difficulty | PASS | delta=-0.110 |
| 5 | E2: image-1.14 <= text-1.14 for jev_drift | PASS | delta=-0.108 |
| 6 | E2: image-1.14 <= text-1.14 for jev_motion | PASS | delta=-0.195 |
| 7 | E2: motion deficit is significant (P<0.05) | PASS | P(image>text)=0.006 |
| 8 | E2: text-1.13 ~ text-1.14 (version not the cause) | PASS | 0.788 vs 0.777 |
| 9 | E3: expensive coarse NOT cheaper online on A100 | PASS | fixed_coarse 4.43 > fixed_fine 3.81 |
| 10 | E3: cheap coarse IS cheaper online on A100 | PASS | fixed_coarse 2.55 < fixed_fine 3.95 |
| 11 | E3: adaptive uses far fewer fine decisions with cheap coarse | PASS | cheap 0.21 < expensive 0.92 |
| 12 | E4 (T4): recovery+hysteresis reduces switches | PASS | 0.50 -> 0.38 |
| 13 | E4 (T4): seam exists (switch L1 > interior L1) | PASS | ratio 1.18 |
| 14 | E4 (A100): recovery+hysteresis reduces switches | PASS | 0.50 -> 0.42 |
| 15 | E4 (A100): seam exists (switch L1 > interior L1) | PASS | ratio 1.98 |
| 16 | E5: image-Jev endpoint agrees with CLIP (same-subject) | PASS | rho=+1.000 |
| 17 | E5: image-Jev consistency agrees with CLIP | PASS | rho=+0.943 |
| 18 | E7: confidence-gating is neutral (|delta|<0.005) | PASS | max |delta|=0.0013 |
