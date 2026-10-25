# E6 — matched T4 boundary (derived)

- T4 coarse `c3-3 s1` vs fine `c1-1 s1`: cheaper = **False** (DiT saved -4.57 s)
- T4 coarse `c3-3 s1` vs fine `c1-1 s2`: cheaper = **True** (DiT saved +7.01 s)
- T4 coarse `c3-3 s1` vs fine `c1-1 s4`: cheaper = **True** (DiT saved +29.70 s)

## Verdict

T4 boundary is NEGATIVE for fine_c1_1_s1: coarse c3-3 s1 saves -4.57 s (i.e. is slower), so adaptive routing cannot save wall-clock time regardless of the signal. The T4 negative is an operating-point boundary, not a signal failure. Coarse only wins on T4 once fine uses >=2 steps.

Confirmed by the prior T4 online arm (expensive coarse), which was negative.

**Note:** a fresh T4 online arm with the CHEAP coarse and fine_s2 is the one configuration that would make T4 positive; it is a candidate Kaggle run, not required for the boundary claim.
