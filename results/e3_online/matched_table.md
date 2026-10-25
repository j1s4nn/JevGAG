# E3 — matched A100 online factorial (same GPU, same 12 prompts)

| policy | EXPENSIVE c3-3 s4: DiT (s) | refL1 | fine% | CHEAP c3-3 s1: DiT (s) | refL1 | fine% |
|---|---|---|---|---|---|---|
| fixed_fine | 3.81 | 0.2812 | 1.00 | 3.95 | 0.2832 | 1.00 |
| fixed_coarse | 4.43 | 0.2874 | 0.00 | 2.55 | 0.2728 | 0.00 |
| fixed_hold | 0.76 | 0.2824 | 0.00 | 0.77 | 0.2674 | 0.00 |
| random50 | 4.12 | 0.2868 | 0.50 | 3.24 | 0.2789 | 0.50 |
| alternating50 | 4.12 | 0.2868 | 0.50 | 3.25 | 0.2784 | 0.50 |
| adaptive_warp | 3.87 | 0.2820 | 0.92 | 2.85 | 0.2733 | 0.21 |

## Operating-point effect (matched)

- EXPENSIVE c3-3 s4: coarse cheaper online? **False** (fixed_coarse 4.43s vs fixed_fine 3.81s)
- CHEAP c3-3 s1: coarse cheaper online? **True** (fixed_coarse 2.55s vs fixed_fine 3.95s)

**Verdict:** on the same A100 and prompts, the adaptive result flips with the coarse operating point alone — cheap coarse enables adaptive savings (fine% 0.21, lower refL1 than fixed_fine); expensive coarse does not (adaptive forced to 92% fine). The old paper's T4-negative/A100-positive comparison was confounded by using expensive coarse on T4 and cheap coarse on A100.
