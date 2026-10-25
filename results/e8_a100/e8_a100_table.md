# E8 (A100) — stability-gated recalibration (n=12 prompts)

| controller | mean refL1 | fine frac | switches | DiT s | total s | peak VRAM GB |
|---|---|---|---|---|---|---|
| threshold_only | 0.2767 | 0.50 | 0.50 | 3.27 | 29.5 | 4.30 |
| full | 0.2772 | 0.52 | 0.42 | 3.31 | 29.8 | 4.30 |
| recalibrate | 0.2884 | 0.50 | 0.50 | 5.35 | 31.9 | 4.30 |

## Verdict

- recalibrate minus full: refL1 +0.0112, DiT +2.04 s.
- **Recalibration does NOT help** (higher/equal refL1 at higher cost).
