# STATUS — JevGAG research repository

_Last updated: 2026-10-25._

## Papers

| Paper | Status | Where |
|---|---|---|
| **JevGAG** — Jev-Grounded Adaptive Temporal Granularity for Causal Video Generation | **complete, audited** (18/18) | [`papers/jevgag/`](papers/jevgag/) |
| **DDAG** — Deadline- and Drift-Aware Adaptive Temporal Granularity for Interactive Causal Video World Models | **in progress** | [`ideas/`](ideas/) |

## Current phase: P2 of DDAG (`plan_Finalv0.1`)

| Phase | Content | Status |
|---|---|---|
| P1 | intrinsic probe, seam repair, closed-loop, cost-model finding | ✅ done |
| **P2** | E1 cost map · E2 CRV verification · E3 drift study · E7 Jev critic · E8 seam/flicker | ⏳ in progress |
| P3 | E4 propagation model · E5 DALG lookahead · E6 DCG deadline | ⏳ |
| P4 | E9 long-horizon · E11 regime map · E12 scale (48×3) | ⏳ |
| P5 | E10 SDGP compiled policy · E13 human study | ⏳ |
| P6 | E14 audit · writing · release | ⏳ |

## Achievements

| # | Achievement | Evidence |
|---|---|---|
| A1 | Intrinsic pre-generation probe **AUROC 0.798** (no external oracle) | `ideas/selfaware-dit` |
| A2 | Training-free seam repair: **−25% seam, +1.1% fidelity** (both CIs exclude 0) | `ideas/seamcontrol` |
| A3 | Free post-generation evidence **0.725**; closed loop **0.690 → 0.737** | `ideas/closedloop-gran` |
| A4 | **Cost model, not the signal, is the binding constraint** (4.8% error at ρ=0.672 vs break-even 0.667) | `ideas/verify-gran` |
| A5 | Reusable substrate: matched capture + DiT hooks + audits + public datasets | whole repo |

## Gaps (being addressed by DDAG)

G1 drift / long-horizon · G2 lookahead controller · G3 deadline scheduling · G4 verification bound ·
G5 Jev critic · G6 A100 results · G7 compiled policy · G8 single seed / 16 prompts.

## Public datasets

See [`DATA.md`](DATA.md). Four public Kaggle datasets; every headline number is audited.

## GPUs

Tesla T4 (Kaggle, free) and NVIDIA A100-SXM4-80GB (Colab). Budget for DDAG: ~35 h total.

## Links

- Site: https://j1s4nn.github.io/JevGAG/
- Repo: https://github.com/j1s4nn/JevGAG
- Base paper (parent, untouched repo): `10.5281/zenodo.23395414`
