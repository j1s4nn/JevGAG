# JevGAG — report (Part 1 of 2)

**Jev-Grounded Adaptive Temporal Granularity for Causal Video Generation** — the *report*: a measured
audit of the three preconditions that any adaptive-granularity scheme silently assumes.

This part is the **empirical foundation**; the method built on it lives in [`../ddag/`](../ddag/).
The parent study is the preprint **"Jev-Routed Temporal Granularity with Recovery and Hysteresis for
Causal Video Generation"**, https://doi.org/10.5281/zenodo.23395414.

---

## What the report establishes

Causal video generators emit one temporal chunk at a time, so spending expensive per-frame ("fine")
compute only where a segment is hard looks attractive. The report shows this rests on **three
preconditions**, each of which is a *regime property* rather than a universal fact:

| Precondition | Question | Report's finding |
|---|---|---|
| **P1 — cost** | is the cheap mode genuinely faster on *this* GPU at *this* schedule? | hardware-conditional: the same mode pair sits on opposite sides of the break-even inequality on T4 vs A100 |
| **P2 — signal** | is there a **pre-generation** quantity that predicts "coarse-unsafe"? | the cheap hand-crafted context signal fails (AUROC 0.491); an off-device **Jev** text oracle succeeds (AUROC 0.788) |
| **P3 — endpoint** | is quality certified by a measure **independent** of the modes compared? | the internal reference flips direction between GPUs; independent endpoints are mandatory |

**Mechanism.** A two-threshold controller with a hysteresis band and a forced-recovery window stabilises
switching, and a *seam* — a measurable discontinuity at the switch boundary — is identified as the cost
of switching.

## Headline results

| E | Finding |
|---|---|
| E1 | Coarse is cheaper only for some `(fine, coarse)` pairs and GPUs; on T4 `c3-3 s4` is never cheaper |
| E2 | Image-Jev does **not** beat text-Jev as a routing signal |
| E3 | The adaptive result flips with the coarse operating point alone (21% vs 92% fine) |
| E4 | Recovery + hysteresis cut switches without worse regret; a switch **seam** exists |
| E5 | Image-Jev endpoint agrees with CLIP subject consistency at **ρ = +1.000** |
| E9 | Claim-to-artifact audit **18/18 PASS** |

## Layout

```
jevgag/
├─ report/      LaTeX source + PDF of the report
├─ code/        one audit script per experiment (e1 … e9) + figure generation
├─ results/     audited result files (small, machine-readable)
├─ figures/     publication figures
└─ compute/     kernel/notebook definitions (colab/, jobs/)
```

## Reproduce (0 GPU)

```bash
cd jevgag
python code/e1_breakeven.py
python code/e2_image_signal.py && python code/e2_text14.py && python code/e2_hybrid.py && python code/e2_audit.py
python code/e3_online_matched.py
python code/e4_rq3_seam.py
python code/e5_endpoint.py && python code/e5_audit.py
python code/e6_t4_boundary.py
python code/e7_confidence_gate.py
python code/e9_claim_audit.py
```

Backbone: Causal-rCM Wan2.1-T2V-1.3B (`ed3cb14`) · GPUs: Tesla T4 and A100-SXM4-80GB · Jev `jev-1.14.0`.

## Data

Audited results are in [`results/`](results/); the public bundle is on Kaggle — see
[`../DATA.md`](../DATA.md).
