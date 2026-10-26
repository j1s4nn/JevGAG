# JevGAG · DDAG — Adaptive Temporal Granularity for Causal Video World Models

![thumbnail](assets/figures/thumbnail.png)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Site](https://img.shields.io/badge/site-j1s4nn.github.io%2FJevGAG-2ea44f)](https://j1s4nn.github.io/JevGAG/)
[![Audited](https://img.shields.io/badge/claim--to--artifact%20audit-18%2F18%20PASS-brightgreen)](jevgag/results/e9_audit/)
[![GPU](https://img.shields.io/badge/GPUs-Tesla%20T4%20%C2%B7%20A100--80GB-6f42c1)](#hardware)

A two-part research programme on **adaptive temporal granularity** for **causal video generation** and
**interactive video world models**: when should a generator spend fine per-frame compute, and when is a
cheaper chunked mode enough?

> **Keywords:** causal video generation · video world models · autoregressive diffusion · adaptive compute
> · temporal granularity · efficient inference · System One / Jev · regime map · boundary analysis.

---

## Prior work

This programme builds directly on the earlier study

> **Hossen Md Jisan**, *Jev-Routed Temporal Granularity with Recovery and Hysteresis for Causal Video
> Generation*, preprint, 2026. **https://doi.org/10.5281/zenodo.23395414**

That study framed adaptive granularity as resting on **three preconditions** (cost, signal, endpoint) and
found the trade-off to be *conditional*. This repository **audits those preconditions** (Part 1) and then
**turns them into a deployable rule** (Part 2).

---

## The programme in one figure

![architecture](assets/figures/fig_architecture.png)

*One row per precondition: the boundary found in the earlier study, the audit that measures it, and the
component that resolves it. The Jev thread is highlighted — Jev grounds the audit.*

---

## Part 1 · JevGAG — the report

A measured audit of the three preconditions on **two GPUs**, with a matched design, an independent
endpoint protocol, and a claim-to-artifact audit.

| Precondition | What the report measures |
|---|---|
| **cost** | the per-forward ratio against the NFE break-even — the same mode pair sits on **opposite sides** of the inequality on T4 and A100 |
| **signal** | cheap context statistics reach only **0.491** AUROC, while an off-device **Jev** text oracle reaches **0.788** |
| **endpoint** | the internal reference **flips** between GPUs, so an endpoint independent of the compared modes is mandatory |

Also established: a two-threshold controller with **hysteresis + forced recovery** stabilises switching,
and a measurable **seam** appears at every switch boundary.

→ Details, code and audited results: [`jevgag/`](jevgag/) · figures: [`jevgag/figures/`](jevgag/figures/)

---

## Part 2 · DDAG — the method

**D**eadline- and **D**rift-**A**ware **G**ranularity turns the preconditions into a **regime map** and a
set of calibrated components, each validated by its own experiment.

| Component | Question | Outcome |
|---|---|---|
| **intrinsic probe** | can the generator predict its own granularity safety? | **0.798 AUROC** from its own prefill activations — no external oracle |
| **convergence residual** | does sampler convergence verify safety? | **0.752** on T4; **probe + residual = 0.842** — the strongest signal in the programme |
| **seam repair** | can the switch seam be reduced without training? | a **25% latent blend cuts the seam 25%** and improves fidelity |
| **regime map** | where does adaptivity actually pay? | it pays only when the cheap mode is **cheaper *and* worse** |

![regime map](assets/figures/fig_regime_map.png)

→ Method, eleven numbered experiments and results: [`ddag/`](ddag/)

---

## Achieved · boundary · open

Every earlier limitation is recorded as a **boundary** — a measured limit of the naive formulation — and
each one either has a resolution or defines the next experiment.

| Boundary (measured) | Resolution in this programme | Status |
|---|---|---|
| cheap context statistics do not predict hard segments (0.491) | intrinsic pre-generation probe (**0.798**) | **resolved** |
| the switch seam is a real cost | training-free seam repair (**−25%** seam, better fidelity) | **resolved** |
| one coarse segment leaves a persistent drift elevation on T4 | drift propagation measured and bounded (linear, not compounding) | **characterised** |
| verification signals **do not transfer** across GPUs (probe 0.798 → 0.143) | the regime map — signals are **regime properties** | **open** |
| a typed Jev critic on raw frames is at chance (0.497) | decision-model critics need **context**, not pixels (reproduces JEVQA) | **open** |
| the repair trades a boundary seam for a small interior flicker | independent endpoint exposes the trade; multi-endpoint protocol required | **open** |

**The programme's honest headline:** across two GPUs and every endpoint measured (latent, decoded-pixel,
temporal flicker), the cheap mode was **never worse** — so the optimal policy is a *fixed* mode, and the
regime map states **exactly where** a router becomes worth deploying. That is a design rule, not a
negative result.

---

## Repository layout

```
jevgag/        Part 1 — the report
├─ report/       LaTeX source + PDF
├─ code/         one audit script per experiment (e1 … e9)
├─ results/      audited result files
├─ figures/      report figures
└─ compute/      kernel / notebook definitions (colab/, jobs/)

ddag/          Part 2 — the method
├─ experiments/  01_intrinsic_probe … 11_flicker_endpoint
├─ analysis/     shared analysis scripts
├─ results/      consolidated machine-readable results
└─ compute/      Colab A100 runners

paper/         the manuscript (Elsevier CAS / CVIU)
assets/        thumbnail + publication figures
docs/          GitHub Pages site
archive/       legacy notes (kept for provenance, not authoritative)
```

## Hardware

Tesla T4 (Kaggle) and NVIDIA A100-SXM4-80GB (Colab). Backbone: **Causal-rCM Wan2.1-T2V-1.3B**
(`ed3cb14`). Oracle: **Jev `jev-1.14.0`**.

## Data

Audited result files live in-repo; large media and full bundles are published as **public Kaggle
datasets** — see [`DATA.md`](DATA.md). Every headline number re-derives from a stored result file.

## Reproduce (0 GPU)

```bash
cd jevgag && python code/e9_claim_audit.py          # report audit
cd ../ddag && python analysis/crv_analysis.py --features results/features_t4.npz --out results
```

## Citation

```bibtex
@software{hossen2026jevgagddag,
  title  = {JevGAG + DDAG: Adaptive Temporal Granularity for Causal Video World Models},
  author = {Hossen, Md Jisan},
  year   = {2026},
  url    = {https://github.com/j1s4nn/JevGAG}
}
```

## Contributing · License

See [`CONTRIBUTING.md`](CONTRIBUTING.md). MIT — see [`LICENSE`](LICENSE).
