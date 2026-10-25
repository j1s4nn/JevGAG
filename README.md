# JevGAG — Jev-Grounded Adaptive Temporal Granularity for Causal Video Generation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Site](https://img.shields.io/badge/site-j1s4nn.github.io%2FJevGAG-2ea44f)](https://j1s4nn.github.io/JevGAG/)
[![Audited](https://img.shields.io/badge/claim--to--artifact%20audit-18%2F18%20PASS-brightgreen)](results/e9_audit/)

**JevGAG** studies **adaptive temporal granularity** for **causal video generation** and **video world
models**: when to generate one frame at a time (**fine**) versus several frames per chunk (**coarse**),
deciding **before** each segment from a signal available off the generation path.

The project contributes:

1. an **off-device, image-grounded System One oracle** (Jev 1.14) used as a routing signal and as an
   independent endpoint,
2. a matched **break-even law** over hardware (Tesla T4 vs A100-80GB) and operating point,
3. a **recovery + hysteresis** controller that stabilizes switching, and
4. a **claim-to-artifact audit** in which every headline number re-derives from a stored result file.

> **Keywords:** Jev · adaptive granularity · temporal granularity · causal video generation · video
> world models · autoregressive diffusion · off-device oracle · System One · efficient inference ·
> adaptive compute · uncertainty quantification.

---

## Overview

Causal video generators produce a rollout one temporal chunk at a time, which invites a simple
efficiency idea: spend fine per-frame computation only where a segment is hard, and use cheaper coarse
chunks (or near-free reuse) elsewhere. Realizing this idea requires three preconditions that are
usually assumed rather than measured — a mode that is genuinely cheaper on the deployment hardware, a
signal that predicts hard segments before generation, and an independent endpoint that certifies the
trade. JevGAG measures all three.

## Headline results

| E | Experiment | Finding |
|---|---|---|
| **E1** | Matched break-even predictor | Coarse is cheaper only for some `(fine, coarse)` pairs and GPUs; on T4 `c3-3 s4` is **never** cheaper; on A100 `c3-3 s1` beats all fine modes |
| **E2** | Image-Jev vs text-Jev signal | Image/hybrid Jev **do not** beat text Jev as a routing signal (honest negative) |
| **E3** | Matched A100 online factorial | The adaptive result **flips with the coarse operating point alone** (cheap: 21% fine; expensive: 92% fine) |
| **E4** | Recovery + hysteresis + seam | Fewer switches, no worse regret; a switch **seam** exists (ratio > 1 on both GPUs) |
| **E5** | Image-Jev endpoint | Agrees with CLIP subject consistency at **Spearman ρ = +1.000** |
| **E6** | Matched T4 boundary | Derived negative: an **operating-point** boundary, not a signal failure |
| **E7** | Confidence-gated routing | Neutral (Δ ≈ 0) |
| **E8** | Stability-gated recalibration | Negative on both GPUs (worse quality at higher cost) |
| **E9** | Claim-to-artifact audit | **18/18 PASS** |

## Method in one line

A pre-generation decision routes each segment to **fine** (`c1-1`, `S` steps), **cheap coarse**
(`c3-3 s1`), **expensive coarse** (`c3-3 s4`), or **hold**, driven by an **off-device Jev 1.14** oracle
reading the actual context, stabilized by **recovery + hysteresis**, and certified by an **independent
image-grounded endpoint**.

## Repository layout

```
paper/      manuscript (IEEEtran)
src/        oracle (Jev image client), controller, endpoint, modes, break-even
scripts/    one audit script per experiment (e1..e9) + figures
config/     locked manifests + Jev question sets
results/    audited results (jsonl/json; media on Kaggle)
colab/      A100 kernels
jobs/       job specifications
docs/       GitHub Pages site
assets/     figures
notes/      internal working notes (design ledger, findings, references)
```

## Reproduce (0 GPU)

```bash
python scripts/e1_breakeven.py
python scripts/e2_image_signal.py && python scripts/e2_text14.py && python scripts/e2_hybrid.py && python scripts/e2_audit.py
python scripts/e3_online_matched.py
python scripts/e4_rq3_seam.py
python scripts/e5_endpoint.py && python scripts/e5_audit.py
python scripts/e6_t4_boundary.py
python scripts/e7_confidence_gate.py
python scripts/e9_claim_audit.py
python scripts/e_figures.py
```

## Models and hardware

- Jev `jev-1.14.0` (image input requires the header `TypeSafe-Danger-Zone-Images: true`).
- Backbone: Causal-rCM Wan2.1-T2V-1.3B, upstream commit `ed3cb14dd936f92cdc9f9381af7369991509b41f`.
- GPUs: Tesla T4 (Kaggle) and NVIDIA A100-SXM4-80GB (Colab).

## Data and artifacts

Audited result files live in [`results/`](results/). Large media and the full audited bundles are
published as public Kaggle datasets (see [`DATA.md`](DATA.md)). Every headline number is re-derived by
[`scripts/e9_claim_audit.py`](scripts/e9_claim_audit.py).

## Citation

```bibtex
@software{hossen2026jevGAG,
  title  = {JevGAG: Jev-Grounded Adaptive Temporal Granularity for Causal Video Generation},
  author = {Hossen, Md Jisan},
  year   = {2026},
  url    = {https://github.com/j1s4nn/JevGAG}
}
```

See [`CITATION.cff`](CITATION.cff) for machine-readable metadata.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). Work happens on topic branches and lands via pull request.

## License

MIT — see [`LICENSE`](LICENSE).
