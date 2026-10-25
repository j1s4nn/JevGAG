# JevGAG — Jev-Grounded Adaptive Temporal Granularity for Causal Video Generation

[![DOI](https://img.shields.io/badge/preprint-10.5281%2Fzenodo.23395414-blue)](https://doi.org/10.5281/zenodo.23395414)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Site](https://img.shields.io/badge/site-j1s4nn.github.io%2FJevGAG-2ea44f)](https://j1s4nn.github.io/JevGAG/)

**JevGAG** studies **adaptive temporal granularity** for **causal video generation / video world
models**: when to generate one frame at a time (fine) versus several frames per chunk (coarse).
It uses **Jev 1.14** as an **off-device, image-grounded System One oracle**, contributes a matched
**break-even law** over hardware and operating point, a **recovery + hysteresis** controller, and a
**validated independent endpoint**.

> Keywords: Jev · adaptive granularity · temporal granularity · causal video generation · video world
> models · autoregressive diffusion · off-device oracle · System One · efficient inference ·
> adaptive compute · uncertainty quantification.

## Headline results

| E | Experiment | Finding |
|---|---|---|
| E1 | Break-even predictor | Coarse is cheaper only for some (fine,coarse) pairs and GPUs; on T4 `c3-3 s4` is **never** cheaper; on A100 `c3-3 s1` beats all fine modes |
| E2 | Image-Jev vs text-Jev signal | Image/hybrid Jev **do not** beat text Jev as a routing signal (honest negative) |
| E3 | Matched A100 online factorial | The adaptive result **flips with the coarse operating point alone** (cheap: 21% fine; expensive: 92% fine) |
| E4 | Recovery + hysteresis + seam | Fewer switches, no worse regret, seam ratio > 1 on **both** GPUs |
| E5 | Image-Jev endpoint | Agrees with CLIP subject consistency at **Spearman ρ = +1.000** |
| E6 | Matched T4 boundary | Derived negative: an **operating-point** boundary, not a signal failure |
| E7 | Confidence-gated routing | Neutral (Δ ≈ 0) |
| E9 | Claim-to-artifact audit | **18/18 PASS** |

## Method in one line
A pre-generation decision: route each segment to **fine** (`c1-1`, S steps), **cheap coarse**
(`c3-3 s1`), **expensive coarse** (`c3-3 s4`), or **hold**, driven by an **off-device Jev 1.14**
oracle reading the actual context frames, stabilized by **recovery + hysteresis**, and certified by
an **independent image-grounded endpoint**.

## Repository layout
```
paper/     manuscript (IEEEtran)
src/       oracle (Jev image client), controller, endpoint, modes, breakeven
scripts/   one audit per experiment (e1..e9) + figures
config/    locked manifests + Jev question sets
results/   audited results (jsonl/json; media on Kaggle)
colab/     A100 kernels
docs/      GitHub Pages site
assets/    figures
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
- Jev `jev-1.14.0` (image input requires header `TypeSafe-Danger-Zone-Images: true`).
- Backbone: Causal-rCM Wan2.1-T2V-1.3B, upstream commit `ed3cb14dd936f92cdc9f9381af7369991509b41f`.
- GPUs: Tesla T4 (Kaggle) and NVIDIA A100-SXM4-80GB (Colab).

## Citation
```bibtex
@software{jisan2026jevGAG,
  title  = {JevGAG: Jev-Grounded Adaptive Temporal Granularity for Causal Video Generation},
  author = {Hossen, Md Jisan},
  year   = {2026},
  url    = {https://github.com/j1s4nn/JevGAG}
}
```
Prior related work (preprint, under review at IEEE TCSVT):
`doi:10.5281/zenodo.23395414`.

## License
MIT — see [LICENSE](LICENSE).
