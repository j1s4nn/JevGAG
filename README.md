# JevGAG — Adaptive Temporal Granularity for Causal Video World Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Site](https://img.shields.io/badge/site-j1s4nn.github.io%2FJevGAG-2ea44f)](https://j1s4nn.github.io/JevGAG/)
[![Audited](https://img.shields.io/badge/claim--to--artifact%20audit-18%2F18%20PASS-brightgreen)](papers/jevgag/results/e9_audit/)

Research repository for **adaptive temporal granularity** in **causal video generation** and
**interactive video world models**: when to spend fine per-frame compute, and when a cheaper coarse
chunk (or near-free reuse) is safe.

> **Keywords:** causal video generation · video world models · autoregressive diffusion · adaptive
> compute · temporal granularity · efficient inference · System One / Jev · deadline scheduling ·
> drift control.

---

## Papers

| Paper | Status | Focus |
|---|---|---|
| **JevGAG** — Jev-Grounded Adaptive Temporal Granularity for Causal Video Generation | complete (audited) | the audit: three preconditions (cost / signal / endpoint), break-even law, recovery + hysteresis, independent endpoint |
| **DDAG** — Deadline- and Drift-Aware Adaptive Temporal Granularity for Interactive Causal Video World Models | in progress | deployable controller: drift-aware lookahead, convergence-residual verification, hard-deadline scheduling, Jev critic, compiled policy |

The JevGAG paper's artifacts live under [`papers/jevgag/`](papers/jevgag/); DDAG is developed under
[`ideas/`](ideas/) and will be promoted to `papers/ddag/` when complete.

## Headline results (JevGAG)

| E | Finding |
|---|---|
| E1 | Coarse is cheaper only for some `(fine, coarse)` pairs and GPUs; on T4 `c3-3 s4` is never cheaper |
| E2 | Image-Jev does **not** beat text-Jev as a routing signal (honest negative) |
| E3 | The adaptive result flips with the coarse operating point alone (21% vs 92% fine) |
| E4 | Recovery + hysteresis cut switches without worse regret; a switch **seam** exists |
| E5 | Image-Jev endpoint agrees with CLIP subject consistency at **ρ = +1.000** |
| E9 | Claim-to-artifact audit **18/18 PASS** |

## Follow-on studies (`ideas/`)

| Study | Result | Dataset |
|---|---|---|
| **SelfAware-DiT** (idea 03) | intrinsic pre-generation probe **AUROC 0.798** (vs hand-crafted 0.491, Jev text 0.788) | [`jevgag-selfaware-dit`](https://www.kaggle.com/datasets/ajjisan/jevgag-selfaware-dit) |
| **SeamControl** (idea 04) | coarse boundary seam **2.9×** fine; training-free **blend0.25** cuts seam 25% **and** improves fidelity | [`jevgag-seamcontrol`](https://www.kaggle.com/datasets/ajjisan/jevgag-seamcontrol) |
| **ClosedLoop-Gran** (idea 01) | free post-generation evidence **0.725**; closed loop 0.690 → **0.737** | [`jevgag-closedloop-gran`](https://www.kaggle.com/datasets/ajjisan/jevgag-closedloop-gran) |
| **Verify-Gran** (gate) | 0-GPU go/no-go gate: **NO-GO** — the cost model, not the signal, is the binding constraint | in `ideas/verify-gran/results/` |

## Repository layout

```
papers/jevgag/     the JevGAG paper: paper/ (LaTeX), scripts/, results/, colab/, jobs/
ideas/             follow-on studies (selfaware-dit, seamcontrol, closedloop-gran, verify-gran)
assets/            figures and social preview
docs/              GitHub Pages site
archive/           legacy working notes (kept for provenance, not authoritative)
```

## Reproduce (JevGAG, 0 GPU)

```bash
cd papers/jevgag
python scripts/e1_breakeven.py
python scripts/e2_image_signal.py && python scripts/e2_text14.py && python scripts/e2_hybrid.py && python scripts/e2_audit.py
python scripts/e3_online_matched.py
python scripts/e4_rq3_seam.py
python scripts/e5_endpoint.py && python scripts/e5_audit.py
python scripts/e6_t4_boundary.py
python scripts/e7_confidence_gate.py
python scripts/e9_claim_audit.py
```

## Models and hardware

- Oracle: Jev `jev-1.14.0` (image input requires the header `TypeSafe-Danger-Zone-Images: true`).
- Backbone: Causal-rCM Wan2.1-T2V-1.3B, upstream commit `ed3cb14dd936f92cdc9f9381af7369991509b41f`.
- GPUs: Tesla T4 (Kaggle) and NVIDIA A100-SXM4-80GB (Colab).

## Data

Audited result files are in-repo; large media and full bundles are published as public Kaggle datasets —
see [`DATA.md`](DATA.md).

## Status and news

- [`STATUS.md`](STATUS.md) — current phase, achievements, and gaps.
- [`NEWS.md`](NEWS.md) — dated changelog.

## Citation

```bibtex
@software{hossen2026jevGAG,
  title  = {JevGAG: Jev-Grounded Adaptive Temporal Granularity for Causal Video Generation},
  author = {Hossen, Md Jisan},
  year   = {2026},
  url    = {https://github.com/j1s4nn/JevGAG}
}
```

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). Work happens on topic branches and lands via pull request.

## License

MIT — see [`LICENSE`](LICENSE).
