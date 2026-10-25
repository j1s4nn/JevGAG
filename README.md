# JevGAG — Jev-Grounded Adaptive Granularity

**Paper:** Jev-Grounded Adaptive Temporal Granularity for Causal Video Generation:
An Image-Grounded Off-Device Oracle with Recovery, Recalibration, and an Independent Endpoint.

## What this repo is
New experiments (E1–E9) for the second paper. Prior work lives in the sibling folder
`../idea_05_adaptive_granularity_WM/04_Experiment/` and is **reused read-only** (see `REFERENCE.md`).
Nothing from the old project is copied here.

## Layout
- `paper/`        manuscript (empty until writing)
- `src/`          oracle (Jev image client), controller, endpoint, modes, breakeven
- `scripts/`      one audit per experiment (e1..e9)
- `config/`       locked manifests + Jev question sets
- `results/`      new results only (jsonl/json; media on Kaggle)
- `colab/`        A100 kernels
- `docs/`         protocols + Pages site
- `assets/`       banner, adapted figures

## Credentials (never committed)
- Jev / TypeSafe: `~/.kilo-automation/typesafe.env` (`TYPESAFE_API_KEY`)
- Kaggle: `~/.config/kilo/secrets/kaggle.json` (user `ajjisan`)
- GitHub: `~/.config/kilo/secrets/github.env` (`GH_TOKEN`)

## Models
- Jev: pin `jev-1.14.0`; images require header `TypeSafe-Danger-Zone-Images: true`.
- Backbone: Causal-rCM Wan2.1-1.3B, upstream commit `ed3cb14dd936f92cdc9f9381af7369991509b41f`.

## Citation
The prior manuscript is a preprint under review; cite as:
`Jisan, H. M. (2026). Hardware-Conditional Adaptive Temporal Granularity ... doi:10.5281/zenodo.23395414 (preprint, under review at IEEE TCSVT).`
