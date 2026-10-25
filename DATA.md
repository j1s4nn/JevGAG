# Data & artifacts

Small, machine-readable result files are versioned in this repository under [`results/`](results/).
Large media and full audited bundles are published as **public Kaggle datasets** so every reported
number can be independently checked ("prove the output").

## Published datasets

| Dataset | Slug | Contents |
|---|---|---|
| JevGAG: Audited Results and Figures | [`ajjisan/jevgag-results`](https://www.kaggle.com/datasets/ajjisan/jevgag-results) | audited `results/` files (E1–E9) and publication `figures/` |
| JevGAG SelfAware-DiT probes | [`ajjisan/jevgag-selfaware-dit`](https://www.kaggle.com/datasets/ajjisan/jevgag-selfaware-dit) | Causal-rCM DiT activations + intrinsic granularity-safety probe results (idea 03) |
| JevGAG SeamControl repair | [`ajjisan/jevgag-seamcontrol`](https://www.kaggle.com/datasets/ajjisan/jevgag-seamcontrol) | boundary-seam measurements + training-free seam-repair ablation (idea 04) |
| JevGAG ClosedLoop-Gran capture | [`ajjisan/jevgag-closedloop-gran`](https://www.kaggle.com/datasets/ajjisan/jevgag-closedloop-gran) | intrinsic prior + free post-generation evidence for closed-loop granularity routing (idea 01) |

## Convention

- Slug pattern: `ajjisan/jevgag-<scope>` (e.g. `jevgag-results`, `jevgag-<idea>-<phase>`).
- Each dataset ships a `README.md` (provenance, license, scripts) and a `dataset-metadata.json`.
- License follows the repository (MIT) unless a dataset states otherwise.
- Datasets are **public**.

## Publishing a new dataset

```bash
# stage results + figures
mkdir -p /tmp/ds/results /tmp/ds/figures
cp -r results/* /tmp/ds/results/
cp assets/figures/*.png /tmp/ds/figures/
# write /tmp/ds/dataset-metadata.json (title, id, licenses, description) and /tmp/ds/README.md
python -m kaggle datasets create -p /tmp/ds --public --dir-mode zip          # first time
python -m kaggle datasets version -p /tmp/ds -m "<change>" --dir-mode zip    # updates
```

## Audit

[`scripts/e9_claim_audit.py`](scripts/e9_claim_audit.py) re-derives every headline number from the
stored result files and reports a claim-to-artifact audit (18/18 PASS at publication).
