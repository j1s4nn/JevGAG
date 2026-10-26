# Data & artifacts

Small, machine-readable result files are versioned in this repository. Large media and full audited
bundles are published as **public Kaggle datasets** so every reported number can be independently
checked.

## Where artifacts live

```
jevgag/results/     the report's audited results (E1–E9)
ddag/results/       the method's consolidated results (experiments 01–11)
ddag/experiments/   per-experiment kernels + results + notes
```

## Published datasets (public)

| Dataset | Slug | Contents |
|---|---|---|
| JevGAG report — audited results and figures | [`ajjisan/jevgag-results`](https://www.kaggle.com/datasets/ajjisan/jevgag-results) | the report's `results/` (E1–E9) + figures |
| Intrinsic probe (exp 01) | [`ajjisan/jevgag-selfaware-dit`](https://www.kaggle.com/datasets/ajjisan/jevgag-selfaware-dit) | DiT activations + probe results |
| Seam repair (exp 02) | [`ajjisan/jevgag-seamcontrol`](https://www.kaggle.com/datasets/ajjisan/jevgag-seamcontrol) | boundary-seam measurements + repair ablation |
| Closed-loop evidence (exp 03) | [`ajjisan/jevgag-closedloop-gran`](https://www.kaggle.com/datasets/ajjisan/jevgag-closedloop-gran) | intrinsic prior + free post-generation evidence |

> Datasets for experiments 05–11 (convergence, drift, online policy, band, pixel endpoint, Jev critic,
> flicker) are consolidated from `ddag/results/` in the same way.

## Convention

- Slug pattern: `ajjisan/jevgag-<scope>`.
- Each dataset ships a `README.md` (provenance, license, scripts) and a `dataset-metadata.json`.
- License follows the repository (MIT) unless a dataset states otherwise.
- Datasets are **public**; the only private dataset is `ajjisan/jev-secrets` (API credentials).

## Audit

- Report: [`jevgag/code/e9_claim_audit.py`](jevgag/code/e9_claim_audit.py) re-derives every headline
  number (18/18 PASS at publication).
- Method: each experiment folder ships its own analysis; `ddag/analysis/` holds the shared scripts.
