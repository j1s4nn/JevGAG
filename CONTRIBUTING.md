# Contributing to JevGAG

Thanks for your interest. This repository follows a **branch → pull request → merge** workflow so
every change is reviewable and reproducible.

## Workflow

1. Branch from `main` using a typed prefix:
   - `feat/<short-name>` — new capability
   - `exp/<e-id>-<short-name>` — an experiment (e.g. `exp/e10-signal-scale`)
   - `fix/<short-name>` — a bug fix
   - `docs/<short-name>` — documentation
   - `chore/<short-name>` — tooling / housekeeping
   - `data/<short-name>` — dataset or artifact publishing
2. Keep the branch focused on one concern.
3. Commit with a conventional message: `type(scope): summary` (e.g. `feat(e10): scale the signal test`).
4. Open a pull request against `main`; fill in the PR template.
5. A change is merged only when the claim-to-artifact audit passes.

## Reproducibility rules

- Every headline number must resolve to a **named script** and a **stored result file**.
- Negative results are **preserved**, never edited out.
- Do not fit any model, threshold, or rule on evaluation prompts. Splits must be disjoint
  (development / calibration / held-out).
- Report prompt-paired bootstrap confidence intervals for paired comparisons.

## Data and artifact convention (Kaggle)

Large media and audited bundles are published as **public Kaggle datasets**:

- slug convention: `ajjisan/jevgag-<scope>` for the report and `ajjisan/ddag-<scope>` for the method (e.g. `ajjisan/jevgag-results`).
- each dataset ships a `dataset-metadata.json` and a `README.md` describing provenance, license, and
  the exact scripts that produced it.
- the repo keeps the small, machine-readable result files in `jevgag/results/` and `ddag/results/`, and links the Kaggle dataset
  from `DATA.md`.

See [`DATA.md`](DATA.md) for the current datasets.

## Secrets

Never commit secrets. API keys are read from the environment (`TYPESAFE_API_KEY`, `GH_TOKEN`,
`KAGGLE_*`). `.env` files and `kaggle.json` are git-ignored.

## License

By contributing you agree that your contributions are licensed under the MIT License (see `LICENSE`).
