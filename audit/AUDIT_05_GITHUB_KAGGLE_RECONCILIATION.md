# AUDIT_05 — GitHub ↔ local ↔ Kaggle reconciliation

**Method.** Kaggle datasets were listed and their file manifests read with the Kaggle CLI (public
datasets only). Numeric verification was performed against the **repository copies** of the result files
(`ddag/results/`, `jevgag/results/`), because those are the files the manuscript's numbers were taken
from. The private datasets were **not** inspected (security policy). No dataset was uploaded,
overwritten or deleted.

---

## 1. Reconciliation table

| Artifact | In GitHub | On Kaggle | Match | Notes |
|---|---|---|---|---|
| `ddag/results/**` (json/npz) | ✅ | ✅ `ddag-experiments` | **match** | consolidated bundle created 26 Oct from the repo working tree |
| `ddag` publication figures (PDF/PNG) | ✅ `assets/figures/` | ✅ in `ddag-experiments` | match | PDF + PNG both shipped |
| qualitative frames | ❌ (only the composed PNG is in git) | ✅ `ddag-matched-frames` | **GitHub incomplete** | 54 raw frames exist only on Kaggle; Fig. 6 is reproducible from the grid PNG but the per-frame evidence is not in git |
| `jevgag/results/**` | ✅ | ✅ `jevgag-results` | match | |
| `jevgag` figures | ✅ `jevgag/figures/` | ✅ | match | |
| report videos | ❌ | ✅ `jev-adaptive-granularity-videos` (412 MB) | GitHub incomplete | media intentionally not versioned; linked from README |
| backbone checkpoints | ❌ | ✅ `causal-rcm-ckpts` (13 GB) | by design | |
| VAE / tokenizer | ❌ | ✅ `wan21-causal-rcm-matched` | **partial** | the bundle contains the HF cache *metadata*; the actual `models_t5_umt5-xxl-enc-bf16.pth` was **not** found, which is why the A100 runner had to pull VAE+T5 from HuggingFace |
| T5 / VAE (private copies) | ❌ | ✅ (private) | n/a | 403 outside the author's kernels |

## 2. Findings

**K-1 (P1) — the A100 reproduction path is not self-contained.**
`ajjisan/wan21-causal-rcm-matched` does not contain the T5 encoder weights, and `ajjisan/wan21-t5` /
`ajjisan/wan21-dit-vae` are private. The A100 runs therefore depend on a **HuggingFace download**
(`Wan-AI/Wan2.1-T2V-1.3B`) that is not documented in the manuscript's data-availability statement.
*Action:* either make the two datasets public, or document the HF fallback in the data-availability
statement and in `ddag/compute/colab/README.md`.

**K-2 (P2) — the consolidated dataset is not a complete mirror.**
`ddag-experiments` was assembled from `ddag/results/` at one point in time; later sub-analyses
(`ddag/results/t4_n32/`, `a100/`, `online_*/`, `band_t4/`) are present in git. Verify that the Kaggle
copy contains the `t4_n32` and `online_*` sub-directories; if not, re-version the dataset.

**K-3 (P1) — the flicker discrepancy propagates to Kaggle.**
`ddag-experiments` contains `README_E8_flicker.md`, which carries the **wrong** flicker means
(fine 0.0254 / none 0.0213) that also appear in the manuscript's Table 2 (see `AUDIT_10`). Fixing the
manuscript without fixing the dataset README leaves an inconsistency between the paper and its
published data. *Action:* correct both, then version-bump the dataset.

**K-4 (P2) — two datasets are referenced nowhere.**
`ajjisan/f4lite-videos` and the report's figure bundle are not referenced by `paper/main.tex` or
`DATA.md`'s main table. Either link or de-list them.

**K-5 (P1) — the audit badge over-claims.**
The README displays a `claim-to-artifact audit 18/18 PASS` badge adjacent to the **new** paper, but
`jevgag/results/e9_audit/claim_audit.json` covers **only the report's** claims (E1–E5, E7). None of the
new paper's headline numbers (0.798, 0.842, 0.752, 0.143, 0.728, 0.536, seam −25 %, drift propagation,
policy table) is inside that audit. *Action:* either extend the audit to the new paper's claims, or
re-word the badge to make its scope explicit.

## 3. What could not be verified

- Whether the Kaggle copies are **byte-identical** to the repo copies (no re-download + hashing was
  performed; the manifests are consistent).
- The private datasets' contents (policy).
- Whether any dataset was created from an **earlier** run than the repo files (creation dates are
  consistent with the run order: 25 Oct for the report bundles, 26 Oct for the DDAG bundles).
