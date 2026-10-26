# AUDIT_00 — Scope and inventory

**Audit date:** 26 Oct 2026
**Repository:** `C:\Users\jisan\OneDrive\Desktop\research\idea_06_jev_grounded_granularity` (= `github.com/j1s4nn/JevGAG`)
**Branch / commit audited:** `main` @ **`0c725c0`** ("Merge pull request #33 from j1s4nn/docs/data-depth")
**Working tree:** clean (no uncommitted files at audit time)

---

## 1. What exists in the repository

### 1.1 Manusripts — **there are TWO**

| Path | Class | Title | Role |
|---|---|---|---|
| `paper/main.tex` (27 KB, 479 lines) | `cas-sc` (Elsevier CAS, single column) | *Regime-Aware Temporal Granularity for Causal Video World Models: When Does Adaptivity Pay?* | **the submission target for CVIU** |
| `jevgag/report/main.tex` (7 KB) | `IEEEtran` (journal) | *Jev-Grounded Adaptive Temporal Granularity for Causal Video Generation: An Image-Grounded Off-Device Oracle with Recovery, Recalibration, and an Independent Endpoint* | **a second, unpublished manuscript** — the "report" |

**Finding I-1 (P1).** The repository contains **two complete, differently titled manuscripts**. The
new paper's §3 ("Background: the report") summarises the report's findings; the report is itself a
submittable IEEEtran manuscript. This is a **self-overlap hazard** that must be resolved before
submission (see `AUDIT_03`, `AUDIT_14`).

### 1.2 Prior work (external, not in this repo)

| Artifact | Status |
|---|---|
| *Jev-Routed Temporal Granularity with Recovery and Hysteresis for Causal Video Generation* | Zenodo preprint `10.5281/zenodo.23395414`; **TCSVT-32491-2026 reported as "Under Review"** by the author; cited as prior work in `paper/main.tex` |

### 1.3 Directory tree (top level)

```
.github/   archive/  assets/  ddag/  docs/  jevgag/  paper/
.gitignore CITATION.cff CODE_OF_CONDUCT.md CONTRIBUTING.md DATA.md
LICENSE NEWS.md README.md SECURITY.md STATUS.md
```

`jevgag/` = Part 1 (report): `report/ code/ results/ figures/ compute/{colab,jobs}`
`ddag/`  = Part 2 (method): `experiments/01…12`, `analysis/`, `results/`, `compute/colab/`
`paper/` = the CVIU manuscript + `refs.bib` + `figures/` + `thumbnails/` + `OUTLINE.md`
`assets/figures/` = publication figures (PDF + PNG) + `thumbnail.png`
`docs/` = GitHub Pages site; `archive/notes/` = legacy working notes

### 1.4 Markdown files: **39**

Root (10): `README.md` (305 L), `STATUS.md`, `NEWS.md`, `DATA.md`, `CONTRIBUTING.md`,
`CODE_OF_CONDUCT.md`, `SECURITY.md`, `.github/*` (3).
`jevgag/` (11): `README.md` + 9 result tables (`e1…e9`) + `claim_audit.md`.
`ddag/` (16): `README.md`, `compute/colab/README.md`, 3 experiment READMEs,
7 result READMEs (`README_E2_T4`, `_E2_crosshw`, `_E3_drift`, `_E5_regime_map`, `_E5b_band`, `_E7_jcd`, `_E8_flicker`).
`paper/OUTLINE.md`, `archive/notes/{DESIGN,FINDINGS,REFERENCE,README}.md`.

### 1.5 Machine-readable results

- `jevgag/results/`: `breakeven_matrix.{csv,json}` (18 rows), `e2_audit.json`, `e3_online.json`,
  `e4.json`, `e5_audit.json`, `claim_audit.json` (**18/18**), plus 9 per-experiment tables.
- `ddag/results/`: `e2_crv.json`, `e2b_residual_variants.json`, `e3_drift.json`, `e7_jcd.json`,
  `e8_flicker.json`, `drift_*.npz` (T4 n=8/24/32, A100), `features_*.npz` (T4, A100, band, online×2, pxq),
  `flicker_t4.npz`, `jcd_t4.npz`, and sub-analyses `a100/`, `t4_n24/`, `t4_n32/`,
  `online_t4/`, `online_a100/`, `band_t4/`.

### 1.6 Figures

`assets/figures/`: `fig_architecture.{pdf,png}`, `fig_regime_map.{pdf,png}`, `fig_signal.{pdf,png}`,
`fig_drift.{pdf,png}`, `fig_seam.{pdf,png}`, `fig_qualitative.png`, `thumbnail.png`.
`jevgag/figures/`: 5 report figures (`fig_e1_breakeven`, `fig_e2_signal`, `fig_e5_endpoint`,
`fig_e7_frontier`, `fig_e8_recalibration`).
Sources: TikZ/pgfplots `.tex` for the new figures; matplotlib for the report's.

### 1.7 Companion packages (outside the repo)

`C:\Users\jisan\OneDrive\Desktop\research\idea_07_novel_directions\`:
`paper/` (45 files), `submission/` (29), `zenodo/` (5), plus planning docs and `ideas/`.

---

## 2. What could NOT be accessed / is uncertain

| Item | Status |
|---|---|
| Kaggle dataset **contents** (per-file numerics) | inspectable via the Kaggle CLI; manifest in `AUDIT_04` — **file-level numeric verification was done against the repository copies, not re-downloaded** |
| Private datasets (`ajjisan/wan21-t5`, `ajjisan/wan21-dit-vae`, `ajjisan/jev-secrets`) | deliberately not inspected (credentials / private) |
| The TCSVT-32491-2026 submission's current status | **contradictory evidence**: an e-mail (16 Oct) says *rescinded — biography missing*; the portal screenshot says *Under Review*. Author confirmed "under review"; unresolved |
| `jevgag/results/e8_recalibration` raw logs | present as `.log`; not re-parsed numerically |
| Drift A100 run | **n = 8 prompts only** (vs n = 32 on T4) — sample-size asymmetry noted |

## 3. Exact versions used in this audit

- Repo commit `0c725c0` (clean tree).
- Manuscript `paper/main.tex` as committed at that commit.
- All result JSON/NPZ read directly from the repository working tree at that commit.
- Figures read as committed PDF/PNG; figure `.tex` sources read where present.

## 4. Audit method and its limits

Every number in `paper/main.tex` was re-derived from the stored result files
(`AUDIT_06`). Equations were re-derived independently (`AUDIT_07`). Figures were checked against their
source data and their plotting scripts (`AUDIT_08`, `AUDIT_10`).

**Limits.** (a) The claim-to-artifact audit (`claim_audit.json`, 18/18) covers the **report**, not the
new manuscript; it is therefore **not** evidence for the new paper's claims. (b) No GPU or API call was
made; no re-computation of rollouts was performed. (c) Kaggle copies were reconciled by manifest, not by
re-downloading and re-hashing every file.
