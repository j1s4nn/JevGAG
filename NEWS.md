# NEWS — changelog

Dated, user-facing changes to this repository. Newest first.

## 2026-10-27

- **Paper released.** *Regime-Aware Temporal Granularity for Causal Video World Models: When Does
  Adaptivity Pay?* is published as a preprint — **https://doi.org/10.5281/zenodo.23693517**. The README,
  the site and [`CITATION.cff`](CITATION.cff) now feature it alongside the prior study
  (**https://doi.org/10.5281/zenodo.23395414**).
- **Housekeeping.** The `paper/` folder is emptied (the journal submission PDF is added on release); the
  legacy, non-authoritative `archive/` notes and the internal `STATUS.md` tracker were removed; unused
  site figures were dropped.
- **Repository-wide scientific audit completed** (15 reports under [`audit/`](audit/)), run against commit
  `0c725c0`, and its **P0/P1 corrections applied** — all from stored result files, with no new computation.
  - **F-1 (P0).** The convergence residual in Eq. (2) was **sign-inverted**: the direct AUROC of
    `v_S/v_1` is **0.240**, not 0.752; the 0.752 came from a single-feature logistic that re-learns the
    sign, and the manuscript's 0.842 appeared in **no** result file. The residual is now defined as
    `κ = v_1/v_S` (direct **0.760**), and the multi-feature model is **stored**
    (`ddag/results/signal_combined.json`): **0.842** on T4 and **0.833** on A100 — the only signal that
    transfers across GPUs.
  - **F-2 (P0).** Two flicker means in Table 1 were wrong (`none` 0.0213 → **0.0148**, `fine` 0.0254 →
    **0.0189**). The trade-off is now *stronger*: the repaired value (0.0218) is the highest of the three.
  - **F-3 (P0).** The regime map now states an explicit constrained objective, and the oracle-router bound
    is reported (no better than fixed coarse at any operating point).
  - **F-4 (P0).** The cost numbers 3.01/1.60/2.33 were the **prior study's**; the paper now uses the
    matched matrix (π = 2.97 / 1.74, break-even 2.68) and the per-segment cost ratio r.
  - **P1.** Added: Limitations/threats-to-validity, Generalisation, Ablations, a protocol table with
    per-experiment sample sizes, the decoded-pixel endpoint result, an ethics statement, corrected
    captions and axis labels; keywords trimmed to six.
- **Corrections propagated** to the README, the site, the experiment READMEs and the Kaggle dataset
  READMEs. `crv_variants.py` now carries an explicit supersession note.

## 2026-10-26

- **DDAG P2 complete** (E2 CRV, E3 drift, E7 Jev critic, E8 seam/flicker), all merged with public
  artifacts. Headline results:
  - **E2 (CRV, T4):** the convergence ratio reaches 0.752 (later corrected — see 27 Oct).
  - **E2 (A100):** the intrinsic probe inverts (0.798 → 0.143) while the combined model transfers.
  - **E3 (drift):** on T4 one coarse segment leaves a significant persistent drift elevation; on A100 the
    comparison reverses.
  - **E7 (JCD):** a typed Jev critic on two raw frames is at chance (0.497) — consistent with JEVQA.
  - **E8 (flicker):** the seam repair reproduces but the independent flicker endpoint reveals a trade-off.
- **Infrastructure:** A100 runs automated through the WSL `colab` CLI; a private `ajjisan/jev-secrets`
  dataset supplies the Jev key to kernels.
- **Repository restructured** into two content-named parts: `jevgag/` (report) and `ddag/` (method).
- **Manuscript** written in the Elsevier CAS format (CVIU target) with a verified bibliography.

## 2026-10-25

- **Repository professionalized** for the new paper: clean root, `papers/` → two-part structure,
  governance files, rebuilt site with SEO, umbrella thumbnail, and the public dataset
  `ajjisan/jevgag-results`.
- **JevGAG report** audited: claim-to-artifact audit **18/18 PASS** (covers the report's claims).
