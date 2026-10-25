# NEWS — changelog

Dated, user-facing changes to this repository. Newest first.

## 2026-10-25

- **DDAG plan finalized.** New paper direction: *Deadline- and Drift-Aware Adaptive Temporal Granularity
  for Interactive Causal Video World Models*, with five new methods (DALG, CRV, DCG, JCD, SDGP), eight
  research questions, and a both-GPU (T4 + A100) evaluation plan. See `STATUS.md`.
- **Repository restructured for a clean, journal-ready layout.**
  - `papers/jevgag/` now holds the JevGAG paper (`paper/`, `scripts/`, `results/`, `colab/`, `jobs/`).
  - `archive/` holds legacy working notes (kept for provenance, not authoritative).
  - Root is limited to `README.md`, `STATUS.md`, `NEWS.md`, `DATA.md`, `CITATION.cff`, `LICENSE`, and
    governance files.
- **Added** `STATUS.md` (phase, achievements, gaps) and `NEWS.md` (this file).
- **README rewritten** with umbrella + DDAG framing, a paper table, and follow-on study results.
- **Follow-on results merged** (all with public Kaggle datasets):
  - *SelfAware-DiT* — intrinsic pre-generation probe **AUROC 0.798**.
  - *SeamControl* — training-free seam repair: **−25% seam, +1.1% fidelity**.
  - *ClosedLoop-Gran* — free post-generation evidence **0.725**; closed loop 0.690 → 0.737.
  - *Verify-Gran* — 0-GPU go/no-go gate returned **NO-GO**: the **cost model**, not the signal, is the
    binding constraint.
- **JevGAG paper artifacts preserved** and still reproducible from `papers/jevgag/`.
