# NEWS — changelog

Dated, user-facing changes to this repository. Newest first.

## 2026-10-25

- **DDAG P2 complete** (E2 CRV, E3 drift, E7 Jev critic, E8 seam/flicker), all merged with public
  artifacts. Headline results:
  - **E2 (CRV, T4):** the free, training-free `step_last/step_first` convergence ratio reaches **0.752**;
    **probe + ratio = 0.842 AUROC**, the best signal in the project.
  - **E2 (A100):** the convergence ratio is **hardware-conditional** (0.752 → **0.485**, chance) while the
    intrinsic probe is **robust** (0.792 → **0.833**).
  - **E3 (drift):** on T4 one coarse segment leaves a **significant persistent** drift elevation
    (4/5 tail segments, +0.021..+0.031) → error propagates; on A100 the comparison **reverses**; drift is
    linear and not predictable.
  - **E7 (JCD):** a typed Jev critic on two raw frames is at **chance** (0.497/0.514) → **refuted**;
    consistent with JEVQA. The Jev image API was fixed (multipart + `assets` map): 0/64 → 64/64 calls.
  - **E8 (flicker):** the seam repair reproduces (−25% seam, +0.6% fidelity) but the independent
    flicker endpoint reveals a **trade-off** (the repair raises interior flicker).
- **Infrastructure:** A100 runs are automated through the WSL `colab` CLI (session → upload → exec →
  download → stop); a private `ajjisan/jev-secrets` dataset supplies the Jev key to kernels.
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
