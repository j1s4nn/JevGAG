# Paper blueprint (P5)

> Target: **Elsevier CVIU** (`els-cas-templates`, single column). **No binding page limit was found** for
> CVIU Regular Papers; the earlier "14–16 pages" was this project's own target, not a journal rule.
> The manuscript compiles to **18 pages in the `cas-sc` review format** (line numbers, generous spacing);
> the typeset length will be shorter. Source of truth for the paper; every figure and number is
> traceable to `jevgag/` or `ddag/`.
>
> **Post-audit revision applied** (see `audit/AUDIT_14_REVISION_ROADMAP.md`): the convergence residual was
> redefined (`κ = v_1/v_S`; the reciprocal is inverted), two flicker means were corrected, the regime map
> gained an explicit objective, the cost paragraph was re-attributed, and Limitations / Generalisation /
> Ablations / Protocol-table / decoded-pixel sections were added.

---

## Title and authors

**Working title:** *Regime-Aware Temporal Granularity for Causal Video World Models: When Does
Adaptivity Pay?*

**Author:** Hossen Md Jisan — Department of Artificial Intelligence, NUIST, Nanjing, China ·
jisanahammed1952@gmail.com · ORCID 0009-0004-9571-1450 · sole author.

---

## Contribution statement (what is new here)

1. **A two-GPU audit of the preconditions** (cost · signal · endpoint) for adaptive temporal granularity
   in causal video world models, with a matched design and an independent-endpoint protocol.
2. **A regime map** that states exactly when adaptive granularity pays — and shows that at both tested
   operating points the cheap mode dominates, so a *fixed* policy is optimal.
3. **An intrinsic pre-generation probe** (AUROC 0.798, no external oracle) and a **convergence-residual**
   signal (probe + residual **0.842**) that beat the cheap context statistics (0.491).
4. **A training-free seam repair** that reduces the switch-boundary discontinuity by **25%** while
   improving fidelity — and an independent endpoint that exposes its cost.
5. **A boundary analysis**: signals are *regime properties* (the probe inverts on A100, 0.798 → 0.143),
   drift is linear and policy-insensitive, and a context-free Jev critic is at chance.

## Non-overlap statement (for the cover letter)

The prior preprint `10.5281/zenodo.23395414` is cited as **prior work**. It established the three
preconditions and a conditional trade-off on one GPU. **This paper's contribution is the two-GPU regime
map, the intrinsic/convergence signals, the seam repair, and the boundary analysis** — none of which
appear in the prior preprint. The prior preprint is **not** restated as a contribution, and no text or
figure is reused.

---

## Section plan (target 14–16 pp)

| § | Content | pp |
|---|---|---|
| I | **Introduction** — interactive causal video world models; the cost/quality lever; the prior study's three preconditions; contributions; the boundary framing | 1.5 |
| II | **Related work** — causal/autoregressive video diffusion; adaptive compute and routing in generative models; early-exit and internal probes; evaluation of video generation; decision models (Jev). *Verified citations only.* | 1.5 |
| III | **Background and the report** — the prior study's preconditions, modes, label, endpoint protocol; what the report measures on T4 and A100 (compressed to background) | 2.0 |
| IV | **The DDAG method** — regime map formalism; intrinsic prefill probe; convergence residual; calibrated gate; training-free seam repair; the policy that the map justifies | 2.5 |
| V | **Experimental protocol** — backbone, hardware, matched design, prompts, statistics (leave-one-prompt-out, grouped bootstrap, permutation nulls), endpoints | 1.5 |
| VI | **Results** — signal; convergence; seam repair; drift propagation; online policy and the regime map; pixel endpoint; Jev critic; flicker | 4.0 |
| VII | **Boundaries and guidance** — what does *not* transfer and why; a practitioner recipe | 1.0 |
| VIII | **Conclusion** | 0.5 |
| — | **References** | 1.5 |
| — | Declarations · CRediT · data availability · biography | 0.5 |

---

## Figure plan (each figure is traceable and vision-verified)

| Fig | Content | Source | Status |
|---|---|---|---|
| 1 | **Architecture** — prior study → report → method, one row per precondition, Jev thread highlighted | `assets/figures/fig_architecture.pdf` | ✅ drawn (TikZ), vision-verified |
| 2 | **Regime map** — cost ratio vs quality gap; the "adaptive pays" quadrant is empty | `fig_regime_map.pdf` | ✅ drawn (pgfplots), vision-verified |
| 3 | **Signals** — AUROC per signal on T4 vs A100 (0.798 / 0.143) | `fig_signal.pdf` | ✅ drawn, vision-verified |
| 4 | **Drift** — trajectories on both GPUs (propagation on T4, reversal on A100) | `fig_drift.pdf` | ✅ drawn, vision-verified |
| 5 | **Seam repair + flicker** — frontier and the independent endpoint | `fig_seam.pdf` | ✅ drawn, vision-verified |
| 6 | **Qualitative** — matched fine / coarse / repaired segments (decoded frames) | *to generate* (small T4 run, saved frames) | ⏳ P6 |
| 7 | (optional) **Online policy frontier** — cost vs quality per policy | `ddag/results/online_*/e5_online.json` | ⏳ P6 |

## Table plan

| T | Content |
|---|---|
| 1 | Preconditions and their status (cost / signal / endpoint) |
| 2 | Signals on both GPUs (direct AUROCs + CIs + permutation p) |
| 3 | Seam-repair variants (seam, fidelity, flicker) |
| 4 | Policies per GPU (cost, quality, verdict) |
| 5 | Boundary ledger (boundary → resolution → status) |

---

## Data and reproducibility

- Public Kaggle datasets: `ajjisan/ddag-experiments`, `ajjisan/jevgag-results`, and the per-experiment
  datasets — see `DATA.md`.
- Code: `https://github.com/j1s4nn/JevGAG`. Every headline number re-derives from a stored result file.

## Declarations (Elsevier requirements)

- **Competing interests:** none.
- **Funding:** none.
- **Ethics:** not applicable (no human/animal subjects).
- **Data availability:** all data are public (Kaggle + GitHub); statement included.
- **CRediT:** single author — Conceptualization, Methodology, Software, Validation, Formal analysis,
  Investigation, Data curation, Writing – original draft, Writing – review & editing, Visualization.
- **Biography:** included (Bachelor in AI, NUIST, Sep 2023 → expected Jun 2027; research interest:
  Machine Vision for Embodied Intelligence; photo supplied).
