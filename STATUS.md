# STATUS — JevGAG · DDAG

_Last updated: 26 Oct 2026._

## Programme

| Part | What it is | Where |
|---|---|---|
| **Prior work** | *Jev-Routed Temporal Granularity with Recovery and Hysteresis for Causal Video Generation* — preprint | https://doi.org/10.5281/zenodo.23395414 |
| **Part 1 · JevGAG** | the **report**: a measured audit of the three preconditions (cost · signal · endpoint) on T4 + A100 | [`jevgag/`](jevgag/) |
| **Part 2 · DDAG** | the **method**: regime-aware temporal granularity, eleven numbered experiments | [`ddag/`](ddag/) |
| **Manuscript** | one paper combining Part 1 + Part 2, Elsevier CAS format, target **CVIU** | [`paper/`](paper/) |

## Experiments

| # | Experiment | Outcome |
|---|---|---|
| 01 | intrinsic probe | **0.798** AUROC, no external oracle |
| 02 | seam repair | **−25%** seam, better fidelity |
| 03 | closed-loop evidence | free evidence **0.725**; closed loop 0.690 → 0.737 |
| 04 | escalation gate | boundary: the **cost model**, not the signal, binds |
| 05 | convergence residual | T4 **0.752**; **probe + residual 0.842**; does not transfer to A100 (0.485) |
| 06 | drift propagation | **significant on T4** (4/5 segments); **reverses on A100** |
| 07 | online policy | boundary: on A100 coarse dominates (−46% cost) |
| 08 | intermediate band | coarse 27% cheaper; the latent endpoint does not penalise it |
| 09 | pixel endpoint | confirmed: coarse cheaper **and** closer by decoded pixels |
| 10 | Jev critic | boundary: chance (0.497) — context is the limit |
| 11 | flicker endpoint | boundary: the repair trades seam for flicker |

## Boundaries and their status

- **Resolved:** weak context signals → intrinsic probe (0.798); switch seam → training-free repair (−25%).
- **Characterised:** drift propagation (real on T4, linear, not policy-predictable).
- **Open:** signal non-transfer across GPUs (probe 0.798 → 0.143); context-free Jev critic; the
  seam↔flicker trade. Each is recorded as a *boundary* and defines a next experiment.

## Infrastructure

- **Kaggle T4** (free) and **Colab A100-80GB** via the WSL `colab` CLI; ~250 compute units remaining.
- Every result is published as a **public Kaggle dataset** — see [`DATA.md`](DATA.md).
- Every headline number re-derives from a stored result file (claim-to-artifact audit).

## Links

- Site: https://j1s4nn.github.io/JevGAG/ · Repo: https://github.com/j1s4nn/JevGAG
- Prior work: https://doi.org/10.5281/zenodo.23395414
