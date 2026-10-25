# STATUS.md — JevGAG master tracker

> **Read this first in any new session.** It is the single source of truth for the plan,
> current phase, what is done, what is waiting, and how to resume.
> Owner: Md Jisan Hossen · Updated: 2026-10-25 (04:35 UTC).

## Goal
Deliver the second paper **“JevGAG: Jev-Grounded Adaptive Temporal Granularity for Causal Video
Generation”** — an off-device, image-grounded System One oracle (Jev 1.14) for adaptive temporal
granularity, with a matched hardware × operating-point break-even law, recovery+hysteresis control,
and a validated independent endpoint. Reuse the prior project (`../idea_05_.../04_Experiment`)
**read-only**. Only **RQ2 is a negative**; everything else is a positive result.

## Phase plan (phase-by-phase)

| Phase | Scope | Status |
|---|---|---|
| **P0 Setup** | repo + SEO + Pages + credentials + Colab CLI control | ✅ **DONE** |
| **P1 0-GPU core** | E1 break-even · E2 image-vs-text signal · E5 image endpoint · E7 confidence | ✅ **DONE** |
| **P2 Matched re-analysis** | E3 matched A100 online · E4 controller+seam · E6 T4 boundary | ✅ **DONE** |
| **P3 Claim audit** | E9 claim-to-artifact audit (18/18 PASS) | ✅ **DONE** |
| **P4 GPU experiments** | E8 recalibration on **A100 (done, negative)** + **Kaggle T4 (done, negative)** | ✅ **DONE** |
| **P5 Figures & tables** | 5 figures (E1/E2/E5/E7/E8) + all audit tables | ✅ **DONE** |
| **P6 Manuscript** | `paper/main.tex` full draft (all E1–E9) + `references.bib` | ✅ **DRAFT** |
| **P7 Release** | Zenodo DOI, git tag v0.1.0, Pages polish, final PRs | ⏳ **WAITING** |

## Experiments inventory

| E | Experiment | Status | Artifact |
|---|---|---|---|
| E1 | Matched break-even predictor | ✅ | `results/e1_breakeven/`, `fig_e1_breakeven.png` |
| E2 | Image-Jev vs text-Jev signal (n=80 paired) | ✅ (negative) | `results/e2_signal/`, `fig_e2_signal.png` |
| E3 | Matched A100 online factorial | ✅ | `results/e3_online/` |
| E4 | Recovery+hysteresis + seam (T4 & A100) | ✅ | `results/e4_controller/` |
| E5 | Image-Jev independent endpoint (ρ=+1.000) | ✅ | `results/e5_endpoint/`, `fig_e5_endpoint.png` |
| E6 | Matched T4 boundary (derived) | ✅ | `results/e6_t4_boundary/` |
| E7 | Confidence-gated routing (neutral) | ✅ | `results/e7_confidence/`, `fig_e7_frontier.png` |
| E8 | Stability-gated recalibration | ✅ (negative, A100 + T4) | `results/e8_a100/`, `results/e8_recalibration/` |
| E9 | Claim-to-artifact audit (18/18 PASS) | ✅ | `results/e9_audit/` |

## Achieved (summary)
- Break-even law over {T4,A100}×{fine s1/s2/s4}×{coarse s1/s2/s4}; resolves the prior paper's
  coarse-operating-point confound (T4 negative was an **operating-point** boundary, not a signal failure).
- Honest negative: image-grounded Jev does **not** beat text Jev as a routing *signal* (Δmotion −0.195, P=0.006).
- Strong positive: image-grounded Jev is an excellent independent *endpoint* (Spearman **+1.000** vs CLIP).
- Matched A100 online: adaptive result flips with the coarse operating point alone (21% vs 92% fine).
- Recovery+hysteresis cuts switches with no worse regret on both GPUs; a switch **seam** exists.
- 18/18 claim-to-artifact audit PASS. Repo + SEO + Pages live; 7 PRs merged, PR #6 open.

## Waiting / in flight
- **None running.** All E1–E9 complete (E8 negative on both GPUs).
- A100 VM `jevgag` **stopped** (balance ~261.8 units, 0 units/hr).

## Next actions (in order)
1. E8 figure + fold E8 into the E3/E4 figures pass.
2. Write the manuscript into `paper/` (all E1–E9 sections).
3. Release: git tag `v0.1.0`, Zenodo-ready bundle, Pages update.

## Environment & credentials
**All keys, accounts, expiry and re-auth steps live in**
`C:\Users\jisan\.config\kilo\secrets\ACCOUNTS.md` (Jev, Kaggle, GitHub, Colab, Windows vs WSL, proxy).
Never commit secrets. Key reminders:
- **Colab CLI runs in WSL2** and needs `http_proxy=http://127.0.0.1:7890`.
- **Kaggle/GitHub/Jev run on Windows.**
- A100 balance ≈ 272 units (~40 h @ 6.77/h). Stop idle sessions (`colab stop -s jevgag`).

## How to resume in a new session
1. Read `STATUS.md` (this file), then `FINDINGS.md`, then `REFERENCE.md`.
2. Read `C:\Users\jisan\.config\kilo\secrets\ACCOUNTS.md` for credentials/env.
3. Check GitHub PRs: `gh pr list -R j1s4nn/JevGAG --state all`.
4. Check running jobs: A100 (`colab sessions`, `background_process list`), Kaggle (`kaggle kernels status`).
5. Continue from the first non-✅ row in the phase table.

## Repo / site
- Repo: https://github.com/j1s4nn/JevGAG (main green) · Site: https://j1s4nn.github.io/JevGAG/
- Layout: `paper/` `src/` `scripts/` `config/` `results/` `colab/` `jobs/` `docs/` `assets/`
