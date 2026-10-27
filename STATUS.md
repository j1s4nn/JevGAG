# STATUS — JevGAG · DDAG

_Last updated: 27 Oct 2026 (post-audit revision)._

## Programme

| Part | What it is | Where |
|---|---|---|
| **Prior work** | *Jev-Routed Temporal Granularity with Recovery and Hysteresis for Causal Video Generation* — preprint, **cited as prior work** | https://doi.org/10.5281/zenodo.23395414 |
| **Part 1 · JevGAG** | the **report**: a measured audit of the three preconditions (cost · signal · endpoint) on T4 + A100. Unpublished repository artifact. | [`jevgag/`](jevgag/) |
| **Part 2 · DDAG** | the **method**: regime-aware temporal granularity, twelve numbered experiments | [`ddag/`](ddag/) |
| **Manuscript** | one paper combining Part 1 + Part 2, Elsevier CAS format, target **CVIU** | [`paper/`](paper/) |
| **Audit** | a repository-wide scientific audit (15 reports) and its applied corrections | [`audit/`](audit/) |

## Audit outcome (27 Oct 2026)

A full audit (`audit/AUDIT_00 … AUDIT_14`) was run at commit `0c725c0`. **Four P0 findings were verified
and corrected**, all from stored result files with no new computation:

| ID | Finding | Correction |
|---|---|---|
| **F-1** | The convergence residual in Eq. (2) was **sign-inverted**: the direct AUROC of `v_S/v_1` is **0.240**, not 0.752; 0.752 came from a single-feature logistic that re-learns the sign, and the 0.842 was in no result file | residual redefined as `κ = v_1/v_S` (direct **0.760**); the multi-feature model is now **stored** (`ddag/results/signal_combined.json`) and re-derives to **0.842** (T4) and **0.833** (A100) — it is the only signal that transfers |
| **F-2** | Two flicker means in Table 1 were wrong (`none` 0.0213, `fine` 0.0254) | corrected to **0.0148** and **0.0189**; the trade-off is now *stronger* (the repaired value is the highest of the three) |
| **F-3** | The regime map asserted "optimal" with no objective | an explicit constrained objective was added (Eq. 2); the oracle-router bound is now reported |
| **F-4** | The cost numbers 3.01/1.60/2.33 belong to the **prior study** | re-attributed; the paper now uses the matched matrix (π = 2.97 / 1.74; break-even 2.68) and the per-segment cost ratio r |

P1 items also applied: limitations/generalisation/ablations sections, a protocol table with per-experiment
sample sizes, the decoded-pixel endpoint result (exp 09) folded in, an ethics statement, keyword count,
figure/table captions, and the F-1/F-2 corrections propagated to the README, the site and the dataset
READMEs.

## Experiments

| # | Experiment | Outcome |
|---|---|---|
| 01 | intrinsic probe | **0.798** direct AUROC (T4), no external oracle |
| 02 | seam repair | **−25%** seam, better fidelity |
| 03 | closed-loop evidence | free evidence 0.725; closed loop 0.690 → 0.737 |
| 04 | escalation gate | boundary: the **cost model**, not the signal, binds |
| 05 | convergence residual | **0.760** direct (T4); probe + residual **0.842 / 0.833**; the reciprocal is inverted (0.240) |
| 06 | drift propagation | **significant on T4** (4/5 segments); **reverses on A100** |
| 07 | online policy | boundary: on A100 coarse dominates (−46% cost); the oracle router adds nothing |
| 08 | intermediate band | coarse 27% cheaper; the latent endpoint does not penalise it |
| 09 | pixel endpoint | confirmed: coarse cheaper **and** closer by decoded pixels |
| 10 | Jev critic | boundary: chance (0.497) — context is the limit |
| 11 | flicker endpoint | boundary: the repair trades seam for flicker (0.0218 > both) |
| 12 | qualitative | modes differ in content; no endpoint ranks coarse worse |

## Infrastructure

- **Kaggle T4** (free) and **Colab A100-80GB** via the WSL `colab` CLI; ~250 compute units remaining.
- Every result is published as a **public Kaggle dataset** — see [`DATA.md`](DATA.md).
- The claim-to-artifact audit (`jevgag/results/e9_audit`) covers the **report's** claims (18/18); the
  method's claims are verified in `audit/AUDIT_06_CLAIM_EVIDENCE_LEDGER.csv`.

## Links

- Site: https://j1s4nn.github.io/JevGAG/ · Repo: https://github.com/j1s4nn/JevGAG
- Prior work: https://doi.org/10.5281/zenodo.23395414
- Audit: [`audit/`](audit/)
