# AUDIT_03 — Manuscript structure and scientific-claim audit

Source: `paper/main.tex` (479 lines) + `jevgag/report/main.tex` (the second, unpublished manuscript).

---

## 1. Section-by-section status

| § | Content | Status | Reason |
|---|---|---|---|
| Title | *Regime-Aware Temporal Granularity for Causal Video World Models: When Does Adaptivity Pay?* | **Complete** | Accurate and specific. |
| Authorship | Md Jisan Hossen, sole author, NUIST, ORCID present | **Complete** | — |
| Abstract | 250-word-style summary; claims: ratio inversion, 0.491/0.798/0.842, endpoint reversal, empty adaptive region, −25 % seam, three boundaries | **Present but incomplete** | No sample sizes, no CIs, and the flicker number is absent (correctly) but the **cost ratio quoted (none) is regime-ambiguous**. |
| Keywords | 7 | **Complete** | Elsevier prefers ≤ 6; minor. |
| 1 Introduction | motivation, prior work, 4 contributions | **Complete** | Clear. Missing: an explicit *research-question* statement and a statement of what would falsify the central claim. |
| 2 Related Work | 5 paragraphs, 38 refs | **Complete** | Good coverage. Weak on: **dynamic compute allocation for diffusion specifically** (no per-step/early-exit diffusion baseline is named as the closest competitor), and no comparison table. |
| 3 Background | preconditions, matched design, label, the report | **Inconsistent with the evidence** | Describes "the report" without disclosing that a **second full manuscript exists in the same repository** (self-overlap). The label formula `y = 1[ℓ(q_c) > ℓ(q_f)]` is written with `ℓ(q)` where `q` is already defined as a quality *functional* — a notational error (see AUDIT_07). |
| 4 Method | regime map (Eq. 1), intrinsic probe, convergence residual (Eq. 2), calibrated gating, seam repair (Eq. 3) | **Present but incomplete** | Eq. 1 defines the policy but **not the objective** the policy optimises; the gating cost algebra omits the probe/prefill cost; the seam-repair equation has no `α` selection rule. |
| 5 Protocol | backbone, hardware, prompts, statistics, endpoints | **Incomplete** | **No reproducibility detail**: no backbone checkpoint IDs, no commit hash, no prompt list, no seed value, no software versions, no per-experiment sample sizes in one place. Prompts are described only by stratum counts. |
| 6 Results | cost, signal, seam, drift, regime map, qualitative | **Inconsistent with the evidence** | §6.1 cost numbers do not match the repository table (R-1/R-4); §6.3 flicker means are wrong (AUDIT_10); §6.4 mixes n = 32 and n = 8 without saying so in the prose; §6.5 asserts optimality without an objective. |
| 7 Boundaries and guidance | boundary ledger + recipe | **Complete** | This is the paper's strongest editorial device. |
| 8 Conclusion | restates the findings | **Present but incomplete** | Does not state the **scope of generalisation** (one backbone, two GPUs, one schedule family) explicitly as a limitation. |
| Declarations | competing interests, funding, data availability, CRediT | **Complete** | Ethics statement absent (Elsevier expects it or an explicit n/a). |
| Biography | present with photo | **Complete** | Not required by Elsevier; harmless. |
| References | 38 entries | **Complete** | Generated from arXiv metadata; 0 missing, 0 uncited (checked). |
| Limitations | — | **MISSING as a section** | Boundaries are scattered in §7; Elsevier/CVIU papers conventionally carry an explicit *Limitations / threats to validity* subsection. |
| Ablations | — | **MISSING** | No ablation of the probe (block choice, layer, pooling), of the gating threshold, or of the seam-repair α beyond the α-sweep table. |
| Compute/energy reporting | — | **MISSING** | No GPU-hours or energy statement (increasingly expected). |
| Appendix / supplementary | — | **MISSING** | No appendix; per-experiment details live only in the repo. |

---

## 2. The four questions the audit was asked to test

### 2.1 Does the paper explain why a *fixed* policy can beat adaptive routing?
**Partially.** §6.5 says fixed modes are "dominant" and §4.1/Eq. 1 defines the regions, but the paper
never states the **objective function** under which fixed beats adaptive. Without it, "optimal" is
undefined (`AUDIT_07`, R-7). **Action:** define the objective (e.g. minimise DiT seconds subject to
quality not worse than fixed-fine) and show the fixed policy is optimal *for that objective*.

### 2.2 Is a regime map useful even when the adaptive region is empty?
**Not argued.** The paper asserts the map is a "design rule" but does not defend the value of a
negative map. **Action:** add 3–4 sentences: the map is a *pre-deployment audit*; it tells a
practitioner not to build a router, and it predicts where a router *would* pay (a different mode pair
or GPU). This is the paper's most defensible framing and should be made explicit.

### 2.3 How does the method differ from the prior JevGAG study?
**Weakly.** §3 summarises the report, but the paper never states the delta crisply. **Action:** add a
one-paragraph "Relationship to the prior study" in §1: the prior study measured the preconditions on
one GPU with an external oracle; **this paper** (a) extends to two GPUs, (b) replaces the external
oracle with an intrinsic signal, (c) adds the convergence residual, (d) adds the seam repair, and
(e) derives the regime map. None of these appear in the prior study.

### 2.4 Is the contribution novel relative to adaptive inference / early exit / speculative decoding?
**Under-argued.** §2 lists related work but never states the **novelty delta** explicitly.
**Action:** add an explicit positioning sentence: the object (temporal chunking of a *generative
video* model), the signal (prefill activations, not classifier confidence), and the finding (the
regime is often empty) are each distinct from prior adaptive-inference work.

### 2.5 What generalises beyond the tested hardware and prompts?
**Not stated.** **Action:** add an explicit generalisation paragraph: the *method* (measure r and Δ;
audit the signal per GPU) generalises; the *numbers* do not. The probe inversion is the concrete
evidence that nothing here should be assumed to transfer.

---

## 3. Overlap audit vs `jevgag/report/main.tex`

| Aspect | Finding |
|---|---|
| Titles | Different. New: *Regime-Aware…*; report: *Jev-Grounded…* |
| Shared content | Both describe the three preconditions, the matched design, the break-even law, the seam, hysteresis/recovery |
| Shared text | No verbatim sentence reuse detected by inspection |
| Shared figures | **None** — the new paper uses new TikZ/pgfplots figures; the report uses matplotlib figures |
| Risk | **The report is a complete manuscript that is not published.** If it is submitted elsewhere, §3 of the new paper duplicates its contribution summary. **Decision needed** (AUDIT_14, F-2) |

---

## 4. Missing standard sections (with the scientific reason, not template-filling)

| Section | Why it is needed here |
|---|---|
| **Limitations / threats to validity** | The paper's own thesis is that everything is regime-dependent; a limitations section is therefore *scientifically load-bearing*, not boilerplate. Must state: one backbone, one schedule family, two GPUs, small n (8 on A100), single seed. |
| **Ablations** | The probe's block choice and the gating threshold are design decisions that a reader cannot evaluate without an ablation. The data already exist (`probe_results.json` has a 30-block sweep; `e5_online.json` has a τ frontier) — **no new computation needed**. |
| **Reproducibility statement / protocol table** | CVIU expects enough detail to reproduce. All values exist in the repo; they simply are not in the paper. |
| **Ethics statement** | Elsevier requires it or an explicit "not applicable". |
| **Objective definition for the regime map** | Without it the central claim is unfalsifiable. |
