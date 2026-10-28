# JevGAG — Report

**Jev-Grounded Adaptive Temporal Granularity for Causal Video Generation: An Audit of the Cost, Signal
and Endpoint Preconditions**

Md Jisan Hossen · Department of Artificial Intelligence, NUIST, Nanjing, China ·
jisanahammed1952@gmail.com · ORCID [0009-0004-9571-1450](https://orcid.org/0009-0004-9571-1450)

> This is the **GitHub-readable rendering** of the report. The LaTeX source and the compiled PDF are in
> this folder ([`main.tex`](main.tex), [`main.pdf`](main.pdf)); the audited result files are in
> [`../results/`](../results/) and the public bundle is on Kaggle (see [`../README.md`](../README.md)).

---

## Abstract

Adaptive temporal granularity — deciding when a causal video generator should produce one frame at a time
(**fine**) versus several frames per chunk (**coarse**) — rests on four preconditions: a real
mode-dependent cost gap, a pre-generation signal, an endpoint independent of the compared modes, and an
oracle calibrated well enough to gate on. This report audits all four with an off-device,
image-grounded System One oracle (**Jev 1.14**) on a matched factorial over two accelerators (Tesla T4 and
A100-80GB). We contribute (i) a **break-even law** conditional on both the hardware and the operating
point, which resolves a confound in the earlier audit of this problem; (ii) a paired test showing that an
**image**-grounded oracle is a **worse routing signal** than a text-grounded one; (iii) a validated
independent **endpoint** — the image-grounded oracle agrees with CLIP subject consistency at
**Spearman ρ = +1.000**; and (iv) **two honest negatives** (confidence gating is neutral; stability-gated
recalibration is worse on both GPUs). Every headline number is re-derived from a stored artifact
(**18/18 claim-to-artifact checks pass**).

**Keywords:** adaptive granularity · temporal granularity · causal video generation · video world models ·
autoregressive diffusion · off-device oracle · System One · efficient inference

---

## 1. Introduction

Causal (autoregressive) video generators can trade computation for quality by changing their **temporal
granularity**: frame-wise generation is accurate but expensive, while chunk-wise generation is cheaper on
some hardware but drifts. A natural system spends fine computation only on hard segments, deciding
*before* each segment is generated. That system is viable only if four things hold — and this report
measures each.

**Contributions.**

1. **A matched break-even law** (§4.1) over `{T4, A100} × {fine s1/s2/s4} × {coarse s1/s2/s4}`. Coarse is
   cheaper **iff** the measured per-forward ratio falls below the schedule break-even, and the same pair
   can sit on **opposite sides** of that inequality on the two GPUs. This resolves a confound in the
   earlier audit, which compared an *expensive* coarse mode on T4 with a *cheap* one on A100.
2. **A paired modality test** (§4.2). On 80 matched segments an image-grounded oracle is a **worse**
   routing signal than a text-grounded one (ΔAUROC −0.110 to −0.195).
3. **A validated independent endpoint** (§4.5). The same image-grounded oracle, used as a quality
   endpoint, agrees with CLIP subject consistency at **ρ = +1.000**.
4. **Two honest negatives** (§4.7, §4.8): confidence-gated routing is neutral; stability-gated
   recalibration is worse on both GPUs.

**Relationship to the prior study.** The prior preprint (`10.5281/zenodo.23395414`) framed the problem
and reported a conditional trade-off on one GPU. This report is the audit that follows: it measures the
preconditions on two GPUs with a matched design, tests the *modality* of the oracle, and validates the
endpoint. The companion method (DDAG) is a separate artifact; no text or figure is reused.

---

## 2. Related work

- **Causal video generation.** Streaming generators roll out chunk by chunk; closing the teacher-forcing
  / self-rolling gap is central (Self Forcing; Causal Forcing). Few-step causal distillation makes the
  setting interactive (Causal-rCM, our backbone); we treat the model as fixed and study its granularity.
- **Adaptive computation.** Introduced for recurrent nets (ACT) and early exiting (BranchyNet; ensemble
  internal classifiers). In generative video, computation is allocated by **step** (DSA), by **chunk
  shape** (Flex-Forcing), by pruning/routing (PARE), and by layer reuse (Foresight). Our object is the
  **temporal chunk size** of a fixed backbone, driven by an **off-device** oracle.
- **Evaluation.** Video generators are evaluated along many axes (VBench, VBench-2.0). We adopt CLIP
  subject consistency and a captioner (BLIP) as independent endpoints.
- **Decision models.** System One models return typed, calibrated answers. Independent evidence suggests
  pixel-only quality prediction with such models is unreliable (JEVQA) — consistent with our signal
  result.

---

## 3. Preconditions and method

| Precondition | Statement |
|---|---|
| **P1 — cost** | coarse must be cheaper than fine **on the deployment hardware and at the chosen operating point** |
| **P2 — signal** | a quantity computable **before** a segment is generated must identify coarse-unsafe segments |
| **P3 — endpoint** | quality must be certified by a measure **independent** of the compared modes |
| **P4 — calibration** | the oracle's confidence must be informative enough to gate on |

**Cost model.** With `t_c`, `t_f` the seconds per single forward and `NFE` the forward evaluations per
segment, coarse is cheaper iff

```
t_c / t_f  <  NFE_f / NFE_c  ≡  ρ_be        (Eq. 1)
```

i.e. the **measured per-forward ratio** must fall below the **schedule break-even**.

**Oracle.** Jev 1.14, queried off-device. A *text* state passes a caption plus motion statistics; an
*image* state passes the last three context frames as assets in a `multipart/form-data` request with the
header `TypeSafe-Danger-Zone-Images: true`. The oracle returns typed answers (difficulty score, drift
risk, motion class) and a confidence.

**Backbone, modes, design.** Backbone: **Causal-rCM Wan2.1-T2V-1.3B** (commit `ed3cb14`). Modes: fine
`c1-1` (s steps), cheap coarse `c3-3 s1`, expensive coarse `c3-3 s4`, and `hold` (reuse the previous
latent). For every prompt and segment we generate **both** modes from the same boundary and the same
noise, so the only difference is the generator; a segment is **coarse-unsafe** when the coarse segment is
farther from the four-step fine reference than the fine segment. Costs are wall-clock DiT seconds with
device synchronisation.

---

## 4. Results

### 4.1 E1 — Break-even is hardware- and operating-point-conditional

![E1 break-even](../figures/fig_e1_breakeven.png)

| GPU | fine / coarse | NFE f/c | ρ_be | measured ratio | coarse cheaper |
|---|---|---|---|---|---|
| T4 | s1 / s1 | 19.0/8.33 | 2.28 | 2.694 | no |
| T4 | s1 / s4 | 19.0/21.33 | 0.891 | 2.197 | no |
| T4 | s2 / s1 | 34.0/8.33 | 4.08 | 3.301 | **yes** |
| T4 | s2 / s2 | 34.0/12.67 | 2.684 | 2.969 | no |
| T4 | s4 / s1 | 64.0/8.33 | 7.68 | 3.841 | **yes** |
| T4 | s4 / s4 | 64.0/21.33 | 3.0 | 3.133 | no |
| A100 | s1 / s1 | 19.0/8.33 | 2.28 | 1.561 | **yes** |
| A100 | s2 / s1 | 34.0/8.33 | 4.08 | 1.95 | **yes** |
| A100 | s2 / s2 | 34.0/12.67 | 2.684 | 1.744 | **yes** |
| A100 | s4 / s4 | 64.0/21.33 | 3.0 | 1.84 | **yes** |

On T4 the cheap coarse `c3-3 s1` beats fine only when fine uses ≥ 2 steps, and the expensive `c3-3 s4` is
**never** cheaper; on A100 the cheap coarse beats all three fine modes. The same pair (fine s2, coarse
s2) therefore lies on **opposite sides** of Eq. 1 on the two GPUs (2.97 vs break-even 2.68 on T4; 1.74 on
A100). **This is the confound resolved.**

### 4.2 E2 — Image-grounded Jev does not beat text Jev as a signal

![E2 signal audit](../figures/fig_e2_signal.png)

| signal | text-1.13 | text-1.14 | image-1.14 | hybrid-1.14 |
|---|---|---|---|---|
| Jev difficulty | 0.788 | 0.777 | 0.667 | 0.687 |
| Jev drift | 0.738 | 0.755 | 0.647 | 0.656 |
| Jev motion | 0.703 | 0.706 | 0.510 | 0.568 |
| warp residual | 0.586 | 0.586 | 0.586 | 0.586 |
| motion stat | 0.607 | 0.607 | 0.607 | 0.607 |
| complexity | 0.368 | 0.368 | 0.368 | 0.368 |

Paired ΔAUROC (image − text): difficulty **−0.110**, drift **−0.108**, motion **−0.195**
(95 % CI [−0.356, −0.030], P(image>text) = **0.006**). Text-1.13 ≈ text-1.14, so the deficit is the
**modality**, not the model version. Honest negative.

### 4.3 E3 — The adaptive outcome flips with the operating point alone

Same A100, same 12 prompts, **only the coarse step differs**:

| policy | expensive s4 DiT (s) | s4 refL1 | s4 fine% | cheap s1 DiT (s) | s1 refL1 | s1 fine% |
|---|---|---|---|---|---|---|
| fixed fine | 3.81 | 0.2812 | 1.00 | 3.95 | 0.2832 | 1.00 |
| fixed coarse | 4.43 | 0.2874 | 0.00 | 2.55 | 0.2728 | 0.00 |
| fixed hold | 0.76 | 0.2824 | 0.00 | 0.77 | 0.2674 | 0.00 |
| random 50 % | 4.12 | 0.2868 | 0.50 | 3.24 | 0.2789 | 0.50 |
| adaptive warp | 3.87 | 0.2820 | 0.92 | 2.85 | 0.2733 | 0.21 |

With the **expensive** coarse the online policy is forced to **92 % fine** and cannot beat fixed fine
(4.43 s vs 3.81 s); with the **cheap** coarse it uses **21 % fine** and is both cheaper and closer to the
reference (2.55 s vs 3.95 s).

### 4.4 E4 — Recovery, hysteresis and the switch seam

| GPU | controller | switches | fine frac | AUH refL1 | seam ratio |
|---|---|---|---|---|---|
| T4 | threshold only | 0.50 | 0.50 | 0.3133 | 1.30 |
| T4 | full (hyst.+rec.) | 0.38 | 0.53 | 0.3116 | 1.18 |
| T4 | full, noisy | 0.62 | 0.66 | 0.3150 | 1.53 |
| A100 | threshold only | 0.50 | 0.50 | 0.2767 | 2.20 |
| A100 | full (hyst.+rec.) | 0.42 | 0.52 | 0.2772 | 1.98 |
| A100 | full, noisy | 0.50 | 0.56 | 0.2718 | 1.75 |

Adding a hysteresis band and a forced-recovery window **reduces switches** on both GPUs without worsening
the area-under-horizon regret, and a switch **seam** is measurable (switch-adjacent L1 > interior L1:
ratio 1.18 on T4, 1.98 on A100).

### 4.5 E5 — Image-grounded Jev is a validated independent endpoint

![E5 endpoint](../figures/fig_e5_endpoint.png)

| policy | Jev consistency | Jev same-subject | CLIP subj. cons. |
|---|---|---|---|
| reference | 3.461 | 0.894 | 0.990 |
| fixed fine | 2.151 | 0.416 | 0.984 |
| adaptive warp | 2.081 | 0.410 | 0.980 |
| random 50 % | 1.978 | 0.286 | 0.965 |
| alternating | 1.626 | 0.268 | 0.959 |
| fixed coarse | 1.775 | 0.230 | 0.949 |

Across six policies and 48 videos, **Spearman ρ = +1.000** (same-subject) and **+0.943** (consistency).
This is the report's strongest positive: the image oracle cannot *drive* the decision (E2) but it can
*certify* it — exactly what P3 requires.

### 4.6 E6 — The T4 negative is an operating-point boundary, not a signal failure

- T4 coarse `c3-3 s1` vs fine `c1-1 s1`: cheaper = **False** (DiT saved **−4.57 s**)
- T4 coarse `c3-3 s1` vs fine `c1-1 s2`: cheaper = **True** (+7.01 s)
- T4 coarse `c3-3 s1` vs fine `c1-1 s4`: cheaper = **True** (+29.70 s)

With fine `s1` the cheap coarse is *slower*, so adaptive routing cannot save wall-clock time regardless
of the signal.

### 4.7 E7 — Confidence gating is neutral

![E7 confidence frontier](../figures/fig_e7_frontier.png)

| policy | 25 % fine | 50 % fine | 75 % fine |
|---|---|---|---|
| random | 0.3044 | 0.2957 | 0.2881 |
| difficulty only | 0.2968 | 0.2875 | 0.2781 |
| confidence gated | 0.2964 | 0.2875 | 0.2794 |
| oracle | 0.2734 | 0.2734 | 0.2734 |

Confidence-gated minus difficulty-only: **|Δ| ≤ 0.0013** at every budget. Honest negative.

### 4.8 E8 — Stability-gated recalibration does not help

![E8 recalibration](../figures/fig_e8_recalibration.png)

| A100 (n = 12) | refL1 | | T4 (n = 8) | refL1 |
|---|---|---|---|---|
| threshold only | 0.2767 | | fixed coarse | 0.3186 |
| full | 0.2772 | | fixed fine | 0.2998 |
| recalibrate | 0.2884 | | adaptive warp | 0.2949 |
| | | | recalibrate | 0.3151 |

A100: recalibrate − full = refL1 **+0.0112** at DiT **+2.04 s**. T4: recalibrate − adaptive_warp =
refL1 **+0.0202** at DiT **+12.73 s**. Honest negative.

### 4.9 E9 — Claim-to-artifact audit

| experiment | claims audited | example | result |
|---|---|---|---|
| E1 | 3 | T4 `c3-3 s4` never cheaper | PASS |
| E2 | 5 | motion deficit significant (P = 0.006) | PASS |
| E3 | 3 | cheap coarse cheaper online on A100 | PASS |
| E4 | 4 | switches 0.50 → 0.38 (T4) | PASS |
| E5 | 2 | endpoint ρ = +1.000 | PASS |
| E7 | 1 | confidence gating neutral | PASS |
| **total** | | | **18/18 PASS** |

---

## 5. Discussion and limitations

**What the report establishes.** The image-grounded oracle is a *poor routing signal* but an *excellent
independent endpoint*; the break-even law (Eq. 1) is the reusable engineering tool; and the adaptive
verdict is conditional on hardware **and** operating point, which explains the earlier confound.

**Limitations.**
1. **One backbone** — a single causal distilled model; the numbers do not transfer to other generators.
2. **Two accelerators, one schedule family** — T4 and A100-80GB only.
3. **Small samples** — the signal test uses 80 paired segments; the recalibration arms use 8–12 prompts.
4. **Off-device dependency** — the oracle is a versioned external service (pinned Jev 1.14); the
   companion method removes this dependency.
5. **One modality comparison** — text vs image; audio/depth untested.
6. **No step-allocation or chunk-shape baseline** — the closest published alternatives are discussed but
   not re-implemented.

**Threats to validity.** E5 rests on 48 videos from six policies, all on T4; E8 is measured on a single
seed.

---

## 6. Conclusion

An off-device, image-grounded System One model can **certify** adaptive temporal granularity even when it
cannot **drive** it, and the cost side obeys a matched break-even law over hardware and operating point.
The routing signal remains the open problem, and we report the two failed policies openly rather than
tuning them to look better.

---

## Declarations

**Competing interests:** none · **Funding:** none · **Ethics:** not applicable · **Data availability:**
all code, figures and audited result files are public at https://github.com/j1s4nn/JevGAG, and the result
bundles are published as public Kaggle datasets · **CRediT:** Md Jisan Hossen — Conceptualization,
Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Visualization,
Writing — original draft, Writing — review and editing.

## References

See [`references.bib`](references.bib) (17 entries, all cited). Key sources: the prior preprint
(zenodo.23395414); Causal-rCM; Self Forcing; Causal Forcing; ACT; BranchyNet; ensemble internal
classifiers; DSA; Flex-Forcing; PARE; Foresight; CLIP; BLIP; VBench; VBench-2.0; JEVQA; Jev in the Wild.
