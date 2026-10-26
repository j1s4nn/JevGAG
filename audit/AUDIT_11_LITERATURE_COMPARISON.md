# AUDIT_11 — Literature comparison and positioning

All sources below were verified against authoritative arXiv metadata (titles, authors, years) during this
audit; no citation was invented and no claim about another paper's contents is made without having read
its abstract.

---

## 1. Closest prior work and the actual delta

| Theme | Closest works (verified) | What they do | What this paper does differently |
|---|---|---|---|
| **Causal video distillation** | Causal-rCM `2606.25473`; Causal Forcing `2602.02214`; Causal Forcing++ `2605.15141`; Self Forcing `2506.08009`; minWM `2605.30263`; MAGI-1 `2505.13211` | Train/distil few-step causal generators; improve quality and speed of the *generator* | **Treats the generator as fixed** and studies its *temporal granularity* as a control variable. Orthogonal: the method could be layered on any of these. |
| **Adaptive compute (classic)** | ACT `1603.08983`; BranchyNet `1709.01686`; SkipNet `1711.09485` | Choose how much compute to spend per input, from internal signals | Same *idea*, different *axis* (temporal chunking in a generative model) and, crucially, a **negative result about when the idea pays**. |
| **Adaptive compute in video diffusion** | DSA `2606.04432` (step allocation); Flex-Forcing `2607.03509` (flexible chunking); PARE `2605.27336` (pruning+routing); Foresight `2506.00329` (layer reuse) | Allocate steps/layers/patches per token or per frame | **DSA and Flex-Forcing are the closest competitors.** DSA reallocates *denoising steps*; Flex-Forcing learns *flexible chunk shapes*; neither reports a **regime analysis of when adaptation pays**, nor an *intrinsic pre-generation* probe, nor a *seam repair*. |
| **Cache-based acceleration** | Flow Caching `2602.10825`; Motion-Aware Caching `2605.01725`; In-Flight KV `2609.32540`; TRAC `2610.02779` | Reuse computation across frames | Complementary; orthogonal to granularity. |
| **Representation-level adaptivity** | AdapTok `2505.17011`; EVATok `2603.12267`; DLFR-VAE `2502.11897`; VFRTok `2505.12053` | Change the *tokeniser* / latent frame rate | Different object (representation vs inference-time mode choice). Worth citing as the "other way" to get temporal adaptivity. |
| **Internal probes / early exit** | Ensemble internal classifiers `2105.13792`; LGViT `2308.00255`; ADEPT `2601.03700`; ConsistentEE `2312.11882`; BERT Loses Patience `2006.04152` | Predict difficulty from internal features | Same mechanism, **new target** (generation-mode safety) and a **new finding** (the probe *inverts* across GPUs). |
| **Speculative decoding** | DFlash `2602.06036`; PEARL `2408.11850`; multimodal spec survey `2608.20743` | Draft-and-verify over tokens | Structurally analogous but over tokens; the paper's escalation gate is the temporal-chunk analogue. |
| **Video evaluation** | VBench `huang2024vbench`; VBench-2.0 `2503.21755`; artefact-aware `2601.20297`; compression-consistency `2605.18378` | Standardised video metrics | The paper adopts a **multi-endpoint** protocol and shows one endpoint (flicker) **contradicts** the seam metric — a direct, useful instance of why VBench-style multi-axis evaluation matters. |
| **Decision models (Jev)** | Jev in the Wild `2609.30216`; Releasing the Inner Jev `2610.17205`; LLM-as-Jev `2610.02076`; **JEVQA `2609.24395`** | Use/produce decision models | The paper **reproduces JEVQA's negative** (pixel-only quality prediction fails) in a new setting, and shows the same family works as a *text* signal (report) — a nuanced, citable contribution. |

---

## 2. Novelty verdict

| Candidate contribution | Novel? | Justification |
|---|---|---|
| Routing *temporal granularity* (not steps/tokens/layers) | **Yes** | No verified work routes the frame-chunk size of a causal generator. DSA (steps) and Flex-Forcing (chunk shape, learned offline) are the nearest; neither is a pre-generation router. |
| **Regime map** as a first-class result | **Yes** | No verified work reports a *decision map* for when adaptive inference pays in generative video; the literature reports wins. |
| **Intrinsic prefill probe** for mode safety | **Yes** | Internal probes are used for classification/LLM difficulty; use for *generation-mode safety* is new. |
| **Convergence residual** | **Yes but currently mis-stated** | The *idea* is new; the *result* is currently sign-inverted (AUDIT_07 M-1) and must be corrected before it can be claimed. |
| **Training-free seam repair** | **Yes** | Seam repair exists for stitching/inpainting/KV anchors, not for the boundary produced by a *mode switch*. |
| **Signal non-transfer across GPUs** | **Yes, and it is the most citable finding** | A probe at 0.798 on one GPU is 0.143 on another. This is a concrete, reproducible cautionary result for the adaptive-inference community. |

**Positioning recommendation.** Lead with the **regime map + non-transfer of signals**, because those are
the parts that no verified prior work contains and that survive the current numerical issues. The
convergence residual must be fixed (M-1) before it can be used as a headline.

---

## 3. What the manuscript should compare against but does not

| Missing comparison | Why it matters | Cost |
|---|---|---|
| **A step-allocation baseline** (DSA-style) | A reviewer will ask "why granularity and not steps?" | Moderate — a different axis; may need a GPU run |
| **A token/chunk-shape baseline** (Flex-Forcing-style) | The closest neighbour | High — requires reimplementing a learned chunker |
| **A trivial "always coarse" and "always fine" baseline** | Already present (Table 2) ✅ | none |
| **A random router** | Establishes the value of the signal | **Zero GPU** — computable from `features_*.npz` |
| **An oracle router** | Upper bound; shows how much a perfect signal could buy | **Zero GPU** — `e5_online.json` already contains `oracle_route` |
| **A calibration curve for the probe** | Reviewers expect calibration, not only AUROC | **Zero GPU** — probabilities can be re-derived |

**Highest-value, zero-compute additions:** the **random-router** and **oracle-router** baselines (both
already in `e5_online.json` as `oracle_route`), and a **calibration/reliability diagram** for the probe.
