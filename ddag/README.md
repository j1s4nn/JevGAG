# DDAG — method (Part 2 of 2)

**Deadline- and Drift-Aware Adaptive Temporal Granularity** for causal video world models — the *method*
built on the report in [`../jevgag/`](../jevgag/).

DDAG turns the report's three preconditions into a **deployable rule**: measure the regime, then choose a
granularity policy that is justified inside that regime. Every claim below is backed by an audited
experiment folder with its own kernel, analysis and results.

---

## The method in one paragraph

For a segment `t`, DDAG (i) forms an **intrinsic pre-generation probe** from the generator's own prefill
activations, (ii) optionally forms a **convergence residual** from the sampler's step-to-step update
magnitudes, (iii) routes to a cheap or fine mode according to a **calibrated gate**, and (iv) applies a
**training-free seam repair** at accepted switches. Which of these is *worth using* is decided by the
**regime map**: adaptive granularity only pays when the cheap mode is cheaper **and** worse.

## The eleven experiments (in reading order)

| # | Folder | Question | Result |
|---|---|---|---|
| 01 | `01_intrinsic_probe` | can the generator predict its own granularity safety? | **yes — AUROC 0.798** with no external oracle (hand-crafted 0.491, Jev text 0.788) |
| 02 | `02_seam_repair` | can the switch seam be reduced without training? | **yes — a 25% latent blend cuts the seam 25% and improves fidelity** |
| 03 | `03_closed_loop_evidence` | does post-generation evidence add signal? | free evidence alone **0.725**; closed loop 0.690 → **0.737** (bounded) |
| 04 | `04_escalation_gate` | is a calibrated escalation gate viable? | **boundary** — the cost model, not the signal, binds |
| 05 | `05_convergence_residual` | does a sampler convergence residual verify safety? | T4 **0.752**; probe+residual **0.842**; does **not** transfer to A100 (0.485) |
| 06 | `06_drift_propagation` | does coarse error propagate? | **yes on T4** — one coarse segment leaves a persistent elevation (4/5 segments) |
| 07 | `07_online_policy` | does routing beat a fixed mode? | **boundary** — on A100 coarse simply dominates (−46% cost) |
| 08 | `08_intermediate_band` | does routing pay inside the band? | coarse 27% cheaper; the latent endpoint does not penalise it |
| 09 | `09_pixel_endpoint` | does a decoded-pixel endpoint change the verdict? | **no** — coarse is cheaper *and* closer by pixels too |
| 10 | `10_jev_critic` | can a typed Jev critic certify safety from pixels? | **boundary** — chance (0.497); context, not the model, is the limit |
| 11 | `11_flicker_endpoint` | does the repair trade seam for flicker? | **yes** — an independent endpoint exposes the trade |

## The regime map (the method's core)

```
                       cheap mode cheaper?        cheap mode worse?
   T4  (small step gap)       NO                        —            -> FIXED FINE
   T4  (large step gap)      YES                       NO            -> FIXED COARSE
   A100                      YES                       NO            -> FIXED COARSE
   ---------- adaptive granularity pays only here ----------
                              YES                      YES
```

Across our two GPUs and every endpoint we measured, the cheap mode was **never worse**, so the optimal
policy is a **fixed mode** and adaptivity adds nothing. The map says exactly *where* a router would
become worth deploying — and that is a design rule, not a failure.

## Boundaries (recorded, never hidden)

- The intrinsic probe **inverts** on A100 (0.143) — signals are regime properties.
- Drift is **linear and not policy-predictable** (R² ≈ 0.01), so lookahead has little headroom.
- A typed Jev critic on raw frames is at **chance** — decision models need context.
- The seam repair trades a boundary discontinuity for a small interior flicker.

## Layout

```
ddag/
├─ experiments/  01_intrinsic_probe … 11_flicker_endpoint  (kernel + results + notes per experiment)
├─ analysis/     shared analysis scripts (one per experiment)
├─ results/      consolidated machine-readable results
└─ compute/      Colab A100 runners
```

## Reproduce (0 GPU analyses)

```bash
cd ddag
python analysis/crv_analysis.py       --features results/features_t4.npz --out results
python analysis/drift_analysis.py     --meas     results/drift_t4_n32.npz --out results/t4_n32
python analysis/online_policy.py      --features results/features_online_a100.npz --out results/online_a100
python analysis/jcd_analysis.py       --meas     results/jcd_t4.npz --out results
python analysis/flicker_analysis.py   --meas     results/flicker_t4.npz --out results
```
