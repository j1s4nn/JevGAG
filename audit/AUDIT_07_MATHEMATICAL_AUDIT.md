# AUDIT_07 — Mathematical and policy-logic audit

All equations were re-derived independently. Values were re-computed from the stored result files
(`ddag/results/*.npz`, `*.json`) without a logistic layer, so that **sign information is preserved**.

---

## M-1 (P0) — The convergence residual has the WRONG SIGN

**Manuscript.** Eq. (2): `κ = v_S / v_1` — "the ratio of the last to the first step magnitude. A sampler
that has settled produces κ ≪ 1; a sampler still moving at its final step produces κ ≈ 1." §6.2 then
reports "the convergence residual of Eq. (2) reaches 0.752".

**Re-derivation.** From `ddag/results/features_t4.npz` (T4, 64 segments), the **direct** AUROC
(no logistic, so the sign is preserved) of each residual definition is:

| definition | direct AUROC | reading |
|---|---|---|
| prefill velocity norm | 0.4964 | chance |
| **first-step magnitude `v_1`** | **0.7101** | predictive |
| last-step magnitude `v_S` | 0.6618 | predictive |
| mean step magnitude | 0.6896 | predictive |
| std of step magnitudes | 0.5676 | weak |
| **`v_S / v_1` (Eq. 2 as written)** | **0.2403** | **INVERTED** |
| `v_1 / v_S` (reciprocal) | 0.7597 | predictive |

**Finding.** The quantity *as defined in Eq. (2)* is **anti-predictive** (0.2403). The value 0.752
quoted in the manuscript is **sign-corrected**: it equals the AUROC of the reciprocal (0.7597), and it
was obtained by passing the 1-D ratio through a logistic regression
(`ddag/analysis/crv_variants.py`, `oof(...)`), which re-learns the sign and reports
`max(AUROC, 1−AUROC)`. This is the *same defect* previously found and corrected in
`crv_analysis.py` but **still present in `crv_variants.py`**, and it propagates into the manuscript.

**Consequences.**
1. The physical story in §4.3 is contradicted by the data: a *large* last/first ratio is associated with
   **safe** segments, not unsafe ones.
2. The manuscript's `probe + residual = 0.842` is a legitimate **multi-feature** model
   (a logistic over `[probe, ratio]` may use the ratio with a negative weight), and it re-derives to
   **0.8418**. However, it is **not stored in any result file** — the stored `e2_crv.json` reports
   `probe+r_conv = 0.7621`, which uses a *different* residual (the last-step magnitude, not the ratio).
3. The abstract, §1 and §4.3 all carry the sign error.

**Recommended correction (evidence-supported, no new compute).** Redefine the residual as either
(i) `κ = v_1 / v_S` (direct AUROC 0.7597), or (ii) the **first-step magnitude** `v_1` (direct AUROC
0.7101), which is the strongest single residual and the simplest to justify ("a segment that starts with
a large update is one the coarse mode has not yet resolved"). Then:
- re-derive and **store** the combined model in a result file;
- update Eq. (2), §4.3, §6.2, the abstract, and Fig. 3;
- add a note that single-scalar signals must be evaluated by *direct* AUROC, never through a
  single-feature logistic.

---

## M-2 (P1) — The regime map has no objective function

**Manuscript.** Eq. (1) partitions the plane into fixed-fine / adaptive / fixed-coarse, and §6.5
concludes "the optimal policy is a fixed mode".

**Finding.** No objective is defined. "Optimal" is therefore undefined. The three candidate objectives
give different answers:
- **cost-optimal** (minimise DiT seconds, ignore quality): the cheapest mode, i.e. coarse where `r < 1`;
- **quality-optimal**: the mode with the lower endpoint distance;
- **Pareto / constrained** (minimise cost subject to quality not worse than fixed-fine): the mode that
  satisfies the constraint.

At all three operating points the cheap mode is *also* not worse, so the three objectives coincide —
which is precisely why the adaptive region is empty. But the paper must say so rather than assert
"optimal".

**Recommended correction.** Add the objective to §4.1, e.g.
`minimise E[DiT seconds] subject to E[quality] ≤ E[quality of fixed fine]`,
then state that the fixed policy is optimal **for this objective**, and note that the three objectives
coincide at every measured operating point (this is the reason the adaptive region is empty).

---

## M-3 (P1) — The gating cost algebra omits the probe cost

**Manuscript.** §4.4: "escalation costs `c_c + ρ c_f` against `c_f` for fixed fine, so a saving requires
`ρ < 1 − r`; routing costs `(1−ρ)c_c + ρ c_f` and saves whenever `r < 1`."

**Re-derivation.**
- Escalation: `c_c + ρ c_f < c_f ⇔ c_c < (1−ρ) c_f ⇔ r < 1−ρ ⇔ ρ < 1−r`. ✅ **correct.**
- Routing: `(1−ρ)c_c + ρ c_f < c_f ⇔ (1−ρ)c_c < (1−ρ)c_f ⇔ r < 1` (for `ρ<1`). ✅ **correct.**

**But** both expressions omit the **probe/prefill cost** `c_p`, which is paid once per segment to obtain
the score. With it, escalation costs `c_c + c_p + ρ c_f` and routing `c_p + (1−ρ)c_c + ρ c_f`, and the
conditions tighten to `ρ < 1 − r − c_p/c_f` and `r < 1 − c_p/c_f` respectively.

**Recommended correction.** State the conditions with `c_p`, and note that `c_p` is one fine-mode forward
per segment (the prefill), i.e. small but not zero. This strengthens §4.4's conclusion that the *cost*
precondition, not the signal, binds.

---

## M-4 (P1) — Two different ratios share one name

**Manuscript.** §6.1 uses "the per-forward ratio" for 3.01 (T4) and 1.60 (A100) — a *hardware* property
(seconds per single forward, coarse vs fine) — and then "the measured ratio to 0.73" for the band run,
which is a *total per-segment cost* ratio `c_c/c_f`. These are different quantities with the same name.

**Recommended correction.** Name them separately: `π` = per-forward ratio (hardware), `r` = per-segment
cost ratio (schedule × hardware, Eq. 1). Re-state §6.1 with `r` values from the repository table
(T4 s2/s2 `r = 2.969/…`; T4 s2/s1 `r = 0.728`; A100 `r = 0.536`) and attribute 3.01/1.60/2.33 to the
**prior study**.

---

## M-5 (P2) — Label notation is ill-typed

**Manuscript.** §3: "`y = 1[ ℓ(q_c) > ℓ(q_f) ]` with `ℓ` the endpoint distance", but `q` was defined in
§3 as "a quality functional (lower is better)". So `ℓ(q)` applies a distance to a functional.

**Recommended correction.** Write `y = 1[ d(m_c) > d(m_f) ]` where `d(·)` is the endpoint distance of a
*generated segment* and `m ∈ {fine, coarse}`; keep `q` for the functional's expectation.

---

## M-6 (P2) — Seam-repair α has no selection rule

**Manuscript.** Eq. (3) blends with `α`, and §6.3 reports a sweep, but no rule is given for choosing α.

**Recommended correction.** State the rule actually used (α = 0.25 selected as the smallest α whose seam
reduction is significant while fidelity does not degrade), and note that α should be calibrated on the
target GPU like the gate threshold.

---

## M-7 (P2) — Break-even direction is stated loosely

**Manuscript.** §3 P1: "the condition is a comparison of the measured per-forward ratio against the
break-even `ρ_be = NFE_f / NFE_c`."

**Re-derivation.** Coarse is cheaper iff `NFE_c · t_c < NFE_f · t_f`, i.e.
`t_c/t_f < NFE_f/NFE_c` — i.e. the **per-forward ratio must be *below* the break-even**. The manuscript's
wording is compatible but should state the inequality direction explicitly. Verified against
`breakeven_matrix.json`: T4 s2/s2 → break-even 2.684, measured 2.969 → not cheaper ✅.

---

## Summary of mathematical findings

| ID | Priority | Issue | Needs new compute? |
|---|---|---|---|
| M-1 | **P0** | convergence residual sign inverted; 0.752 is a sign-corrected artifact; 0.842 not stored | no (re-derive + store) |
| M-2 | **P1** | regime map has no objective | no |
| M-3 | **P1** | gating algebra omits probe cost | no |
| M-4 | **P1** | two ratios, one name | no |
| M-5 | P2 | label notation ill-typed | no |
| M-6 | P2 | α selection rule absent | no |
| M-7 | P2 | inequality direction implicit | no |
