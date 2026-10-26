# AUDIT_13 — CVIU / Elsevier compliance (verified)

**Verification status.** The CVIU Guide for Authors (ScienceDirect) returned HTTP 400 to a direct fetch;
the requirements below were assembled from (a) the ScienceDirect journal pages, (b) the Elsevier CAS
template that ships with MiKTeX (`els-cas-templates`), and (c) the Elsevier author-centre pages, all
retrieved 26 Oct 2026. Items marked ⚠️ should be re-checked on the live Guide for Authors immediately
before submission.

---

## 1. Binding requirements

| Requirement | Status | Evidence |
|---|---|---|
| **Article type** | Regular Paper (as declared in `paper/OUTLINE.md`) | journal scope |
| **Template** | **Elsevier CAS** (`els-cas-templates`): `cas-sc.cls` (single column) or `cas-dc.cls` (double column). The manuscript uses `cas-sc` with the `review` option (line numbers) | CVIU guide references `els-cas-templates.zip`; `cas-sc.cls` verified to compile under MiKTeX |
| **Page limit** | **No binding limit found** for Regular Papers. The "14–16 pages" in `paper/OUTLINE.md` is the project's own target, **not** a CVIU rule ⚠️ | no limit stated on the journal pages retrieved |
| **Abstract** | ~250 words, no references/abbreviations | Elsevier/CVIU author guidance |
| **Keywords** | present; Elsevier typically allows up to 6 — the manuscript has **7** | minor |
| **Highlights** | 3–5 bullets, ≤ 85 characters each (optional for submission, required at revision for many Elsevier journals) ⚠️ | Elsevier author centre |
| **Graphical abstract** | optional ⚠️ | Elsevier |
| **Declarations** | competing interests, funding, **ethics**, data availability, CRediT | Elsevier |
| **Ethics statement** | **MISSING** from the manuscript | Elsevier requires it or an explicit n/a |
| **ORCID** | present on the title page and renders correctly | verified in the compiled PDF |
| **Author biography** | present (not required by Elsevier, but kept because a previous IEEE submission was rescinded for its absence) | — |
| **Reference style** | CAS numbered (`cas-model2-names.bst`), `natbib` numbers+sort&compress | compiles; 38 entries, 0 missing, 0 uncited |
| **Editable sources** | `.tex`, `.bib`, figures must be supplied | present in `submission/` |
| **Data availability** | statement present; must name the repositories | present, but see AUDIT_05 K-1 (the A100 input path is not self-contained) |

## 2. Cost model (verified)

CVIU is **hybrid**: the **subscription route is free** (no APC); Gold OA is **USD 2,390** (excluding
taxes). Overlength page charges are not applicable to Elsevier journals in the same way as IEEE
Transactions; there is no confirmed per-page charge for CVIU ⚠️.

## 3. Findings

**C-1 (P1) — the "14–16 page" figure is a project target, not a journal rule.** `paper/OUTLINE.md`
states it as a target, but nothing in the paper or repo says so. The manuscript is currently **14 pages**
in `cas-sc` review format, which is comfortable under any plausible limit. **No action required**, but do
not describe it as a CVIU limit in the cover letter.

**C-2 (P1) — the ethics statement is missing.** Elsevier expects an ethics statement or an explicit
"not applicable". Add one line (no human/animal subjects).

**C-3 (P2) — 7 keywords vs a typical maximum of 6.** Drop one (suggest removing "efficient inference",
which is implied by "adaptive compute").

**C-4 (P2) — Highlights are not in the manuscript or the package** (they exist only in
`submission/highlights.md`). That is correct for Elsevier (highlights are entered in the portal), but
confirm the portal field accepts them.

**C-5 (P2) — the `review` class option adds line numbers.** Appropriate for submission; remove for the
camera-ready.

**C-6 (P2) — no supplementary-material declaration.** If REC-3/REC-7/REC-9/REC-10 from `AUDIT_12` are
produced, they should be declared as supplementary material.

## 4. Pre-submission compliance checklist

- [x] CAS template, single column
- [x] Abstract ≤ 250 words
- [ ] Keywords ≤ 6 — **fix (7 present)**
- [ ] Ethics statement — **add**
- [x] Competing interests
- [x] Funding
- [x] Data availability (needs the K-1 fix)
- [x] CRediT
- [x] ORCID
- [x] References complete and consistent
- [ ] Highlights in the portal
- [ ] Graphical abstract (optional)
- [ ] Supplementary declaration if used
- [ ] Remove the `review` option for the final version
