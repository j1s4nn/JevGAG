"""ClosedLoop-Gran E1 — does post-generation evidence improve granularity routing? (0 GPU)

Input : features.npz from the closed-loop capture kernel
        (prefill activations + per-segment free post-generation evidence + labels)
Output: results/e1_closedloop.json

Signals
-------
prior_g    : intrinsic pre-generation probe score from the prefill activations (as in idea 03).
evidence_g : FREE post-generation evidence from running the segment in coarse mode
             (boundary seam + interior jitter; no reference rollout needed).
disc_g     : realized coarse-vs-fine discrepancy (NOT free: requires both modes) -- upper bound.

Closed-loop tests (leave-one-prompt-out CV, grouped by prompt)
  A. prior_g                      (open loop, idea 03)
  B. evidence_{g-1}               (evidence only, causal)
  C. prior_g + evidence_{g-1}     (CLOSED LOOP, causal)
  D. prior_g + evidence_g         (non-causal upper bound)
  E. prior_g + evidence_{g-1} + disc_{g-1}   (closed loop + costly discrepancy)
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


def oof(X, y, groups, C=0.05):
    s = np.zeros(len(y))
    for g in np.unique(groups):
        tr, te = groups != g, groups == g
        if len(np.unique(y[tr])) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=C, max_iter=5000, class_weight="balanced").fit(sc.transform(X[tr]), y[tr])
        s[te] = clf.decision_function(sc.transform(X[te]))
    return s


def auroc(y, s):
    return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float("nan")


def boot_ci(y, s, groups, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    gs = np.unique(groups)
    v = []
    for _ in range(n):
        pick = rng.choice(gs, size=len(gs), replace=True)
        idx = np.concatenate([np.where(groups == g)[0] for g in pick])
        if len(np.unique(y[idx])) > 1:
            v.append(roc_auc_score(y[idx], s[idx]))
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else [float("nan")] * 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    d = np.load(args.features)
    pre = d["prefill"].astype(np.float32)
    y = d["labels"].astype(np.int64)
    groups = d["prompt_idx"].astype(np.int64)
    seg = d["segment"].astype(np.int64)
    n, nb, dim = pre.shape

    # prior: best single block probe (block 11 as in idea 03; re-derived here)
    prior = oof(pre.reshape(n, nb * dim), y, groups)  # concat probe as a stable prior

    ev = np.stack([d["seam_coarse"], d["interior_coarse"]], axis=1).astype(np.float32)
    disc = d["disc"].astype(np.float32).reshape(-1, 1)

    # build causal pairs: for segment g>=2 use evidence from g-1
    idx_prev, idx_cur = [], []
    for i in range(n):
        if seg[i] >= 2:
            j = i - 1
            if j >= 0 and groups[j] == groups[i] and seg[j] == seg[i] - 1:
                idx_prev.append(j)
                idx_cur.append(i)
    idx_prev = np.array(idx_prev, dtype=int)
    idx_cur = np.array(idx_cur, dtype=int)

    rep = {"n_segments": int(n), "n_causal_pairs": int(len(idx_cur)), "n_positive": int(y.sum()), "tests": {}}

    def test(name, X, yy, gg):
        s = oof(X, yy, gg)
        rep["tests"][name] = {"n": int(len(yy)), "auroc": auroc(yy, s), "ci": boot_ci(yy, s, gg)}
        print(f"{name:38s} n={len(yy):3d} AUROC={rep['tests'][name]['auroc']:.3f} CI={rep['tests'][name]['ci']}")

    # A: open loop (prior only) on all segments
    test("A_prior_open_loop", prior.reshape(-1, 1), y, groups)

    # B/C: causal pairs
    yc = y[idx_cur]
    gc = groups[idx_cur]
    test("B_evidence_prev_only", ev[idx_prev], yc, gc)
    test("C_closed_loop_prior+ev_prev", np.hstack([prior[idx_cur].reshape(-1, 1), ev[idx_prev]]), yc, gc)
    test("A2_prior_on_causal_subset", prior[idx_cur].reshape(-1, 1), yc, gc)

    # D: non-causal upper bound (same-segment evidence)
    test("D_prior+ev_same_noncausal", np.hstack([prior.reshape(-1, 1), ev]), y, groups)

    # E: closed loop + costly discrepancy from the previous segment
    test("E_closed_loop+disc_prev", np.hstack([prior[idx_cur].reshape(-1, 1), ev[idx_prev], disc[idx_prev]]), yc, gc)

    rep["reference_points"] = {"chance": 0.5, "hand_crafted": 0.491, "jev_text": 0.788,
                               "selfaware_dit_prefill_probe": 0.798}
    (out / "e1_closedloop.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        keys = [k for k in rep["tests"] if not k.startswith("A2")]
        vals = [rep["tests"][k]["auroc"] for k in keys]
        err = [[rep["tests"][k]["auroc"] - rep["tests"][k]["ci"][0] for k in keys],
               [rep["tests"][k]["ci"][1] - rep["tests"][k]["auroc"] for k in keys]]
        fig, ax = plt.subplots(figsize=(7.4, 3.8), dpi=150)
        ax.bar(range(len(keys)), vals, yerr=err, capsize=4, color="#1668d6")
        ax.axhline(0.5, ls=":", c="gray"); ax.axhline(0.798, ls="--", c="#2a7", lw=1)
        ax.text(len(keys) - 0.5, 0.80, "idea 03 probe", color="#2a7", fontsize=7, ha="right")
        ax.set_xticks(range(len(keys))); ax.set_xticklabels([k.split("_")[0] for k in keys])
        ax.set_ylabel("AUROC (leave-one-prompt-out)"); ax.set_ylim(0.4, 1.0)
        ax.set_title("ClosedLoop-Gran: does post-generation evidence help?")
        ax.grid(alpha=0.25, axis="y")
        fig.tight_layout(); fig.savefig(out / "fig_closedloop.png")
        print("figure written")
    except Exception as e:
        print("figure skipped:", e)


if __name__ == "__main__":
    main()
