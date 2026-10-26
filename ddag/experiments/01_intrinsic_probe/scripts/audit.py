"""Claim-to-artifact audit for SelfAware-DiT.

Re-derives the headline AUROC from features.npz and checks it against the stored
probe_results.json. Exits non-zero on mismatch.

  python ideas/selfaware-dit/scripts/audit.py \
      --features ideas/selfaware-dit/results/features.npz \
      --results  ideas/selfaware-dit/results/probe_results.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


def oof_scores(X, y, groups, C=0.05):
    s = np.zeros(len(y), dtype=np.float64)
    for g in np.unique(groups):
        tr, te = groups != g, groups == g
        if len(np.unique(y[tr])) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=C, max_iter=5000, class_weight="balanced").fit(sc.transform(X[tr]), y[tr])
        s[te] = clf.decision_function(sc.transform(X[te]))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--tol", type=float, default=1e-6)
    args = ap.parse_args()

    d = np.load(args.features)
    y = d["labels"].astype(np.int64)
    groups = d["prompt_idx"].astype(np.int64)
    pre = d["prefill"].astype(np.float32)
    rep = json.loads(Path(args.results).read_text(encoding="utf-8"))

    b = rep["best_prefill_block"]
    s = oof_scores(pre[:, b, :], y, groups)
    auroc = float(roc_auc_score(y, s))

    checks = []
    checks.append(("n_segments", int(d["prefill"].shape[0]), rep["n_segments"], None))
    checks.append(("n_positive", int(y.sum()), rep["n_positive"], None))
    checks.append(("best_block", b, rep["best_prefill_block"], None))
    checks.append(("auroc", round(auroc, 10), round(rep["best_prefill_auroc"], 10), args.tol))

    ok = True
    for name, got, want, tol in checks:
        good = (got == want) if tol is None else (abs(got - want) <= tol)
        ok = ok and good
        print(f"[{'PASS' if good else 'FAIL'}] {name}: re-derived={got} stored={want}")

    # verdict
    verdict = "SUPPORTED" if rep["best_prefill_auroc"] >= 0.70 and rep["best_prefill_perm_p"] < 0.05 else (
        "PARTIAL" if rep["best_prefill_auroc"] >= 0.55 else "NEGATIVE")
    print(f"[VERDICT] {verdict} (AUROC={rep['best_prefill_auroc']:.3f}, perm_p={rep['best_prefill_perm_p']:.4f})")
    print(f"[REFERENCE] chance=0.500 hand_crafted=0.491 jev_text=0.788")
    print("AUDIT", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
