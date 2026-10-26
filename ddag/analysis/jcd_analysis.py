"""DDAG E7 — JCD analysis: does a typed Jev critic independently predict granularity safety? (0 GPU)

Jev sees only pixels (two frames of the coarse segment) plus the prompt, and answers two typed
Noul questions. We test whether those answers predict the measured coarse-unsafe label.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def auroc(y, s):
    m = np.isfinite(s)
    if m.sum() < 4 or len(np.unique(y[m])) < 2:
        return float("nan")
    return float(roc_auc_score(y[m], s[m]))


def boot_ci(y, s, groups, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    gs = np.unique(groups)
    v = []
    for _ in range(n):
        pick = rng.choice(gs, size=len(gs), replace=True)
        idx = np.concatenate([np.where(groups == g)[0] for g in pick])
        if len(np.unique(y[idx])) > 1 and np.isfinite(s[idx]).sum() > 3:
            v.append(roc_auc_score(y[idx], s[idx]))
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else [float("nan")] * 2


def perm_p(y, s, n=1000, seed=1):
    m = np.isfinite(s)
    base = auroc(y, s)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        yy = y.copy()
        yy[m] = rng.permutation(y[m])
        vals.append(auroc(yy, s))
    vals = np.asarray([v for v in vals if not np.isnan(v)])
    return float(np.mean(vals >= base)) if len(vals) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meas", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    d = np.load(args.meas)
    y = d["labels"].astype(np.int64)
    groups = d["prompt_idx"].astype(np.int64)
    disc = d["jev_discontinuity"].astype(np.float64)
    plaus = d["jev_plausible"].astype(np.float64)

    rep = {"n_segments": int(len(y)), "n_positive": int(y.sum()),
           "n_jev_ok": int(np.isfinite(disc).sum()),
           "jev_mean": {"discontinuity": float(np.nanmean(disc)), "plausible": float(np.nanmean(plaus))},
           "tests": {}}
    tests = {
        "jev_discontinuity": disc,
        "jev_plausible": plaus,
        "jev_plausible_inverted": -plaus,
        "jev_disc+inv_plaus": disc - plaus,
    }
    for name, s in tests.items():
        rep["tests"][name] = {"auroc": auroc(y, s), "ci": boot_ci(y, s, groups), "perm_p": perm_p(y, s)}
        print(f"{name:24s} AUROC={rep['tests'][name]['auroc']:.3f} "
              f"CI={[round(c,3) for c in rep['tests'][name]['ci']]} p={rep['tests'][name]['perm_p']:.3f}")

    # correlation of the two Jev answers
    rep["corr_disc_plaus"] = float(np.corrcoef(disc[np.isfinite(disc)], plaus[np.isfinite(disc)])[0, 1])
    rep["reference"] = {"chance": 0.5, "hand_crafted": 0.491, "jev_text_base_paper": 0.788,
                        "selfaware_probe": 0.792, "crv_ratio_t4": 0.752}
    best = max(rep["tests"], key=lambda k: (rep["tests"][k]["auroc"] if not np.isnan(rep["tests"][k]["auroc"]) else -1))
    rep["best"] = best
    rep["verdict"] = ("H6_supported" if rep["tests"][best]["auroc"] >= 0.65 and rep["tests"][best]["perm_p"] < 0.05
                      else "H6_weak" if rep["tests"][best]["auroc"] >= 0.58 else "H6_refuted")
    print("best:", best, round(rep["tests"][best]["auroc"], 3), "->", rep["verdict"])
    (out / "e7_jcd.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
