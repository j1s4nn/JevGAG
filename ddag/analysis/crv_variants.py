"""DDAG E2b (careful) — residual-variant sweep with CIs and a permutation null (0 GPU).

Per coarse segment the captured calls are: [prefill, step1, ..., stepK, append].
We separate them and test well-defined convergence statistics, with leave-one-prompt-out CV,
a prompt-grouped bootstrap CI, and a label-permutation null. An inverted-looking AUROC (<0.5)
is reported as-is: with a 1-D feature and LOO-CV the logistic model learns the sign on the
training folds, so a value far below 0.5 indicates a *sign-inconsistent* (confounded) feature,
not a usable signal.
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


def perm_p(y, s, groups, n=1000, seed=1):
    base = auroc(y, s)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        yy = rng.permutation(y)
        vals.append(auroc(yy, s))
    vals = np.asarray([v for v in vals if not np.isnan(v)])
    return float(np.mean(vals >= base)) if len(vals) else float("nan")


def split_calls(vn):
    """Return (prefill, steps, append) arrays per segment from the padded [n,maxc,2] norms."""
    pref, steps, app = [], [], []
    for i in range(vn.shape[0]):
        nz = np.nonzero(vn[i, :, 0])[0]
        vals = vn[i, nz, 0]
        if len(vals) >= 3:
            pref.append(vals[0])
            steps.append(vals[1:-1])
            app.append(vals[-1])
        elif len(vals) == 2:
            pref.append(vals[0]); steps.append(vals[1:]); app.append(vals[-1])
        else:
            pref.append(vals[0] if len(vals) else 0.0); steps.append(np.array([0.0])); app.append(0.0)
    return np.asarray(pref, dtype=np.float32), steps, np.asarray(app, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    d = np.load(args.features)
    y = d["labels"].astype(np.int64)
    groups = d["prompt_idx"].astype(np.int64)
    pref, steps, app = split_calls(d["vn_coarse"])

    def stepstat(fn):
        return np.asarray([fn(s) for s in steps], dtype=np.float32).reshape(-1, 1)

    variants = {
        "prefill_norm": pref.reshape(-1, 1),
        "step_first": stepstat(lambda s: s[0]),
        "step_last": stepstat(lambda s: s[-1]),
        "step_mean": stepstat(lambda s: s.mean()),
        "step_std": stepstat(lambda s: s.std()),
        "step_max": stepstat(lambda s: s.max()),
        "step_range": stepstat(lambda s: s.max() - s.min()),
        "step_last_over_first": stepstat(lambda s: s[-1] / (s[0] + 1e-6)),
    }

    rep = {"n_segments": int(len(y)), "n_positive": int(y.sum()), "variants": {}}
    for name, X in variants.items():
        s = oof(X, y, groups)
        rep["variants"][name] = {"auroc": auroc(y, s), "ci": boot_ci(y, s, groups), "perm_p": perm_p(y, s, groups)}
        print(f"{name:22s} AUROC={rep['variants'][name]['auroc']:.3f} "
              f"CI={[round(c,3) for c in rep['variants'][name]['ci']]} p={rep['variants'][name]['perm_p']:.3f}")

    valid = {k: v for k, v in rep["variants"].items() if v["auroc"] >= 0.5}
    rep["best"] = max(valid, key=lambda k: valid[k]["auroc"]) if valid else None
    rep["best_auroc"] = valid[rep["best"]]["auroc"] if valid else None
    rep["verdict"] = ("H1_supported" if rep["best_auroc"] and rep["best_auroc"] >= 0.70
                      else "H1_weak" if rep["best_auroc"] and rep["best_auroc"] >= 0.60 else "H1_refuted")
    rep["note"] = "AUROC far below 0.5 indicates a sign-inconsistent (confounded) feature, not a usable signal."
    rep["reference"] = {"chance": 0.5, "hand_crafted": 0.491, "selfaware_probe": 0.798}
    (out / "e2b_residual_variants.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print("best:", rep["best"], rep["best_auroc"], "->", rep["verdict"])


if __name__ == "__main__":
    main()
