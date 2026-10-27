"""DDAG — combined signal study with DIRECT AUROCs (audit F-1).

Replaces the sign-corrected reporting in `crv_variants.py`. Every scalar signal is scored by its
**direct** AUROC (no logistic layer), so an inverted signal is visible as AUROC < 0.5 rather than
being silently flipped.

Signals (all free, computed from quantities the sampler already produces):
  v1              first denoising-step magnitude of the coarse segment
  vS              last denoising-step magnitude
  mean            mean step magnitude
  std             std of step magnitudes
  ratio_last_first  vS / v1   (the definition in the manuscript's Eq. 2)
  ratio_first_last  v1 / vS   (the reciprocal)
  prefill_norm    magnitude of the prefill velocity
  probe           logistic probe on the pooled block-11 prefill activations (leave-one-prompt-out)

Combined models use a **multi-feature** logistic (where learning the sign is legitimate) and are
stored so the manuscript's numbers are reproducible from a result file.

Output: ddag/results/signal_combined.json
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


def direct(y, s):
    m = np.isfinite(s)
    return float(roc_auc_score(y[m], s[m])) if len(np.unique(y[m])) > 1 else float("nan")


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


def perm_p(y, s, n=1000, seed=1):
    base = direct(y, s)
    rng = np.random.default_rng(seed)
    vals = [direct(rng.permutation(y), s) for _ in range(n)]
    vals = np.asarray([v for v in vals if not np.isnan(v)])
    return float(np.mean(vals >= base)) if len(vals) else float("nan")


def steps_of(vn):
    out = []
    for i in range(vn.shape[0]):
        nz = np.nonzero(vn[i, :, 0])[0]
        v = vn[i, nz, 0]
        out.append(v[1:-1] if len(v) >= 3 else v)   # drop prefill (call 0) and append (last call)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--block", type=int, default=11)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    d = np.load(args.features)
    y = d["labels"].astype(np.int64)
    g = d["prompt_idx"].astype(np.int64)
    pre = d["prefill"].astype(np.float32)
    st = steps_of(d["vn_coarse"])
    vn = d["vn_coarse"]
    prefill_norm = np.asarray([vn[i, np.nonzero(vn[i, :, 0])[0][0], 0] for i in range(vn.shape[0])], dtype=np.float64)

    def f(fn):
        return np.asarray([fn(s) for s in st], dtype=np.float64)

    sig = {
        "v1_first_step": f(lambda s: s[0]),
        "vS_last_step": f(lambda s: s[-1]),
        "mean_step": f(lambda s: s.mean()),
        "std_step": f(lambda s: s.std()),
        "ratio_last_first_Eq2": f(lambda s: s[-1] / (s[0] + 1e-6)),
        "ratio_first_last": f(lambda s: s[0] / (s[-1] + 1e-6)),
        "prefill_norm": prefill_norm,
    }
    probe = oof(pre[:, args.block, :], y, g)
    sig["probe_block11"] = probe

    rep = {"n_segments": int(len(y)), "n_positive": int(y.sum()),
           "n_prompts": int(len(np.unique(g))), "block": args.block,
           "note": "All AUROCs are DIRECT (no logistic on a single scalar).",
           "direct_auroc": {}, "combined": {}}

    for k, s in sig.items():
        rep["direct_auroc"][k] = {"auroc": direct(y, s), "ci": boot_ci(y, s, g), "perm_p": perm_p(y, s)}
        print(f"{k:24s} direct AUROC={rep['direct_auroc'][k]['auroc']:.4f} "
              f"CI={[round(c,3) for c in rep['direct_auroc'][k]['ci']]} p={rep['direct_auroc'][k]['perm_p']:.3f}")

    # combined multi-feature models (legitimate: the model may use either sign)
    combos = {
        "probe+v1": np.stack([probe, sig["v1_first_step"]], axis=1),
        "probe+ratio_first_last": np.stack([probe, sig["ratio_first_last"]], axis=1),
        "probe+ratio_last_first": np.stack([probe, sig["ratio_last_first_Eq2"]], axis=1),
    }
    for k, X in combos.items():
        s = oof(X, y, g)
        rep["combined"][k] = {"auroc": direct(y, s), "ci": boot_ci(y, s, g), "perm_p": perm_p(y, s)}
        print(f"{k:24s} combined AUROC={rep['combined'][k]['auroc']:.4f} "
              f"CI={[round(c,3) for c in rep['combined'][k]['ci']]} p={rep['combined'][k]['perm_p']:.3f}")

    best = max(rep["direct_auroc"], key=lambda k: rep["direct_auroc"][k]["auroc"])
    rep["best_single"] = best
    rep["verdict"] = {
        "Eq2_ratio_is_inverted": rep["direct_auroc"]["ratio_last_first_Eq2"]["auroc"] < 0.5,
        "best_single_signal": best,
        "best_single_auroc": rep["direct_auroc"][best]["auroc"],
    }
    (out / "signal_combined.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep["verdict"], indent=2))
    print("wrote", out / "signal_combined.json")


if __name__ == "__main__":
    main()
