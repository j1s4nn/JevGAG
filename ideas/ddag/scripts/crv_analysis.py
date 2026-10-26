"""DDAG E2 — CRV: does the final-step convergence residual verify coarse-safety? (0 GPU)

Input : features.npz from ideas/ddag/jobs/crv (prefill probe features + per-call velocity
        norms for the matched coarse and fine segments + labels).
Output: results/e2_crv.json + fig_e2_crv.png

Signals
-------
r_conv   = ||v|| at the FINAL denoise step of the coarse segment (convergence residual).
           For a distilled few-step sampler this measures "how far from done" the sample is.
probe    = intrinsic pre-generation probe score (best block from idea 03).
label    = coarse-unsafe = 1[ refL1_coarse > refL1_fine ] (latent reference).

Tests: AUROC of r_conv alone, probe alone, probe+r_conv, probe+r_conv+free evidence.
Calibration: with a threshold tau on r_conv, what error rate is reached at what escalation rho?
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


def final_step_norm(vn):
    """vn: [n, maxc, 2]; take the norm of the second-to-last real call (last step, not the append)."""
    out = np.zeros(vn.shape[0], dtype=np.float32)
    for i in range(vn.shape[0]):
        nz = np.nonzero(vn[i, :, 0])[0]
        if len(nz) >= 2:
            out[i] = vn[i, nz[-2], 0]
        elif len(nz) == 1:
            out[i] = vn[i, nz[-1], 0]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-err", type=float, default=0.15)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    d = np.load(args.features)
    y = d["labels"].astype(np.int64)
    groups = d["prompt_idx"].astype(np.int64)
    pre = d["prefill"].astype(np.float32)
    n, nb, dim = pre.shape

    rconv = final_step_norm(d["vn_coarse"]).reshape(-1, 1)
    rconv_f = final_step_norm(d["vn_fine"]).reshape(-1, 1)
    ev = np.stack([d["seam_coarse"], d["interior_coarse"]], axis=1).astype(np.float32)
    prior = oof(pre[:, 11, :], y, groups)
    # NOTE: a logistic trained on a *single* score (e.g. oof(prior)) re-learns the sign and would
    # report max(AUROC, 1-AUROC), hiding an inverted signal. Always report the DIRECT AUROC.
    prior_auroc = auroc(y, prior)

    rep = {"n_segments": int(n), "n_positive": int(y.sum()), "n_prompts": int(len(np.unique(groups))),
           "probe_block11_direct_auroc": prior_auroc,
           "r_conv_stats": {"mean": float(rconv.mean()), "std": float(rconv.std()),
                            "corr_with_label": float(np.corrcoef(rconv[:, 0], y)[0, 1])},
           "auroc": {}}

    tests = {
        "r_conv_alone": rconv,
        "probe_alone": None,  # reported directly above (avoid the sign-flip artifact)
        "probe+r_conv": np.hstack([prior.reshape(-1, 1), rconv]),
        "r_conv+free_evidence": np.hstack([rconv, ev]),
        "probe+r_conv+evidence": np.hstack([prior.reshape(-1, 1), rconv, ev]),
    }
    rep["auroc"]["probe_alone"] = {"auroc": prior_auroc, "ci": boot_ci(y, prior, groups),
                                   "note": "direct AUROC of the block-11 probe score"}
    print(f"{'probe_alone(direct)':26s} AUROC={prior_auroc:.3f}")
    for name, X in tests.items():
        if X is None:
            continue
        s = oof(X, y, groups)
        rep["auroc"][name] = {"auroc": auroc(y, s), "ci": boot_ci(y, s, groups)}
        print(f"{name:26s} AUROC={rep['auroc'][name]['auroc']:.3f} CI={rep['auroc'][name]['ci']}")

    # calibration: threshold on r_conv -> escalation rho and error among accepted
    r = rconv[:, 0]
    cal = []
    for q in np.linspace(0.05, 0.95, 19):
        tau = np.quantile(r, q)
        acc = r <= tau
        if acc.sum() == 0:
            continue
        cal.append({"quantile": float(q), "tau": float(tau), "rho": float(1 - acc.mean()),
                    "error_among_accepted": float(y[acc].mean())})
    rep["calibration"] = cal
    ok = [c for c in cal if c["error_among_accepted"] <= args.target_err and c["rho"] < 0.667]
    rep["feasible_at_target"] = bool(ok)
    rep["best_feasible"] = ok[0] if ok else None

    rep["reference"] = {"chance": 0.5, "hand_crafted": 0.491, "jev_text": 0.788, "selfaware_probe": 0.798}
    rep["verdict"] = {
        "H1_supported": rep["auroc"]["r_conv_alone"]["auroc"] >= 0.65,
        "beats_probe": rep["auroc"]["probe+r_conv"]["auroc"] > rep["auroc"]["probe_alone"]["auroc"],
        "cost_win_feasible": rep["feasible_at_target"],
    }
    (out / "e2_crv.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep["verdict"], indent=2))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=150)
        names = list(rep["auroc"].keys())
        vals = [rep["auroc"][k]["auroc"] for k in names]
        err = [[rep["auroc"][k]["auroc"] - rep["auroc"][k]["ci"][0] for k in names],
               [rep["auroc"][k]["ci"][1] - rep["auroc"][k]["auroc"] for k in names]]
        ax.bar(range(len(names)), vals, yerr=err, capsize=4, color="#1668d6")
        ax.axhline(0.798, ls="--", c="#2a7", lw=1)
        ax.axhline(0.5, ls=":", c="gray")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.replace("+", "+\n") for n in names], fontsize=7)
        ax.set_ylabel("AUROC (leave-one-prompt-out)"); ax.set_ylim(0.4, 1.0)
        ax.set_title("DDAG E2: convergence-residual verification (CRV)")
        ax.grid(alpha=0.25, axis="y")
        fig.tight_layout(); fig.savefig(out / "fig_e2_crv.png")
        print("figure written")
    except Exception as e:
        print("figure skipped:", e)


if __name__ == "__main__":
    main()
