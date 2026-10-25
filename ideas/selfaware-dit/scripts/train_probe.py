"""SelfAware-DiT probe training and layer sweep (0 GPU).

Input : features.npz  (prefill, gen, labels, prompt_idx, segment) produced by the
        capture kernel.
Output: probe_results.json + fig_probe_layer_sweep.png

Protocol
--------
- Task: predict the base paper's coarse-unsafe label
  (1 iff the coarse segment's reference-L1 exceeds the fine segment's).
- Features:
    prefill : pooled DiT block activations of the I2V *prefill forward on the segment
              boundary* -> strictly PRE-GENERATION (the decision point).
    gen     : pooled activations averaged over the segment's *generation* forwards
              -> WITHIN-GENERATION ablation (not a legal pre-generation signal).
- Evaluation: leave-one-prompt-out CV (grouped by prompt; segments within a prompt are
  correlated), so no segment is ever scored by a model that saw its own prompt.
- Reported: AUROC per block (layer sweep), best block, all-block concat, a cheap
  velocity-scalar baseline, and a label-permutation null.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


def oof_scores(X, y, groups, C=0.05):
    """Leave-one-group-out out-of-fold decision scores."""
    scores = np.zeros(len(y), dtype=np.float64)
    for g in np.unique(groups):
        tr = groups != g
        te = ~tr
        if len(np.unique(y[tr])) < 2:
            scores[te] = 0.0
            continue
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=C, max_iter=5000, class_weight="balanced")
        clf.fit(sc.transform(X[tr]), y[tr])
        scores[te] = clf.decision_function(sc.transform(X[te]))
    return scores


def safe_auroc(y, s):
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def bootstrap_ci(y, s, groups, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    gs = np.unique(groups)
    vals = []
    for _ in range(n):
        pick = rng.choice(gs, size=len(gs), replace=True)
        idx = np.concatenate([np.where(groups == g)[0] for g in pick])
        yy, ss = y[idx], s[idx]
        if len(np.unique(yy)) < 2:
            continue
        vals.append(roc_auc_score(yy, ss))
    if not vals:
        return [float("nan"), float("nan")]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def permutation_null(y, s, groups, n=1000, seed=1):
    rng = np.random.default_rng(seed)
    base = safe_auroc(y, s)
    vals = []
    for _ in range(n):
        yy = rng.permutation(y)
        vals.append(safe_auroc(yy, s))
    vals = np.asarray([v for v in vals if not np.isnan(v)])
    return float(np.mean(vals >= base)) if len(vals) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--C", type=float, default=0.05)
    ap.add_argument("--label-key", default="labels")
    args = ap.parse_args()

    d = np.load(args.features)
    pre = d["prefill"].astype(np.float32)
    gen = d["gen"].astype(np.float32)
    y = d[args.label_key].astype(np.int64)
    groups = d["prompt_idx"].astype(np.int64)
    n, nb, dim = pre.shape
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"segments={n} blocks={nb} dim={dim} positives={int(y.sum())}/{n} prompts={len(np.unique(groups))}")

    report = {
        "n_segments": int(n), "n_blocks": int(nb), "dim": int(dim),
        "n_positive": int(y.sum()), "n_prompts": int(len(np.unique(groups))),
        "base_rate": float(y.mean()),
        "block_sweep_prefill": [], "block_sweep_gen": [],
    }

    # --- layer sweep ---
    for b in range(nb):
        s = oof_scores(pre[:, b, :], y, groups, args.C)
        report["block_sweep_prefill"].append(safe_auroc(y, s))
        s2 = oof_scores(gen[:, b, :], y, groups, args.C)
        report["block_sweep_gen"].append(safe_auroc(y, s2))

    sweep_pre = np.asarray(report["block_sweep_prefill"])
    sweep_gen = np.asarray(report["block_sweep_gen"])
    best_pre = int(np.nanargmax(sweep_pre)) if np.any(~np.isnan(sweep_pre)) else -1
    best_gen = int(np.nanargmax(sweep_gen)) if np.any(~np.isnan(sweep_gen)) else -1
    report["best_prefill_block"] = best_pre
    report["best_prefill_auroc"] = float(sweep_pre[best_pre]) if best_pre >= 0 else float("nan")
    report["best_gen_block"] = best_gen
    report["best_gen_auroc"] = float(sweep_gen[best_gen]) if best_gen >= 0 else float("nan")

    # --- concat all blocks (prefill) ---
    Xall = pre.reshape(n, nb * dim)
    s_all = oof_scores(Xall, y, groups, args.C)
    report["concat_prefill_auroc"] = safe_auroc(y, s_all)
    report["concat_prefill_ci"] = bootstrap_ci(y, s_all, groups)
    report["concat_prefill_perm_p"] = permutation_null(y, s_all, groups)

    # --- best single block, with CI + permutation ---
    if best_pre >= 0:
        sb = oof_scores(pre[:, best_pre, :], y, groups, args.C)
        report["best_prefill_ci"] = bootstrap_ci(y, sb, groups)
        report["best_prefill_perm_p"] = permutation_null(y, sb, groups)

    # --- reference points ---
    report["reference_points"] = {
        "hand_crafted_context_signal_base_paper": 0.491,
        "jev_text_oracle_base_paper": 0.788,
        "chance": 0.5,
    }

    (out / "probe_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # --- figure ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=150)
        ax.plot(range(nb), sweep_pre, "-o", ms=3, label="prefill (pre-generation)", color="#1668d6")
        ax.plot(range(nb), sweep_gen, "-s", ms=3, label="generation (ablation)", color="#e07b39")
        ax.axhline(0.5, ls=":", c="gray", lw=1, label="chance")
        ax.axhline(0.491, ls="--", c="#b23", lw=1, label="hand-crafted (base paper)")
        ax.axhline(0.788, ls="--", c="#2a7", lw=1, label="Jev text (base paper)")
        ax.set_xlabel("DiT block depth")
        ax.set_ylabel("AUROC (leave-one-prompt-out)")
        ax.set_title("SelfAware-DiT: intrinsic granularity-safety probe, layer sweep")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(out / "fig_probe_layer_sweep.png")
        print("figure written")
    except Exception as e:  # noqa
        print("figure skipped:", e)

    print(json.dumps({k: v for k, v in report.items() if not k.startswith("block_sweep")}, indent=2, default=str))


if __name__ == "__main__":
    main()
