"""SeamControl E2 — analysis of the boundary-seam / repair measurements (0 GPU).

Input : measurements.npz from the repair kernel
Output: results/e2_repair.json, results/fig_seam_repair.png

Variants: none | blend(a in {0.25,0.5,0.75}) : x1 <- (1-a)x1 + a*b | anchor : x1 <- b
where b is the boundary latent frame. Metrics: boundary seam L1(b, x1) and fidelity
(latent L1 of the 3-frame segment to the c1-1 s4 reference).
"""

import argparse
import json
from pathlib import Path

import numpy as np


def boot_ci(x, groups, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    by = {g: x[groups == g] for g in np.unique(groups)}
    gs = sorted(by)
    vals = []
    for _ in range(n):
        pick = rng.choice(gs, size=len(gs), replace=True)
        cat = np.concatenate([by[g] for g in pick])
        vals.append(cat.mean())
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def boot_delta_ci(a, b, groups, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    ga = {g: a[groups == g] for g in np.unique(groups)}
    gb = {g: b[groups == g] for g in np.unique(groups)}
    gs = sorted(set(ga) | set(gb))
    vals = []
    for _ in range(n):
        pick = rng.choice(gs, size=len(gs), replace=True)
        aa = np.concatenate([ga[g] for g in pick if g in ga])
        bb = np.concatenate([gb[g] for g in pick if g in gb])
        vals.append(aa.mean() - bb.mean())
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meas", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    d = np.load(args.meas)
    variants = [str(v) for v in d["variants"]]
    seams, fids = d["seams"], d["fids"]
    groups = d["prompt_idx"]
    n = seams.shape[1]
    idx_none = variants.index("none")

    rep = {"n_segments": int(n), "n_prompts": int(len(np.unique(groups))), "variants": {}}
    for i, v in enumerate(variants):
        rep["variants"][v] = {
            "seam_mean": float(seams[i].mean()),
            "seam_ci": boot_ci(seams[i], groups),
            "fidelity_mean": float(fids[i].mean()),
            "fidelity_ci": boot_ci(fids[i], groups),
            "seam_delta_vs_none": float(seams[i].mean() - seams[idx_none].mean()),
            "seam_delta_ci": boot_delta_ci(seams[i], seams[idx_none], groups),
            "fidelity_delta_vs_none": float(fids[i].mean() - fids[idx_none].mean()),
            "fidelity_delta_ci": boot_delta_ci(fids[i], fids[idx_none], groups),
        }
    # fine-mode reference seam (no switch -> what fine's own boundary looks like)
    rep["reference"] = {
        "seam_fine_mean": float(d["seam_fine"].mean()),
        "interior_fine_mean": float(d["interior_fine"].mean()),
        "seam_coarse_mean": float(d["seam_coarse"].mean()),
        "interior_coarse_mean": float(d["interior_coarse"].mean()),
        "fid_fine_mean": float(d["fid_fine"].mean()),
        "fid_coarse_mean": float(d["fid_coarse"].mean()),
    }
    (out / "e2_repair.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")

    print(json.dumps(rep, indent=2))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=150)
        for i, v in enumerate(variants):
            ax.scatter(seams[i].mean(), fids[i].mean(), s=60, label=v, zorder=3)
        ax.errorbar(seams[idx_none].mean(), fids[idx_none].mean(), xerr=[[seams[idx_none].mean() - rep["variants"]["none"]["seam_ci"][0]], [rep["variants"]["none"]["seam_ci"][1] - seams[idx_none].mean()]], fmt="none", ecolor="gray", zorder=1)
        ax.annotate("blend0.25 dominates none", xy=(seams[1].mean(), fids[1].mean()),
                    xytext=(seams[1].mean() + 0.06, fids[1].mean() - 0.02),
                    arrowprops=dict(arrowstyle="->", color="#1668d6"), fontsize=8, color="#1668d6")
        ax.set_xlabel("boundary seam  L1(b, first frame)  (lower better)")
        ax.set_ylabel("fidelity  L1(segment, s4 ref)  (lower better)")
        ax.set_title("SeamControl: training-free seam repair frontier")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "fig_seam_repair.png")
        print("figure written")
    except Exception as e:
        print("figure skipped:", e)


if __name__ == "__main__":
    main()
