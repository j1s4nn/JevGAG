"""DDAG E3 — drift accumulation analysis (0 GPU).

Input : measurements.npz with per-prompt drift trajectories for three chains
        A_all_fine, B_all_coarse, C_coarse_then_fine  (shape [n_prompts, n_segments]).
Output: results/e3_drift.json + fig_e3_drift.png

Tests
  H2 (compounding): does coarse drift grow super-linearly, and does a single coarse segment
                    leave a PERSISTENT elevation (propagation) vs the all-fine chain?
  H3 (predictable) : can the next drift increment be predicted from the current state?
"""

import argparse
import json
from pathlib import Path

import numpy as np


def boot_ci(x, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    v = [np.mean(rng.choice(x, size=len(x), replace=True)) for _ in range(n)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def paired_ci(a, b, n=4000, seed=0):
    """bootstrap CI of mean(a - b) over prompts."""
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    v = [np.mean(rng.choice(d, size=len(d), replace=True)) for _ in range(n)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meas", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    d = np.load(args.meas)
    A = d["A_all_fine"].astype(np.float64)
    B = d["B_all_coarse"].astype(np.float64)
    C = d["C_coarse_then_fine"].astype(np.float64)
    n, S = A.shape

    rep = {"n_prompts": int(n), "n_segments": int(S),
           "mean_drift": {"A_all_fine": A.mean(0).tolist(),
                          "B_all_coarse": B.mean(0).tolist(),
                          "C_coarse_then_fine": C.mean(0).tolist()}}

    # H2a: coarse vs fine drift, per segment (paired over prompts)
    rep["per_segment_delta_coarse_minus_fine"] = []
    for t in range(S):
        ci = paired_ci(B[:, t], A[:, t])
        rep["per_segment_delta_coarse_minus_fine"].append(
            {"segment": t + 1, "delta": float((B[:, t] - A[:, t]).mean()), "ci": ci,
             "significant": bool(ci[0] > 0 or ci[1] < 0)})

    # H2b: super-linearity — fit drift_B(t) vs t; compare linear vs quadratic R^2
    t = np.arange(1, S + 1, dtype=float)
    yB = B.mean(0)
    lin = np.polyfit(t, yB, 1)
    quad = np.polyfit(t, yB, 2)

    def r2(p, tt, yy):
        pred = np.polyval(p, tt)
        return float(1 - ((yy - pred) ** 2).sum() / (((yy - yy.mean()) ** 2).sum() + 1e-12))
    rep["superlinearity"] = {"linear_r2": r2(lin, t, yB), "quadratic_r2": r2(quad, t, yB),
                             "quad_coef": float(quad[0]),
                             "superlinear": bool(quad[0] > 0 and r2(quad, t, yB) > r2(lin, t, yB))}

    # H2c: PROPAGATION — after the single coarse segment (t=1), does chain C stay above chain A?
    rep["propagation"] = []
    for t in range(1, S):
        ci = paired_ci(C[:, t], A[:, t])
        rep["propagation"].append({"segment": t + 1, "delta_C_minus_A": float((C[:, t] - A[:, t]).mean()),
                                   "ci": ci, "significant": bool(ci[0] > 0 or ci[1] < 0)})
    tail = [p for p in rep["propagation"]]
    rep["propagation_persistent"] = bool(sum(p["significant"] for p in tail) >= max(1, len(tail) // 2))

    # H3: predict next increment from current drift (leave-one-prompt-out linear fit)
    X = np.concatenate([B[:, :-1].reshape(-1, 1), A[:, :-1].reshape(-1, 1)], axis=1)
    y = (B[:, 1:] - B[:, :-1]).reshape(-1)
    preds = np.zeros(len(y))
    pid = np.repeat(np.arange(n), S - 1)
    for g in range(n):
        tr, te = pid != g, pid == g
        coef, *_ = np.linalg.lstsq(np.hstack([X[tr], np.ones((tr.sum(), 1))]), y[tr], rcond=None)
        preds[te] = np.hstack([X[te], np.ones((te.sum(), 1))]) @ coef
    ss_res = ((y - preds) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    rep["H3_increment_r2"] = float(1 - ss_res / (ss_tot + 1e-12))

    rep["verdict"] = {
        "H2_coarse_worse": rep["per_segment_delta_coarse_minus_fine"][-1]["delta"] > 0,
        "H2_superlinear": rep["superlinearity"]["superlinear"],
        "H2_propagation": rep["propagation_persistent"],
        "H3_predictable_r2>0.3": rep["H3_increment_r2"] > 0.3,
    }
    (out / "e3_drift.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep["verdict"], indent=2))
    print("superlinearity:", rep["superlinearity"])
    print("H3 increment R2:", round(rep["H3_increment_r2"], 3))
    print("mean drift A:", [round(v, 4) for v in rep["mean_drift"]["A_all_fine"]])
    print("mean drift B:", [round(v, 4) for v in rep["mean_drift"]["B_all_coarse"]])
    print("mean drift C:", [round(v, 4) for v in rep["mean_drift"]["C_coarse_then_fine"]])

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=150)
        for k, m in (("A_all_fine", A), ("B_all_coarse", B), ("C_coarse_then_fine", C)):
            mu = m.mean(0)
            se = m.std(0) / np.sqrt(m.shape[0])
            ax.errorbar(range(1, S + 1), mu, yerr=se, marker="o", ms=3, capsize=3, label=k)
        ax.set_xlabel("segment"); ax.set_ylabel("latent L1 to s4 reference (drift)")
        ax.set_title("DDAG E3: drift accumulation (coarse vs fine vs one-coarse-then-fine)")
        ax.grid(alpha=0.25); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(out / "fig_e3_drift.png")
        print("figure written")
    except Exception as e:
        print("figure skipped:", e)


if __name__ == "__main__":
    main()
