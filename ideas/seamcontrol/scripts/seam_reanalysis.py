"""SeamControl E1 — seam re-analysis on reused controller traces (0 GPU).

Reads the audited RQ3 controller rollouts from the base paper project (read-only):
  T4   : idea_05 .../results/rq3_controller/results.jsonl
  A100 : idea_05 .../results/a100_rq3/results.jsonl

Quantifies the switch seam and asks whether the pre-generation context signal (psi)
predicts elevated adjacent-frame L1.

Outputs ideas/seamcontrol/results/e1_seam.json
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

T4 = r"C:\Users\jisan\OneDrive\Desktop\research\idea_05_adaptive_granularity_WM\04_Experiment\results\rq3_controller\results.jsonl"
A100 = r"C:\Users\jisan\OneDrive\Desktop\research\idea_05_adaptive_granularity_WM\04_Experiment\results\a100_rq3\results.jsonl"


def load(path, hw):
    rows = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r.get("controller") in (None, "reference") or r.get("status") != "complete":
            continue
        segs = r.get("segments") or []
        modes = r.get("mode_sequence") or []
        psi = r.get("psi") or []
        if len(segs) != 4 or len(modes) != 4:
            continue
        prev = "C"  # shared coarse prefix
        for i, s in enumerate(segs):
            rows.append({
                "hw": hw, "controller": r["controller"], "prompt": r["prompt"], "stratum": r.get("stratum"),
                "group": s["group"], "mode": modes[i], "switched": int(modes[i] != prev),
                "psi": psi[i] if i < len(psi) else np.nan,
                "prev_psi": psi[i - 1] if i >= 1 and i - 1 < len(psi) else np.nan,
                "adj_l1": s.get("adjacent_pixel_l1"), "ref_l1": s.get("reference_l1"),
            })
            prev = modes[i]
    return rows


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 4:
        return float("nan")
    rx = np.argsort(np.argsort(x[m]))
    ry = np.argsort(np.argsort(y[m]))
    return float(np.corrcoef(rx, ry)[0, 1])


def boot_mean_diff(a, a_groups, b, b_groups, n=3000, seed=0):
    """Prompt-grouped bootstrap of mean(a) - mean(b)."""
    rng = np.random.default_rng(seed)
    a_by = {g: a[a_groups == g] for g in np.unique(a_groups)}
    b_by = {g: b[b_groups == g] for g in np.unique(b_groups)}
    gs = sorted(set(a_by) | set(b_by))
    d = []
    for _ in range(n):
        pick = rng.choice(gs, size=len(gs), replace=True)
        aa = np.concatenate([a_by[g] for g in pick if g in a_by and len(a_by[g])])
        bb = np.concatenate([b_by[g] for g in pick if g in b_by and len(b_by[g])])
        if len(aa) and len(bb):
            d.append(aa.mean() - bb.mean())
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))] if d else [float("nan")] * 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = load(T4, "T4") + load(A100, "A100")
    print(f"rows={len(rows)} (T4={sum(r['hw']=='T4' for r in rows)}, A100={sum(r['hw']=='A100' for r in rows)})")

    rep = {"n_rows": len(rows), "by_hw": {}}
    for hw in ("T4", "A100"):
        rr = [r for r in rows if r["hw"] == hw]
        sw = np.array([r["adj_l1"] for r in rr if r["switched"] == 1], float)
        it = np.array([r["adj_l1"] for r in rr if r["switched"] == 0], float)
        groups = np.array([r["prompt"] for r in rr if r["switched"] in (0, 1)])
        psi = np.array([r["psi"] for r in rr], float)
        adj = np.array([r["adj_l1"] for r in rr], float)
        sw_g = np.array([r["prompt"] for r in rr if r["switched"] == 1])
        it_g = np.array([r["prompt"] for r in rr if r["switched"] == 0])
        ci = boot_mean_diff(sw, sw_g, it, it_g) if len(sw) and len(it) else [float("nan")] * 2
        rep["by_hw"][hw] = {
            "n": len(rr),
            "n_switched_segments": int(len(sw)),
            "mean_adj_l1_switched": float(sw.mean()) if len(sw) else None,
            "mean_adj_l1_interior": float(it.mean()) if len(it) else None,
            "seam_ratio": float(sw.mean() / it.mean()) if len(sw) and len(it) and it.mean() else None,
            "seam_diff_ci": ci,
            "spearman_psi_vs_adj": spearman(psi, adj),
            "mean_ref_l1": float(np.mean([r["ref_l1"] for r in rr])),
        }
        print(hw, json.dumps(rep["by_hw"][hw], indent=1))

    (out / "e1_seam.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print("wrote", out / "e1_seam.json")


if __name__ == "__main__":
    main()
