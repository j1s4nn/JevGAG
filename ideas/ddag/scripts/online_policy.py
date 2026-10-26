"""DDAG E5 — online granularity policy from captured per-segment cost + quality (0 GPU).

The capture ran BOTH modes for every segment (matched design), so every policy is computable
offline from the recorded per-segment DiT time and latent refL1.

Policies
  fixed_fine      : always fine
  fixed_coarse    : always coarse
  probe_route     : choose fine if the intrinsic probe says "unsafe", else coarse   (pre-generation)
  probe_escalate  : always coarse, then recompute in fine if the probe says "unsafe"
  oracle_route    : choose fine iff the segment is truly coarse-unsafe (upper bound)
  oracle_escalate : always coarse, recompute in fine iff truly coarse-unsafe (upper bound)

The probe score is computed with leave-one-prompt-out CV (grouped by prompt), so no segment is
scored by a model that saw its own prompt. tau is swept over the score quantiles to trace a frontier.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--block", type=int, default=11)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    d = np.load(args.features)
    y = d["labels"].astype(np.int64)                      # 1 = coarse-unsafe
    g = d["prompt_idx"].astype(np.int64)
    pre = d["prefill"].astype(np.float32)
    cf = d["dit_fine_s"].astype(np.float64)               # DiT seconds, fine segment
    cc = d["dit_coarse_s"].astype(np.float64)             # DiT seconds, coarse segment
    qf = d["ll1_fine"].astype(np.float64)                 # quality proxy (lower better)
    qc = d["ll1_coarse"].astype(np.float64)
    n = len(y)

    score = oof(pre[:, args.block, :], y, g)              # higher = more likely unsafe
    auroc = float(roc_auc_score(y, score)) if len(np.unique(y)) > 1 else float("nan")

    rep = {"n_segments": int(n), "n_prompts": int(len(np.unique(g))), "n_positive": int(y.sum()),
           "probe_block": args.block, "probe_auroc": auroc,
           "mean_dit_fine_s": float(cf.mean()), "mean_dit_coarse_s": float(cc.mean()),
           "cost_ratio_coarse_over_fine": float(cc.mean() / cf.mean()),
           "mean_q_fine": float(qf.mean()), "mean_q_coarse": float(qc.mean())}

    def policy(cost, qual):
        return {"dit_total_s": float(cost.sum()), "dit_mean_s": float(cost.mean()),
                "quality_mean": float(qual.mean()), "n_fine": int(np.sum(qual == qf))}

    rep["fixed_fine"] = policy(cf, qf)
    rep["fixed_coarse"] = policy(cc, qc)
    rep["oracle_route"] = policy(np.where(y == 1, cf, cc), np.where(y == 1, qf, qc))
    rep["oracle_escalate"] = policy(cc + np.where(y == 1, cf, 0.0), np.where(y == 1, qf, qc))

    base_fine_cost = rep["fixed_fine"]["dit_total_s"]
    base_fine_q = rep["fixed_fine"]["quality_mean"]

    frontier = []
    for q in np.linspace(0.0, 1.0, 41):
        tau = np.quantile(score, q)
        esc = score > tau
        r_cost = np.where(esc, cf, cc)
        r_q = np.where(esc, qf, qc)
        e_cost = cc + np.where(esc, cf, 0.0)
        e_q = np.where(esc, qf, qc)
        frontier.append({
            "quantile": float(q), "tau": float(tau), "escalation_rate": float(esc.mean()),
            "route_dit_total_s": float(r_cost.sum()), "route_quality": float(r_q.mean()),
            "escalate_dit_total_s": float(e_cost.sum()), "escalate_quality": float(e_q.mean()),
            "route_saves_vs_fine": bool(r_cost.sum() < base_fine_cost),
            "route_quality_ok": bool(r_q.mean() <= base_fine_q),
            "escalate_saves_vs_fine": bool(e_cost.sum() < base_fine_cost),
            "escalate_quality_ok": bool(e_q.mean() <= base_fine_q),
        })
    rep["frontier"] = frontier

    wins_route = [f for f in frontier if f["route_saves_vs_fine"] and f["route_quality_ok"]]
    wins_esc = [f for f in frontier if f["escalate_saves_vs_fine"] and f["escalate_quality_ok"]]
    rep["best_route"] = min(wins_route, key=lambda f: f["route_dit_total_s"]) if wins_route else None
    rep["best_escalate"] = min(wins_esc, key=lambda f: f["escalate_dit_total_s"]) if wins_esc else None
    rep["verdict"] = {
        "coarse_cheaper": rep["cost_ratio_coarse_over_fine"] < 1.0,
        "probe_route_beats_fixed_fine": bool(rep["best_route"]),
        "probe_escalate_beats_fixed_fine": bool(rep["best_escalate"]),
        "route_saving_pct": (100.0 * (base_fine_cost - rep["best_route"]["route_dit_total_s"]) / base_fine_cost) if rep["best_route"] else 0.0,
    }
    (out / "e5_online.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k != "frontier"}, indent=2)[:2500])


if __name__ == "__main__":
    main()
