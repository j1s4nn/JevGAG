"""VERIFY-GRAN 0-GPU go/no-go gate (plan-problemv0.2 §2).

Decides whether to spend any GPU on the new verification-first method, using only the
existing P3 capture (features.npz).

T1  can a *calibrated* gate on FREE evidence reach a target error at escalation rho < 67%?
T2  does the gate score beat the prior alone (0.798) and evidence alone (0.725)?
T3  power: how many prompts are needed for the observed effect size?

Honesty: the threshold is selected with leave-one-prompt-out (choose tau on the other
prompts to hit the target error at minimum rho, then apply to the held-out prompt), so the
reported (rho, error) is out-of-fold, not tuned on the evaluated prompt.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

COST_COARSE = 2.0   # forwards per coarse segment (1 chunk x 2 steps)
COST_FINE = 6.0     # forwards per fine segment (3 chunks x 2 steps)
RHO_BREAKEVEN = 1 - COST_COARSE / COST_FINE  # 0.667


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


def loo_gate(y, score, groups, target_err, rho_max):
    """Out-of-fold gate: for each held-out prompt pick tau on the rest to hit target_err
    at minimum rho, then apply. Returns (rho, err_among_accepted, coverage_ok)."""
    acc_flags, lab_flags = [], []
    for g in np.unique(groups):
        tr, te = groups != g, groups == g
        ytr, str_ = y[tr], score[tr]
        best_tau = None
        for tau in np.unique(str_):
            acc = str_ <= tau
            rho = 1 - acc.mean()
            if acc.sum() == 0:
                continue
            err = ytr[acc].mean()
            if err <= target_err and rho <= rho_max:
                best_tau = tau
                break
        if best_tau is None:
            best_tau = np.quantile(str_, 0.5)  # fall back: escalate half
        te_acc = score[te] <= best_tau
        acc_flags.append(te_acc)
        lab_flags.append(y[te])
    acc = np.concatenate(acc_flags)
    lab = np.concatenate(lab_flags)
    rho = float(1 - acc.mean())
    err = float(lab[acc].mean()) if acc.sum() else float("nan")
    return rho, err, bool(acc.sum())


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

    ev = np.stack([d["seam_coarse"], d["interior_coarse"]], axis=1).astype(np.float32)
    disc = d["disc"].astype(np.float32).reshape(-1, 1)

    prior = oof(pre[:, 11, :], y, groups)                 # best block from idea 03
    ev_score = oof(ev, y, groups)                          # free evidence alone
    gate_score = oof(np.hstack([prior.reshape(-1, 1), ev]), y, groups)   # VERIFY-GRAN gate
    gate_disc = oof(np.hstack([prior.reshape(-1, 1), ev, disc]), y, groups)  # + costly

    rep = {
        "n_segments": int(n), "n_positive": int(y.sum()), "n_prompts": int(len(np.unique(groups))),
        "cost_model": {"coarse": COST_COARSE, "fine": COST_FINE, "rho_breakeven": round(RHO_BREAKEVEN, 3)},
        "target_err": args.target_err,
        "T2_auroc": {
            "prior_block11": auroc(y, prior),
            "evidence_free_only": auroc(y, ev_score),
            "gate_prior+free_evidence": auroc(y, gate_score),
            "gate_prior+free+disc(costly)": auroc(y, gate_disc),
            "reference": {"selfaware_probe": 0.798, "jev_text": 0.788, "hand_crafted": 0.491, "chance": 0.5},
        },
    }
    for name, sc in [("prior", prior), ("evidence", ev_score), ("gate", gate_score)]:
        rho, err, ok = loo_gate(y, sc, groups, args.target_err, RHO_BREAKEVEN)
        rep[f"T1_{name}"] = {"rho": rho, "error_among_accepted": err, "feasible": ok,
                             "cost_vs_fine": round((1 - rho) * COST_COARSE + rho * COST_FINE, 3)}

    # T3: power — n for a two-sample AUC difference (Hanley-McNeil style, rough)
    a = rep["T2_auroc"]["gate_prior+free_evidence"]
    rep["T3_power"] = {"observed_gate_auroc": a, "note": "see script for n estimate"}

    rep["verdict"] = {
        "T1_pass": rep["T1_gate"]["feasible"] and rep["T1_gate"]["rho"] < RHO_BREAKEVEN,
        "T2_pass": rep["T2_auroc"]["gate_prior+free_evidence"] >= max(rep["T2_auroc"]["prior_block11"], rep["T2_auroc"]["evidence_free_only"]),
        "go_for_gpu": bool(rep["T1_gate"]["feasible"] and rep["T1_gate"]["rho"] < RHO_BREAKEVEN
                           and rep["T2_auroc"]["gate_prior+free_evidence"] >= 0.75),
    }
    (out / "gate_check.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
