"""DDAG E4 — error-propagation model (0 GPU).

Question: can the next drift level be predicted from (current drift, policy, position)?
This is the model DALG would plan with. Uses the E3 trajectories (A all-fine, B all-coarse,
C coarse-then-fine) at 81 frames / 6 segments.

Outputs results/e4_propagation.json
"""

import argparse
import json
from pathlib import Path

import numpy as np


def fit_loo(X, y, groups, ridge=1e-3):
    pred = np.zeros(len(y))
    for g in np.unique(groups):
        tr, te = groups != g, groups == g
        Xtr = np.hstack([X[tr], np.ones((tr.sum(), 1))])
        coef = np.linalg.solve(Xtr.T @ Xtr + ridge * np.eye(Xtr.shape[1]), Xtr.T @ y[tr])
        pred[te] = np.hstack([X[te], np.ones((te.sum(), 1))]) @ coef
    return pred


def r2(y, p):
    return float(1 - ((y - p) ** 2).sum() / (((y - y.mean()) ** 2).sum() + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meas", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    d = np.load(args.meas)
    chains = {"A_all_fine": d["A_all_fine"].astype(np.float64),
              "B_all_coarse": d["B_all_coarse"].astype(np.float64),
              "C_coarse_then_fine": d["C_coarse_then_fine"].astype(np.float64)}
    n, S = chains["A_all_fine"].shape

    # build a transition dataset: (d_t, mode_{t+1} in {0=fine,1=coarse}, t) -> d_{t+1}
    X, y, g = [], [], []
    for ci, (name, M) in enumerate(chains.items()):
        modes = {"A_all_fine": [0] * S, "B_all_coarse": [1] * S,
                 "C_coarse_then_fine": [1] + [0] * (S - 1)}[name]
        for p in range(n):
            for t in range(S - 1):
                X.append([M[p, t], float(modes[t + 1]), float(t + 1), float(ci)])
                y.append(M[p, t + 1])
                g.append(p)
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    g = np.asarray(g)

    rep = {"n_transitions": int(len(y)), "n_prompts": int(n), "n_segments": int(S)}

    # M0: persistence (predict d_{t+1} = d_t)
    rep["M0_persistence_r2"] = r2(y, X[:, 0])
    # M1: current drift only
    rep["M1_current_drift_r2"] = r2(y, fit_loo(X[:, [0]], y, g))
    # M2: current drift + next mode
    rep["M2_drift_plus_mode_r2"] = r2(y, fit_loo(X[:, [0, 1]], y, g))
    # M3: full (drift, mode, position, chain)
    rep["M3_full_r2"] = r2(y, fit_loo(X, y, g))
    # increment predictability
    inc = y - X[:, 0]
    rep["increment_mean"] = float(inc.mean())
    rep["increment_std"] = float(inc.std())
    rep["M4_increment_from_mode_r2"] = r2(inc, fit_loo(X[:, [1, 2, 3]], inc, g))

    rep["verdict"] = {
        "next_drift_predictable": rep["M3_full_r2"] > 0.5,
        "mode_matters_beyond_drift": rep["M2_drift_plus_mode_r2"] - rep["M1_current_drift_r2"] > 0.01,
        "increment_predictable": rep["M4_increment_from_mode_r2"] > 0.3,
    }
    (out / "e4_propagation.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
