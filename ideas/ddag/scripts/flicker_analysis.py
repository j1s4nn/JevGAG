import argparse, json
from pathlib import Path
import numpy as np


def boot_ci(x, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    v = [np.mean(rng.choice(x, size=len(x), replace=True)) for _ in range(n)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def paired_ci(a, b, n=4000, seed=0):
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    v = [np.mean(rng.choice(d, size=len(d), replace=True)) for _ in range(n)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


ap = argparse.ArgumentParser()
ap.add_argument("--meas", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
d = np.load(a.meas)
variants = [str(v) for v in d["variants"]]
seams, fids = d["seams"], d["fids"]
fl = {k: d[f"flicker_{k}"].astype(np.float64) for k in ("fine", "none", "blend0.25")}

rep = {"n_segments": int(seams.shape[1]), "variants": {}}
for i, v in enumerate(variants):
    rep["variants"][v] = {"seam": float(seams[i].mean()), "fidelity": float(fids[i].mean())}
rep["seam_delta_blend0.25"] = float(seams[1].mean() - seams[0].mean())
rep["fidelity_delta_blend0.25"] = float(fids[1].mean() - fids[0].mean())
rep["fidelity_delta_ci_blend0.25"] = paired_ci(fids[1], fids[0])
rep["flicker"] = {}
for k, x in fl.items():
    rep["flicker"][k] = {"mean": float(x.mean()), "ci": boot_ci(x)}
rep["flicker_delta_none_minus_fine"] = float(fl["none"].mean() - fl["fine"].mean())
rep["flicker_delta_none_minus_fine_ci"] = paired_ci(fl["none"], fl["fine"])
rep["flicker_delta_blend_minus_none"] = float(fl["blend0.25"].mean() - fl["none"].mean())
rep["flicker_delta_blend_minus_none_ci"] = paired_ci(fl["blend0.25"], fl["none"])
rep["verdict"] = {
    "seam_reduced": rep["seam_delta_blend0.25"] < 0,
    "fidelity_not_worse": rep["fidelity_delta_blend0.25"] <= 0,
    "coarse_flickers_more_than_fine": rep["flicker_delta_none_minus_fine"] > 0,
    "blend_reduces_flicker": rep["flicker_delta_blend_minus_none"] < 0,
}
(out / "e8_flicker.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
print(json.dumps({k: round(v, 5) if isinstance(v, float) else v for k, v in rep.items() if k != "variants"}, indent=2))
print("variants:", json.dumps({k: {kk: round(vv, 5) for kk, vv in v.items()} for k, v in rep["variants"].items()}))
