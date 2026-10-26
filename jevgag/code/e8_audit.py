"""E8 (Kaggle T4) audit — recalibration vs fixed and adaptive baselines.

Reads results/e8_recalibration/results/e8_recalibration/results.jsonl.
Policies: fixed_coarse, fixed_fine, adaptive_warp, recalibrate.
Writes results/e8_recalibration/e8_table.md + e8.json
"""
import json, os, statistics
from datetime import datetime, timezone

NEW = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(NEW, "results", "e8_recalibration", "results", "e8_recalibration", "results.jsonl")
OUT = os.path.join(NEW, "results", "e8_recalibration")
POLICIES = ["fixed_coarse", "fixed_fine", "adaptive_warp", "recalibrate"]


def main():
    rows = [json.loads(l) for l in open(SRC, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r.get("status") == "complete" and r.get("policy") in POLICIES]
    by = {}
    for r in rows:
        by.setdefault(r["policy"], []).append(r)
    rep = {"phase": "E8-Kaggle-T4", "generated_utc": datetime.now(timezone.utc).isoformat(),
           "n_prompts": len(by.get("fixed_fine", [])), "policies": {}}
    for p in POLICIES:
        rs = by.get(p, [])
        if not rs:
            continue
        rep["policies"][p] = {
            "mean_reference_l1": round(statistics.fmean(r["mean_reference_l1"] for r in rs), 4),
            "mean_fine_fraction": round(statistics.fmean(r["fine_fraction"] for r in rs), 3),
            "mean_switch_count": round(statistics.fmean(r["switch_count"] for r in rs), 3),
            "mean_dit_s": round(statistics.fmean(r["timing"]["dit_s"] for r in rs), 2)}
    json.dump(rep, open(os.path.join(OUT, "e8.json"), "w", encoding="utf-8"), indent=2)
    lines = [f"# E8 (Kaggle T4) — recalibration (n={rep['n_prompts']} prompts)", "",
             "| policy | mean refL1 | fine frac | switches | DiT s |", "|---|---|---|---|---|"]
    for p in POLICIES:
        if p in rep["policies"]:
            v = rep["policies"][p]
            lines.append(f"| {p} | {v['mean_reference_l1']:.4f} | {v['mean_fine_fraction']:.2f} | "
                         f"{v['mean_switch_count']:.2f} | {v['mean_dit_s']:.2f} |")
    if "adaptive_warp" in rep["policies"] and "recalibrate" in rep["policies"]:
        dq = rep["policies"]["recalibrate"]["mean_reference_l1"] - rep["policies"]["adaptive_warp"]["mean_reference_l1"]
        dc = rep["policies"]["recalibrate"]["mean_dit_s"] - rep["policies"]["adaptive_warp"]["mean_dit_s"]
        lines += ["", "## Verdict", "",
                  f"- recalibrate minus adaptive_warp: refL1 {dq:+.4f}, DiT {dc:+.2f} s.",
                  ("- **Recalibration does NOT help** (worse refL1 at higher cost)." if dq >= 0
                   else "- Recalibration reduces refL1.")]
    open(os.path.join(OUT, "e8_table.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
