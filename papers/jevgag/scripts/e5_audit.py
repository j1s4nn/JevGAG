"""E5 step 2 — does the image-Jev endpoint agree with CLIP subject consistency?
Aggregates per policy and computes Spearman agreement across the 6 f4lite policies.
Writes results/e5_endpoint/e5_audit.json + e5_table.md
"""
import json, os
from datetime import datetime, timezone

NEW = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLD = r"C:\Users\jisan\OneDrive\Desktop\research\idea_05_adaptive_granularity_WM\04_Experiment"
ANS = os.path.join(NEW, "results", "e5_endpoint", "endpoint_answers.jsonl")
CLIP = os.path.join(OLD, "results", "f4lite", "f4lite_quality.json")
OUT = os.path.join(NEW, "results", "e5_endpoint")


def spearman(x, y):
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v); i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
                j += 1
            avg = (i + j) / 2.0
            for k in range(i, j + 1):
                r[o[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else None


def main():
    rows = [json.loads(l) for l in open(ANS, encoding="utf-8") if l.strip() and json.loads(l).get("answer")]
    by = {}
    for r in rows:
        a = r["answer"]["answers"]
        by.setdefault(r["policy"], []).append({
            "same_subject": a["same_subject"]["noul"],
            "consistency": a["consistency"]["score"],
            "scene_drift": a["scene_drift"]["noul"]})
    clip = json.load(open(CLIP, encoding="utf-8"))["by_policy"]
    pol = [p for p in by if p in clip]
    jev_cons = [sum(x["consistency"] for x in by[p]) / len(by[p]) for p in pol]
    jev_same = [sum(x["same_subject"] for x in by[p]) / len(by[p]) for p in pol]
    clip_sc = [clip[p]["subject_consistency"] for p in pol]
    rep = {"phase": "E5-endpoint", "generated_utc": datetime.now(timezone.utc).isoformat(),
           "n_videos": len(rows), "policies": pol,
           "jev_consistency": dict(zip(pol, jev_cons)), "jev_same_subject": dict(zip(pol, jev_same)),
           "clip_subject_consistency": dict(zip(pol, clip_sc)),
           "spearman_jev_consistency_vs_clip": spearman(jev_cons, clip_sc),
           "spearman_jev_same_vs_clip": spearman(jev_same, clip_sc)}
    json.dump(rep, open(os.path.join(OUT, "e5_audit.json"), "w", encoding="utf-8"), indent=2)
    lines = ["# E5 — image-Jev endpoint vs CLIP subject consistency", "",
             "| policy | Jev consistency | Jev same-subject | CLIP subj.cons. |", "|---|---|---|---|"]
    for i, p in enumerate(pol):
        lines.append(f"| {p} | {jev_cons[i]:.3f} | {jev_same[i]:.3f} | {clip_sc[i]:.3f} |")
    lines += ["", f"Spearman(Jev consistency, CLIP) = **{rep['spearman_jev_consistency_vs_clip']:+.3f}**",
              f"Spearman(Jev same-subject, CLIP) = **{rep['spearman_jev_same_vs_clip']:+.3f}**"]
    open(os.path.join(OUT, "e5_table.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
