"""E2 step 4 — fair paired audit: image-1.14 vs text-1.14 (and text-1.13 reference).

Arms (same 80 segments, same questions):
  text13  : old baseline, Jev 1.13, caption state   (read-only prior result)
  text14  : Jev 1.14, caption state                 (this project)
  image14 : Jev 1.14, raw-frame state               (this project)
Fair comparison is image14 vs text14 (model version held fixed).
Writes results/e2_signal/e2_audit.json, results/e2_signal/e2_table.md
"""
import json
import os
import random
from datetime import datetime, timezone

NEW = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLD = r"C:\Users\jisan\OneDrive\Desktop\research\idea_05_adaptive_granularity_WM\04_Experiment"
ARMS = {
    "text13": os.path.join(OLD, "results", "jev_signal", "jev_answers.jsonl"),
    "text14": os.path.join(NEW, "results", "e2_signal", "text14_answers.jsonl"),
    "image14": os.path.join(NEW, "results", "e2_signal", "image_answers.jsonl"),
    "hybrid14": os.path.join(NEW, "results", "e2_signal", "hybrid_answers.jsonl"),
}
OUT = os.path.join(NEW, "results", "e2_signal")
MOTION_SCORE = {"static": 0.0, "moderate": 1.0, "fast": 2.0, "scene_change": 3.0}
BOOTSTRAP, SEED = 5000, 20261026
NAMES = ["jev_difficulty", "jev_drift", "jev_motion", "warp", "motion", "complexity"]


def auroc(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores); i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    rsp = sum(ranks[i] for i in range(len(scores)) if labels[i] == 1)
    return (rsp - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))


def load(p):
    return {r["segment_id"]: r for r in (json.loads(l) for l in open(p, encoding="utf-8") if l.strip()) if r.get("answer")}


def sig(r, name):
    a = r["answer"]["answers"]
    if name == "jev_difficulty":
        return a["difficulty"]["score"]
    if name == "jev_drift":
        return a["coarse_drift"]["noul"]
    if name == "jev_motion":
        return MOTION_SCORE.get(a["motion"]["choice"], 1.0)
    return r[name]


def gboot(s, y, g, rng):
    by = {}
    for a, b, c in zip(s, y, g):
        by.setdefault(c, []).append((a, b))
    ps = list(by); vals = []
    for _ in range(BOOTSTRAP):
        ss, yy = [], []
        for p in rng.choices(ps, k=len(ps)):
            for a, b in by[p]:
                ss.append(a); yy.append(b)
        v = auroc(ss, yy)
        if v is not None:
            vals.append(v)
    vals.sort()
    return (vals[int(.025 * len(vals))], vals[int(.975 * len(vals)) - 1]) if vals else (None, None)


def paired(sa, sb, y, g, rng):
    by = {}
    for a, b, c, d in zip(sa, sb, y, g):
        by.setdefault(d, []).append((a, b, c))
    ps = list(by); diffs = []
    for _ in range(BOOTSTRAP):
        aa, bb, yy = [], [], []
        for p in rng.choices(ps, k=len(ps)):
            for x, z, c in by[p]:
                aa.append(x); bb.append(z); yy.append(c)
        va, vb = auroc(aa, yy), auroc(bb, yy)
        if va is not None and vb is not None:
            diffs.append(vb - va)
    diffs.sort()
    return (diffs[int(.025 * len(diffs))], diffs[int(.975 * len(diffs)) - 1],
            sum(1 for d in diffs if d > 0) / len(diffs)) if diffs else (None, None, None)


def main():
    data = {k: load(v) for k, v in ARMS.items()}
    ids = sorted(set(data["image14"]) & set(data["text14"]) & set(data["text13"]))
    labels = [data["image14"][i]["coarse_unsafe"] for i in ids]
    groups = [data["image14"][i]["prompt_idx"] for i in ids]
    rng = random.Random(SEED)
    rep = {"phase": "E2-fair", "generated_utc": datetime.now(timezone.utc).isoformat(),
           "n": len(ids), "positives": sum(labels), "arms": {}}
    for arm, store in data.items():
        for n in NAMES:
            s = [sig(store[i], n) for i in ids]
            a = auroc(s, labels); lo, hi = gboot(s, labels, groups, rng)
            rep["arms"][f"{arm}_{n}"] = {"auroc": a, "ci_low": lo, "ci_high": hi}
    rep["paired"] = {}
    for n in ["jev_difficulty", "jev_drift", "jev_motion"]:
        t = [sig(data["text14"][i], n) for i in ids]
        im = [sig(data["image14"][i], n) for i in ids]
        lo, hi, pw = paired(t, im, labels, groups, rng)
        rep["paired"][n] = {"delta_image_minus_text14": rep["arms"][f"image14_{n}"]["auroc"] - rep["arms"][f"text14_{n}"]["auroc"],
                            "ci_low": lo, "ci_high": hi, "p_image_gt_text": pw}
    json.dump(rep, open(os.path.join(OUT, "e2_audit.json"), "w", encoding="utf-8"), indent=2)

    lines = [f"# E2 fair audit (n={len(ids)} paired segments)", "",
             "| signal | text-1.13 | text-1.14 | image-1.14 | hybrid-1.14 |", "|---|---|---|---|---|"]
    for n in NAMES:
        g = lambda a: rep["arms"][f"{a}_{n}"]["auroc"]
        lines.append(f"| {n} | {g('text13'):.3f} | {g('text14'):.3f} | {g('image14'):.3f} | {g('hybrid14'):.3f} |")
    lines += ["", "## Paired delta (image-1.14 - text-1.14)", "",
              "| signal | delta | 95% CI | P(image>text) |", "|---|---|---|---|"]
    for n, d in rep["paired"].items():
        lines.append(f"| {n} | {d['delta_image_minus_text14']:+.3f} | [{d['ci_low']:+.3f},{d['ci_high']:+.3f}] | {d['p_image_gt_text']:.3f} |")
    lines += ["", "## Paired delta (hybrid-1.14 - text-1.14)", ""]
    rng2 = random.Random(SEED + 1)
    for n in ["jev_difficulty", "jev_drift", "jev_motion"]:
        t = [sig(data["text14"][i], n) for i in ids]
        hy = [sig(data["hybrid14"][i], n) for i in ids]
        lo, hi, pw = paired(t, hy, labels, groups, rng2)
        lines.append(f"| {n} | {rep['arms'][f'hybrid14_{n}']['auroc'] - rep['arms'][f'text14_{n}']['auroc']:+.3f} | [{lo:+.3f},{hi:+.3f}] | {pw:.3f} |")
    open(os.path.join(OUT, "e2_table.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
